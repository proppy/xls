// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/ir/llvm_ir_jit.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_evaluator_test.h"
#include "re2/re2.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

INSTANTIATE_TEST_SUITE_P(
    LlvmIrJitTest, IrEvaluatorTest,
    testing::Values(IrEvaluatorTestParam(
        [](Function* function,
           const std::vector<Value>& args) -> xabsl::StatusOr<Value> {
          XLS_ASSIGN_OR_RETURN(auto jit, LlvmIrJit::Create(function));
          return jit->Run(args);
        },
        [](Function* function,
           const absl::flat_hash_map<std::string, Value>& kwargs)
            -> xabsl::StatusOr<Value> {
          XLS_ASSIGN_OR_RETURN(auto jit, LlvmIrJit::Create(function));
          return jit->Run(kwargs);
        })));

// This test verifies that a compiled JIT function can be re-used.
TEST(LlvmIrJitTest, ReuseTest) {
  Package package("my_package");
  std::string ir_text = R"(
  fn get_identity(x: bits[8]) -> bits[8] {
    ret identity.1: bits[8] = identity(x)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, LlvmIrJit::Create(function));
  EXPECT_THAT(jit->Run({Value(UBits(2, 8))}), IsOkAndHolds(Value(UBits(2, 8))));
  EXPECT_THAT(jit->Run({Value(UBits(4, 8))}), IsOkAndHolds(Value(UBits(4, 8))));
  EXPECT_THAT(jit->Run({Value(UBits(7, 8))}), IsOkAndHolds(Value(UBits(7, 8))));
}

// Verifies that the QuickCheck mechanism can find counter-examples for a simple
// erroneous function.
//
// Chances of this succeeding erroneously are (1/2)^1000.
TEST(LlvmIrJitTest, QuickCheckBits) {
  Package package("bad_bits_property");
  std::string ir_text = R"(
  fn adjacent_bits(x: bits[2]) -> bits[1] {
    first_bit: bits[1] = bit_slice(x, start=0, width=1)
    second_bit: bits[1] = bit_slice(x, start=1, width=1)
    ret eq_value: bits[1] = eq(first_bit, second_bit)
  }
  )";
  int64 seed = 0;
  int64 num_tests = 1000;
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto quickcheck_info,
                           CreateAndQuickCheck(function, seed, num_tests));
  std::vector<Value> results = quickcheck_info.second;
  // If a counter-example was found, the last result will be 0.
  EXPECT_EQ(results.back(), Value(UBits(0, 1)));
}

// Chances of this succeeding erroneously are (1/256)^1000.
TEST(LlvmIrJitTest, QuickCheckArray) {
  Package package("bad_array_property");
  std::string ir_text = R"(
  fn adjacent_elements(x: bits[8][5]) -> bits[1] {
    index.0: bits[32] = literal(value=0)
    index.1: bits[32] = literal(value=1)
    first_element: bits[8] = array_index(x, index.0)
    second_element: bits[8] = array_index(x, index.1)
    ret eq_value: bits[1] = eq(first_element, second_element)
  }
  )";
  int64 seed = 0;
  int64 num_tests = 1000;
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto quickcheck_info,
                           CreateAndQuickCheck(function, seed, num_tests));
  std::vector<Value> results = quickcheck_info.second;
  EXPECT_EQ(results.back(), Value(UBits(0, 1)));
}

// Chances of this succeeding erroneously are (1/256)^1000.
TEST(LlvmIrJitTest, QuickCheckTuple) {
  Package package("bad_tuple_property");
  std::string ir_text = R"(
  fn adjacent_elements(x: (bits[8], bits[8])) -> bits[1] {
    first_member: bits[8] = tuple_index(x, index=0)
    second_member: bits[8] = tuple_index(x, index=1)
    ret eq_value: bits[1] = eq(first_member, second_member)
  }
  )";
  int64 seed = 0;
  int64 num_tests = 1000;
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto quickcheck_info,
                           CreateAndQuickCheck(function, seed, num_tests));
  std::vector<Value> results = quickcheck_info.second;
  EXPECT_EQ(results.back(), Value(UBits(0, 1)));
}

// If the QuickCheck mechanism can't find a falsifying example, we expect
// the argsets and results vectors to have lengths of 'num_tests'.
TEST(LlvmIrJitTest, NumTests) {
  Package package("always_true");
  std::string ir_text = R"(
  fn ret_true(x: bits[32]) -> bits[1] {
    ret eq_value: bits[1] = eq(x, x)
  }
  )";
  int64 seed = 0;
  int64 num_tests = 5050;
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto quickcheck_info,
                           CreateAndQuickCheck(function, seed, num_tests));

  std::vector<std::vector<Value>> argsets = quickcheck_info.first;
  std::vector<Value> results = quickcheck_info.second;
  EXPECT_EQ(argsets.size(), 5050);
  EXPECT_EQ(results.size(), 5050);
}

// Given a constant seed, we expect the same argsets and results vectors from
// two runs through the QuickCheck mechanism.
//
// We expect this test to fail with a probability of (1/128)^1000.
TEST(LlvmIrJitTest, Seeding) {
  Package package("sometimes_false");
  std::string ir_text = R"(
  fn gt_one(x: bits[8]) -> bits[1] {
    literal.2: bits[8] = literal(value=1)
    ret ugt.3: bits[1] = ugt(x, literal.2)
  }
  )";
  int64 seed = 12345;
  int64 num_tests = 1000;
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto quickcheck_info1,
                           CreateAndQuickCheck(function, seed, num_tests));
  XLS_ASSERT_OK_AND_ASSIGN(auto quickcheck_info2,
                           CreateAndQuickCheck(function, seed, num_tests));

  std::vector<std::vector<Value>> argsets1 = quickcheck_info1.first;
  std::vector<Value> results1 = quickcheck_info1.second;

  std::vector<std::vector<Value>> argsets2 = quickcheck_info2.first;
  std::vector<Value> results2 = quickcheck_info2.second;

  EXPECT_EQ(argsets1, argsets2);
  EXPECT_EQ(results1, results2);
}

}  // namespace
}  // namespace xls
