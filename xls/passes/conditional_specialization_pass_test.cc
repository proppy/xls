// Copyright 2021 The XLS Authors
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

#include "xls/passes/conditional_specialization_pass.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::A;
using ::testing::Optional;

class ConditionalSpecializationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(FunctionBase* f, bool use_bdd = true,
                           bool optimize_for_best_case_throughput = false) {
    PassResults results;
    OptimizationPassOptions options;
    options.optimize_for_best_case_throughput =
        optimize_for_best_case_throughput;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(
        bool changed, ConditionalSpecializationPass(use_bdd).RunOnFunctionBase(
                          f, options, &results, &context));
    return changed;
  }
  absl::StatusOr<bool> Run(Package* p, bool use_bdd = true,
                           bool optimize_for_best_case_throughput = false) {
    PassResults results;
    OptimizationPassOptions options;
    options.optimize_for_best_case_throughput =
        optimize_for_best_case_throughput;
    OptimizationContext context;
    return ConditionalSpecializationPass(use_bdd).Run(p, options, &results,
                                                      &context);
  }
};

TEST_F(ConditionalSpecializationPassTest, SpecializeSelectSimple) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[1], b: bits[31], z: bits[32]) -> bits[32] {
  concat: bits[32] = concat(a, b)
  ret sel.2: bits[32] = sel(a, cases=[z, concat])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("a"),
                {m::Param("z"), m::Concat(m::Literal(1), m::Param("b"))}));
}

TEST_F(ConditionalSpecializationPassTest, SpecializeSelectMultipleBranches) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  add.1: bits[32] = add(a, x)
  add.2: bits[32] = add(a, y)
  add.3: bits[32] = add(a, z)
  ret sel.4: bits[32] = sel(a, cases=[add.1, add.2, add.3], default=a)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("a"),
                        {m::Add(m::Literal(0), m::Param("x")),
                         m::Add(m::Literal(1), m::Param("y")),
                         m::Add(m::Literal(2), m::Param("z"))},
                        m::Param("a")));
}

TEST_F(ConditionalSpecializationPassTest, SpecializeSelectSelectorExpression) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32], x: bits[1]) -> bits[1] {
  literal.1: bits[32] = literal(value=7)
  ult.2: bits[1] = ult(a, literal.1)
  not.3: bits[1] = not(ult.2)
  ret sel.4: bits[1] = sel(ult.2, cases=[not.3, x])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::ULt(m::Param("a"), m::Literal(7)),
                                           {m::Literal(1), m::Param("x")}));
}

TEST_F(ConditionalSpecializationPassTest, SpecializeSelectNegative0) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32], x: bits[32], y: bits[32]) -> bits[32] {
  not.1: bits[32] = not(a)
  add.2: bits[32] = add(not.1, x)
  add.3: bits[32] = add(not.1, y)
  ret sel.4: bits[32] = sel(a, cases=[add.2, add.3], default=a)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("a"),
                        /*cases=*/
                        {m::Add(m::Literal(0xffffffff), m::Param("x")),
                         m::Add(m::Literal(0xfffffffe), m::Param("y"))},
                        /*default_value=*/m::Param("a")));
}

TEST_F(ConditionalSpecializationPassTest, SpecializeSelectNegative1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32], x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(a, y)
  sel.2: bits[32] = sel(a, cases=[x, add.1], default=a)
  ret add.3: bits[32] = add(add.1, sel.2)
}
  )",
                                                       p.get()));
  // Similar to the negative test above, the select arm could be specialized
  // by creating a separate copy of the add.1 Node to be used in the return
  // value, and then replacing only the one used in the select arm.
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest,
       SpecializeSelectWithDuplicateCaseArms) {
  // If an expression is used as more than one arm of the select it should not
  // be transformed because the same expression is used for multiple case
  // values.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32], y: bits[32]) -> bits[32] {
  add: bits[32] = add(a, y)
  ret sel: bits[32] = sel(a, cases=[add, add], default=a)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, Consecutive2WaySelects) {
  //
  //  a   b
  //   \ /
  //   sel1 ----+-- p         a
  //    |       |       =>    |
  //    |  c    |             |  c
  //    | /     |             | /
  //   sel0 ----+            sel0 ----- p
  //    |                     |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue pred = fb.Param("pred", p->GetBitsType(1));

  BValue sel1 = fb.Select(pred, {a, b});
  fb.Select(pred, {sel1, c});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Select(m::Param("pred"),
                                           /*cases=*/
                                           {m::Param("a"), m::Param("c")}));
}

TEST_F(ConditionalSpecializationPassTest,
       Consecutive2WaySelectsWithImplicitUse) {
  //
  //    a
  //    |  c
  //    | /
  //   sel1 ----+ p
  //    |       |
  //   neg      |
  //    |       |
  //    |  d    |
  //    | /     |
  //   sel0 ----+
  //    |
  //
  // Where neg and sel0 are next-state elements of a proc. This should prevent
  // any transformations.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  Type* u1 = p->GetBitsType(1);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a,
      p->CreateStreamingChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_b,
      p->CreateStreamingChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_c,
      p->CreateStreamingChannel("c", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_pred,
      p->CreateStreamingChannel("pred", ChannelOps::kReceiveOnly, u1));

  TokenlessProcBuilder pb(TestName(), "tkn", p.get());
  BValue a = pb.Receive(ch_a);
  BValue b = pb.Receive(ch_b);
  BValue c = pb.Receive(ch_c);
  BValue pred = pb.Receive(ch_pred);

  pb.StateElement("st0", Value(UBits(0, 32)));
  pb.StateElement("st1", Value(UBits(0, 32)));

  BValue sel1 = pb.Select(pred, {a, b});
  BValue neg = pb.Negate(sel1);
  BValue sel0 = pb.Select(pred, {neg, c});

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({sel0, neg}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, Consecutive2WaySelectsCase2) {
  //
  //    a   b
  //     \ /
  //     sel1 -+-- p          b
  //      |    |              |
  //   c  |    |      =>   c  |
  //    \ |    |            \ |
  //     sel0 -+             sel0 ---- p
  //      |                   |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue pred = fb.Param("pred", p->GetBitsType(1));

  BValue sel1 = fb.Select(pred, {a, b});
  fb.Select(pred, {c, sel1});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("pred"),
                        /*cases=*/{m::Param("c"), m::Param("b")}));
}

TEST_F(ConditionalSpecializationPassTest, DuplicateArmSpecialization) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(s: bits[1], x: bits[8], y: bits[8]) -> bits[8] {
   sel0: bits[8] = sel(s, cases=[x,y])
   neg_sel0: bits[8] = neg(sel0)
   sel1: bits[8] = sel(s, cases=[neg_sel0, y])
   neg_sel1: bits[8] = neg(sel1)
   ret sel2: bits[8] = sel(s, cases=[neg_sel1, y])
}
  )",
                                                       p.get()));
  // 's' operand of sel0 can be specialized 0 due to sel1 *and* sel2 arm
  // specialization.  This should not cause a crash.
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("s"), {m::Neg(m::Neg(m::Param("x"))), m::Param("y")}));
}

TEST_F(ConditionalSpecializationPassTest, SpecializePrioritySelectSimple) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[2], b: bits[30], case0: bits[32], case_d: bits[32]) -> bits[32] {
  case1: bits[32] = concat(a, b)
  ret priority_sel.2: bits[32] = priority_sel(a, cases=[case0, case1], default=case_d)
}
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::Param("a"),
                        /*cases=*/
                        {m::Param("case0"),
                         m::Concat(m::Literal("bits[2]:0b10"), m::Param("b"))},
                        /*default_value=*/m::Param("case_d")));
}

TEST_F(ConditionalSpecializationPassTest,
       SpecializePrioritySelectMultipleBranches) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[3], x: bits[32], y: bits[32], z: bits[32], q: bits[32]) -> bits[32] {
  a_p0: bits[1] = bit_slice(a, start=0, width=1)
  addend0: bits[32] = zero_ext(a_p0, new_bit_count=32)
  a_p1: bits[2] = bit_slice(a, start=0, width=2)
  addend1: bits[32] = zero_ext(a_p1, new_bit_count=32)
  addend2: bits[32] = zero_ext(a, new_bit_count=32)
  case0: bits[32] = add(addend0, x)
  case1: bits[32] = add(addend1, y)
  case2: bits[32] = add(addend2, z)
  case_d: bits[32] = add(addend2, q)
  ret priority_sel.4: bits[32] = priority_sel(a, cases=[case0, case1, case2], default=case_d)
}
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Param("a"),
                                {m::Add(m::Literal(1), m::Param("x")),
                                 m::Add(m::Literal(2), m::Param("y")),
                                 m::Add(m::Literal(4), m::Param("z"))},
                                m::Add(m::Literal(0), m::Param("q"))));
}

TEST_F(ConditionalSpecializationPassTest,
       SpecializePrioritySelectSelectorExpression) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32], x: bits[1]) -> bits[1] {
  literal.1: bits[32] = literal(value=7)
  ult.2: bits[1] = ult(a, literal.1)
  not.3: bits[1] = not(ult.2)
  selector: bits[2] = concat(not.3, ult.2)
  ret priority_sel.5: bits[1] = priority_sel(selector, cases=[ult.2, x], default=x)
}
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Concat(m::Not(), m::ULt(m::Param("a"), m::Literal(7))),
                  {m::Literal(1), m::Param("x")}, m::Param("x")));
}

TEST_F(ConditionalSpecializationPassTest,
       SpecializePrioritySelectSelectorExpressionNegative) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32], x: bits[1]) -> bits[1] {
  literal.1: bits[32] = literal(value=7)
  ult.2: bits[1] = ult(a, literal.1)
  not.3: bits[1] = not(ult.2)
  selector: bits[2] = concat(ult.2, not.3)
  ret priority_sel.5: bits[1] = priority_sel(selector, cases=[ult.2, x], default=x)
}
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Concat(m::ULt(m::Param("a"), m::Literal(7)), m::Not()),
                  {m::Literal(0), m::Param("x")}, m::Param("x")));
}

TEST_F(ConditionalSpecializationPassTest,
       SpecializePrioritySelectWithDuplicateCaseArms) {
  // If an expression is used as more than one arm of the select it should not
  // be transformed because the same expression is used for multiple case
  // values.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[2], y: bits[32]) -> bits[32] {
  zero_ext.1: bits[32] = zero_ext(a, new_bit_count=32)
  add: bits[32] = add(zero_ext.1, y)
  ret sel: bits[32] = priority_sel(a, cases=[add, add], default=add)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, SpecializeSelectOnArrayUpdate) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32][8], i: bits[16], z: bits[32], q: bits[32]) -> bits[32][8] {
  array_bound: bits[16] = literal(value=8)
  is_in_bounds: bits[1] = ult(i, array_bound)
  update_val: bits[32] = sel(is_in_bounds, cases=[q, z])
  ret new_a: bits[32][8] = array_update(a, update_val, indices=[i])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Param("a"), m::Param("z"), {m::Param("i")}));
}

TEST_F(ConditionalSpecializationPassTest,
       SpecializeSelectOnNonPowerOfTwoArrayUpdate) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[32][7], i: bits[16], z: bits[32], q: bits[32]) -> bits[32][7] {
  array_bound: bits[16] = literal(value=7)
  is_in_bounds: bits[1] = ult(i, array_bound)
  update_val: bits[32] = sel(is_in_bounds, cases=[q, z])
  ret new_a: bits[32][7] = array_update(a, update_val, indices=[i])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Param("a"), m::Param("z"), {m::Param("i")}));
}

TEST_F(ConditionalSpecializationPassTest, LongSelectChain) {
  // Build a long transformable select chain (first and last select share a
  // selector).
  //
  //  s0 = sel(s[0], cases=[a, b])
  //  s1 = sel(s[1], cases=[x[0], s0)
  //  s2 = sel(s[2], cases=[x[1], s1)
  //  ...
  //  s{n-1} = sel(s[n-1], cases=[x[n], s{n-2}])
  //  s{n} = sel(s[0], cases[x[n+1], s{n-1}])
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  const int64_t kChainSize = 50;
  BValue s = fb.Param("s", p->GetBitsType(kChainSize));
  BValue x = fb.Param("x", p->GetBitsType(kChainSize + 1));
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  std::vector<BValue> selects;
  BValue sel;
  for (int64_t i = 0; i <= kChainSize; ++i) {
    if (i == 0) {
      sel = fb.Select(fb.BitSlice(s, /*start=*/0, /*width=*/1), {a, b});
    } else if (i == kChainSize) {
      sel = fb.Select(fb.BitSlice(s, /*start=*/0, /*width=*/1),
                      {fb.BitSlice(x, /*start=*/i, /*width=*/1), sel});
    } else {
      sel = fb.Select(fb.BitSlice(s, /*start=*/i, /*width=*/1),
                      {fb.BitSlice(x, /*start=*/i, /*width=*/1), sel});
    }
    selects.push_back(sel);
  }

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // The second select should have it's case-1 redirected to "b".
  EXPECT_THAT(selects[1].node(),
              m::Select(m::BitSlice(), {m::BitSlice(), m::Select()}));

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(selects[1].node(),
              m::Select(m::BitSlice(), {m::BitSlice(), m::Param("b")}));
}

TEST_F(ConditionalSpecializationPassTest, TooLongSelectChain) {
  // Build an otherwise transformable but too long select chain (first and last
  // select share a selector) but the chain is too long so the condition set
  // size is maxed out and the transformation doesn't occur.
  //
  //  s0 = sel(s[0], cases=[a, b])
  //  s1 = sel(s[1], cases=[x[0], s0)
  //  s2 = sel(s[2], cases=[x[1], s1)
  //  ...
  //  s{n-1} = sel(s[n-1], cases=[x[n], s{n-2}])
  //  s{n} = sel(s[0], cases[x[n+1], s{n-1}])
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  const int64_t kChainSize = 100;
  BValue s = fb.Param("s", p->GetBitsType(kChainSize));
  BValue x = fb.Param("x", p->GetBitsType(kChainSize + 1));
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue sel;
  for (int64_t i = 0; i <= kChainSize; ++i) {
    if (i == 0) {
      sel = fb.Select(fb.BitSlice(s, /*start=*/0, /*width=*/1), {a, b});
    } else if (i == kChainSize) {
      sel = fb.Select(fb.BitSlice(s, /*start=*/0, /*width=*/1),
                      {fb.BitSlice(x, /*start=*/i, /*width=*/1), sel});
    } else {
      sel = fb.Select(fb.BitSlice(s, /*start=*/i, /*width=*/1),
                      {fb.BitSlice(x, /*start=*/i, /*width=*/1), sel});
    }
  }

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedSelectorValueUsingOr) {
  // r implies r|s.
  //
  //    a   b
  //     \ /
  //     sel1 ---- r|s         b
  //      |                    |
  //   c  |              => c  |
  //    \ |                  \ |
  //     sel0 ---- r         sel0 ---- r
  //      |                    |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue r = fb.Param("r", p->GetBitsType(1));
  BValue s = fb.Param("s", p->GetBitsType(1));

  BValue r_or_s = fb.Or(r, s);
  BValue sel1 = fb.Select(r_or_s, {a, b});
  BValue sel0 = fb.Select(r, {c, sel1});

  // Keep r_or_s alive to the return value to avoid replacing r in the
  // expression with one (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({sel0, r_or_s})));

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::Select(m::Param("r"),
                        /*cases=*/{m::Param("c"), m::Param("b")}));
}

TEST_F(ConditionalSpecializationPassTest, NotImpliedSelectorValueUsingAnd) {
  // No transformation because r does not imply r&s&t is true or false.
  //
  //    a   b
  //     \ /
  //     sel1 ---- r&s&t
  //      |
  //   c  |
  //    \ |
  //     sel0 ---- r
  //      |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue r = fb.Param("r", p->GetBitsType(1));
  BValue s = fb.Param("s", p->GetBitsType(1));
  BValue t = fb.Param("t", p->GetBitsType(1));

  BValue r_and_s_and_t = fb.And({r, s, t});
  BValue sel1 = fb.Select(r_and_s_and_t, {a, b});
  BValue sel0 = fb.Select(r, {c, sel1});

  // Keep r_and_s alive to the return value to avoid replacing r in the
  // expression with one (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({sel0, r_and_s_and_t})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, NotImpliedSelectorValueUsingOr) {
  // No transformation because !r does not imply r|s|t is true or false.
  //
  //    a   b
  //     \ /
  //     sel1 ---- r|s|t
  //      |
  //      |   c
  //      |  /
  //     sel0 ---- r
  //      |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue r = fb.Param("r", p->GetBitsType(1));
  BValue s = fb.Param("s", p->GetBitsType(1));
  BValue t = fb.Param("t", p->GetBitsType(1));

  BValue r_or_s_or_t = fb.Or({r, s, t});
  BValue sel1 = fb.Select(r_or_s_or_t, {a, b});
  BValue sel0 = fb.Select(r, {sel1, c});

  // Keep r_or_s_or_t alive to the return value to avoid replacing r in the
  // expression with zero (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({sel0, r_or_s_or_t})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedSelectorValueUsingAnd) {
  // !r implies !(r&s).
  //
  //    a   b
  //     \ /
  //     sel1 ---- r&s         a
  //      |                    |
  //      |  c           =>    |  c
  //      | /                  | /
  //     sel0 ---- r          sel0 ---- r
  //      |                    |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue r = fb.Param("r", p->GetBitsType(1));
  BValue s = fb.Param("s", p->GetBitsType(1));

  BValue r_and_s = fb.And(r, s);
  BValue sel1 = fb.Select(r_and_s, {a, b});
  BValue sel0 = fb.Select(r, {sel1, c});

  // Keep r_and_s alive to the return value to avoid replacing r in the
  // expression with zero (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({sel0, r_and_s})));

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::Select(m::Param("r"),
                        /*cases=*/{m::Param("a"), m::Param("c")}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedSelectorValueWithOtherUses) {
  // r implies r|s.
  //
  //    a   b                a   b ------------
  //     \ /                  \ /              \
  //     sel1 ---- r|s        sel1 ---- r|s     \
  //      | \                  |                |
  //   c  |  ...         =>   ...            c  |
  //    \ |                                   \ |
  //     sel0 ---- r                          sel0 ---- r
  //      |                                     |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue r = fb.Param("r", p->GetBitsType(1));
  BValue s = fb.Param("s", p->GetBitsType(1));

  BValue r_or_s = fb.Or(r, s);
  BValue sel1 = fb.Select(r_or_s, {a, b});
  BValue sel0 = fb.Select(r, {c, sel1});

  // Keep r_or_s alive to the return value to avoid replacing r in the
  // expression with one (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({sel0, sel1, r_or_s})));

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::Select(m::Param("r"),
                        /*cases=*/{m::Param("c"), m::Param("b")}));
  EXPECT_THAT(f->return_value()->operand(1),
              m::Select(m::Or(m::Param("r"), m::Param("s")),
                        /*cases=*/{m::Param("a"), m::Param("b")}));
}

TEST_F(ConditionalSpecializationPassTest,
       ImpliedPrioritySelectorValueWithOtherUses) {
  // r implies r|s.
  //
  //        a                    a ------------
  //       /                    /              \
  //     sel1 ---- r|s        sel1 ---- r|s     \
  //      | \                  |                |
  //      |  ...         =>   ...               |
  //      |                                     |
  //     sel0 ---- r                          sel0 ---- r
  //      |                                     |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue r = fb.Param("r", p->GetBitsType(2));
  BValue s = fb.Param("s", p->GetBitsType(2));

  BValue r_or_s = fb.Or(r, s);
  BValue other_value = fb.Literal(UBits(42, 32));
  BValue sel1 =
      fb.PrioritySelect(r_or_s, {a, other_value}, fb.Literal(UBits(0, 32)));
  BValue sel0 =
      fb.PrioritySelect(r, {sel1, other_value}, fb.Literal(UBits(0, 32)));

  // Keep r_or_s alive to the return value to avoid replacing r in the
  // expression with one (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({sel0, sel1, r_or_s})));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::PrioritySelect(m::Param("r"),
                                /*cases=*/{m::Param("a"), m::Literal(42)},
                                /*default_value=*/m::Literal(0)));
  EXPECT_THAT(f->return_value()->operand(1),
              m::PrioritySelect(m::Or(m::Param("r"), m::Param("s")),
                                /*cases=*/{m::Param("a"), m::Literal(42)},
                                /*default_value=*/m::Literal(0)));
}

TEST_F(ConditionalSpecializationPassTest,
       ImpliedPrioritySelectorDefaultValueWithOtherUses) {
  // r implies r|s.
  //
  //        d                    d ------------
  //       /                    /              \
  //     sel1 ---- r&s        sel1 ---- r&s     \
  //      | \                  |                |
  //      |  ...         =>   ...               |
  //      |                                     |
  //     sel0 ---- r                          sel0 ---- r
  //      |                                     |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue r = fb.Param("r", p->GetBitsType(2));
  BValue s = fb.Param("s", p->GetBitsType(2));

  BValue r_and_s = fb.And(r, s);
  BValue other_value = fb.Literal(UBits(42, 32));
  BValue sel1 = fb.PrioritySelect(r_and_s, {a, other_value},
                                  /*default_value=*/fb.Literal(UBits(85, 32)));
  BValue sel0 = fb.PrioritySelect(r, {a, other_value},
                                  /*default_value=*/sel1);

  // Keep r_and_s alive to the return value to avoid replacing r in the
  // expression (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({sel0, sel1, r_and_s})));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::PrioritySelect(m::Param("r"),
                                /*cases=*/{m::Param("a"), m::Literal(42)},
                                /*default_value=*/m::Literal(85)));
  EXPECT_THAT(f->return_value()->operand(1),
              m::PrioritySelect(m::And(m::Param("r"), m::Param("s")),
                                /*cases=*/{m::Param("a"), m::Literal(42)},
                                /*default_value=*/m::Literal(85)));
}

TEST_F(ConditionalSpecializationPassTest,
       ImpliedOneHotSelectorValueWithOtherUses) {
  // r implies r|s.
  //
  //        a                    a ------------
  //       /                    /              \
  //     sel1 ---- r|s        sel1 ---- r|s     \
  //      | \                  |                |
  //      |  ...         =>   ...               |
  //      |                                     |
  //     sel0 ---- r                          sel0 ---- r
  //      |                                     |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue r = fb.Param("r", p->GetBitsType(2));
  BValue s = fb.Param("s", p->GetBitsType(2));

  BValue r_or_s = fb.Or(r, s);
  BValue other_value = fb.Literal(UBits(42, 32));
  BValue sel1 = fb.OneHotSelect(fb.BitSlice(r, /*start=*/0, /*width=*/1), {a});
  BValue sel0 =
      fb.PrioritySelect(r, {sel1, other_value}, fb.Literal(UBits(0, 32)));

  // Keep r_or_s alive to the return value to avoid replacing r in the
  // expression with one (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({sel0, sel1, r_or_s})));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::PrioritySelect(m::Param("r"),
                                /*cases=*/{m::Param("a"), m::Literal(42)},
                                /*default_value=*/m::Literal(0)));
  EXPECT_THAT(
      f->return_value()->operand(1),
      m::OneHotSelect(m::BitSlice(m::Param("r"), /*start=*/0, /*width=*/1),
                      /*cases=*/{m::Param("a")}));
}

TEST_F(ConditionalSpecializationPassTest,
       ImpliedOneHotSelectorDefaultValueWithOtherUses) {
  // !r implies !(r&s).
  //
  //        d                    d ------------
  //       /                    /              \
  //     sel1 ---- r&s        sel1 ---- r&s     \
  //      | \                  |                |
  //      |  ...         =>   ...               |
  //      |                                     |
  //     sel0 ---- r                          sel0 ---- r
  //      |                                     |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue r = fb.Param("r", p->GetBitsType(2));
  BValue s = fb.Param("s", p->GetBitsType(2));

  BValue r_and_s = fb.And(r, s);
  BValue other_value = fb.Literal(UBits(42, 32));
  BValue sel1 = fb.OneHotSelect(r_and_s, {a, other_value});
  BValue sel0 = fb.PrioritySelect(r, {a, other_value},
                                  /*default_value=*/sel1);

  // Keep r_and_s alive to the return value to avoid replacing r in the
  // expression (this is not what we're testing).
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({sel0, sel1, r_and_s})));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::PrioritySelect(m::Param("r"),
                                /*cases=*/{m::Param("a"), m::Literal(42)},
                                /*default_value=*/m::Literal(0)));
  EXPECT_THAT(f->return_value()->operand(1),
              m::OneHotSelect(m::And(m::Param("r"), m::Param("s")),
                              /*cases=*/{m::Param("a"), m::Literal(42)}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedValueThroughAnd) {
  // r implies r|s.
  //
  //    r   a                r   a ------------
  //     \ /                  \ /              \
  //     and                  and               \
  //      | \                  |                |
  //      |  ...         =>   ...               |
  //      |                                     |
  //     sel0 ---- r                          sel0 ---- r
  //      |                                     |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue r = fb.Param("r", u1);

  BValue r_and_a = fb.And(r, a);
  BValue sel0 = fb.PrioritySelect(r, {r_and_a}, fb.Literal(UBits(0, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({sel0, r_and_a})));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::PrioritySelect(m::Param("r"),
                                /*cases=*/{m::Param("a")},
                                /*default_value=*/m::Literal(0)));
  EXPECT_THAT(f->return_value()->operand(1),
              m::And(m::Param("r"), m::Param("a")));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedValueThroughOr) {
  //
  //    r    a               r    a -----------
  //     \  /                 \  /             \
  //      or                   or               \
  //      | \                  |                |
  //      |  ...         =>   ...               |
  //      |                                     |
  //     sel0 ---- r                          sel0 ---- r
  //      |                                     |
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue r = fb.Param("r", u1);

  BValue r_or_a = fb.Or(r, a);
  BValue sel0 = fb.PrioritySelect(r, {fb.Literal(UBits(0, 1))}, r_or_a);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({sel0, r_or_a})));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value()->operand(0),
              m::PrioritySelect(m::Param("r"),
                                /*cases=*/{m::Literal(0)},
                                /*default_value=*/m::Param("a")));
  EXPECT_THAT(f->return_value()->operand(1),
              m::Or(m::Param("r"), m::Param("a")));
}

TEST_F(ConditionalSpecializationPassTest, SendNoChangeLiteralPred) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> p,
                           ParsePackageNoVerify(R"(
      package my_package
      chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

      top proc Delay_proc(value1: bits[32], value2: bits[32], init={1, 2}) {
        tkn: token = literal(value=token, id=1000)
        literal.2: bits[1] = literal(value=1, id=2)
        literal.5: bits[32] = literal(value=400, id=5)
        eq.3: bits[1] = eq(value1, literal.5, id=3)
        sel.4: bits[32] = sel(eq.3, cases=[literal.5, value2], id=4)
        send.1: token = send(tkn, sel.4, predicate=literal.2, channel=out, id=1)
      }
    )"));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, SendNoChangeUnprovable) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> p,
                           ParsePackageNoVerify(R"(
      package my_package
      chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

      top proc Delay_proc(value1: bits[32], value2: bits[32], init={1, 2}) {
        tkn: token = literal(value=token, id=1000)
        literal.2: bits[1] = literal(value=1, id=2)
        literal.5: bits[32] = literal(value=400, id=5)
        literal.6: bits[32] = literal(value=55, id=6)
        eq.3: bits[1] = eq(value1, literal.5, id=3)
        ugt.4: bits[1] = ugt(value1, literal.6, id=4)
        sel.4: bits[32] = sel(eq.3, cases=[literal.5, value2], id=4)
        send.1: token = send(tkn, sel.4, predicate=ugt.4, channel=out, id=1)
      }
    )"));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, SendChange) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> p,
                           ParsePackageNoVerify(R"(
      package my_package
      chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

      top proc Delay_proc(value1: bits[32], value2: bits[32], init={1, 2}) {
        tkn: token = literal(value=token, id=1000)
        literal.2: bits[1] = literal(value=1, id=2)
        literal.5: bits[32] = literal(value=400, id=5)
        eq.3: bits[1] = eq(value1, literal.5, id=3)
        sel.4: bits[32] = sel(eq.3, cases=[literal.5, value2], id=4)
        send.1: token = send(tkn, sel.4, predicate=eq.3, channel=out, id=1)
        next (value1, value2)
      }
    )"));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * f, p->GetProc("Delay_proc"));
  EXPECT_THAT(f->GetNode("send.1"),
              IsOkAndHolds(m::Send(
                  /*token=*/A<const Node*>(), /*data=*/m::StateRead("value2"),
                  /*predicate=*/A<const Node*>())));
}

TEST_F(ConditionalSpecializationPassTest, NextValueNoChangeLiteralPred) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> p,
                           ParsePackageNoVerify(R"(
      package my_package

      top proc Delay_proc(value1: bits[32], value2: bits[32], init={1, 2}) {
        tkn: token = literal(value=token, id=1000)
        literal.1: bits[1] = literal(value=1, id=1)
        literal.2: bits[32] = literal(value=400, id=2)
        eq.3: bits[1] = eq(value1, literal.2, id=3)
        sel.4: bits[32] = sel(eq.3, cases=[literal.2, value2], id=4)
        next_value.5: () = next_value(param=value1, value=sel.4, predicate=literal.1, id=5)
      }
    )"));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, NextValueNoChangeUnprovable) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> p,
                           ParsePackageNoVerify(R"(
      package my_package

      top proc Delay_proc(value1: bits[32], value2: bits[32], init={1, 2}) {
        tkn: token = literal(value=token, id=1000)
        literal.1: bits[32] = literal(value=400, id=1)
        literal.2: bits[32] = literal(value=55, id=2)
        eq.3: bits[1] = eq(value1, literal.1, id=3)
        ugt.4: bits[1] = ugt(value1, literal.2, id=4)
        sel.5: bits[32] = sel(eq.3, cases=[literal.1, value2], id=5)
        next_value.6: () = next_value(param=value1, value=sel.5, predicate=ugt.4, id=6)
      }
    )"));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ConditionalSpecializationPassTest, NextValueChange) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> p,
                           ParsePackageNoVerify(R"(
      package my_package

      top proc Delay_proc(value1: bits[32], value2: bits[32], init={1, 2}) {
        tkn: token = literal(value=token, id=1000)
        literal.1: bits[32] = literal(value=400, id=1)
        eq.2: bits[1] = eq(value1, literal.1, id=2)
        sel.3: bits[32] = sel(eq.2, cases=[literal.1, value2], id=3)
        next_value.4: () = next_value(param=value1, value=sel.3, predicate=eq.2, id=4)
      }
    )"));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * f, p->GetProc("Delay_proc"));
  EXPECT_THAT(f->next_values(f->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(m::StateRead("value1"),
                                  m::StateRead("value2"), m::Eq())));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedConditionThroughNot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue s = fb.Not(a);
  BValue result = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f, /*use_bdd=*/false), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Not(m::Param("a")), {m::Literal(1), m::Param("b")}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedConditionThroughEq) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue s = fb.Eq(b, fb.Literal(UBits(1, 1)));
  BValue result = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f, /*use_bdd=*/false), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::Eq(m::Param("b"), m::Literal(1)),
                                           {m::Param("a"), m::Literal(1)}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedConditionThroughNe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue s = fb.Ne(a, fb.Literal(UBits(1, 1)));
  BValue result = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f, /*use_bdd=*/false), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::Ne(m::Param("a"), m::Literal(1)),
                                           {m::Literal(1), m::Param("b")}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedConditionThroughOr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue s = fb.Or(a, b);
  BValue result = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f, /*use_bdd=*/false), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::Or(m::Param("a"), m::Param("b")),
                                           {m::Literal(0), m::Param("b")}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedConditionThroughAnd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue s = fb.And(a, b);
  BValue result = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f, /*use_bdd=*/false), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::And(m::Param("a"), m::Param("b")),
                                           {m::Param("a"), m::Literal(1)}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedConditionThroughNor) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue s = fb.Nor(a, b);
  BValue result = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f, /*use_bdd=*/false), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::Nor(m::Param("a"), m::Param("b")),
                                           {m::Param("a"), m::Literal(0)}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedConditionThroughNand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue s = fb.Nand(a, b);
  BValue result = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f, /*use_bdd=*/false), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Nand(m::Param("a"), m::Param("b")),
                        {m::Literal(1), m::Param("b")}));
}

TEST_F(ConditionalSpecializationPassTest, ImpliedConditionThroughIteratedNot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue s = fb.Not(fb.Not(b));
  BValue result = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::z3::ScopedVerifyEquivalence sve{f};
  EXPECT_THAT(Run(f, /*use_bdd=*/false), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::Not(m::Not(m::Param("b"))),
                                           {m::Param("a"), m::Literal(1)}));
}

TEST_F(ConditionalSpecializationPassTest, StateReadSpecialization) {
  auto p = CreatePackage();

  TokenlessProcBuilder pb(TestName(), "tkn", p.get());
  BValue index = pb.StateElement("index", UBits(0, 1));
  BValue index_is_0 = pb.Not(index);
  BValue index_is_1 = index;
  BValue counter0 = pb.StateElement("counter0", UBits(0, 32));
  BValue counter1 = pb.StateElement("counter1", UBits(5, 32));
  BValue selected_counter =
      pb.Select(index_is_1, {counter1}, /*on_false=*/counter0);
  BValue incremented_counter =
      pb.Add(selected_counter, pb.Literal(UBits(1, 32)));
  pb.Next(index, index_is_0);
  // counter0 has an explicit increment, and an implicit unchanged state.
  pb.Next(counter0, incremented_counter, /*pred=*/index_is_0);
  // counter1 has an explicit unchanged state and an explicit increment.
  pb.Next(counter1, counter1, /*pred=*/index_is_0);
  pb.Next(counter1, incremented_counter, /*pred=*/index_is_1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ScopedRecordIr sri(p.get());
  EXPECT_THAT(
      Run(proc, /*use_bdd=*/true, /*optimize_for_best_case_throughput=*/true),
      IsOkAndHolds(true));

  EXPECT_THAT(
      proc->GetStateRead(*proc->GetStateElement("counter0"))->predicate(),
      Optional(m::Not(m::StateRead("index"))));
  EXPECT_THAT(
      proc->GetStateRead(*proc->GetStateElement("counter1"))->predicate(),
      Optional(m::StateRead("index")));
}

TEST_F(ConditionalSpecializationPassTest, HarderStateReadSpecialization) {
  auto p = CreatePackage();

  TokenlessProcBuilder pb(TestName(), "tkn", p.get());
  BValue index = pb.StateElement("index", UBits(0, 2));
  BValue counter0 = pb.StateElement("counter0", UBits(0, 32));
  BValue counter1 = pb.StateElement("counter1", UBits(5, 32));
  BValue counter2 = pb.StateElement("counter2", UBits(10, 32));
  BValue index_not_0 = pb.Ne(index, pb.Literal(UBits(0, 2)));
  BValue index_not_1 = pb.Ne(index, pb.Literal(UBits(1, 2)));
  BValue selected_counter = pb.Select(
      index_not_0, {counter0, pb.Select(index_not_1, {counter1, counter2})});
  BValue incremented_counter =
      pb.Add(selected_counter, pb.Literal(UBits(1, 32)));
  BValue index_is_0 = pb.Eq(index, pb.Literal(UBits(0, 2)));
  BValue index_is_1 = pb.Eq(index, pb.Literal(UBits(1, 2)));
  pb.Next(index, pb.Add(index, pb.Literal(UBits(1, 2))));
  pb.Next(counter0, incremented_counter, /*pred=*/index_is_0);
  pb.Next(counter1, incremented_counter, /*pred=*/index_is_1);
  pb.Next(counter2, incremented_counter,
          /*pred=*/pb.And(index_not_0, index_not_1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  EXPECT_THAT(
      Run(proc, /*use_bdd=*/true, /*optimize_for_best_case_throughput=*/true),
      IsOkAndHolds(true));

  EXPECT_THAT(
      proc->GetStateRead(*proc->GetStateElement("counter0"))->predicate(),
      Optional(m::Eq(m::StateRead("index"), m::Literal(0))));
  EXPECT_THAT(
      proc->GetStateRead(*proc->GetStateElement("counter1"))->predicate(),
      Optional(m::Eq(m::StateRead("index"), m::Literal(1))));
  EXPECT_THAT(
      proc->GetStateRead(*proc->GetStateElement("counter2"))->predicate(),
      Optional(m::And(
          // High bit of index is set
          m::Eq(m::BitSlice(m::StateRead("index"), 1, 1), m::Literal(1)),
          // index is not 0 or 1; this is redundant with the high-bit check, but
          // can be simplified away by other passes.
          m::Ne(m::StateRead("index"), m::Literal(1)),
          m::Ne(m::StateRead("index"), m::Literal(0)))));
}

TEST_F(ConditionalSpecializationPassTest, StateReadSpecializationDisabled) {
  auto p = CreatePackage();

  TokenlessProcBuilder pb(TestName(), "tkn", p.get());
  BValue index = pb.StateElement("index", UBits(0, 1));
  BValue index_is_0 = pb.Not(index);
  BValue index_is_1 = index;
  BValue counter0 = pb.StateElement("counter0", UBits(0, 32));
  BValue counter1 = pb.StateElement("counter1", UBits(5, 32));
  BValue selected_counter =
      pb.Select(index_is_1, {counter1}, /*on_false=*/counter0);
  BValue incremented_counter =
      pb.Add(selected_counter, pb.Literal(UBits(1, 32)));
  pb.Next(index, index_is_0);
  pb.Next(counter0, incremented_counter, /*pred=*/index_is_0);
  pb.Next(counter1, incremented_counter, /*pred=*/index_is_1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  EXPECT_THAT(
      Run(proc, /*use_bdd=*/true, /*optimize_for_best_case_throughput=*/false),
      IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
