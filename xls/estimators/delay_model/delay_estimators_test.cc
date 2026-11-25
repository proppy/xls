// Copyright 2020 The XLS Authors
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

#include "xls/estimators/delay_model/delay_estimators.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

class DelayEstimatorsTest : public IrTestBase {};

TEST_F(DelayEstimatorsTest, UnitDelayModel) {
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           GetDelayEstimator("unit"));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue add = fb.Add(x, x);
  BValue negate = fb.Negate(x);
  BValue mul = fb.UMul(x, x);
  BValue and_op = fb.And(x, x);
  BValue tuple = fb.Tuple({x, add, mul});

  EXPECT_THAT(estimator->GetOperationDelayInPs(x.node()), IsOkAndHolds(0));
  EXPECT_THAT(estimator->GetOperationDelayInPs(add.node()), IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationDelayInPs(negate.node()), IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationDelayInPs(mul.node()), IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationDelayInPs(and_op.node()), IsOkAndHolds(1));
  EXPECT_THAT(estimator->GetOperationDelayInPs(tuple.node()), IsOkAndHolds(1));
}

TEST_F(DelayEstimatorsTest, SyntheticDelayModelWithoutPrecomputedProto) {
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           GetDelayEstimator("synthetic_test_delay"));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue add = fb.Add(x, x);
  BValue sub = fb.Subtract(x, x);
  BValue negate = fb.Negate(x);

  EXPECT_THAT(estimator->GetOperationDelayInPs(x.node()), IsOkAndHolds(0));
  EXPECT_THAT(estimator->GetOperationDelayInPs(add.node()), IsOkAndHolds(42));
  EXPECT_THAT(estimator->GetOperationDelayInPs(sub.node()), IsOkAndHolds(160));
  EXPECT_THAT(estimator->GetOperationDelayInPs(negate.node()),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(DelayEstimatorsTest, Gf180McuDelayModels) {
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator_5v,
                           GetDelayEstimator("gf180mcu_5v0"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator_3v3,
                           GetDelayEstimator("gf180mcu_3v3"));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y);
  BValue mul = fb.UMul(x, y);

  EXPECT_THAT(estimator_5v->GetOperationDelayInPs(x.node()), IsOkAndHolds(0));
  EXPECT_THAT(estimator_3v3->GetOperationDelayInPs(x.node()), IsOkAndHolds(0));

  XLS_ASSERT_OK_AND_ASSIGN(int64_t add_5v,
                           estimator_5v->GetOperationDelayInPs(add.node()));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t mul_5v,
                           estimator_5v->GetOperationDelayInPs(mul.node()));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t add_3v3,
                           estimator_3v3->GetOperationDelayInPs(add.node()));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t mul_3v3,
                           estimator_3v3->GetOperationDelayInPs(mul.node()));

  EXPECT_GT(add_5v, 0);
  EXPECT_GT(mul_5v, add_5v);
  EXPECT_GT(add_3v3, add_5v);
  EXPECT_GT(mul_3v3, mul_5v);
}

}  // namespace
}  // namespace xls
