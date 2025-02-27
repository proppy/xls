// Copyright 2025 The XLS Authors
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

#include "xls/passes/query_engine_checker.h"

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"

namespace xls {

absl::Status QueryEngineChecker::Run(Package* p,
                                     const OptimizationPassOptions& options,
                                     PassResults* results,
                                     OptimizationContext* context) const {
  for (QueryEngine* query_engine : context->ListQueryEngines()) {
    XLS_RETURN_IF_ERROR(query_engine->CheckConsistency());
  }
  return absl::OkStatus();
}

}  // namespace xls
