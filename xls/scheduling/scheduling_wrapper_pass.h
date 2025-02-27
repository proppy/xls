// Copyright 2022 The XLS Authors
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

#ifndef XLS_SCHEDULING_SCHEDULING_WRAPPER_PASS_H_
#define XLS_SCHEDULING_SCHEDULING_WRAPPER_PASS_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/passes/optimization_pass.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

// A scheduling pass wrapper which wraps a Pass. This is useful for adding an
// optimization or transformation pass to the scheduling pipeline. The wrapped
// pass is run on the underlying package and then any nodes that were removed
// by the pass are removed from the schedule.
// If nodes were added by the pass, the behavior is determined by
// reschedule_new_nodes, which is false by default. If !reschedule_new_nodes,
// nodes added by the pass are detected and an error is raised. If
// reschedule_new_nodes is true, the current schedule is deleted and a
// scheduling pass must be rerun after this wrapped pass.
// TODO(allight): Do a refactor to move opt-level into pass-options similar to
// what's done with optimization-pass
class SchedulingWrapperPass : public SchedulingPass {
 public:
  explicit SchedulingWrapperPass(std::unique_ptr<OptimizationPass> wrapped_pass,
                                 OptimizationContext* context,
                                 int64_t opt_level, bool eliminate_noop_next,
                                 bool reschedule_new_nodes = false)
      : SchedulingPass(
            absl::StrFormat("scheduling_%s", wrapped_pass->short_name()),
            absl::StrFormat("%s (scheduling)", wrapped_pass->long_name())),
        wrapped_pass_(std::move(wrapped_pass)),
        context_(context),
        opt_level_(opt_level),
        eliminate_noop_next_(eliminate_noop_next),
        reschedule_new_nodes_(reschedule_new_nodes) {}
  ~SchedulingWrapperPass() override = default;

 protected:
  absl::StatusOr<bool> RunInternal(
      SchedulingUnit* unit, const SchedulingPassOptions& options,
      SchedulingPassResults* results) const override;

 private:
  std::unique_ptr<OptimizationPass> wrapped_pass_;
  OptimizationContext* context_;
  int64_t opt_level_;
  bool eliminate_noop_next_;
  bool reschedule_new_nodes_;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_WRAPPER_PASS_H_
