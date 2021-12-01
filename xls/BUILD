# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# XLS: Accelerator Synthesis

# XLS-internal-only.
package_group(
    name = "xls_internal",
    packages = [
        "-//xls/public/tests/...",
        "//xls/...",
        "//third_party/xls_berkeley_softfloat/...",
        "//third_party/xls_colors/...",
        "//third_party/xls_go_math/...",
        "//third_party/xls_machsuite/...",
    ],
)

# The group of known users of XLS.
# The targets listed with this visibility comprise the intended "public API" of
# XLS, and are be supported and are "reasonably" stable, i.e., significant
# changes are expected to be rare and coordinated.
# Ultimately, this is where all non-XLS-internal users should be listed, once
# the transition described in the TODOs below are complete.
package_group(
    name = "xls_public",
    # Everything the public can see is also visible internally.
    includes = [":xls_internal"],
    packages = [
        "//...",
        "//xls/public/tests/...",
    ],
)

# TODO(leary): 2020-06-29 This structure is a work-around to publically expose
# XLS APIs. Once the public XLS API is determined, those packages should be
# explicitly marked with 'xls_public' visibility, and this should be changed so
# that 'xls_public' includes 'xls_internal' and not vice versa.
package_group(
    name = "xls_users",
    # Everything users can see is also visible internally.
    includes = [":xls_internal"],
    packages = ["//..."],
)

# Users of various XLS utility helpers, e.g. things like logging, testing
# targets. Some of these will likely be eliminated when logging is
# no longer XLS specific.
#
# We call this out to distinguish it from properly public headers that expose
# fundamental XLS functionality.
package_group(
    name = "xls_utility_users",
    # Everything utility users can see is also visible internally.
    includes = [":xls_internal"],
    packages = ["//..."],
)

# "Best effort" users are ones that are using some XLS internals
# (ideally temporarily) and are agreeing to be broken at any
# time if we can't figure out how to fix them, since we need to
# be able to evolve XLS innards as necessary.
package_group(
    name = "xls_best_effort_users",
    # Everything best effort users can see is also visible internally.
    includes = [":xls_internal"],
    packages = ["//..."],
)

package(
    default_visibility = [":xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "dist",
    extension = "tar.gz",
    srcs = [
        "//xls/tools:wrap_io_main",
        "//xls/tools:solver",
        "//xls/tools:smtlib_emitter_main",
        "//xls/tools:simulate_module_main",
        "//xls/tools:repl",
        "//xls/tools:proto_to_dslx_main",
        "//xls/tools:parse_ir",
        "//xls/tools:netlist_interpreter_main",
        "//xls/tools:llvm_main_generator",
        "//xls/tools:lec_main",
        "//xls/tools:ir_stats_main",
        "//xls/tools:ir_minimizer_main",
        "//xls/tools:opt_main",
        "//xls/tools:extract_stage_main",
        "//xls/tools:eval_proc_main",
        "//xls/tools:eval_ir_main",
        "//xls/tools:drpc_main",
        "//xls/tools:delay_info_main",
        "//xls/tools:codegen_main",
        "//xls/tools:check_ir_equivalence_main",
        "//xls/tools:cell_library_extract_formula",
        "//xls/tools:booleanify_main",
        "//xls/tools:benchmark_main",
        "//xls/tools:bdd_stats",
        "//xls/dslx:typecheck_main",
        "//xls/dslx:ir_converter_main",
        "//xls/dslx:interpreter_main",
        "//xls/dslx:highlight_main",
        "//xls/dslx:cpp_transpiler_main",
    ],
    package_dir = "/xls-tools-dist"
)

exports_files(["LICENSE"])
