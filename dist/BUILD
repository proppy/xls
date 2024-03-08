# Copyright 2024 The XLS Authors
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

load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files", "strip_prefix")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load(":dist.bzl", "xls_release")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

filegroup(
    name = "third_party_notices",
    srcs = [
	"@boringssl//:LICENSE",
	"@com_github_cares_cares//:LICENSE.md",
	"@com_github_facebook_zstd//:LICENSE",
	"@com_github_grpc_grpc//:LICENSE",
	"@com_google_absl//:LICENSE",
	"@com_google_ortools//:LICENSE",
	"@com_google_protobuf//:LICENSE",
	"@com_googlesource_code_re2//:LICENSE",
	"@llvm-project//llvm:LICENSE.TXT",
	"@utf8_range//:LICENSE",
	"@zlib//:LICENSE",
    ],
)

genrule(
    name = "third_party_notices_txt",
    srcs = [
	":third_party_notices",
    ],
    outs = [
	"THIRD_PARTY_NOTICES.txt",
    ],
    cmd = "cat $(locations :third_party_notices) > $@",
)

string_flag(
    name = "version",
    build_setting_default = "dev",
)

string_flag(
    name = "os",
    build_setting_default = "unknown",
)

string_flag(
    name = "arch",
    build_setting_default = "unknown",
)

xls_release(
    name = "xls_release_tar_vars",
    version = ":version",
    os = ":os",
    arch = ":arch",
)


py_package(
    name = "xls_pkg",
    deps = [
      "//xls/scheduling:pipeline_schedule_py_pb2",
      "//xls/codegen:xls_metrics_py_pb2",
      "//xls/codegen:module_signature_py_pb2",
      "//xls/ir:op_py_pb2",
    ],
)

py_wheel(
    name = "xls_dist_wheel",
    distribution = "xls",
    python_tag = "py3",
    version = "0.0.0",
    deps = [":xls_pkg"],
)

pkg_files(
    name = "xls_stdlib_dslx",
    srcs = [
	"//xls/dslx/stdlib:x_files",
    ],
    prefix = "xls/dslx/stdlib",
    strip_prefix = strip_prefix.from_pkg(),
)

pkg_tar(
    name = "xls_dist_tar",
    extension = "tar.gz",
    package_file_name = "xls-{version}-{os}-{arch}.tar.gz",
    package_dir = "xls-{version}-{os}-{arch}",
    srcs = [
	":THIRD_PARTY_NOTICES.txt",
	"//:LICENSE",
	"//xls/dslx:interpreter_main",
	"//xls/dslx/ir_convert:ir_converter_main",
	"//xls/tools:codegen_main",
	"//xls/tools:opt_main",
	"//xls/tools:proto_to_dslx_main",
	":xls_stdlib_dslx",
    ],
    stamp = 1,
    package_variables = ":xls_release_tar_vars",
)
