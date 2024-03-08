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
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load(":dist.bzl", "xls_release")

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
    name = "xls_package_vars",
    version = ":version",
    os = ":os",
    arch = ":arch",
)

pkg_tar(
    name = "xls_dist_tar",
    extension = "tar.gz",
    package_file_name = "xls-{version}-{os}-{arch}.tar.gz",
    package_dir = "xls-{version}-{os}-{arch}",
    include_runfiles = True,
    srcs = [
	":THIRD_PARTY_NOTICES.txt",
	"//:LICENSE",
	"//xls/dslx:interpreter_main",
	"//xls/dslx/ir_convert:ir_converter_main",
	"//xls/tools:codegen_main",
	"//xls/tools:opt_main",
	"//xls/tools:proto_to_dslx_main",
	"//xls/scheduling:pipeline_schedule_py_pb2",
	"//xls/codegen:xls_metrics_py_pb2",
	"//xls/codegen:module_signature_py_pb2",	
	"//xls/ir:op_py_pb2",
    ],
    symlinks = {
       "./interpreter_main": "interpreter_main.runfiles/xls/dslx/interpreter_main",
       "./ir_converter_main": "ir_converter_main.runfiles/xls/dslx/ir_convert/ir_converter_main",
       "./codegen_main": "codegen_main.runfiles/xls/tools/codegen_main",
       "./opt_main": "opt_main.runfiles/xls/tools/opt_main",
       "./proto_to_dslx_main": "proto_to_dslx_main.runfiles/xls/tools/proto_to_dslx_main",
       "./xls/codegen/module_signature_pb2.py": "../../op_pb2.py.runfiles/xls/codegen/module_signature_pb2.py",
       "./xls/ir/op_pb2.py": "../../op_pb2.py.runfiles/xls/ir/op_pb2.py",
       "./xls/scheduling/pipeline_schedule_pb2.py": "../../pipeline_schedule_pb2.py.runfiles/xls/scheduling/pipeline_schedule_pb2.py",
       "./xls/codegen/xls_metrics_pb2.py": "../../op_pb2.py.runfiles/xls/codegen/xls_metrics_pb2.py",
       "./xls/ir/channel_pb2.py": "../../op_pb2.py.runfiles/xls/ir/channel_pb2.py",
       "./google/prototobuf/descriptor_pb2.py": "../../com_google_protobuf/python/google/protobuf/descriptor_pb2.py",
       "./google/api/field_behavior_pb2.py": "../../com_google_googleapis/google/api/field_behavior_pb2.py",
       "./xls/ir/foreign_function_data_pb2.py": "../../op_pb2.py.runfiles/xls/ir/foreign_function_data_pb2.py", 
       "./xls/ir/xls_type_pb2.py": "../../op_pb2.py.runfiles/xls/ir/xls_type_pb2.py",
    },
    stamp = 1,
    package_variables = ":xls_package_vars",
)
