load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//xls:xls_internal"],
)

license(
    name = "license",
    package_name = "xls_pixels",
)

licenses(["notice"])

exports_files(["LICENSE"])

cc_binary(
    name = "xls_pixels",
    srcs = ["xls_pixels.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_helpers",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)
