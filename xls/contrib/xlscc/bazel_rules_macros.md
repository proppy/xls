<!-- DO NOT EDIT THIS FILE -->
<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Bazel Rules And Macros

<!-- DO NOT REMOVE! Placeholder for TOC. -->

<a id="xls_cc_ir"></a>

## xls_cc_ir

<pre>
load("//xls/contrib/xlscc/build_rules:xlscc_build_defs.bzl", "xls_cc_ir")

xls_cc_ir(<a href="#xls_cc_ir-name">name</a>, <a href="#xls_cc_ir-src">src</a>, <a href="#xls_cc_ir-block">block</a>, <a href="#xls_cc_ir-block_pb_out">block_pb_out</a>, <a href="#xls_cc_ir-block_from_class">block_from_class</a>, <a href="#xls_cc_ir-src_deps">src_deps</a>, <a href="#xls_cc_ir-xlscc_args">xlscc_args</a>,
          <a href="#xls_cc_ir-enable_generated_file">enable_generated_file</a>, <a href="#xls_cc_ir-enable_presubmit_generated_file">enable_presubmit_generated_file</a>, <a href="#xls_cc_ir-metadata_out">metadata_out</a>, <a href="#xls_cc_ir-kwargs">**kwargs</a>)
</pre>

A macro that instantiates a build rule generating an IR file from a C/C++ source file.

The macro instantiates a rule that converts a C/C++ source file to an IR
file and the 'enable_generated_file_wrapper' function. The generated files
are listed in the outs attribute of the rule.

Examples:

1) A simple IR conversion example. Assume target 'a_block_pb' is defined.

```
    xls_cc_ir(
        name = "a_ir",
        src = "a.cc",
        block = ":a_block_pb",
    )
```


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="xls_cc_ir-name"></a>name |  The name of the rule.   |  none |
| <a id="xls_cc_ir-src"></a>src |  The C/C++ source file containing the top level block. A single source file must be provided. The file must have a '.cc' extension.   |  none |
| <a id="xls_cc_ir-block"></a>block |  Protobuf describing top-level block interface. A single source file single source file must be provided. The file must have a '.protobin' or a '.binarypb' extension. To create this protobuf automatically from your C++ source file, use 'block_from_class' instead. Exactly one of 'block' or 'block_from_class' should be specified.   |  `None` |
| <a id="xls_cc_ir-block_pb_out"></a>block_pb_out |  Protobuf describing top-level block interface, same as block, but an output used with block-from-class.   |  `None` |
| <a id="xls_cc_ir-block_from_class"></a>block_from_class |  Filename of the generated top-level block interface protobuf created from a C++ class. To manually specify this protobuf, use 'block' instead. Exactly one of 'block' or 'block_from_class' should be specified.   |  `None` |
| <a id="xls_cc_ir-src_deps"></a>src_deps |  Additional source files for the rule. The file must have a '.cc', '.h' or '.inc' extension.   |  `[]` |
| <a id="xls_cc_ir-xlscc_args"></a>xlscc_args |  Arguments of the XLSCC conversion tool.   |  `{}` |
| <a id="xls_cc_ir-enable_generated_file"></a>enable_generated_file |  See 'enable_generated_file' from 'enable_generated_file_wrapper' function.   |  `True` |
| <a id="xls_cc_ir-enable_presubmit_generated_file"></a>enable_presubmit_generated_file |  See 'enable_presubmit_generated_file' from 'enable_generated_file_wrapper' function.   |  `False` |
| <a id="xls_cc_ir-metadata_out"></a>metadata_out |  Generated metadata proto.   |  `None` |
| <a id="xls_cc_ir-kwargs"></a>kwargs |  Keyword arguments. Named arguments.   |  none |


<a id="xls_cc_verilog"></a>

## xls_cc_verilog

<pre>
load("//xls/contrib/xlscc/build_rules:xlscc_build_defs.bzl", "xls_cc_verilog")

xls_cc_verilog(<a href="#xls_cc_verilog-name">name</a>, <a href="#xls_cc_verilog-src">src</a>, <a href="#xls_cc_verilog-block">block</a>, <a href="#xls_cc_verilog-verilog_file">verilog_file</a>, <a href="#xls_cc_verilog-src_deps">src_deps</a>, <a href="#xls_cc_verilog-xlscc_args">xlscc_args</a>, <a href="#xls_cc_verilog-opt_ir_args">opt_ir_args</a>, <a href="#xls_cc_verilog-codegen_args">codegen_args</a>,
               <a href="#xls_cc_verilog-enable_generated_file">enable_generated_file</a>, <a href="#xls_cc_verilog-enable_presubmit_generated_file">enable_presubmit_generated_file</a>, <a href="#xls_cc_verilog-kwargs">**kwargs</a>)
</pre>

A macro that instantiates a build rule generating a Verilog file from a C/C++ source file.

The macro instantiates a build rule that generates a Verilog file from
a DSLX source file. The build rule executes the core functionality of
following macros:

1. xls_cc_ir (converts a C/C++ file to an IR),
1. xls_ir_opt_ir (optimizes the IR), and,
1. xls_ir_verilog (generated a Verilog file).

Examples:

1) A simple example. Assume target 'a_block_pb' is defined.

```
    xls_cc_verilog(
        name = "a_verilog",
        src = "a.cc",
        block = ":a_block_pb",
        codegen_args = {
            "generator": "combinational",
            "module_name": "A",
            "top": "A_proc",
        },
    )
```


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="xls_cc_verilog-name"></a>name |  The name of the rule.   |  none |
| <a id="xls_cc_verilog-src"></a>src |  The C/C++ source file containing the top level block. A single source file must be provided. The file must have a '.cc' extension.   |  none |
| <a id="xls_cc_verilog-block"></a>block |  Protobuf describing top-level block interface. A single source file single source file must be provided. The file must have a '.protobin' , '.pbtxt', or a '.binarypb' extension.   |  none |
| <a id="xls_cc_verilog-verilog_file"></a>verilog_file |  The filename of Verilog file generated. The filename must have a '.v' extension.   |  none |
| <a id="xls_cc_verilog-src_deps"></a>src_deps |  Additional source files for the rule. The file must have a '.cc', '.h' or '.inc' extension.   |  `[]` |
| <a id="xls_cc_verilog-xlscc_args"></a>xlscc_args |  Arguments of the XLSCC conversion tool.   |  `{}` |
| <a id="xls_cc_verilog-opt_ir_args"></a>opt_ir_args |  Arguments of the IR optimizer tool. For details on the arguments, refer to the opt_main application at //xls/tools/opt_main.cc. Note: the 'top' argument is not assigned using this attribute.   |  `{}` |
| <a id="xls_cc_verilog-codegen_args"></a>codegen_args |  Arguments of the codegen tool. For details on the arguments, refer to the codegen_main application at //xls/tools/codegen_main.cc.   |  `{}` |
| <a id="xls_cc_verilog-enable_generated_file"></a>enable_generated_file |  See 'enable_generated_file' from 'enable_generated_file_wrapper' function.   |  `True` |
| <a id="xls_cc_verilog-enable_presubmit_generated_file"></a>enable_presubmit_generated_file |  See 'enable_presubmit_generated_file' from 'enable_generated_file_wrapper' function.   |  `False` |
| <a id="xls_cc_verilog-kwargs"></a>kwargs |  Keyword arguments. Named arguments.   |  none |


