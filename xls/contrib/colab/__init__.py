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

import os
import pathlib

from xls.codegen import module_signature_pb2
from xls.codegen import xls_metrics_pb2
from xls.ir import op_pb2
from xls.scheduling import pipeline_schedule_pb2

default_work_dir = pathlib.Path('xls_work_dir')
default_work_dir.mkdir(exist_ok=True)

def xls_path(p):
  path = pathlib.Path(p)
  return path.resolve() if path.exists() else p

interpreter_main = xls_path('interpreter_main')
ir_converter_main = xls_path('ir_converter_main')
opt_main = xls_path('opt_main')
codegen_main = xls_path('codegen_main')

# TODO(proppy): bundle varint simulation tool
# TODO(proppy): add back dslx_parameters
# TODO(proppy): unify colabtools.shell and subprocess.run

import subprocess
from google.colab import widgets
from typing import Any, Callable, Dict, Optional, Union
from IPython.core import magic_arguments
from IPython.core.magic import register_cell_magic
import dataclasses
import pandas as pd
from google.protobuf import text_format
import enum
import json

pdk = 'asap7'

class XlsRuntimeError(RuntimeError):
  pass

def parse_dslx_argstring(
    magic: Callable[[str, str], None], line: str
) -> (Dict[str, Any], Dict[str, int]):
  """Parse parameters from the %%dslx cell args and the global python context.

  Args:
    magic: %%dslx magic.
    line: %%dslx magic line w/ arguments.

  Returns:
    A tuple containing:
    - A dict of toolchain parameters.
  """
  args = magic_arguments.parse_argstring(magic, line)
  magic_kwargs = {k: v for (k, v) in args._get_kwargs() if v is not None}  # pylint: disable=protected-access
  toolchain_kwargs = magic_kwargs
  return toolchain_kwargs

def register_dslx_magic():
  """Register the %%dslx magic in the IPython context."""
  return register_cell_magic(dslx)

@magic_arguments.magic_arguments()
@magic_arguments.argument('--top')
@magic_arguments.argument('--reset')
@magic_arguments.argument('--pipeline_stages')
@magic_arguments.argument('--clock_period_ps')
@magic_arguments.argument('--clock_margin_percent')
@magic_arguments.argument('--io_constraints')
@magic_arguments.argument('--flop_inputs')
@magic_arguments.argument('--flop_outputs')
@magic_arguments.argument('--worst_case_throughput')
@magic_arguments.argument('--generator')
@magic_arguments.argument('--pdk')
def dslx(line: str, cell: str):
  """Run the content of the cell thru the XLS toolchain.

  Args:
    line: optionally takes command-line style args for the XLS tools.
    cell: content of the cell.

  Usage:
    %%dslx --top ENTRYPOINT [--pipeline_stages N]
      [--design_parameters=PARAM1:VALUE1,PARAM2:VALUE2]
  """
  toolchain_kwargs = parse_dslx_argstring(dslx, line)
  selected_pdk = toolchain_kwargs.pop('pdk', pdk)
  delay_model = selected_pdk
  tb = widgets.TabBar(
      [
          'interpreter',
          'ir',
          'opt',
          'verilog',
          'schedule',
          'bom',
      ]
  )
  try:
    run_xls_toolchain(
        cell,
        use_system_verilog='true',
        delay_model=delay_model,
        tb=tb,
        **toolchain_kwargs,
    )
  except XlsRuntimeError as e:
    display(e)
    pass  # don't re-raise exception to keep tabs readable
    # this has the unfortunate effect of not marking the cell as failed.

@dataclasses.dataclass(frozen=True)
class XlsToolchainOutputs:
  top: str
  top_x: pathlib.Path
  top_ir: pathlib.Path
  top_opt_ir: pathlib.Path
  top_block_ir: pathlib.Path
  top_v: pathlib.Path
  top_schedule_proto: pathlib.Path
  top_signature_proto: pathlib.Path

@dataclasses.dataclass(frozen=True)
class XlsToolchainMetrics:
  schedule: pd.DataFrame
  parts: pd.DataFrame
  bom: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class XlsToolchainResults:
  outputs: XlsToolchainOutputs
  metrics: XlsToolchainMetrics


def xls_test_dslx(
    top_x: pathlib.Path,
    work_dir: Optional[pathlib.Path] = None,
    silent: bool = False,
):
  """Run tests with the DSLX interpreter."""
  if work_dir is None:
    work_dir = top_x.parent
  p = subprocess.run([
        interpreter_main,
        f'--dslx_path={work_dir}',
        top_x
    ],
    capture_output=True,
    check=False
  )
  print(p.stderr.decode('utf-8'))
  parse_or_typecheck_error = b'[        FAILED ]' not in p.stderr
  if p.returncode != 0 and parse_or_typecheck_error:
    raise XlsRuntimeError(p.stderr)


def xls_ir_conversion(
    *,
    top: str,
    top_ir: pathlib.Path,
    top_x: pathlib.Path,
    work_dir: Optional[pathlib.Path] = None,
    silent: bool = False,
):
  """Run XLS IR conversion."""
  if work_dir is None:
    work_dir = top_ir.parent
  p = subprocess.run([
        ir_converter_main,
        f'--top={top}',
        f'--dslx_path={work_dir}',
        top_x
    ],
    capture_output=True,
    check=False,
  )
  if p.returncode != 0:
    raise XlsRuntimeError(p.stderr)
  print(p.stdout.decode('utf-8'))
  with open(top_ir, 'wb') as f:
    f.write(p.stdout)


def xls_ir_opt(
    *, top_opt_ir: pathlib.Path, top_ir: pathlib.Path, silent: bool = False
):
  """Run XLS IR opt."""
  p = subprocess.run([
          opt_main,
          top_ir
      ],
      capture_output=True,
      check=False
  )
  if p.returncode != 0:
    raise XlsRuntimeError(p.stderr)
  print(p.stdout.decode('utf-8'))
  with open(top_opt_ir, 'wb') as f:
    f.write(p.stdout)


@dataclasses.dataclass(frozen=True)
class XlsCodegenOutputs:
  top_v: pathlib.Path
  top_schedule_proto: pathlib.Path
  top_signature_proto: pathlib.Path
  top_block_ir: pathlib.Path


def xls_codegen(
    *,
    top_v: pathlib.Path,
    top_opt_ir: pathlib.Path,
    work_dir: Optional[pathlib.Path] = None,
    silent: bool = False,
    **kwargs: Dict[str, str],
) -> XlsCodegenOutputs:
  """Run XLS codegen."""
  if work_dir is None:
    work_dir = top_v.parent
  top_schedule_proto = work_dir / 'user_module_schedule.prototext'
  top_signature_proto = work_dir / 'user_module_signature.prototext'
  top_block_ir = work_dir / 'user_module_block.ir'
  codegen_args = [
      'delay_model',
      'clock_period_ps',
      'pipeline_stages',
      'reset',
      'worst_case_throughput',
      'io_constraints',
      'flop_inputs',
      'flop_outputs',
      'generator',
      'use_system_verilog',
      'streaming_channel_data_suffix',
      'streaming_channel_ready_suffix',
      'streaming_channel_valid_suffix',
  ]
  codegen_args = [
      f'--{k}={kwargs.pop(k)}' for k in codegen_args if k in kwargs
  ]
  if kwargs:
    display(f'Unexpected codegen args {kwargs}')
    raise XlsRuntimeError(f'Unexpected codegen args {kwargs}')

  p = subprocess.run([
      codegen_main,
      '--module_name=user_module',
      f'--output_block_ir_path={top_block_ir}',
      f'--output_schedule_path={top_schedule_proto}',
      f'--output_signature_path={top_signature_proto}',
      '--streaming_channel_data_suffix=_data',
      '--streaming_channel_valid_suffix=_valid',
      '--streaming_channel_ready_suffix=_ready',
  ] + codegen_args + [top_opt_ir],
      capture_output=True,
      check=False,
  )
  if p.returncode != 0:
    raise XlsRuntimeError(p.stderr)
  print(p.stdout.decode('utf-8'))
  with open(top_v, 'wb') as f:
    f.write(p.stdout)

  return XlsCodegenOutputs(
      top_v=top_v,
      top_schedule_proto=top_schedule_proto,
      top_signature_proto=top_signature_proto,
      top_block_ir=top_block_ir,
  )

def load_schedule(codegen_outputs: XlsCodegenOutputs) -> pd.DataFrame:
  """Load schedule from proto."""
  # compute pipeline delays
  with codegen_outputs.top_schedule_proto.open('r') as f:
    proto = pipeline_schedule_pb2.PackagePipelineSchedulesProto()
    text_format.Parse(f.read(), proto)

  def pipeline_schedule_delays(proto):
    for s in proto.stages:
      if not s.timed_nodes:
        yield s.stage, 'noop', 0
      for n in s.timed_nodes:
        yield s.stage, n.node, n.node_delay_ps, n.path_delay_ps

  df_schedule = pd.DataFrame.from_records(
      pipeline_schedule_delays(list(proto.schedules.values())[0]),
      columns=['stage', 'node', 'node_delay_ps', 'path_delay_ps'],
  )
  return df_schedule


def load_parts_and_bom(
    codegen_outputs: XlsCodegenOutputs,
) -> (pd.DataFrame, pd.DataFrame):
  """Load parts and BOM from module signature."""
  with codegen_outputs.top_signature_proto.open('r') as f:
    proto = module_signature_pb2.ModuleSignatureProto()
    text = f.read()
    text_format.Parse(text, proto)

    def bom_parts(module_signature_proto):
      for _ in range(proto.metrics.block_metrics.flop_count):
        yield 'FLOP', 'MISC', None, None
      for p in module_signature_proto.metrics.block_metrics.bill_of_materials:
        yield (
            op_pb2.OpProto.Name(p.op).replace('OP_', ''),
            xls_metrics_pb2.BomKindProto.Name(p.kind).replace('BOM_KIND_', ''),
            p.maximum_input_width,
            p.output_width,
            p.number_of_arguments,
        )

    df_parts = pd.DataFrame.from_records(
        bom_parts(proto),
        columns=['op', 'kind', 'output', 'input', 'arguments'],
    ).convert_dtypes()
    df_bom = (
        df_parts.groupby('kind')
        .value_counts(['op', 'output', 'input', 'arguments'])
        .to_frame('count')
    )
  return df_parts, df_bom


def run_xls_toolchain(
    code: str,
    top: str,
    tb: widgets.TabBar,
    work_dir: pathlib.Path = default_work_dir,
    **kwargs: Dict[str, str],
) -> XlsToolchainResults:
  """Run the XLS toolchain.

  Args:
    code: DSLX code.
    top: Name of the top DSLX function/proc.
    tb: TabBar to display intermediate results on. Must have 'interpreter',
      'ir', 'opt' and 'verilog' tabs.
    work_dir: Directory to put outputs.
    **kwargs: XLS toolchain args.

  Returns:
    XLS outputs and metrics.
  """
  # run DSLX interpreter
  top_x = work_dir / 'user_module.x'
  with top_x.open('w') as f:
    f.write('// dslx \n')  # add placeholder line to fix line count
    f.write(code)
  with tb.output_to('interpreter', select=True):
    xls_test_dslx(top_x)
  # run XLS IR converter
  with tb.output_to('ir', select=False):
    top_ir = work_dir / 'user_module.ir'
    xls_ir_conversion(top=top, top_ir=top_ir, top_x=top_x)
  # run XLS IR optimizer
  with tb.output_to('opt', select=False):
    top_opt_ir = work_dir / 'user_module_opt.ir'
    xls_ir_opt(top_opt_ir=top_opt_ir, top_ir=top_ir)
  # run verilog codegen
  with tb.output_to('verilog', select=False):
    top_v = work_dir / 'user_module.sv'
    codegen_outputs = xls_codegen(
        top_v=top_v,
        top_opt_ir=top_opt_ir,
        **kwargs,
    )

  # visualize pipeline delays
  df_schedule = load_schedule(codegen_outputs)
  with tb.output_to('schedule', select=False):
    display(
        df_schedule.sort_values(by=['stage', 'path_delay_ps'])
        .style.hide(axis='index')
        .background_gradient(subset=['stage'], cmap='tab20', vmax=20)
        .background_gradient(subset=['node_delay_ps'], cmap='Oranges')
        .bar(subset=['path_delay_ps'], color='lightblue')
    )
  # visualize bom
  df_parts, df_bom = load_parts_and_bom(codegen_outputs)

  with tb.output_to('bom', select=False):
    display(
        df_bom.style.format(precision=2).bar(
            subset=['count'], color='lightblue'
        )
    )

  return XlsToolchainResults(
      outputs=XlsToolchainOutputs(
          top=top,
          top_x=top_x,
          top_ir=top_ir,
          top_opt_ir=top_opt_ir,
          top_block_ir=codegen_outputs.top_block_ir,
          top_v=codegen_outputs.top_v,
          top_schedule_proto=codegen_outputs.top_schedule_proto,
          top_signature_proto=codegen_outputs.top_signature_proto,
      ),
      metrics=XlsToolchainMetrics(
          schedule=df_schedule,
          parts=df_parts,
          bom=df_bom,
      ),
  )
