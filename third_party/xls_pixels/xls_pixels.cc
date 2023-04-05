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

#include <filesystem>
#include <vector>
#include <fstream>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/dslx/parse_and_typecheck.h"

const char kUsage[] = R"(
Evaluate a DSLX file implementing a pixel color function (returnining an ARGB color for a given pixel coordinate) and dump result into a raw texture file.

   Examples:
   xls_pixels DSLX_FILE
)";

ABSL_FLAG(std::string, entry, "main", "Entry function to evaluate.");
ABSL_FLAG(std::string, output_path, "texture.raw", "Output file.");

namespace xls {

absl::Status RealMain(
    std::filesystem::path dslx_path,
    std::filesystem::path texture_path,
    std::string_view entry_fn_name) {
  dslx::ImportData import_data(
      dslx::CreateImportData(kDefaultDslxStdlibPath, {}));

  XLS_ASSIGN_OR_RETURN(std::string dslx_text, GetFileContents(dslx_path));
  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule tm,
      dslx::ParseAndTypecheck(dslx_text, std::string(dslx_path), "the_module",
                              &import_data));
  XLS_ASSIGN_OR_RETURN(
      dslx::Function * f,
      tm.module->GetMemberOrError<dslx::Function>(entry_fn_name));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::BytecodeFunction> bf,
                       dslx::BytecodeEmitter::Emit(&import_data, tm.type_info,
                                                   f, std::nullopt));

  std::ofstream ofs(texture_path);
  for (uint32_t y = 0; y < 240; ++y) {
    for (uint32_t x = 0; x < 320; ++x) {
      std::vector<dslx::InterpValue> args = {
        dslx::InterpValue::MakeU32(x),
        dslx::InterpValue::MakeU32(y)
      };
      XLS_ASSIGN_OR_RETURN(
          dslx::InterpValue result,
          dslx::BytecodeInterpreter::Interpret(&import_data, bf.get(), args));
      for (dslx::InterpValue value : result.GetValuesOrDie()) {
        ofs << value.GetBitsOrDie().ToBytes()[0];
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);
  XLS_QCHECK_EQ(positional_arguments.size(), 1) << absl::StreamFormat(
      "Expected invocation: %s <DSLX path>", argv[0]);

  std::string entry_fn_name = absl::GetFlag(FLAGS_entry);
  std::string output_path = absl::GetFlag(FLAGS_output_path);
  XLS_QCHECK_OK(xls::RealMain(positional_arguments[0],
                              output_path,
                              entry_fn_name));
  return 0;
}
