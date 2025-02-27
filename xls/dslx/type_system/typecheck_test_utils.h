// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPECHECK_TEST_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPECHECK_TEST_UTILS_H_

#include <memory>
#include <string_view>

#include "gmock/gmock.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "re2/re2.h"

namespace xls::dslx {

struct TypecheckResult {
  // If `import_data` is not dependency injected into the `Typecheck` routine,
  // we create import data, and it owns objects with lifetimes we need for the
  // `TypecheckedModule` (e.g. the `FileTable`) so we provide it in the result.
  std::unique_ptr<ImportData> import_data;
  TypecheckedModule tm;
};

// Helper for parsing/typechecking a snippet of DSLX text.
//
// If `import_data` is not provided one is created for internal use.
absl::StatusOr<TypecheckResult> Typecheck(std::string_view text);

// Variant that prepends the `type_inference_v2` DSLX module attribute to
// `program` to force the use of type_system_v2.
absl::StatusOr<TypecheckResult> TypecheckV2(std::string_view program);

// Verifies that a failed typecheck status message indicates a type mismatch
// between the given two types in string format.
MATCHER_P2(HasTypeMismatch, type1, type2, "") {
  return ExplainMatchResult(
      AnyOf(::testing::ContainsRegex(absl::Substitute(R"($0.*\n?vs\.? $1)",
                                                      RE2::QuoteMeta(type1),
                                                      RE2::QuoteMeta(type2))),
            ::testing::ContainsRegex(absl::Substitute(R"($1.*\n?vs\.? $0)",
                                                      RE2::QuoteMeta(type1),
                                                      RE2::QuoteMeta(type2)))),
      arg, result_listener);
}

// Verifies that a failed typecheck status message indicates a size mismatch
// between the given two types in string format.
MATCHER_P2(HasSizeMismatch, type1, type2, "") {
  return ExplainMatchResult(
      AnyOf(::testing::ContainsRegex(absl::Substitute(R"($0.*\n?vs\.? $1)",
                                                      RE2::QuoteMeta(type1),
                                                      RE2::QuoteMeta(type2))),
            ::testing::ContainsRegex(absl::Substitute(R"($1.*\n?vs\.? $0)",
                                                      RE2::QuoteMeta(type1),
                                                      RE2::QuoteMeta(type2)))),
      arg, result_listener);
}

// Verifies that a failed typecheck status message indicates a cast error
// between the given two types in string format.
MATCHER_P2(HasCastError, from_type, to_type, "") {
  return ExplainMatchResult(
      ::testing::ContainsRegex(
          absl::Substitute("Cannot cast from type `$0` to type `$1`",
                           RE2::QuoteMeta(from_type), RE2::QuoteMeta(to_type))),
      arg, result_listener);
}

// Verifies that a failed typecheck status message indicates a signedness
// mismatch between the given two types in string format.
MATCHER_P2(HasSignednessMismatch, type1, type2, "") {
  return ExplainMatchResult(
      AnyOf(::testing::ContainsRegex(absl::Substitute(
                R"(signed vs\.? unsigned mismatch.*$0.* vs. $1)",
                RE2::QuoteMeta(type1), RE2::QuoteMeta(type2))),
            ::testing::ContainsRegex(absl::Substitute(
                R"(signed vs\.? unsigned mismatch.*$1.* vs. $0)",
                RE2::QuoteMeta(type1), RE2::QuoteMeta(type2)))),
      arg, result_listener);
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPECHECK_TEST_UTILS_H_
