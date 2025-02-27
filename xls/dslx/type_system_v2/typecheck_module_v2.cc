// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system_v2/typecheck_module_v2.h"

#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_to_type_info.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// A visitor that walks an AST and populates an `InferenceTable` with the
// encountered info.
class PopulateInferenceTableVisitor : public AstNodeVisitorWithDefault {
 public:
  PopulateInferenceTableVisitor(Module& module, InferenceTable& table,
                                const FileTable& file_table)
      : module_(module), table_(table), file_table_(file_table) {}

  absl::Status HandleConstantDef(const ConstantDef* node) override {
    VLOG(5) << "HandleConstantDef: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(const NameRef* variable,
                         DefineTypeVariableForVariableOrConstant(node));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->value(), variable));
    return DefaultHandler(node);
  }

  absl::Status HandleParam(const Param* node) override {
    VLOG(5) << "HandleParam: " << node->ToString();
    XLS_RETURN_IF_ERROR(DefineTypeVariableForVariableOrConstant(node).status());
    return DefaultHandler(node);
  }

  absl::Status HandleNameRef(const NameRef* node) override {
    VLOG(5) << "HandleNameRef: " << node->ToString();
    return PropagateDefToRef(node);
  }

  absl::Status HandleColonRef(const ColonRef* node) override {
    VLOG(5) << "HandleColonRef: " << node->ToString() << " of subject kind: "
            << AstNodeKindToString(ToAstNode(node->subject())->kind());
    if (std::holds_alternative<NameRef*>(node->subject())) {
      if (const NameDef* name_def = dynamic_cast<const NameDef*>(
              ToAstNode(std::get<NameRef*>(node->subject())->name_def()))) {
        if (const auto* struct_def =
                dynamic_cast<const StructDefBase*>(name_def->definer())) {
          XLS_ASSIGN_OR_RETURN(std::optional<const AstNode*> def,
                               HandleStructAttributeReferenceInternal(
                                   node, *struct_def, {}, node->attr()));
          if (def.has_value()) {
            return PropagateDefToRef(*def, node);
          }
        }
      }
    }
    if (std::holds_alternative<TypeRefTypeAnnotation*>(node->subject())) {
      // This is something like `S<parametrics>::CONSTANT` or
      // `S<parametrics>::static_fn`. We can't fully resolve these things on
      // the spot, so we do some basic validation and then produce a
      // `MemberTypeAnnotation` for deferred resolution.
      const auto* annotation =
          std::get<TypeRefTypeAnnotation*>(node->subject());
      XLS_RETURN_IF_ERROR(annotation->Accept(this));
      XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                           GetStructOrProcRef(annotation, file_table_));
      if (struct_or_proc_ref.has_value()) {
        XLS_RETURN_IF_ERROR(HandleStructAttributeReferenceInternal(
                                node, *struct_or_proc_ref->def,
                                struct_or_proc_ref->parametrics, node->attr())
                                .status());
      }
      return table_.SetTypeAnnotation(
          node, module_.Make<MemberTypeAnnotation>(annotation, node->attr()));
    }
    return absl::UnimplementedError(
        "Type inference version 2 is a work in progress and has limited "
        "support for colon references so far.");
  }

  absl::Status HandleNumber(const Number* node) override {
    VLOG(5) << "HandleNumber: " << node->ToString();
    TypeAnnotation* annotation = node->type_annotation();
    if (annotation == nullptr) {
      XLS_ASSIGN_OR_RETURN(annotation,
                           CreateAnnotationSizedToFit(module_, *node));
      // Treat `true` and `false` like they have intrinsic bool annotations.
      // Otherwise, consider an annotation we add to be an auto-annotation that
      // is "negotiable".
      if (node->number_kind() != NumberKind::kBool) {
        table_.MarkAsAutoLiteral(annotation);
      }
    } else {
      XLS_RETURN_IF_ERROR(annotation->Accept(this));
    }
    return table_.SetTypeAnnotation(node, annotation);
  }

  absl::Status HandleBinop(const Binop* node) override {
    VLOG(5) << "HandleBinop: " << node->ToString();

    // Any `Binop` should be a descendant of some context-setting node and
    // should have a type that was set when its parent was visited.
    const NameRef* type_variable = *table_.GetTypeVariable(node);
    if (GetBinopSameTypeKinds().contains(node->binop_kind()) ||
        GetBinopLogicalKinds().contains(node->binop_kind())) {
      // In the example `const C = a + b;`, the `ConstantDef` establishes a type
      // variable that is just propagated down to `a` and `b` here, meaning that
      // `a`, `b`, and the result must ultimately be the same type.
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), type_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->rhs(), type_variable));
    } else if (GetBinopComparisonKinds().contains(node->binop_kind())) {
      // In a comparison example, like `const C = a > b;`, the `>` establishes a
      // new type variable for `a` and `b` (meaning the two of them must be the
      // same type), and attaches a bool annotation to the overall expression,
      // which will then be assumed by the type variable for the `ConstantDef`.
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node, CreateBoolAnnotation(module_, node->span())));
      XLS_ASSIGN_OR_RETURN(
          const NameRef* operand_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Binop*>(node),
              GenerateInternalTypeVariableName(node)));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(node->lhs(), operand_variable));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(node->rhs(), operand_variable));
    } else if (GetBinopShifts().contains(node->binop_kind())) {
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), type_variable));
      XLS_ASSIGN_OR_RETURN(const NameRef* rhs_variable,
                           table_.DefineInternalVariable(
                               InferenceVariableKind::kType, node->rhs(),
                               GenerateInternalTypeVariableName(node->rhs())));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->rhs(), rhs_variable));
    } else if (node->binop_kind() == BinopKind::kConcat) {
      // The type of a concat is
      //   ArrayType(ElementType(lhs),
      //             element_count<lhs_var>() + element_count<rhs_var>())
      //
      // which is bits-like if the element type amounts to a built-in bits type;
      // otherwise, it's a real array.
      //
      // There is a nontrivial set of rules for what input types are actually
      // allowed, and the application of those rules is deferred until
      // `ValidateConcreteType` at the end.
      XLS_ASSIGN_OR_RETURN(
          const NameRef* lhs_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
              GenerateInternalTypeVariableName(node->lhs())));
      XLS_ASSIGN_OR_RETURN(
          const NameRef* rhs_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(node->rhs()),
              GenerateInternalTypeVariableName(node->rhs())));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), lhs_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->rhs(), rhs_variable));
      auto* lhs_tvta = module_.Make<TypeVariableTypeAnnotation>(lhs_variable);
      auto* rhs_tvta = module_.Make<TypeVariableTypeAnnotation>(rhs_variable);
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node, module_.Make<ArrayTypeAnnotation>(
                    node->span(),
                    module_.Make<ElementTypeAnnotation>(
                        lhs_tvta, /*tuple_index=*/std::nullopt,
                        /*allow_bit_vector_destructuring=*/true),
                    CreateElementCountSum(module_, lhs_tvta, rhs_tvta))));
    } else {
      return absl::UnimplementedError(
          absl::StrCat("Type inference version 2 is a work in progress and "
                       "does not yet support the expression: ",
                       node->ToString()));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleUnop(const Unop* node) override {
    VLOG(5) << "HandleUnop: " << node->ToString();

    // Any `Unop` should be a descendant of some context-setting node and
    // should have a type that was set when its parent was visited.
    const NameRef* type_variable = *table_.GetTypeVariable(node);
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->operand(), type_variable));

    return DefaultHandler(node);
  }

  absl::Status HandleCast(const Cast* node) override {
    VLOG(5) << "HandleCast: " << node->ToString();

    // Create a new type variable for the casted expression.
    XLS_ASSIGN_OR_RETURN(const NameRef* casted_variable,
                         table_.DefineInternalVariable(
                             InferenceVariableKind::kType, node->expr(),
                             GenerateInternalTypeVariableName(node->expr())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->expr(), casted_variable));

    // The cast node has the target type annotation (assuming it is valid, which
    // will be checked at conversion time).
    const TypeAnnotation* target_type = node->type_annotation();
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, target_type));
    return DefaultHandler(node);
  }

  absl::Status HandleConditional(const Conditional* node) override {
    VLOG(5) << "HandleConditional: " << node->ToString();
    // In the example `const D = if (a) {b} else {c};`, the `ConstantDef`
    // establishes a type variable that is just propagated down to `b` and
    // `c` here, meaning that `b`, `c`, and the result must ultimately be
    // the same type as 'D'. The test 'a' must be a bool, so we annotate it as
    // such.
    const NameRef* type_variable = *table_.GetTypeVariable(node);
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->consequent(), type_variable));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(ToAstNode(node->alternate()), type_variable));

    // Mark the test as bool.
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->test(), CreateBoolAnnotation(module_, node->test()->span())));
    XLS_ASSIGN_OR_RETURN(
        const NameRef* test_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->test()),
            GenerateInternalTypeVariableName(node->test())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->test(), test_variable));

    return DefaultHandler(node);
  }

  absl::Status HandleMatch(const Match* node) override {
    VLOG(5) << "HandleMatch: " << node->ToString();
    // Any `match` should be a descendant of some context-setting node and
    // should have a type that was set when its parent was visited. Each
    // arm of the `match` must match the type of the `match` itself.
    const NameRef* arm_type = *table_.GetTypeVariable(node);

    XLS_ASSIGN_OR_RETURN(
        const NameRef* matched_var_type,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, node->matched(),
            GenerateInternalTypeVariableName(node->matched())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->matched(), matched_var_type));

    for (const MatchArm* arm : node->arms()) {
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(arm->expr(), arm_type));
      for (const NameDefTree* pattern : arm->patterns()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(pattern, matched_var_type));
        if (pattern->is_leaf()) {
          XLS_RETURN_IF_ERROR(table_.SetTypeVariable(ToAstNode(pattern->leaf()),
                                                     matched_var_type));
        }
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleXlsTuple(const XlsTuple* node) override {
    VLOG(5) << "HandleXlsTuple: " << node->ToString();

    // When we come in here with an example like:
    //   const FOO: (u32, (s8, u32)) = (4, (-2, 5));
    //
    // the table will look like this before descent into this function:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                (u32, (s8, u32))    T0
    //   (4, (-2, 5))                           T0
    //
    // and this function will make it look like this:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                (u32, (s8, u32))    T0
    //   (4, (-2, 5))       (var:M0, var:M1)    T0
    //   4                  u32                 M0
    //   (-2, 5)            (s8, u32)           M1
    //
    // Recursive descent will ultimately put auto annotations for the literals
    // in the table. Upon conversion of the table to type info, unification of
    // the LHS annotation with the variable-based RHS annotation will be
    // attempted.

    XLS_ASSIGN_OR_RETURN(
        std::optional<const TypeAnnotation*> tuple_annotation,
        GetDeclarationTypeAnnotation<TupleTypeAnnotation>(node));

    // Create the M0, M1, ... variables and apply them to the members.
    std::vector<TypeAnnotation*> member_types;
    member_types.reserve(node->members().size());
    for (int i = 0; i < node->members().size(); ++i) {
      Expr* member = node->members()[i];
      std::optional<TypeAnnotation*> element_annotation;
      if (tuple_annotation.has_value()) {
        element_annotation = module_.Make<ElementTypeAnnotation>(
            *tuple_annotation,
            module_.Make<Number>((*tuple_annotation)->span(), absl::StrCat(i),
                                 NumberKind::kOther,
                                 /*type_annotation=*/nullptr));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(member, *element_annotation));
      }
      XLS_ASSIGN_OR_RETURN(
          const NameRef* member_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, member,
              GenerateInternalTypeVariableName(member), element_annotation));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(member, member_variable));
      member_types.push_back(
          module_.Make<TypeVariableTypeAnnotation>(member_variable));
    }
    // Annotate the whole tuple expression as (var:M0, var:M1, ...).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<TupleTypeAnnotation>(node->span(), member_types)));
    return DefaultHandler(node);
  }

  absl::Status HandleRestOfTuple(const RestOfTuple* node) override {
    VLOG(5) << "HandleRestOfTuple: " << node->ToString();
    TypeAnnotation* any_type =
        module_.Make<AnyTypeAnnotation>(/*multiple=*/true);
    return table_.SetTypeAnnotation(node, any_type);
  }

  // Recursively define tuple type annotations for the `NameDefTree`. This is
  // similar to `HandleXlsTuple` but without access to an explicit type
  // annotation. It's only necessary when a `NameDefTree` is not associated with
  // a tuple (e.g., outside a `let` assignment).
  absl::Status HandleNameDefTree(const NameDefTree* node) override {
    VLOG(5) << "HandleNameDefTree: " << node->ToString();
    if (node->is_leaf()) {
      return DefaultHandler(node);
    }
    std::vector<TypeAnnotation*> member_types;
    for (const NameDefTree* child_node : node->nodes()) {
      XLS_RETURN_IF_ERROR(child_node->Accept(this));
      XLS_ASSIGN_OR_RETURN(TypeAnnotation * member_type,
                           GetOrMakeTypeAnnotationForNDF(child_node));

      member_types.push_back(member_type);
    }
    TupleTypeAnnotation* tuple_annotation =
        module_.Make<TupleTypeAnnotation>(node->span(), member_types);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, tuple_annotation));
    return DefaultHandler(node);
  }

  absl::Status HandleArrayTypeAnnotation(
      const ArrayTypeAnnotation* node) override {
    VLOG(5) << "HandleArrayTypeAnnotation: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(
        const NameRef* dim_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->dim()),
            GenerateInternalTypeVariableName(node->dim())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->dim(), dim_variable));
    if (const auto* builtin_element =
            dynamic_cast<const BuiltinTypeAnnotation*>(node->element_type());
        builtin_element != nullptr &&
        builtin_element->builtin_type() == BuiltinType::kXN) {
      // For an `xN[S][N]`-style annotation, there is one ArrayTypeAnnotation
      // wrapping another, and so we get into this function twice. The "outer"
      // one has the dimension `N` and an ArrayTypeAnnotation for the element
      // type, and does not come into this if-statement. The "inner" one has the
      // dimension `S` and the `BuiltinType` `kXN` for the element type.
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node->dim(), CreateBoolAnnotation(module_, node->dim()->span())));
    } else {
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node->dim(), CreateU32Annotation(module_, node->dim()->span())));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* node) override {
    VLOG(5) << "HandleTypeRefTypeAnnotation: " << node->ToString();

    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(node, file_table_));
    if (!struct_or_proc_ref.has_value() ||
        struct_or_proc_ref->parametrics.empty()) {
      return DefaultHandler(node);
    }
    const StructDefBase* struct_def = struct_or_proc_ref->def;
    if (struct_or_proc_ref->parametrics.size() >
        struct_def->parametric_bindings().size()) {
      return ArgCountMismatchErrorStatus(
          node->span(),
          absl::Substitute(
              "Too many parametric values supplied; limit: $0 given: $1",
              struct_def->parametric_bindings().size(),
              struct_or_proc_ref->parametrics.size()),
          file_table_);
    }

    // If any parametrics are explicitly specified, then they must all be
    // explicit or defaulted. We technically could infer the rest, as with
    // functions, but historically we choose not to. We must also constrain the
    // actual parametric values to the binding type.
    for (int i = 0; i < struct_def->parametric_bindings().size(); i++) {
      const ParametricBinding* binding = struct_def->parametric_bindings()[i];
      if (i < struct_or_proc_ref->parametrics.size()) {
        const Expr* actual_expr =
            i < struct_or_proc_ref->parametrics.size()
                ? std::get<Expr*>(struct_or_proc_ref->parametrics[i])
                : binding->expr();
        XLS_ASSIGN_OR_RETURN(
            const NameRef* actual_expr_variable,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType, const_cast<Expr*>(actual_expr),
                GenerateInternalTypeVariableName(actual_expr)));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeVariable(actual_expr, actual_expr_variable));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(actual_expr, binding->type_annotation()));
      } else if (binding->expr() == nullptr) {
        return ArgCountMismatchErrorStatus(
            node->span(),
            absl::Substitute("No parametric value provided for `$0` in `$1`",
                             binding->identifier(), struct_def->identifier()),
            file_table_);
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleSelfTypeAnnotation(
      const SelfTypeAnnotation* node) override {
    VLOG(5) << "HandleSelfTypeAnnotation: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* real_type,
                         GetRealTypeAnnotationForSelf(node, file_table_));
    VLOG(5) << "Real TypeAnnotation for Self: " << real_type->ToString();
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(real_type, file_table_));
    // There are two paths for handling of `Self`.
    // - Within a parametric struct, it gets left alone here, and when the
    //   conversion step scrubs struct parametrics via
    //   GetParametricFreeStructMemberType, we finally turn it into
    //   `TheStruct<ActualParametricValues>`.
    // - Within a non-parametric struct, we just equate it to `TheStruct` now,
    //   because the conversion logic will not send it down the parametric
    //   scrubbing path.
    if (!struct_or_proc_ref->def->IsParametric()) {
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, real_type));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStructInstance(const StructInstance* node) override {
    // As far as we're concerned here, type-checking a struct instance is like
    // type-checking a function invocation (see `HandleFreeFunctionInvocation`),
    // but with named arguments instead of parallel ordering. The naming of
    // arguments creates additional pitfalls, like erroneously naming two
    // different arguments the same thing.
    VLOG(5) << "HandleStructInstance: " << node->ToString();

    XLS_RETURN_IF_ERROR(node->struct_ref()->Accept(this));
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(node->struct_ref(), file_table_));
    if (!struct_or_proc_ref.has_value()) {
      return TypeInferenceErrorStatusForAnnotation(
          node->span(), node->struct_ref(),
          absl::Substitute(
              "Attempted to instantiate non-struct type `$0` as a struct.",
              node->struct_ref()->ToString()),
          file_table_);
    }
    if (struct_or_proc_ref->def->kind() == AstNodeKind::kProcDef) {
      return TypeInferenceErrorStatusForAnnotation(
          node->span(), node->struct_ref(),
          "Impl-style procs are a work in progress and cannot yet be "
          "instantiated.",
          file_table_);
    }

    const StructDef* struct_def =
        dynamic_cast<const StructDef*>(struct_or_proc_ref->def);
    const NameRef* type_variable = *table_.GetTypeVariable(node);
    const TypeAnnotation* struct_variable_type =
        module_.Make<TypeVariableTypeAnnotation>(type_variable);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node,
        CreateStructAnnotation(module_, const_cast<StructDef*>(struct_def),
                               struct_or_proc_ref->parametrics, node)));
    XLS_RETURN_IF_ERROR(ValidateStructInstanceMemberNames(*node, *struct_def));

    std::vector<std::pair<std::string, Expr*>> members =
        node->GetOrderedMembers(struct_def);
    for (int i = 0; i < members.size(); i++) {
      const auto& [name, actual_member] = members[i];
      const StructMemberNode* formal_member = struct_def->members()[i];
      TypeAnnotation* member_type = module_.Make<MemberTypeAnnotation>(
          struct_variable_type, formal_member->name());
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(actual_member, member_type));
      XLS_ASSIGN_OR_RETURN(
          const NameRef* member_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(actual_member),
              GenerateInternalTypeVariableName(formal_member, actual_member),
              member_type));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(actual_member, member_type_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          actual_member, module_.Make<MemberTypeAnnotation>(
                             struct_variable_type, formal_member->name())));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleAttr(const Attr* node) override {
    // Establish a context for the unification of the struct type.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* struct_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
            GenerateInternalTypeVariableName(node->lhs())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->lhs(), struct_type_variable));

    // The type of the node itself is basically
    // decltype(struct_type_variable.member).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node,
        module_.Make<MemberTypeAnnotation>(
            module_.Make<TypeVariableTypeAnnotation>(struct_type_variable),
            node->attr())));
    return DefaultHandler(node);
  }

  absl::Status HandleArray(const Array* node) override {
    VLOG(5) << "HandleArray: " << node->ToString();

    // When we come in here with an example like:
    //   const FOO = [u32:4, u32:5];
    //
    // the table will look like this before descent into this function:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                                    T0
    //   [u32:4, u32:5]                         T0
    //
    // and this function will make it look like this:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                                    T0
    //   [u32:4, u32:5]     var:T1[2]           T0
    //   u32:4                                  T1
    //   u32:5                                  T1
    //
    // Recursive descent will ultimately put annotations on the elements in the
    // table. Upon conversion of the table to type info, unification of any LHS
    // annotation with the variable-based RHS annotation will be attempted, and
    // this unification will fail if the array is inadequately annotated (e.g.
    // no explicit annotation on a zero-size or elliptical array).

    XLS_ASSIGN_OR_RETURN(
        std::optional<const TypeAnnotation*> array_annotation,
        GetDeclarationTypeAnnotation<ArrayTypeAnnotation>(node));

    // An empty array can't end with an ellipsis, even if unification is
    // possible.
    if (node->has_ellipsis() && node->members().empty()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          "Array cannot have an ellipsis (`...`) without an element to repeat.",
          file_table_);
    }

    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, node->type_annotation()));

      // If it's a zero-length array literal with a type annotation directly
      // attached, we can at least presume it's meant to be a zero-length array
      // of the element type in the annotation. Otherwise, we know nothing about
      // it, and the early return below will just let it be unified with any LHS
      // annotation later.
      if (node->members().empty()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            node,
            module_.Make<ArrayTypeAnnotation>(
                node->span(),
                module_.Make<ElementTypeAnnotation>(node->type_annotation()),
                CreateUntypedZero(module_, node->span()))));
      }
    }

    if (node->members().empty()) {
      return absl::OkStatus();
    }

    // Create a variable for the element type, and assign it to all the
    // elements.
    std::optional<TypeAnnotation*> element_annotation;
    if (array_annotation.has_value()) {
      element_annotation =
          module_.Make<ElementTypeAnnotation>(*array_annotation);
    }
    XLS_ASSIGN_OR_RETURN(
        const NameRef* element_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Array*>(node),
            GenerateInternalTypeVariableName(node), element_annotation));
    for (Expr* member : node->members()) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(member, element_type_variable));
      if (element_annotation.has_value()) {
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(member, *element_annotation));
      }
    }
    Expr* element_count = module_.Make<Number>(
        node->span(), absl::StrCat(node->members().size()), NumberKind::kOther,
        /*type_annotation=*/nullptr);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node,
        module_.Make<ArrayTypeAnnotation>(
            node->span(),
            module_.Make<TypeVariableTypeAnnotation>(element_type_variable),
            element_count,
            /*dim_is_min=*/node->has_ellipsis())));
    return DefaultHandler(node);
  }

  absl::Status HandleIndex(const Index* node) override {
    // Whether it's a normal index op or a slice, the LHS, which is the original
    // array, always has its own unification context.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* lhs_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
            GenerateInternalTypeVariableName(node->lhs())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), lhs_variable));
    auto* lhs_tvta = module_.Make<TypeVariableTypeAnnotation>(lhs_variable);

    if (std::holds_alternative<Slice*>(node->rhs()) ||
        std::holds_alternative<WidthSlice*>(node->rhs())) {
      XLS_ASSIGN_OR_RETURN(StartAndWidthExprs start_and_width,
                           CreateSliceStartAndWidthExprs(
                               module_, lhs_tvta, ToAstNode(node->rhs())));

      // A slice is kind of the opposite of a concat binop, producing an array
      // with the element type of the original, and an element count that should
      // be at most that of the original. We can tell the requested width of the
      // new array here, at least as an `Expr`, but we rely on
      // `ValidateConcreteType` to eventually decide if that width, given the
      // requested start index, stays within the bounds of the original array.
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node, module_.Make<ArrayTypeAnnotation>(
                    node->span(),
                    module_.Make<ElementTypeAnnotation>(
                        lhs_tvta, /*tuple_index=*/std::nullopt,
                        /*allow_bit_vector_destructuring=*/true),
                    start_and_width.width)));

      XLS_RETURN_IF_ERROR(
          table_.SetSliceStartAndWidthExprs(node, start_and_width));

      // Type-check the nontrivial, fabricated expressions, constraining them to
      // s32.
      VLOG(6) << "Type-checking expanded slice start: "
              << start_and_width.start->ToString()
              << " and width: " << start_and_width.width->ToString();
      XLS_RETURN_IF_ERROR(HandleSliceBoundInternal(start_and_width.start));
      XLS_RETURN_IF_ERROR(HandleSliceBoundInternal(start_and_width.width));

      return DefaultHandler(node);
    }

    // A node like `array[i]` is basically a binary operator with independent
    // contexts on the LHS and RHS. The RHS is constrained to u32, while the LHS
    // must be some kind of array. The "some kind of array" part is not
    // capturable in the table, but readily verifiable at the end of type
    // inference, so we defer that.
    Expr* index = std::get<Expr*>(node->rhs());
    XLS_ASSIGN_OR_RETURN(
        const NameRef* rhs_variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<Expr*>(index),
                                      GenerateInternalTypeVariableName(index)));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(index, rhs_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        index, CreateU32Annotation(module_, index->span())));

    // The type of the entire expr is then ElementType(lhs_variable).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<ElementTypeAnnotation>(lhs_tvta)));
    return DefaultHandler(node);
  }

  absl::Status HandleTupleIndex(const TupleIndex* node) override {
    VLOG(5) << "HandleTupleIndex: " << node->ToString();

    // Establish a context for the unification of the tuple type.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* tuple_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
            GenerateInternalTypeVariableName(node->lhs())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->lhs(), tuple_type_variable));

    // The type of the entire expr is then ElementType(tuple_type_variable,
    // index).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<ElementTypeAnnotation>(
                  module_.Make<TypeVariableTypeAnnotation>(tuple_type_variable),
                  node->index())));
    return DefaultHandler(node);
  }

  absl::Status HandleFunction(const Function* node) override {
    VLOG(5) << "HandleFunction: " << node->ToString()
            << ", parametric: " << node->IsParametric();
    for (const ParametricBinding* binding : node->parametric_bindings()) {
      XLS_RETURN_IF_ERROR(binding->Accept(this));
    }

    const TypeAnnotation* return_type = GetReturnType(module_, *node);
    XLS_RETURN_IF_ERROR(return_type->Accept(this));
    for (const Param* param : node->params()) {
      XLS_RETURN_IF_ERROR(param->Accept(this));
    }

    const FunctionTypeAnnotation* function_type_annotation =
        CreateFunctionTypeAnnotation(module_, *node);

    // Create a variable for the function return type, and use it to unify the
    // formal return type and what is returned by the actual body.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* return_type_variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<Function*>(node),
                                      GenerateInternalTypeVariableName(node),
                                      function_type_annotation->return_type()));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->body(), return_type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->body(), function_type_annotation->return_type()));

    // Only apply a type annotation to the function itself if it's
    // non-parametric. This is to avoid leaking types like `uN[N]` into type
    // variables that are outside the function.
    if (!node->IsParametric()) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, function_type_annotation));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node->name_def(), function_type_annotation));
    }

    // Descend into the function body.
    XLS_RETURN_IF_ERROR(node->body()->Accept(this));
    return absl::OkStatus();
  }

  absl::Status HandleParametricBinding(const ParametricBinding* node) override {
    VLOG(5) << "HandleParametricBinding: " << node->ToString();
    XLS_RETURN_IF_ERROR(table_.DefineParametricVariable(*node).status());
    if (node->expr() != nullptr) {
      // To handle the default expression correctly, we need to impose a type
      // variable pretending that there is something like a `let` or `const`
      // LHS, and the expression type will later have to be unified to that.
      XLS_ASSIGN_OR_RETURN(const NameRef* type_of_parametric,
                           table_.DefineInternalVariable(
                               InferenceVariableKind::kType, node->expr(),
                               GenerateInternalTypeVariableName(node->expr())));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(node->expr(), type_of_parametric));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node->expr(), node->type_annotation()));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStatement(const Statement* node) override {
    VLOG(5) << "HandleStatement: " << node->ToString();
    // If it's just an expr, assign it a type variable.
    if (std::holds_alternative<Expr*>(node->wrapped())) {
      Expr* expr = std::get<Expr*>(node->wrapped());
      std::optional<const NameRef*> type_variable =
          table_.GetTypeVariable(expr);
      if (!type_variable.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            const NameRef* type_variable,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType, const_cast<Statement*>(node),
                GenerateInternalTypeVariableName(expr)));
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(expr, type_variable));
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStatementBlock(const StatementBlock* node) override {
    // A statement block may have a type variable imposed at a higher level of
    // the tree. For example, in
    //     `const X = { statement0; ...; statementN }`
    // or
    //     `fn foo() -> u32 { statement0; ...; statementN }`
    //
    // we will have imposed a type variable on the statement block upon hitting
    // the `ConstantDef` or `Function`. In such cases, we need to propagate the
    // statement block's type variable to `statementN`, if it is an `Expr`, in
    // order for unification to ensure that it's producing the expected type.
    std::optional<const NameRef*> variable = table_.GetTypeVariable(node);
    if (node->trailing_semi()) {
      // A statement block implicitly produces a unit tuple if the last
      // statement ends with a semicolon.
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node, CreateUnitTupleAnnotation(module_, node->span())));
    } else if (!node->statements().empty() && variable.has_value()) {
      const Statement* last_statement =
          node->statements()[node->statements().size() - 1];
      if (std::holds_alternative<Expr*>(last_statement->wrapped())) {
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(
            std::get<Expr*>(last_statement->wrapped()), *variable));
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleInvocation(const Invocation* node) override {
    // When we come in here with an example like:
    //   let x: u32 = foo(a, b);
    //
    // the table will look like this before descent into this function:
    //   Node               Annotation             Variable
    //   --------------------------------------------------
    //   x                  u32                    T0
    //   foo(a, b)                                 T0
    //
    // and this function will make it look like this:
    //   Node               Annotation             Variable
    //   --------------------------------------------------
    //   x                  u32                    T0
    //   foo(a, b)          ReturnType(T3)         T0
    //   a                  ParamType(T3, 0)       T1
    //   b                  ParamType(T3, 1)       T2
    //   foo                (T1, T2) -> T0         T3
    //
    // The core task here is to produce a `FunctionTypeAnnotation` for the
    // actual arguments/return type: the `(T1, T2) -> T0` annotation in the
    // example. By the time of conversion of the invocation node to type info,
    // a formal `FunctionTypeAnnotation` for the resolved target `Function`
    // object will have been determined, and the two annotations will be
    // unified.

    VLOG(5) << "HandleInvocation: " << node->ToString();

    for (ExprOrType parametric : node->explicit_parametrics()) {
      if (std::holds_alternative<Expr*>(parametric)) {
        const Expr* parametric_value_expr = std::get<Expr*>(parametric);
        XLS_ASSIGN_OR_RETURN(
            const NameRef* type_variable,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType,
                const_cast<Expr*>(parametric_value_expr),
                GenerateInternalTypeVariableName(parametric_value_expr)));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeVariable(parametric_value_expr, type_variable));
        XLS_RETURN_IF_ERROR(parametric_value_expr->Accept(this));
      } else {
        XLS_RETURN_IF_ERROR(
            std::get<TypeAnnotation*>(parametric)->Accept(this));
      }
    }

    const NameRef* return_type_variable = *table_.GetTypeVariable(node);
    XLS_ASSIGN_OR_RETURN(
        const NameRef* function_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->callee()),
            absl::StrCat(GenerateInternalTypeVariableName(node->callee()),
                         "_callee")));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->callee(), function_type_variable));

    std::vector<TypeAnnotation*> arg_types;
    arg_types.reserve(node->args().size());
    int self_arg_offset = 0;
    if (node->callee()->kind() == AstNodeKind::kAttr) {
      // An invocation like foo.bar(args), which is targeting an instance
      // function of a struct, needs the actual object type added to the
      // signature in place of the formal `Self`.
      const Attr* attr = dynamic_cast<const Attr*>(node->callee());
      XLS_ASSIGN_OR_RETURN(
          const NameRef* obj_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(attr->lhs()),
              absl::StrCat(GenerateInternalTypeVariableName(attr->lhs()),
                           "_target_obj")));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(attr->lhs(), obj_type_variable));
      XLS_RETURN_IF_ERROR(attr->lhs()->Accept(this));
      arg_types.push_back(
          module_.Make<TypeVariableTypeAnnotation>(obj_type_variable));
      self_arg_offset = 1;
    }
    for (int i = 0; i < node->args().size(); i++) {
      const Expr* arg = node->args()[i];
      // In a case like `foo.fn(arg0, arg1)`, `foo` is the implicit first actual
      // argument, hence `arg0` and `arg1` are actually at index 1 and 2 among
      // the params in the `FunctionTypeAnnotation`.
      const int arg_index_including_implicit_self = i + self_arg_offset;
      TypeAnnotation* arg_annotation = module_.Make<ParamTypeAnnotation>(
          module_.Make<TypeVariableTypeAnnotation>(function_type_variable),
          arg_index_including_implicit_self);
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(arg, arg_annotation));
      XLS_ASSIGN_OR_RETURN(
          const NameRef* arg_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(arg),
              absl::Substitute("$0_actual_arg_$1",
                               GenerateInternalTypeVariableName(arg), i),
              arg_annotation));
      arg_types.push_back(
          module_.Make<TypeVariableTypeAnnotation>(arg_type_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(arg, arg_type_variable));

      XLS_RETURN_IF_ERROR(arg->Accept(this));
    }

    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->callee(), module_.Make<FunctionTypeAnnotation>(
                            arg_types, module_.Make<TypeVariableTypeAnnotation>(
                                           return_type_variable))));

    // The specific way we formulate this annotation indicates that the type of
    // the node is the return type of the unification of the function type;
    // hence, the formal return type, which has not been encountered yet, will
    // govern it.
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node,
        module_.Make<ReturnTypeAnnotation>(
            module_.Make<TypeVariableTypeAnnotation>(function_type_variable))));
    return node->callee()->Accept(this);
  }

  absl::Status HandleZeroOrOneMacro(const AstNode* node, ExprOrType type) {
    if (std::holds_alternative<TypeAnnotation*>(type)) {
      // If the "type" is a type annotation, that's the type.
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, std::get<TypeAnnotation*>(type)));
      return DefaultHandler(node);
    }
    // If it's an "expr", that's an error (just like in V1)
    return TypeInferenceErrorStatus(
        *node->GetSpan(), nullptr,
        absl::Substitute("Expected a type argument in `$0`; saw `$1`.",
                         node->ToString(), std::get<Expr*>(type)->ToString()),
        file_table_);
  }

  absl::Status HandleZeroMacro(const ZeroMacro* node) override {
    VLOG(5) << "HandleZeroMacro: " << node->ToString();
    return HandleZeroOrOneMacro(node, node->type());
  }

  absl::Status HandleAllOnesMacro(const AllOnesMacro* node) override {
    VLOG(5) << "HandleAllOnesMacro: " << node->ToString();
    return HandleZeroOrOneMacro(node, node->type());
  }

  absl::Status HandleConstAssert(const ConstAssert* node) override {
    VLOG(5) << "HandleConstAssert: " << node->ToString();
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->arg(), CreateBoolAnnotation(module_, node->span())));
    XLS_ASSIGN_OR_RETURN(
        const NameRef* operand_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->arg()),
            GenerateInternalTypeVariableName(node->arg())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->arg(), operand_variable));
    return DefaultHandler(node);
  }

  absl::Status HandleLet(const Let* node) override {
    VLOG(5) << "HandleLet: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(
        const NameRef* variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Let*>(node),
            GenerateInternalTypeVariableName(node),
            node->type_annotation() == nullptr
                ? std::nullopt
                : std::make_optional(node->type_annotation())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->rhs(), variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node, variable));
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, node->type_annotation()));
    }
    if (node->name_def_tree()->is_leaf()) {
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(
          ToAstNode(node->name_def_tree()->leaf()), variable));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleTypeAlias(const TypeAlias* node) override {
    VLOG(5) << "HandleTypeAlias: " << node->ToString();
    XLS_RETURN_IF_ERROR(
        table_.SetTypeAnnotation(&node->name_def(), &node->type_annotation()));
    return DefaultHandler(node);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

 private:
  // Determines the target of the given `ColonRef` that is already known to be
  // referencing a member with the name `attribute` of the given `struct_def`.
  // Associates the target node with the `ColonRef` in the `InferenceTable` for
  // later reference, and returns it.
  absl::StatusOr<std::optional<const AstNode*>>
  HandleStructAttributeReferenceInternal(
      const ColonRef* node, const StructDefBase& struct_def,
      const std::vector<ExprOrType>& parametrics, std::string_view attribute) {
    if (!struct_def.impl().has_value()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::Substitute("Struct '$0' has no impl defining '$1'",
                           struct_def.identifier(), attribute),
          file_table_);
    }
    std::optional<ImplMember> member =
        (*struct_def.impl())->GetMember(attribute);
    if (!member.has_value()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::Substitute(
              "Name '$0' is not defined by the impl for struct '$1'.",
              attribute, struct_def.identifier()),
          file_table_);
    }
    if (struct_def.IsParametric()) {
      // The type-checking of a `TypeRefTypeAnnotation` containing any
      // parametrics will prove that there aren't too many parametrics given.
      // However, for general validation, a type reference does not need all
      // bindings satisfied. In a case like `S { a, b }`, we can infer some or
      // all `S` binding values from `a` and `b` at conversion time. However, in
      // `S::SOME_CONSTANT` or `S::static_fn(a)`, we will not infer the `S`
      // bindings; only the bindings for `static_fn` itself, if it has any.
      // Hence all the `S` bindings must be satisfied.
      XLS_RETURN_IF_ERROR(VerifyAllParametricsSatisfied(
          struct_def.parametric_bindings(), parametrics,
          struct_def.identifier(), node->span(), file_table_));
    }
    table_.SetColonRefTarget(node, ToAstNode(*member));
    if (std::holds_alternative<ConstantDef*>(*member) ||
        std::holds_alternative<Function*>(*member)) {
      return ToAstNode(*member);
    }
    return std::nullopt;
  }

  // Helper that creates an internal type variable for a `ConstantDef`, `Param`,
  // or similar type of node that contains a `NameDef` and optional
  // `TypeAnnotation`.
  template <typename T>
  absl::StatusOr<const NameRef*> DefineTypeVariableForVariableOrConstant(
      const T* node) {
    XLS_ASSIGN_OR_RETURN(
        const NameRef* variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<T*>(node),
            GenerateInternalTypeVariableName(node),
            node->type_annotation() == nullptr
                ? std::nullopt
                : std::make_optional(node->type_annotation())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node, variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->name_def(), variable));
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node->name_def(), node->type_annotation()));
    }
    return variable;
  }

  // Helper to either read a type annotation or member variable for a
  // `NameDefNode` or create one if it doesn't exist.
  absl::StatusOr<TypeAnnotation*> GetOrMakeTypeAnnotationForNDF(
      const NameDefTree* node) {
    std::optional<const TypeAnnotation*> type = table_.GetTypeAnnotation(node);
    if (type.has_value()) {
      return const_cast<TypeAnnotation*>(*type);
    }
    const AstNode* definer_node =
        node->is_leaf() ? ToAstNode(node->leaf()) : node;
    std::optional<const NameRef*> member_var =
        table_.GetTypeVariable(definer_node);
    if (!member_var.has_value()) {
      if (node->is_leaf()) {
        XLS_ASSIGN_OR_RETURN(
            member_var,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType, ToAstNode(node->leaf()),
                GenerateInternalTypeVariableName(node)));
      } else {
        XLS_ASSIGN_OR_RETURN(
            member_var,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType, const_cast<NameDefTree*>(node),
                GenerateInternalTypeVariableName(node)));
      }
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(definer_node, *member_var));
    }
    return module_.Make<TypeVariableTypeAnnotation>(*member_var);
  }

  // Generates a name for an internal inference variable that will be used as
  // the type for the given node. The name is only relevant for traceability.
  template <typename T>
  std::string GenerateInternalTypeVariableName(const T* node) {
    return absl::Substitute("internal_type_$0_at_$1_in_$2", node->identifier(),
                            node->span().ToString(file_table_), module_.name());
  }
  // Specialization for `Expr` nodes, which do not have an identifier.
  template <>
  std::string GenerateInternalTypeVariableName(const Expr* node) {
    return absl::Substitute("internal_type_expr_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }

  // Specialization for `Let` nodes, which do not have an identifier.
  template <>
  std::string GenerateInternalTypeVariableName(const Let* node) {
    return absl::Substitute("internal_type_let_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }
  // Specialization for `Array` nodes.
  template <>
  std::string GenerateInternalTypeVariableName(const Array* node) {
    return absl::Substitute("internal_type_array_element_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }
  // Variant for an actual struct member expr.
  std::string GenerateInternalTypeVariableName(
      const StructMemberNode* formal_member, const Expr* actual_member) {
    return absl::Substitute(
        "internal_type_actual_member_$0_at_$1_in_$2", formal_member->name(),
        actual_member->span().ToString(file_table_), module_.name());
  }
  // Variant for operands of a binary operator.
  std::string GenerateInternalTypeVariableName(const Binop* binop) {
    return absl::Substitute("internal_type_operand_$0_at_$1_in_$2",
                            BinopKindToString(binop->binop_kind()),
                            binop->span().ToString(file_table_),
                            module_.name());
  }
  // Variant for `NameDefTree`.
  std::string GenerateInternalTypeVariableName(const NameDefTree* node) {
    return absl::Substitute("internal_type_ndf_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }

  // Propagates the type from the def for `ref`, to `ref` itself in the
  // inference table. This may result in a `TypeAnnotation` being added to the
  // table, but never a variable. If the type of the def is governed by a
  // variable, then `ref` will get a `TypeVariableTypeAnnotation`. This allows
  // the caller to assign a variable to `ref` which unifies it with its
  // context, while also carrying the type information over from its def.
  template <typename T>
  absl::Status PropagateDefToRef(const T* ref) {
    const AstNode* def;
    if constexpr (is_variant<decltype(ref->name_def())>::value) {
      def = ToAstNode(ref->name_def());
    } else {
      def = ref->name_def();
    }
    return PropagateDefToRef(def, ref);
  }

  absl::Status PropagateDefToRef(const AstNode* def, const AstNode* ref) {
    std::optional<const NameRef*> variable = table_.GetTypeVariable(def);
    if (variable.has_value()) {
      return table_.SetTypeAnnotation(
          ref, module_.Make<TypeVariableTypeAnnotation>(*variable));
    }
    std::optional<const TypeAnnotation*> annotation =
        table_.GetTypeAnnotation(def);
    if (annotation.has_value()) {
      return table_.SetTypeAnnotation(ref, *annotation);
    }
    return absl::OkStatus();
  }

  // Ensures that a `StructInstance` nodes provides exprs for all the names in a
  // struct definition, with no extraneous or duplicate names.
  absl::Status ValidateStructInstanceMemberNames(const StructInstance& instance,
                                                 const StructDefBase& def) {
    std::vector<std::string> formal_name_vector = def.GetMemberNames();
    absl::btree_set<std::string> formal_names(formal_name_vector.begin(),
                                              formal_name_vector.end());
    absl::btree_set<std::string> actual_names;
    for (const auto& [name, expr] : instance.GetUnorderedMembers()) {
      if (!formal_names.contains(name)) {
        return TypeInferenceErrorStatus(
            expr->span(), nullptr,
            absl::Substitute("Struct `$0` has no member `$1`, but it was "
                             "provided by this instance.",
                             def.identifier(), name),
            file_table_);
      }
      if (!actual_names.insert(name).second) {
        return TypeInferenceErrorStatus(
            expr->span(), nullptr,
            absl::Substitute(
                "Duplicate value seen for `$0` in this `$1` struct instance.",
                name, def.identifier()),
            file_table_);
      }
    }
    if (actual_names.size() != formal_names.size()) {
      absl::btree_set<std::string> missing_set;
      absl::c_set_difference(formal_names, actual_names,
                             std::inserter(missing_set, missing_set.begin()));
      std::vector<std::string> missing(missing_set.begin(), missing_set.end());
      return TypeInferenceErrorStatus(
          instance.span(), nullptr,
          absl::Substitute(
              "Instance of struct `$0` is missing member(s): $1",
              def.identifier(),
              absl::StrJoin(missing, ", ",
                            [](std::string* out, const std::string& piece) {
                              absl::StrAppendFormat(out, "`%s`", piece);
                            })),
          file_table_);
    }
    return absl::OkStatus();
  }

  // Gets the explicit type annotation (expected to be of type `T` if it is
  // direct) for a node by querying the type variable that it shares with a
  // declaration, if any. This must be done before imposing any synthetic type
  // annotation on the value.
  template <typename T>
  absl::StatusOr<std::optional<const TypeAnnotation*>>
  GetDeclarationTypeAnnotation(const AstNode* node) {
    std::optional<const NameRef*> type_variable = table_.GetTypeVariable(node);
    if (!type_variable.has_value()) {
      return std::nullopt;
    }
    XLS_ASSIGN_OR_RETURN(std::optional<const TypeAnnotation*> annotation,
                         table_.GetDeclarationTypeAnnotation(*type_variable));
    // Constraining the annotation type here improves error messages in
    // situations where there is a type mismatch for an entire array/tuple.
    // We allow indirect member/element annotations through at this point,
    // because we can't yet prove whether they amount to something expected.
    if (annotation.has_value()) {
      if (dynamic_cast<const T*>(*annotation) ||
          dynamic_cast<const MemberTypeAnnotation*>(*annotation) ||
          dynamic_cast<const ElementTypeAnnotation*>(*annotation) ||
          dynamic_cast<const ParamTypeAnnotation*>(*annotation)) {
        return annotation;
      }
      VLOG(5) << "Declaration type is unsupported kind: "
              << (*annotation)->ToString() << " for " << node->ToString();
    }
    return std::nullopt;
  }

  // Type-checks a fabricated longhand slice bound expression. Expressions in
  // this context are always constrained to s32.
  absl::Status HandleSliceBoundInternal(const Expr* bound) {
    XLS_ASSIGN_OR_RETURN(
        const NameRef* variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<Expr*>(bound),
                                      GenerateInternalTypeVariableName(bound)));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(bound, variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        bound, CreateS32Annotation(module_, bound->span())));
    return bound->Accept(this);
  }

  Module& module_;
  InferenceTable& table_;
  const FileTable& file_table_;
};

}  // namespace

absl::StatusOr<TypeInfo*> TypecheckModuleV2(Module* module,
                                            ImportData* import_data,
                                            WarningCollector* warnings) {
  std::unique_ptr<InferenceTable> table = InferenceTable::Create(*module);

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> builtins_module,
                       LoadBuiltinStubs());
  PopulateInferenceTableVisitor builtins_visitor(*builtins_module, *table,
                                                 import_data->file_table());
  XLS_RETURN_IF_ERROR((*builtins_module).Accept(&builtins_visitor));

  PopulateInferenceTableVisitor visitor(*module, *table,
                                        import_data->file_table());
  XLS_RETURN_IF_ERROR(module->Accept(&visitor));

  return InferenceTableToTypeInfo(*table, *module, *import_data, *warnings,
                                  import_data->file_table(),
                                  std::move(builtins_module));
}

}  // namespace xls::dslx
