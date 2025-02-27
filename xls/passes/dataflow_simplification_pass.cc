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

#include "xls/passes/dataflow_simplification_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

// Data-structure describing the source of leaf element of a node in the
// graph. If the source cannot be determined statically then the source of the
// leaf element is itself. Example NodeSources after dataflow analysis
//
//   x: u32 = param(...)          // NodeSource(x, {})
//   y: u32 = param(...)          // NodeSource(y, {})
//   z: (u32, u32) = param(...)   // (NodeSource(z, {0}), NodeSource(z, {1}))
//   a: u32 = identity(x)         // NodeSource(x, {})
//   b: u32 = tuple_index(z, 1)   // NodeSource(z, {1})
//   c: u32 = sel(..., {x, y})    // NodeSource(c, {})
//   d: u32 = sel(..., {x, x})    // NodeSource(x, {})
class NodeSource {
 public:
  NodeSource() = default;
  NodeSource(Node* node, std::vector<int64_t> tree_index)
      : node_(node), tree_index_(std::move(tree_index)) {}

  Node* node() const { return node_; }
  absl::Span<const int64_t> tree_index() const { return tree_index_; }

  std::string ToString() const {
    if (tree_index().empty()) {
      return node()->GetName();
    }
    return absl::StrFormat("%s{%s}", node()->GetName(),
                           absl::StrJoin(tree_index(), ","));
  }

  friend bool operator==(const NodeSource&, const NodeSource&) = default;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const NodeSource& source) {
    absl::Format(&sink, "%s", source.ToString());
  }

  template <typename H>
  friend H AbslHashValue(H h, const NodeSource& ns) {
    return H::combine(std::move(h), ns.node(), ns.tree_index());
  }

 private:
  Node* node_;
  std::vector<int64_t> tree_index_;
};

class NodeSourceDataflowVisitor : public DataflowVisitor<NodeSource> {
 public:
  absl::Status DefaultHandler(Node* node) override {
    LeafTypeTree<NodeSource> result(node->GetType());
    XLS_RETURN_IF_ERROR(leaf_type_tree::ForEachIndex(
        result.AsMutableView(), [&](Type* element_type, NodeSource& element,
                                    absl::Span<const int64_t> index) {
          element = NodeSource(node, std::vector(index.begin(), index.end()));
          return absl::OkStatus();
        }));
    return SetValue(node, std::move(result));
  }

 protected:
  absl::StatusOr<NodeSource> JoinElements(
      Type* element_type, absl::Span<const NodeSource* const> data_sources,
      absl::Span<const LeafTypeTreeView<NodeSource>> control_sources,
      Node* node, absl::Span<const int64_t> index) override {
    if (std::all_of(
            data_sources.begin(), data_sources.end(),
            [&](const NodeSource* n) { return *n == *data_sources.front(); })) {
      return *data_sources.front();
    }
    return NodeSource(node, std::vector(index.begin(), index.end()));
  }
};

// Returns true if `type` has no non-zero-width bits components or tokens.
bool IsEmptyType(Type* type) {
  return type->GetFlatBitCount() == 0 && !TypeHasToken(type);
}

// Try to replace `node` with an Array operation whose elements consist entirely
// of other node in the graph. Returns true if the transformation is successful.
// Note this will not generate an array consisting entirely of 'array-index'
// operations from a single source array in order since this is equivalent to
// the original array and will cause an infinite loop with
// array_simplification_pass.cc SimplifyArray.
absl::StatusOr<bool> MaybeReplaceWithArrayOfExistingNodes(
    Node* node, LeafTypeTreeView<NodeSource> node_source,
    const absl::flat_hash_map<LeafTypeTreeView<NodeSource>, Node*>&
        source_map) {
  // Skip literals, arrays, and empty types. Replacing these is not beneficial
  // and may result in inf-loops. Avoid ArrayIndex specifically for an infinite
  // loop with array_simplification_pass.
  // TODO(allight): It would be nice to have a better way to avoid this infinite
  // loop since it can be a beneficial transform.
  if (!node->GetType()->IsArray() || node->Is<Literal>() || node->Is<Array>() ||
      node->Is<ArrayIndex>() || IsEmptyType(node->GetType())) {
    return false;
  }

  // Every element value of the array must exist as a node in the graph.
  ArrayType* array_type = node->GetType()->AsArrayOrDie();
  std::vector<Node*> sources;
  sources.reserve(array_type->size());
  for (int64_t i = 0; i < array_type->size(); ++i) {
    auto it = source_map.find(node_source.AsView({i}));
    if (it == source_map.end()) {
      return false;
    }
    sources.push_back(it->second);
  }
  XLS_ASSIGN_OR_RETURN(
      Node * new_array,
      node->ReplaceUsesWithNew<Array>(sources, array_type->element_type()));
  VLOG(2) << "Replaced " << node << " with " << new_array;
  return true;
}

}  // namespace

absl::StatusOr<bool> DataflowSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext* context) const {
  NodeSourceDataflowVisitor visitor;
  XLS_RETURN_IF_ERROR(func->Accept(&visitor));
  bool changed = false;
  // Hashmap from the LTT<NodeSource> of a node to the Node*. If two nodes have
  // the same LTT<NodeSource> they are necessarily equivalent.
  absl::flat_hash_map<LeafTypeTreeView<NodeSource>, Node*> source_map;
  for (Node* node : context->TopoSort(func)) {
    LeafTypeTreeView<NodeSource> source = visitor.GetValue(node);
    VLOG(3) << absl::StrFormat("Considering `%s`: %s", node->GetName(),
                               source.ToString());
    auto [it, inserted] = source_map.insert({source, node});
    VLOG(3) << "Inserted: " << inserted;
    // Skip empty tuples (and tuples of empty tuples, etc) as these carry no
    // data and are trivially substitutable with other nodes of the same
    // type. This can result in inflooping as, for example, parameters replace
    // each others uses.
    if (!inserted && !IsEmptyType(node->GetType())) {
      // An equivalent node exists in the graph. Repace this node with its
      // equivalent.
      VLOG(2) << absl::StrFormat("Replacing `%s` with equivalent `%s`",
                                 node->GetName(), it->second->GetName());
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(it->second));
      changed = true;
      continue;
    }
    if (node->GetType()->IsArray()) {
      XLS_ASSIGN_OR_RETURN(bool replaced, MaybeReplaceWithArrayOfExistingNodes(
                                              node, source, source_map));
      changed = changed || replaced;
    }
  }
  return changed;
}

REGISTER_OPT_PASS(DataflowSimplificationPass);

}  // namespace xls
