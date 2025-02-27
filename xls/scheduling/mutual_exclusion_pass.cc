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

#include "xls/scheduling/mutual_exclusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/graph_coloring.h"
#include "xls/data_structures/transitive_closure.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/token_provenance_analysis.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3_api.h"

namespace xls {

namespace {

bool FunctionIsOneBit(Node* node) {
  return node->GetType()->IsBits() &&
         node->GetType()->AsBitsOrDie()->bit_count() == 1;
}

absl::Status AddSendReceivePredicates(Predicates* p, FunctionBase* f) {
  for (Node* node : f->nodes()) {
    if (node->Is<Send>()) {
      if (std::optional<Node*> pred = node->As<Send>()->predicate()) {
        XLS_RETURN_IF_ERROR(AddPredicate(p, node, pred.value()).status());
      }
    } else if (node->Is<Receive>()) {
      if (std::optional<Node*> pred = node->As<Receive>()->predicate()) {
        XLS_RETURN_IF_ERROR(AddPredicate(p, node, pred.value()).status());
      }
    }
  }
  return absl::OkStatus();
}

template <typename T>
bool HasIntersection(const absl::flat_hash_set<T>& lhs,
                     const absl::flat_hash_set<T>& rhs) {
  const absl::flat_hash_set<T>& smaller = lhs.size() > rhs.size() ? rhs : lhs;
  const absl::flat_hash_set<T>& bigger = lhs.size() > rhs.size() ? lhs : rhs;
  return std::any_of(smaller.begin(), smaller.end(),
                     [&bigger](T element) { return bigger.contains(element); });
}

template <typename T>
absl::flat_hash_set<T> Intersection(const absl::flat_hash_set<T>& lhs,
                                    const absl::flat_hash_set<T>& rhs) {
  absl::flat_hash_set<T> result;
  const absl::flat_hash_set<T>& smaller = lhs.size() > rhs.size() ? rhs : lhs;
  const absl::flat_hash_set<T>& bigger = lhs.size() > rhs.size() ? lhs : rhs;
  for (const T& element : smaller) {
    if (bigger.contains(element)) {
      result.insert(element);
    }
  }
  return result;
}

Z3_lbool RunSolver(Z3_context c, Z3_ast asserted) {
  Z3_solver solver = solvers::z3::CreateSolver(c, 1);
  Z3_solver_assert(c, solver, asserted);
  Z3_lbool satisfiable = Z3_solver_check(c, solver);
  Z3_solver_dec_ref(c, solver);
  return satisfiable;
}

// Returns a list of all predicates in a deterministic order, paired with their
// index in the list.
std::vector<std::pair<Node*, int64_t>> PredicateNodes(Predicates* p,
                                                      FunctionBase* f) {
  std::vector<std::pair<Node*, int64_t>> result;

  int64_t i = 0;
  for (Node* node : f->nodes()) {
    if (!p->GetNodesPredicatedBy(node).empty()) {
      result.push_back({node, i});
      ++i;
    }
  }

  return result;
}

bool IsHeavyOp(Op op) { return op == Op::kSend || op == Op::kReceive; }

std::string_view GetChannelName(Node* node) {
  if (node->Is<Send>()) {
    return node->As<Send>()->channel_name();
  }
  if (node->Is<Receive>()) {
    return node->As<Receive>()->channel_name();
  }
  return "";
}

std::optional<Node*> GetPredicate(Node* node) {
  std::optional<Node*> predicate;
  if (node->Is<Send>()) {
    predicate = node->As<Send>()->predicate();
  }
  if (node->Is<Receive>()) {
    predicate = node->As<Receive>()->predicate();
  }
  return predicate;
};

using NodeRelation = absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>;

// Find all nodes that the given node transitively depends on.
absl::flat_hash_set<Node*> DependenciesOf(Node* root) {
  std::vector<Node*> stack;
  stack.push_back(root);
  absl::flat_hash_set<Node*> discovered;
  while (!stack.empty()) {
    Node* popped = stack.back();
    stack.pop_back();
    if (!discovered.contains(popped)) {
      discovered.insert(popped);
      for (Node* child : popped->operands()) {
        stack.push_back(child);
      }
    }
  }
  return discovered;
}

// Find the largest connected subgraph of the given token DAG, such that it is
// a rooted DAG whose root is the given node, and all nodes in the subgraph
// satisfy the given predicate.
absl::flat_hash_set<Node*> LargestConnectedSubgraph(
    Node* root, const TokenDAG& dag,
    const std::function<bool(Node*)>& predicate) {
  std::vector<Node*> stack;
  stack.push_back(root);
  absl::flat_hash_set<Node*> discovered;
  while (!stack.empty()) {
    Node* popped = stack.back();
    stack.pop_back();
    if (predicate(popped) && !discovered.contains(popped)) {
      discovered.insert(popped);
      if (dag.contains(popped)) {
        for (Node* child : dag.at(popped)) {
          stack.push_back(child);
        }
      }
    }
  }
  return discovered;
}

// Computes a symmetric relation that decides whether two side-effectful nodes
// can be merged. The principle is as follows:
//
// 1. A connected subgraph consisting of all the same kind of effect (sends on
//    all the same channel; receives on all the same channel) of the token DAG
//    can be merged.
// 2. Nodes of the same type that are unrelated in the transitive token
//    dependency relation can be merged.
absl::StatusOr<NodeRelation> ComputeMergableEffects(FunctionBase* f) {
  absl::flat_hash_set<std::string> multi_op_channels;
  absl::flat_hash_set<std::string> channel_names;
  for (Node* node : f->nodes()) {
    if (node->Is<Send>() || node->Is<Receive>()) {
      std::string_view channel_name = GetChannelName(node);
      if (auto it = channel_names.find(channel_name);
          it == channel_names.end()) {
        channel_names.insert(it, std::string(channel_name));
      } else if (auto it = multi_op_channels.find(channel_name);
                 it == multi_op_channels.end()) {
        multi_op_channels.insert(it, std::string(channel_name));
      }
    }
  }
  if (multi_op_channels.empty()) {
    return NodeRelation();
  }

  XLS_ASSIGN_OR_RETURN(TokenDAG token_dag, ComputeTokenDAG(f));
  absl::flat_hash_set<Node*> token_nodes;
  for (const auto& [node, children] : token_dag) {
    token_nodes.insert(node);
    for (const auto& child : children) {
      token_nodes.insert(child);
    }
  }

  NodeRelation result;
  NodeRelation transitive_closure = TransitiveClosure<Node*>(token_dag);
  for (Node* node : ReverseTopoSort(f)) {
    if (node->Is<Send>() || node->Is<Receive>()) {
      std::string_view channel_name = GetChannelName(node);
      if (!multi_op_channels.contains(channel_name)) {
        continue;
      }
      absl::flat_hash_set<Node*> subgraph =
          LargestConnectedSubgraph(node, token_dag, [&](Node* n) -> bool {
            return n->op() == node->op() && GetChannelName(n) == channel_name;
          });
      for (Node* x : subgraph) {
        for (Node* y : subgraph) {
          // Ensure that x and y are not data-dependent on each other (but they
          // can be token-dependent). The only way for two sends or two receives
          // to have a data dependency is through the predicate.
          if (std::optional<Node*> pred_x = GetPredicate(x)) {
            absl::flat_hash_set<Node*> dependencies_of_pred_x =
                DependenciesOf(pred_x.value());
            if (dependencies_of_pred_x.contains(y)) {
              continue;
            }
          }
          if (std::optional<Node*> pred_y = GetPredicate(y)) {
            absl::flat_hash_set<Node*> dependencies_of_pred_y =
                DependenciesOf(pred_y.value());
            if (dependencies_of_pred_y.contains(x)) {
              continue;
            }
          }
          result[x].insert(y);
          result[y].insert(x);
        }
      }
    }
  }
  for (Node* x : token_nodes) {
    for (Node* y : token_nodes) {
      if (!(transitive_closure.contains(x) &&
            transitive_closure.at(x).contains(y)) &&
          !(transitive_closure.contains(y) &&
            transitive_closure.at(y).contains(x))) {
        result[x].insert(y);
        result[y].insert(x);
      }
    }
  }
  return result;
}

// This computes a partition of a subset of all nodes into merge classes.
// Nodes that are not in this partition can be assumed to be in a merge class of
// size 1 including only themselves.
// A merge class is a set of nodes that are all jointly mutually exclusive.
absl::StatusOr<std::vector<absl::flat_hash_set<Node*>>> ComputeMergeClasses(
    Predicates* p, FunctionBase* f, const ScheduleCycleMap& scm) {
  XLS_ASSIGN_OR_RETURN(NodeRelation mergable_effects,
                       ComputeMergableEffects(f));
  if (mergable_effects.empty()) {
    return std::vector<absl::flat_hash_set<Node*>>();
  }

  absl::flat_hash_set<Node*> nodes;
  std::vector<Node*> ordered_nodes;
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> neighborhoods;

  for (Node* node : TopoSort(f)) {
    if (IsHeavyOp(node->op())) {
      nodes.insert(node);
      neighborhoods[node];
      ordered_nodes.push_back(node);
    }
  }

  auto is_mergable = [&](Node* x, Node* y) -> bool {
    return mergable_effects.contains(x) && mergable_effects.at(x).contains(y) &&
           scm.at(x) == scm.at(y);
  };

  for (Node* x : nodes) {
    if (!(p->GetPredicate(x).has_value())) {
      continue;
    }
    Node* px = p->GetPredicate(x).value();
    for (Node* y : nodes) {
      if (!(p->GetPredicate(y).has_value())) {
        continue;
      }
      Node* py = p->GetPredicate(y).value();
      if (x->op() != y->op()) {
        continue;
      }
      if ((x->op() == Op::kSend) &&
          (!is_mergable(x, y) ||
           (x->As<Send>()->channel_name() != y->As<Send>()->channel_name()))) {
        continue;
      }
      if ((x->op() == Op::kReceive) &&
          (!is_mergable(x, y) || (x->As<Receive>()->channel_name() !=
                                  y->As<Receive>()->channel_name()))) {
        continue;
      }
      if (p->QueryMutuallyExclusive(px, py) == std::make_optional(true)) {
        neighborhoods[x].insert(y);
        neighborhoods[y].insert(x);
      }
    }
  }

  // The complement of the `neighborhoods` graph
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> inverted_neighborhoods;

  for (Node* node : nodes) {
    inverted_neighborhoods[node] = nodes;
    for (Node* neighbor : neighborhoods.at(node)) {
      inverted_neighborhoods[node].erase(neighbor);
    }
  }

  absl::flat_hash_map<Node*, int64_t> node_to_index;
  for (int64_t i = 0; i < ordered_nodes.size(); ++i) {
    node_to_index[ordered_nodes[i]] = i;
  }

  std::vector<int64_t> iota(ordered_nodes.size());
  std::iota(iota.begin(), iota.end(), 0);
  std::vector<absl::flat_hash_set<int64_t>> coloring_indices =
      RecursiveLargestFirstColoring<int64_t>(
          absl::flat_hash_set<int64_t>(iota.begin(), iota.end()),
          [&](int64_t node_index) -> absl::flat_hash_set<int64_t> {
            absl::flat_hash_set<Node*> inv_neighbors =
                inverted_neighborhoods.at(ordered_nodes.at(node_index));
            absl::flat_hash_set<int64_t> result;
            for (Node* inv_neighbor : inv_neighbors) {
              result.insert(node_to_index.at(inv_neighbor));
            }
            return result;
          });

  std::vector<absl::flat_hash_set<Node*>> coloring;
  for (const absl::flat_hash_set<int64_t>& color_class : coloring_indices) {
    absl::flat_hash_set<Node*> color_node_class;
    for (int64_t index : color_class) {
      color_node_class.insert(ordered_nodes[index]);
    }
    coloring.push_back(color_node_class);
  }

  for (const absl::flat_hash_set<Node*>& color_class : coloring) {
    CHECK(!color_class.empty());
    Op op = (*(color_class.cbegin()))->op();
    for (Node* node : color_class) {
      CHECK_EQ(node->op(), op);
    }
  }

  return coloring;
}

// Given a sequence of nodes, returns a new sequence of nodes that comprises the
// predicates of each of the input nodes. If an input node does not have a
// predicate, a literal `true` (1-bit value containing 1) node is used.
absl::StatusOr<std::vector<Node*>> PredicateVectorFromNodes(
    Predicates* p, FunctionBase* f, absl::Span<Node* const> nodes) {
  XLS_ASSIGN_OR_RETURN(Node * literal_true,
                       f->MakeNode<Literal>(SourceInfo(), Value(UBits(1, 1))));

  std::vector<Node*> predicates;
  predicates.reserve(nodes.size());
  for (Node* node : nodes) {
    if (std::optional<Node*> pred = p->GetPredicate(node)) {
      predicates.push_back(pred.value());
    } else {
      predicates.push_back(literal_true);
    }
  }

  return predicates;
}

// Get the token produced by the given receive node. This avoids creating a new
// `tuple_index` node if one already exists.
absl::StatusOr<Node*> GetTokenOfReceive(Node* receive) {
  FunctionBase* f = receive->function_base();
  for (Node* user : receive->users()) {
    if (user->Is<TupleIndex>()) {
      if (user->As<TupleIndex>()->index() == 0) {
        return user;
      }
    }
  }
  return f->MakeNode<TupleIndex>(SourceInfo(), receive, 0);
}

// Given a set of nodes, returns all nodes in the token dag that feed into this
// set but are not contained within it. The ordering of the result is guaranteed
// to be deterministic.
absl::StatusOr<std::vector<Node*>> ComputeTokenInputs(
    FunctionBase* f, absl::Span<Node* const> nodes) {
  XLS_ASSIGN_OR_RETURN(TokenDAG token_dag, ComputeTokenDAG(f));

  absl::flat_hash_set<Node*> token_inputs_unsorted;

  {
    absl::flat_hash_set<Node*> nodes_set(nodes.begin(), nodes.end());

    for (Node* node : nodes) {
      if (token_dag.contains(node)) {
        absl::flat_hash_set<Node*> children = token_dag.at(node);
        for (Node* child : children) {
          if (!nodes_set.contains(child)) {
            Node* token = child;
            if (child->Is<Receive>()) {
              XLS_ASSIGN_OR_RETURN(token, GetTokenOfReceive(child));
            }
            token_inputs_unsorted.insert(token);
          }
        }
      }
    }
  }

  // Ensure determinism of output.
  return SetToSortedVector(token_inputs_unsorted);
}

// Returns whether there is a path from any of the nodes in `sources` to any of
// the nodes in `sinks`.
bool HasPath(const absl::flat_hash_set<Node*>& sources,
             const absl::flat_hash_set<Node*>& sinks) {
  if (HasIntersection(sources, sinks)) {
    return true;
  }

  std::vector<Node*> to_visit(sources.begin(), sources.end());
  absl::flat_hash_set<Node*> visited;
  while (!to_visit.empty()) {
    Node* node = to_visit.back();
    to_visit.pop_back();
    auto [_, inserted] = visited.insert(node);
    if (!inserted) {
      continue;
    }
    if (sinks.contains(node)) {
      return true;
    }
    absl::c_copy_if(node->users(), std::back_inserter(to_visit),
                    [&](Node* user) { return !visited.contains(user); });
  }

  return false;
}

bool IsProvenMutuallyExclusiveChannel(ChannelRef channel_ref) {
  if (std::holds_alternative<Channel*>(channel_ref)) {
    Channel* channel = std::get<Channel*>(channel_ref);
    return channel->kind() == ChannelKind::kStreaming &&
           down_cast<StreamingChannel*>(channel)->GetStrictness() ==
               ChannelStrictness::kProvenMutuallyExclusive;
  }

  CHECK(std::holds_alternative<ChannelReference*>(channel_ref));
  ChannelReference* channel_reference =
      std::get<ChannelReference*>(channel_ref);
  return channel_reference->kind() == ChannelKind::kStreaming &&
         channel_reference->strictness() ==
             ChannelStrictness::kProvenMutuallyExclusive;
}

std::string_view GetChannelName(ChannelRef channel_ref) {
  if (std::holds_alternative<Channel*>(channel_ref)) {
    return std::get<Channel*>(channel_ref)->name();
  }

  CHECK(std::holds_alternative<ChannelReference*>(channel_ref));
  return std::get<ChannelReference*>(channel_ref)->name();
}

absl::StatusOr<bool> MergeSends(Predicates* p, FunctionBase* f,
                                absl::Span<Node* const> to_merge) {
  if (to_merge.size() <= 1) {
    return false;
  }
  XLS_ASSIGN_OR_RETURN(ChannelRef channel_ref,
                       to_merge.front()->As<Send>()->GetChannelRef());

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> token_inputs,
                       ComputeTokenInputs(f, to_merge));

  // Collect all inputs to & users of the merged send, so we can check whether
  // we can do this merge without creating a cycle.
  absl::flat_hash_set<Node*> inputs;
  absl::flat_hash_set<Node*> users;
  inputs.reserve(token_inputs.size() + 2 * to_merge.size());
  inputs.insert(token_inputs.begin(), token_inputs.end());
  for (Node* node : to_merge) {
    if (std::optional<Node*> predicate = p->GetPredicate(node);
        predicate.has_value()) {
      inputs.insert(*predicate);
    }
    inputs.insert(node->As<Send>()->data());
  }
  users.reserve(to_merge.size());
  for (Node* node : to_merge) {
    absl::c_copy(node->users(), std::inserter(users, users.end()));
  }

  if (HasPath(users, inputs)) {
    // Check if this was a required merge due to channel strictness.
    if (IsProvenMutuallyExclusiveChannel(channel_ref)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Unable to merge operations on proven-mutually-exclusive channel "
          "%s in proc %s without creating a cycle.",
          GetChannelName(channel_ref), f->name()));
    }

    // We can't merge these sends without forming a cycle.
    VLOG(1) << "Unable to merge nodes without creating a cycle: "
            << absl::StrJoin(to_merge, ", ", [](std::string* out, Node* node) {
                 absl::StrAppend(out, node->GetName());
               });
    return false;
  }

  absl::btree_set<SourceLocation> source_locations_set;
  for (Node* send : to_merge) {
    CHECK(channel_ref == *send->As<Send>()->GetChannelRef())
        << "Channel mismatch; attempted to merge sends on channels "
        << GetChannelName(channel_ref) << " and "
        << GetChannelName(*send->As<Send>()->GetChannelRef());
    absl::c_copy(
        send->loc().locations,
        std::inserter(source_locations_set, source_locations_set.end()));
  }
  SourceInfo merged_source_info(std::vector<SourceLocation>(
      source_locations_set.begin(), source_locations_set.end()));

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> predicates,
                       PredicateVectorFromNodes(p, f, to_merge));

  std::vector<Node*> args;
  args.reserve(to_merge.size());
  for (Node* node : to_merge) {
    args.push_back(node->As<Send>()->data());
  }
  // OneHotSelect takes the cases in reverse order (LSB-to-MSB).
  std::reverse(args.begin(), args.end());

  XLS_ASSIGN_OR_RETURN(Node * token,
                       f->MakeNode<AfterAll>(merged_source_info, token_inputs));

  XLS_ASSIGN_OR_RETURN(Node * selector,
                       f->MakeNode<Concat>(merged_source_info, predicates));

  XLS_ASSIGN_OR_RETURN(Node * predicate,
                       f->MakeNode<BitwiseReductionOp>(
                           merged_source_info, selector, Op::kOrReduce));

  XLS_ASSIGN_OR_RETURN(Node * data, f->MakeNode<OneHotSelect>(
                                        merged_source_info, selector, args));

  XLS_ASSIGN_OR_RETURN(
      Node * send, f->MakeNode<Send>(merged_source_info, token, data, predicate,
                                     GetChannelName(channel_ref)));

  for (Node* node : to_merge) {
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(send));
    XLS_RETURN_IF_ERROR(f->RemoveNode(node));
  }

  return true;
}

absl::StatusOr<bool> MergeReceives(Predicates* p, FunctionBase* f,
                                   absl::Span<Node* const> to_merge) {
  if (to_merge.size() <= 1) {
    return false;
  }
  XLS_ASSIGN_OR_RETURN(ChannelRef channel_ref,
                       to_merge.front()->As<Receive>()->GetChannelRef());
  bool is_blocking = to_merge.front()->As<Receive>()->is_blocking();

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> token_inputs,
                       ComputeTokenInputs(f, to_merge));

  // Collect all inputs to & users of the merged send, so we can check whether
  // we can do this merge without creating a cycle.
  absl::flat_hash_set<Node*> inputs;
  absl::flat_hash_set<Node*> users;
  inputs.reserve(token_inputs.size() + to_merge.size());
  inputs.insert(token_inputs.begin(), token_inputs.end());
  for (Node* node : to_merge) {
    if (std::optional<Node*> predicate = p->GetPredicate(node);
        predicate.has_value()) {
      inputs.insert(*predicate);
    }
  }
  users.reserve(to_merge.size());
  for (Node* node : to_merge) {
    absl::c_copy(node->users(), std::inserter(users, users.end()));
  }

  if (HasPath(users, inputs)) {
    // Check if this was a required merge due to channel strictness.
    if (IsProvenMutuallyExclusiveChannel(channel_ref)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Unable to merge operations on proven-mutually-exclusive channel "
          "%s in proc %s without creating a cycle.",
          GetChannelName(channel_ref), f->name()));
    }

    // We can't merge these sends without forming a cycle.
    VLOG(1) << "Unable to merge nodes without creating a cycle: "
            << absl::StrJoin(to_merge, ", ", [](std::string* out, Node* node) {
                 absl::StrAppend(out, node->GetName());
               });
    return false;
  }

  absl::btree_set<SourceLocation> source_locations_set;
  for (Node* receive : to_merge) {
    CHECK(channel_ref == *receive->As<Receive>()->GetChannelRef())
        << "Channel mismatch; attempted to merge receives on channels "
        << GetChannelName(channel_ref) << " and "
        << GetChannelName(*receive->As<Receive>()->GetChannelRef());
    CHECK_EQ(is_blocking, receive->As<Receive>()->is_blocking());
    absl::c_copy(
        receive->loc().locations,
        std::inserter(source_locations_set, source_locations_set.end()));
  }
  SourceInfo merged_source_info(std::vector<SourceLocation>(
      source_locations_set.begin(), source_locations_set.end()));

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> predicates,
                       PredicateVectorFromNodes(p, f, to_merge));

  XLS_ASSIGN_OR_RETURN(Node * token,
                       f->MakeNode<AfterAll>(merged_source_info, token_inputs));

  XLS_ASSIGN_OR_RETURN(
      Node * predicate,
      f->MakeNode<NaryOp>(merged_source_info, predicates, Op::kOr));

  XLS_ASSIGN_OR_RETURN(
      Node * receive,
      f->MakeNode<Receive>(merged_source_info, token, predicate,
                           GetChannelName(channel_ref), is_blocking));

  XLS_ASSIGN_OR_RETURN(Node * token_output,
                       f->MakeNode<TupleIndex>(SourceInfo(), receive, 0));

  XLS_ASSIGN_OR_RETURN(Node * value_output,
                       f->MakeNode<TupleIndex>(SourceInfo(), receive, 1));

  XLS_ASSIGN_OR_RETURN(
      Node * zero,
      f->MakeNode<Literal>(SourceInfo(), ZeroOfType(value_output->GetType())));

  std::vector<Node*> gated_output;
  for (int64_t i = 0; i < to_merge.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * gated,
        f->MakeNode<Select>(SourceInfo(), predicates[i],
                            std::vector<Node*>{zero, value_output},
                            std::nullopt));
    XLS_ASSIGN_OR_RETURN(
        Node * gated_with_token,
        f->MakeNode<Tuple>(SourceInfo(),
                           std::vector<Node*>{token_output, gated}));
    gated_output.push_back(gated_with_token);
  }

  for (int64_t i = 0; i < to_merge.size(); ++i) {
    XLS_RETURN_IF_ERROR(to_merge[i]->ReplaceUsesWith(gated_output[i]));
    XLS_RETURN_IF_ERROR(f->RemoveNode(to_merge[i]));
  }

  return true;
}

absl::StatusOr<bool> MergeNodes(Predicates* p, FunctionBase* f,
                                const absl::flat_hash_set<Node*>& merge_class) {
  if (merge_class.size() <= 1) {
    return false;
  }

  std::vector<Node*> to_merge = SetToSortedVector(merge_class);

  Op op = to_merge.front()->op();
  if (op == Op::kSend) {
    return MergeSends(p, f, to_merge);
  }
  if (op == Op::kReceive) {
    return MergeReceives(p, f, to_merge);
  }

  return false;
}

// Returns the set of proven_mutually_exclusive channels that have operations
// predicated by `node`.
absl::StatusOr<absl::flat_hash_set<Channel*>>
GetControlledProvenMutuallyExclusiveChannels(Node* node, Predicates* p,
                                             FunctionBase* f) {
  absl::flat_hash_set<Channel*> controlled_channels;
  for (Node* predicated_node : p->GetNodesPredicatedBy(node)) {
    Channel* channel = nullptr;
    if (predicated_node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(channel,
                           f->package()->GetChannel(
                               predicated_node->As<Send>()->channel_name()));
    } else if (predicated_node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(channel,
                           f->package()->GetChannel(
                               predicated_node->As<Receive>()->channel_name()));
    } else {
      continue;
    }

    if (channel->kind() != ChannelKind::kStreaming) {
      continue;
    }
    if (down_cast<StreamingChannel*>(channel)->GetStrictness() ==
        ChannelStrictness::kProvenMutuallyExclusive) {
      controlled_channels.insert(channel);
    }
  }
  return controlled_channels;
}

// Checks whether `node` controls a use of a `proven_mutually_exclusive`
// streaming channel that is contended by another use of that channel. For
// example, this can be used to determine whether a failure to prove `node`
// mutually exclusive with another node might cause the channel operations to be
// illegal.
absl::StatusOr<bool> ControlsContendedProvenMutuallyExclusiveChannel(
    Node* node, Predicates* p, FunctionBase* f) {
  absl::flat_hash_set<Node*> predicated_nodes = p->GetNodesPredicatedBy(node);

  // First, check if this node might control two independent operations on the
  // same proven-mutually-exclusive channel... and collect the set of
  // proven-mutually-exclusive channels it controls while we're at it.
  absl::flat_hash_set<Channel*> predicated_channels;
  for (Node* predicated_node : predicated_nodes) {
    Channel* channel = nullptr;
    if (predicated_node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(channel,
                           f->package()->GetChannel(
                               predicated_node->As<Send>()->channel_name()));
    } else if (predicated_node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(channel,
                           f->package()->GetChannel(
                               predicated_node->As<Receive>()->channel_name()));
    } else {
      continue;
    }

    if (channel->kind() != ChannelKind::kStreaming ||
        down_cast<StreamingChannel*>(channel)->GetStrictness() !=
            ChannelStrictness::kProvenMutuallyExclusive) {
      continue;
    }
    if (predicated_channels.contains(channel)) {
      return true;
    }
    predicated_channels.insert(channel);
  }
  if (predicated_channels.empty()) {
    return false;
  }

  // Next, for each proven-mutually-exclusive channel this node *does* control,
  // check if there's any other node that controls an operation on that channel.
  for (Node* other_node : f->nodes()) {
    if (predicated_nodes.contains(other_node)) {
      // Controlled by the current node; we already accounted for that above.
      continue;
    }

    std::optional<Node*> predicate = p->GetPredicate(other_node);
    if (!predicate.has_value() || *predicate == node) {
      continue;
    }

    Channel* channel;
    if (other_node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(
          channel,
          f->package()->GetChannel(other_node->As<Send>()->channel_name()));
    } else if (other_node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(
          channel,
          f->package()->GetChannel(other_node->As<Receive>()->channel_name()));
    } else {
      continue;
    }
    if (predicated_channels.contains(channel)) {
      return true;
    }
  }
  return false;
}

}  // namespace

void Predicates::SetPredicate(Node* node, Node* pred) {
  if (predicated_by_.contains(node)) {
    Node* replaced_predicate = predicated_by_.at(node);
    predicate_of_[replaced_predicate].erase(node);
    if (predicate_of_[replaced_predicate].empty()) {
      predicate_of_.erase(replaced_predicate);
    }
  }
  predicated_by_[node] = pred;
  predicate_of_[pred].insert(node);
}

std::optional<Node*> Predicates::GetPredicate(Node* node) const {
  return predicated_by_.contains(node)
             ? std::make_optional(predicated_by_.at(node))
             : std::nullopt;
}

absl::flat_hash_set<Node*> Predicates::GetNodesPredicatedBy(Node* node) const {
  return predicate_of_.contains(node) ? predicate_of_.at(node)
                                      : absl::flat_hash_set<Node*>();
}

absl::Status Predicates::MarkMutuallyExclusive(Node* pred_a, Node* pred_b) {
  XLS_RET_CHECK_NE(pred_a, pred_b);
  XLS_RET_CHECK(FunctionIsOneBit(pred_a));
  XLS_RET_CHECK(FunctionIsOneBit(pred_b));
  mutual_exclusion_[pred_a][pred_b] = true;
  mutual_exclusion_[pred_b][pred_a] = true;
  return absl::OkStatus();
}

absl::Status Predicates::MarkNotMutuallyExclusive(Node* pred_a, Node* pred_b) {
  XLS_RET_CHECK_NE(pred_a, pred_b);
  XLS_RET_CHECK(FunctionIsOneBit(pred_a));
  XLS_RET_CHECK(FunctionIsOneBit(pred_b));
  mutual_exclusion_[pred_a][pred_b] = false;
  mutual_exclusion_[pred_b][pred_a] = false;
  return absl::OkStatus();
}

std::optional<bool> Predicates::QueryMutuallyExclusive(Node* pred_a,
                                                       Node* pred_b) const {
  if (!mutual_exclusion_.contains(pred_a)) {
    return std::nullopt;
  }
  if (!mutual_exclusion_.at(pred_a).contains(pred_b)) {
    return std::nullopt;
  }
  return mutual_exclusion_.at(pred_a).at(pred_b);
}

absl::flat_hash_map<Node*, bool> Predicates::MutualExclusionNeighbors(
    Node* pred) const {
  return mutual_exclusion_.contains(pred) ? mutual_exclusion_.at(pred)
                                          : absl::flat_hash_map<Node*, bool>();
}

void Predicates::ReplaceNode(Node* original, Node* replacement) {
  if (original == replacement) {
    return;
  }
  if (predicated_by_.contains(original)) {
    Node* predicate = predicated_by_.at(original);
    predicated_by_.erase(original);
    predicated_by_[replacement] = predicate;
    predicate_of_.at(predicate).erase(original);
    predicate_of_.at(predicate).insert(replacement);
  }
  if (predicate_of_.contains(original)) {
    for (Node* node : predicate_of_.at(original)) {
      predicate_of_[replacement].insert(node);
    }
    predicate_of_.erase(original);
  }
  if (mutual_exclusion_.contains(original)) {
    absl::flat_hash_map<Node*, bool> neighbors = mutual_exclusion_.at(original);
    mutual_exclusion_.erase(original);
    mutual_exclusion_[replacement] = neighbors;
    for (const auto& [neighbor, boolean] : neighbors) {
      mutual_exclusion_.at(neighbor).erase(original);
      mutual_exclusion_.at(neighbor)[replacement] = boolean;
    }
  }
}

absl::StatusOr<Node*> AddPredicate(Predicates* p, Node* node, Node* pred) {
  CHECK_EQ(node->function_base(), pred->function_base());
  FunctionBase* f = node->function_base();

  std::optional<Node*> existing_predicate_maybe = p->GetPredicate(node);

  if (existing_predicate_maybe.has_value()) {
    Node* existing_predicate = existing_predicate_maybe.value();
    XLS_ASSIGN_OR_RETURN(
        Node * pred_and_existing,
        f->MakeNode<NaryOp>(SourceInfo(),
                            std::vector<Node*>({existing_predicate, pred}),
                            Op::kAnd));
    p->SetPredicate(node, pred_and_existing);

    for (const auto [neighbor, boolean] :
         p->MutualExclusionNeighbors(existing_predicate)) {
      if (boolean) {
        XLS_RETURN_IF_ERROR(
            p->MarkMutuallyExclusive(pred_and_existing, neighbor));
      }
    }

    return pred_and_existing;
  }

  p->SetPredicate(node, pred);
  return pred;
}

absl::Status ComputeMutualExclusion(Predicates* p, FunctionBase* f,
                                    int64_t z3_rlimit) {
  if (f->IsBlock()) {
    return absl::OkStatus();
  }

  std::vector<std::pair<Node*, int64_t>> predicate_nodes = PredicateNodes(p, f);
  if (VLOG_IS_ON(3)) {
    for (const auto& [node, index] : predicate_nodes) {
      VLOG(3) << "Predicate: " << node;
    }
  }

  absl::flat_hash_map<Node*, absl::flat_hash_set<Op>> ops_for_pred;
  for (const auto& [node, index] : predicate_nodes) {
    bool includes_heavy_op = false;
    for (Node* predicated_by : p->GetNodesPredicatedBy(node)) {
      ops_for_pred[node].insert(predicated_by->op());
      if (IsHeavyOp(predicated_by->op())) {
        includes_heavy_op = true;
      }
    }
    if (!includes_heavy_op) {
      // Irrelevant; doesn't affect any heavy operation.
      ops_for_pred.erase(node);
    }
  }

  // Remove any predicate nodes that don't participate in any potential mutual
  // exclusion computations.
  std::erase_if(predicate_nodes,
                [&](const std::pair<Node*, int64_t>& predicate_node) {
                  const auto& [node, index] = predicate_node;
                  return !ops_for_pred.contains(node);
                });
  std::erase_if(
      predicate_nodes, [&](const std::pair<Node*, int64_t>& predicate_node) {
        const auto& [node, index] = predicate_node;
        const absl::flat_hash_set<Op>& ops = ops_for_pred.at(node);
        return absl::c_none_of(
            predicate_nodes, [&](const std::pair<Node*, int64_t>& other) {
              const auto& [other_node, other_index] = other;
              if (other_node == node) {
                // Skip if this is the same instance - but otherwise, if the
                // nodes match, the ops definitely intersect!
                return other_index != index;
              }
              return HasIntersection(ops, ops_for_pred.at(other_node));
            });
      });

  if (predicate_nodes.empty()) {
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<solvers::z3::IrTranslator> translator,
                       solvers::z3::IrTranslator::CreateAndTranslate(f, true));

  Z3_context ctx = translator->ctx();

  solvers::z3::ScopedErrorHandler seh(ctx);

  // Determine for each predicate whether it is always false using Z3.
  // Dead nodes are mutually exclusive with all other nodes, so this can reduce
  // the runtime by doing only a linear amount of Z3 calls to remove
  // quadratically many Z3 calls.
  for (const auto& [node, index] : predicate_nodes) {
    Z3_ast translated = translator->GetTranslation(node);
    // Check whether it's possible for `node` to need to be proven mutually
    // exclusive with some other node in order for channel operations to be
    // legal; if so, we remove the rlimit on the prover.
    XLS_ASSIGN_OR_RETURN(
        bool required_for_compilation,
        ControlsContendedProvenMutuallyExclusiveChannel(node, p, f));
    if (required_for_compilation) {
      LOG(INFO) << "Removing Z3's rlimit for always-false check on "
                << node->GetName()
                << " as mutual exclusion is required for compilation.";
    }
    translator->SetRlimit(z3_rlimit);
    if (RunSolver(ctx, solvers::z3::BitVectorToBoolean(ctx, translated)) ==
        Z3_L_FALSE) {
      VLOG(3) << "Proved that " << node << " is always false";
      // A constant false node is mutually exclusive with all other nodes.
      for (const auto& [other, other_index] : predicate_nodes) {
        if (index != other_index) {
          XLS_RETURN_IF_ERROR(p->MarkMutuallyExclusive(node, other));
        }
      }
    }
  }

  int64_t known_false = 0;
  int64_t known_true = 0;
  int64_t unknown = 0;

  for (const auto& [node_a, index_a] : predicate_nodes) {
    XLS_ASSIGN_OR_RETURN(
        absl::flat_hash_set<Channel*> channels_a,
        GetControlledProvenMutuallyExclusiveChannels(node_a, p, f));
    for (const auto& [node_b, index_b] : predicate_nodes) {
      // This prevents checking `a NAND b` and then later checking `b NAND a`.
      if (index_a >= index_b) {
        continue;
      }

      // Skip this pair if we already know whether they are mutually exclusive.
      if (p->QueryMutuallyExclusive(node_a, node_b).has_value()) {
        continue;
      }

      if (!ops_for_pred.contains(node_a) || !ops_for_pred.contains(node_b) ||
          !HasIntersection(ops_for_pred.at(node_a), ops_for_pred.at(node_b))) {
        continue;
      }

      Z3_ast z3_a = translator->GetTranslation(node_a);
      Z3_ast z3_b = translator->GetTranslation(node_b);

      // We try to find out if `a ∧ b` is satisfiable, which is true iff
      // `a NAND b` is not valid.
      Z3_ast a_and_b =
          solvers::z3::BitVectorToBoolean(ctx, Z3_mk_bvand(ctx, z3_a, z3_b));

      // Check whether `a` and `b` must be proven mutually exclusive in order
      // for channel operations to be legal; if so, we remove the rlimit on the
      // prover.
      XLS_ASSIGN_OR_RETURN(
          absl::flat_hash_set<Channel*> channels_b,
          GetControlledProvenMutuallyExclusiveChannels(node_b, p, f));
      bool required_for_compilation = HasIntersection(channels_a, channels_b);
      translator->SetRlimit(required_for_compilation ? 0 : z3_rlimit);
      if (required_for_compilation) {
        LOG(INFO) << "Removing Z3's rlimit for mutual exclusion between "
                  << node_a->GetName() << " and " << node_b->GetName()
                  << " as mutual exclusion is required for compilation.";
      }
      Z3_lbool satisfiable = RunSolver(ctx, a_and_b);

      if (satisfiable == Z3_L_FALSE) {
        known_true += 1;
        XLS_RETURN_IF_ERROR(p->MarkMutuallyExclusive(node_a, node_b));
      } else if (satisfiable == Z3_L_TRUE) {
        known_false += 1;
        XLS_RETURN_IF_ERROR(p->MarkNotMutuallyExclusive(node_a, node_b));
        if (required_for_compilation) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "Proved that %s and %s, which control operations on "
              "proven-mutually-exclusive channels (%s), are not mutually "
              "exclusive.",
              node_a->GetName(), node_b->GetName(),
              absl::StrJoin(Intersection(channels_a, channels_b), ", ",
                            [](std::string* out, Channel* channel) {
                              absl::StrAppend(out, channel->name());
                            })));
        }
      } else {
        unknown += 1;
        VLOG(3) << "Z3 ran out of time checking mutual exclusion of "
                << node_a->GetName() << " and " << node_b->GetName();
        if (required_for_compilation) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "Z3 failed to prove that %s and %s, which control operations on "
              "proven-mutually-exclusive channels (%s), are mutually "
              "exclusive.",
              node_a->GetName(), node_b->GetName(),
              absl::StrJoin(Intersection(channels_a, channels_b), ", ",
                            [](std::string* out, Channel* channel) {
                              absl::StrAppend(out, channel->name());
                            })));
        }
      }
    }
  }

  VLOG(3) << "known_false = " << known_false;
  VLOG(3) << "known_true  = " << known_true;
  VLOG(3) << "unknown     = " << unknown;

  XLS_RETURN_IF_ERROR(seh.status());

  return absl::OkStatus();
}

absl::StatusOr<bool> MutualExclusionPass::RunOnFunctionBaseInternal(
    FunctionBase* f, SchedulingUnit* unit, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  ScheduleCycleMap scm;
  if (unit->schedules().contains(f)) {
    if (f != unit->schedules().at(f).function_base()) {
      return false;
    }
    scm = unit->schedules().at(f).GetCycleMap();
  } else {
    for (Node* node : f->nodes()) {
      scm[node] = 0;
    }
  }

  // Sets limit on z3 "solver resources", so that the pass doesn't take too long
  const int64_t z3_rlimit =
      options.scheduling_options.mutual_exclusion_z3_rlimit().value_or(5000);

  Predicates p;
  XLS_RETURN_IF_ERROR(AddSendReceivePredicates(&p, f));
  XLS_RETURN_IF_ERROR(ComputeMutualExclusion(&p, f, z3_rlimit));
  XLS_ASSIGN_OR_RETURN(std::vector<absl::flat_hash_set<Node*>> merge_classes,
                       ComputeMergeClasses(&p, f, scm));

  if (VLOG_IS_ON(3)) {
    for (const absl::flat_hash_set<Node*>& merge_class : merge_classes) {
      if (merge_class.size() <= 1) {
        continue;
      }
      Op op = (*(merge_class.cbegin()))->op();
      std::vector<std::string> name_pred_pairs;
      for (Node* node : merge_class) {
        CHECK_EQ(node->op(), op);
        name_pred_pairs.push_back(
            absl::StrFormat("%s [%s]", node->GetName(),
                            p.GetPredicate(node).value()->GetName()));
      }
      VLOG(3) << "Merge class: " << merge_class.size() << ", op = " << op;
      VLOG(3) << "    " << absl::StrJoin(name_pred_pairs, ", ");
    }
  }

  VLOG(3) << "Successfully computed mutual exclusion for " << f->name();

  bool changed = false;
  for (const absl::flat_hash_set<Node*>& merge_class : merge_classes) {
    XLS_ASSIGN_OR_RETURN(bool subpass_changed, MergeNodes(&p, f, merge_class));
    changed = changed || subpass_changed;
  }

  if (changed) {
    unit->schedules().clear();
  }

  return changed;
}

}  // namespace xls
