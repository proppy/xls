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

#include "xls/contrib/mlir/tools/xls_translate/xls_stitch.h"

#include <cassert>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "xls/codegen/vast/vast.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/ir/source_location.h"

namespace mlir::xls {
namespace {
namespace vast = ::xls::verilog;
using ::llvm::any_of;
using ::xls::SourceInfo;

struct ChannelPortNames {
  std::string data;
  std::string ready;
  std::string valid;
};

ChannelPortNames getChannelPortNames(ChanOp chan,
                                     const XlsStitchOptions& options) {
  ChannelPortNames result;
  result.data = absl::StrCat(chan.getName().str(), options.data_port_suffix);
  result.ready = absl::StrCat(chan.getName().str(), options.ready_port_suffix);
  result.valid = absl::StrCat(chan.getName().str(), options.valid_port_suffix);
  return result;
}

struct ChannelLogicRefs {
  vast::LogicRef* data;
  vast::LogicRef* ready;
  vast::LogicRef* valid;
};
}  // namespace

LogicalResult XlsStitch(ModuleOp op, llvm::raw_ostream& output,
                        XlsStitchOptions options) {
  vast::VerilogFile f(vast::FileType::kSystemVerilog);
  vast::Module* top = f.AddModule(op.getName().value_or("top"), SourceInfo());
  SymbolTableCollection symbolTable;
  SymbolUserMap symbolUsers(symbolTable, op);

  vast::DataType* i1 = f.BitVectorType(1, SourceInfo());

  vast::LogicRef* clk =
      top->AddInput(options.clock_signal_name, i1, SourceInfo());
  vast::LogicRef* rst =
      top->AddInput(options.reset_signal_name, i1, SourceInfo());

  DenseMap<ChanOp, ChannelLogicRefs> channelRefs;
  op->walk([&](ChanOp chan) {
    // If the channel is used by a discardable eproc, then it never appears
    // during stitching.
    bool isEphemeralChannel =
        any_of(symbolUsers.getUsers(chan), [](Operation* user) {
          auto eproc = user->getParentOfType<EprocOp>();
          return eproc && eproc.getDiscardable();
        });
    if (isEphemeralChannel) {
      return;
    }

    ChannelPortNames names = getChannelPortNames(chan, options);
    vast::DataType* dataType =
        f.BitVectorType(chan.getType().getIntOrFloatBitWidth(), SourceInfo());
    ChannelLogicRefs refs;
    if (chan.getSendSupported() && chan.getRecvSupported()) {
      // Interior port; this becomes a wire.
      refs.data = top->AddWire(names.data, dataType, SourceInfo());
      refs.ready = top->AddWire(names.ready, i1, SourceInfo());
      refs.valid = top->AddWire(names.valid, i1, SourceInfo());
    } else if (chan.getSendSupported()) {
      // Output port; this becomes an output port.
      refs.data = top->AddOutput(names.data, dataType, SourceInfo());
      refs.ready = top->AddInput(names.ready, i1, SourceInfo());
      refs.valid = top->AddOutput(names.valid, i1, SourceInfo());
    } else {
      assert(chan.getRecvSupported());
      // Input port; this becomes an input port.
      refs.data = top->AddInput(names.data, dataType, SourceInfo());
      refs.ready = top->AddOutput(names.ready, i1, SourceInfo());
      refs.valid = top->AddInput(names.valid, i1, SourceInfo());
    }
    channelRefs[chan] = refs;
  });

  DenseMap<StringRef, int> instantiationCount;
  op->walk([&](InstantiateEprocOp op) {
    std::vector<vast::Connection> connections;
    for (auto [local, global] :
         llvm::zip(op.getLocalChannels(), op.getGlobalChannels())) {
      ChanOp localChan = symbolTable.lookupNearestSymbolFrom<ChanOp>(
          op, cast<FlatSymbolRefAttr>(local));
      ChanOp globalChan = symbolTable.lookupNearestSymbolFrom<ChanOp>(
          op, cast<FlatSymbolRefAttr>(global));
      ChannelPortNames localNames = getChannelPortNames(localChan, options);
      connections.push_back(vast::Connection{
          .port_name = localNames.data,
          .expression = channelRefs[globalChan].data,
      });
      connections.push_back(vast::Connection{
          .port_name = localNames.ready,
          .expression = channelRefs[globalChan].ready,
      });
      connections.push_back(vast::Connection{
          .port_name = localNames.valid,
          .expression = channelRefs[globalChan].valid,
      });
    }
    connections.push_back(vast::Connection{
        .port_name = options.clock_signal_name,
        .expression = clk,
    });
    connections.push_back(vast::Connection{
        .port_name = options.reset_signal_name,
        .expression = rst,
    });
    std::string instanceName = absl::StrCat(
        op.getEproc().str(), "_", instantiationCount[op.getEproc()]++);
    top->Add<vast::Instantiation>(
        SourceInfo(), op.getEproc().str(), instanceName,
        /*parameters=*/absl::Span<const vast::Connection>(), connections);
  });

  output << f.Emit();
  return success();
}

}  // namespace mlir::xls