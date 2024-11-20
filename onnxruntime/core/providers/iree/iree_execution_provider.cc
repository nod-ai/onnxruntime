// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/iree/iree_execution_provider.h"

#include "core/common/inlined_containers.h"
#include "core/framework/compute_capability.h"
#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"

#include "core/providers/iree/compiler/jit_compiler.h"

#include <cassert>
#include <codecvt>
#include <fstream>
#include <istream>

namespace onnxruntime {

IREEExecutionProvider::IREEExecutionProvider(const ProviderOptions& info)
    : IExecutionProvider{onnxruntime::kIreeExecutionProvider}, info_(info) {
  registry_ = std::make_shared<KernelRegistry>();
  rt_instance_ = std::make_shared<iree_ep_rt::Instance>();
  auto status = Initialize();
  if (!status.IsOK()) {
    LOGS_DEFAULT(FATAL) << "IREEExecutionProvider failed to initialize: " << status.ToString();
  }
}

IREEExecutionProvider::~IREEExecutionProvider() {
}

common::Status IREEExecutionProvider::Initialize() {
  if (info_.find("device") == info_.end())
    info_["device"] = "local-task";
  ORT_RETURN_IF_ERROR(iree_ep_rt::HandleIREEStatus(rt_instance_->Initialize(info_["device"])));
  return common::Status::OK();
}

std::shared_ptr<KernelRegistry> IREEExecutionProvider::GetKernelRegistry() const { return registry_; }

std::vector<std::unique_ptr<ComputeCapability>> IREEExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& kernel_lookup) const {
  if (graph_viewer.IsSubgraph()) {
    LOGS(*GetLogger(), INFO) << "IREEExecutionProvider::GetCapability() FAIL: IsSubgraph()";
    return {};
  }

  // Assume all are valid. If this turns out to not be true, then we need to filter instead of doing
  // this fallback action.
  // This implementation adapted from other EPs, most notably TVM's, which has a similarly simple
  // heuristic.
  std::vector<std::unique_ptr<ComputeCapability>> result;
  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();
  std::unordered_set<std::string> required_initializers;

  auto sorted_nodes = graph_viewer.GetNodesInTopologicalOrder();
  auto sub_graph = std::make_unique<onnxruntime::IndexedSubGraph>();
  for (NodeIndex& node_idx : sorted_nodes) {
    auto* node = graph_viewer.GetNode(node_idx);
    LOGS(*GetLogger(), INFO) << "  add to subgraph: node = " << node->OpType() << " (" << node->Name() << ")";

    node->ForEachDef([&required_initializers, &init_tensors](const NodeArg& node_arg, bool is_input) {
              if(is_input && init_tensors.count(node_arg.Name())) {
                  required_initializers.insert(node_arg.Name());
              } }, true);
  }

  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "IREE";
  meta_def->domain = "IREE";
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  for (auto& nodeArgPtr : graph_viewer.GetInputs()) {
    inputs.push_back(nodeArgPtr->Name());
  }

  for (auto& nodeArgPtr : graph_viewer.GetOutputs()) {
    outputs.push_back(nodeArgPtr->Name());
  }

  meta_def->inputs = inputs;
  meta_def->outputs = outputs;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  sub_graph->SetMetaDef(std::move(meta_def));
  sub_graph->nodes = sorted_nodes;
  result.push_back(
      std::make_unique<ComputeCapability>(std::move(sub_graph)));

  return result;
}

common::Status IREEExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                              std::vector<NodeComputeInfo>& node_compute_funcs) {
  iree_ep_jit::CompilerSession compiler(*GetLogger());
  // TODO: The target needs to be synchronized with the runtime based on EP options.
  // TODO: We should just be adding the target to the module instead of specifying via
  // flags.
  std::string device_flag = "--iree-hal-target-device=";
  if (info_.find("hal_target_device") == info_.end()) {
    // In case device info is absent, set `llvm-cpu` as default hal-target-device.
    device_flag.append("llvm-cpu");
  } else {
    device_flag.append(info_["hal_target_device"]);
  }
  LOGS(*GetLogger(), INFO) << "IREEExecutionProvider compile: setting flag " << device_flag;
  ORT_RETURN_IF_ERROR(compiler.SetFlag(device_flag.c_str()));

  // Set all the compile-time flags.
  // TODO(Shukla-Gaurav): Use ireeCompilerSessionSetFlags API to set all the flags at once.
  // TODO(Shukla-Gaurav): support more than one extra flags by parsing the input string.
  if (info_.find("compile_time_flags") != info_.end()) {
    std::string extra_flag = info_["compile_time_flags"];
    LOGS(*GetLogger(), INFO) << "IREEExecutionProvider compile: setting flag " << extra_flag;
    ORT_RETURN_IF_ERROR(compiler.SetFlag(extra_flag.c_str()));
  }

  ORT_RETURN_IF_ERROR(compiler.Initialize());
  std::string module_name = "ort";
  iree_ep_jit::CompilerInvocation inv(compiler, module_name.c_str());

  // This loop is often single-trip but can be used for batch compilation.
  // We import each fused node by name as a top-level function, which produces a more parallelized
  // compilation.
  std::vector<std::string> entrypoint_names;
  for (auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_view = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;
    const std::string& func_name = fused_node.Name();
    ORT_RETURN_IF_ERROR(inv.ImportSubgraph(graph_view, func_name));
    // The fully qualified name is the {module_name}.{func_name}. This is what we look up at
    // runtime.
    std::string fq_name(module_name);
    fq_name.append(".");
    fq_name.append(func_name);
    entrypoint_names.push_back(fq_name);
  }

  // Compile aggregate module to a VMFB membuffer.
  iree_ep_jit::CompilerOutput vmfb_output;
  if (auto* err = ireeCompilerOutputOpenMembuffer(&vmfb_output.output)) {
    return iree_ep_jit::ErrorToStatus(err, "Failure opening compiler output buffer: ");
  }

  // This will save the compiled module to temporary directory.
  fs::path save_to = fs::temp_directory_path();
  if (info_.find("save_to") != info_.end() && fs::is_directory(info_["save_to"]))
    save_to = fs::path(info_["save_to"]);

  fs::path file_name("compiled_model.vmfb");
  fs::path vmfb_path = save_to / file_name;

  ORT_RETURN_IF_ERROR(inv.CompileAndOutputVMFB(vmfb_output.output, vmfb_path));
  LOGS(*GetLogger(), INFO) << "IREEExecutionProvider compiled vmfb saved at this location " << vmfb_path;

  // Map raw memory.
  // void* vmfb_contents = nullptr;
  // uint64_t vmfb_size = 0;
  // TODO(Shukla-Gaurav): Map memory instead of storing the compiled module as a file
  // ORT_RETURN_IF_ERROR(vmfb_output.MapMemory(&vmfb_contents, &vmfb_size));

  // Create a new runtime session.
  auto rt_session = std::make_shared<iree_ep_rt::Session>(rt_instance_);
  // In case device info is absent, set `local-task` as default device.
  ORT_RETURN_IF_ERROR(iree_ep_rt::HandleIREEStatus(rt_session->Initialize()));

  // Release hal device after session initialization.
  if (rt_instance_->device) {
    iree_hal_device_release(rt_instance_->device);
  }

  // Load the compiled module, releasing our ownership of the CompilerOutput.
  ORT_RETURN_IF_ERROR(iree_ep_rt::HandleIREEStatus(rt_session->AppendBytecodeModule(vmfb_path,
                                                                                    vmfb_output.Release(vmfb_path))));

  for (auto& entrypoint_name : entrypoint_names) {
    node_compute_funcs.push_back(CreateNodeComputeFunc(entrypoint_name, rt_session));
  }

  return common::Status::OK();
}

NodeComputeInfo IREEExecutionProvider::CreateNodeComputeFunc(
    std::string entrypoint_name, std::shared_ptr<iree_ep_rt::Session> session) {
  // Note that arguments are necessarily passed by value, since we have to capture
  // them into lambdas (by value). This is an unfortunate way that ORT chose to
  // do all of this.
  NodeComputeInfo info;
  info.create_state_func = [](ComputeContext*, FunctionState*) -> int {
    return 0;
  };
  info.compute_func = [entrypoint_name, session](
                          FunctionState, const OrtApi* ort_api, OrtKernelContext* ort_context) -> common::Status {
    return session->Call(entrypoint_name.c_str(), ort_api, ort_context);
  };
  info.release_state_func = [](FunctionState) {
  };

  return info;
}

}  // namespace onnxruntime
