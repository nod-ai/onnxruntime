// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Interface to the IREE compiler. This is not included in minimal builds
// (which require a pre-compilation step).

#include "core/providers/iree/compiler/jit_compiler.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/providers/iree/compiler/torch-mlir-import-onnx/OnnxImporter.h"
#include "mlir-c/BuiltinAttributes.h"

#include <cstring>

namespace onnxruntime::iree_ep_jit {

namespace {

bool InitializeCompiler() {
  // TODO: Come up with something better for this.
  static bool initialized = ([]() {
    ireeCompilerGlobalInitialize();
    return true;
  })();
  return initialized;
}

inline MlirStringRef toMlirStringRef(const char* s) {
  return mlirStringRefCreate(s, std::strlen(s));
}

std::string MlirOperationToString(MlirOperation op, bool generic = false, bool debug_info = true,
                                  bool elide_large = false) {
  std::string s;
  auto callback = +[](MlirStringRef sr, void* userdata) {
    std::string* s = static_cast<std::string*>(userdata);
    s->append(sr.data, sr.length);
  };
  MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
  mlirOpPrintingFlagsEnableDebugInfo(flags, debug_info, false);
  if (generic) {
    mlirOpPrintingFlagsPrintGenericOpForm(flags);
  }
  if (elide_large) {
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, 100);
  }
  mlirOperationPrintWithFlags(op, flags, callback, static_cast<void*>(&s));
  mlirOpPrintingFlagsDestroy(flags);
  return s;
}

}  // namespace

common::Status ErrorToStatus(iree_compiler_error_t* err, std::string message_prefix) {
  if (!err) {
    return common::Status::OK();
  }

  const char* message = ireeCompilerErrorGetMessage(err);
  message_prefix.append(message, std::strlen(message));
  ireeCompilerErrorDestroy(err);
  return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, std::move(message_prefix));
}

CompilerSession::CompilerSession(const onnxruntime::logging::Logger& logger) : logger(logger) {
  InitializeCompiler();
  session = ireeCompilerSessionCreate();
  context = ireeCompilerSessionBorrowContext(session);
}

CompilerSession::~CompilerSession() {
  ireeCompilerSessionDestroy(session);
}

common::Status CompilerSession::Initialize() {
  // Loads a dialect into the context. If building IR programmatically (vs parsing from ASM), it is necessary to
  // load all dialects that will be generated.
  if (mlirDialectIsNull(mlirContextGetOrLoadDialect(context, toMlirStringRef("builtin")))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Dialect 'builtin' not registered in the IREE compiler");
  }
  if (mlirDialectIsNull(mlirContextGetOrLoadDialect(context, toMlirStringRef("func")))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Dialect 'func' not registered in the IREE compiler");
  }
  if (mlirDialectIsNull(mlirContextGetOrLoadDialect(context, toMlirStringRef("torch")))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION,
                           "Dialect 'torch' not registered in the IREE compiler "
                           "(this could mean that IREE was not built with TORCH support enabled)");
  }

  // TODO: Make it possible to set the input type on the module vs as a flag as it will produce better
  // reproducers and error messages.
  ORT_RETURN_IF_ERROR(SetFlag("--iree-input-type=onnx"));
  return common::Status::OK();
}

common::Status CompilerSession::SetFlag(const char* flag) {
  if (auto err = ireeCompilerSessionSetFlags(session, 1, &flag)) {
    return ErrorToStatus(err, "Error setting IREE compiler flag: ");
  }

  return common::Status::OK();
}

CompilerInvocation::CompilerInvocation(CompilerSession& session, const char* module_name) : session(session) {
  inv = ireeCompilerInvocationCreate(session.session);

  // Set up diagnostics.
  ireeCompilerInvocationEnableCallbackDiagnostics(
      inv, /*flags=*/0, +[](enum iree_compiler_diagnostic_severity_t severity, const char* message, size_t messageSize, void* userData) {
        CompilerInvocation* self = static_cast<CompilerInvocation*>(userData);
        self->diagnostics.emplace_back(DiagnosticRecord{severity, std::string(message, messageSize)});
        // VLOG it.
        VLOGS(session.logger, INFO) << self->diagnostics.back().ToString();
      },
      static_cast<void*>(this));

  // Set up crash handler.
  ireeCompilerInvocationSetCrashHandler(
      inv, /*genLocalReproducer=*/false,
      +[](iree_compiler_output_t** out_output, void* userdata) -> iree_compiler_error_t* {
        auto* self = static_cast<CompilerInvocation*>(userdata);
        // TODO: We need to have better configuration for how to dump such reproducers.
        auto output_path = std::filesystem::temp_directory_path() / "ort_iree_reproducer.mlir";
        std::string output_path_str = output_path;
        LOGS(self->session.logger, ERROR) << "IREE compiler crash. Writing reproducer to: " << output_path_str;
        return ireeCompilerOutputOpenFile(output_path_str.c_str(), out_output);
      },
      static_cast<void*>(this));

  // Ownership of the module is immediately transferred to the invocation.
  // Note that this implicitly initializes defaults on the invocation, so any setup or callbacks must
  // be done prior.
  MlirModule stolen_module = mlirModuleCreateEmpty(mlirLocationUnknownGet(session.context));
  module_op = mlirModuleGetOperation(stolen_module);
  MlirAttribute module_name_attr = mlirStringAttrGet(session.context, toMlirStringRef(module_name));
  mlirOperationSetInherentAttributeByName(module_op, toMlirStringRef("sym_name"), module_name_attr);
  ireeCompilerInvocationImportStealModule(inv, module_op);
}

CompilerInvocation::~CompilerInvocation() {
  ireeCompilerInvocationDestroy(inv);
}

common::Status CompilerInvocation::ImportSubgraph(const onnxruntime::GraphViewer& graph_view, const std::string& func_name) {
  // Note that we just use a synthetic top-level ModelProto and forego main
  // graph initialization. Since we are operating on a subgraph view, we
  // initialize from the backing Graph proto but initialize it ourselves.
  // TODO: Refactor upstream to make it clear that this is a supported way
  // of using.
  torch_mlir_onnx::ModelInfo model_info;

  // Populate the domain to version map from the GraphView.
  for (auto it : graph_view.DomainToVersionMap()) {
    auto* opset_import = model_info.model_proto().add_opset_import();
    if (!it.first.empty()) {
      opset_import->set_domain(it.first);
    }
    opset_import->set_version(it.second);
  }

  ONNX_NAMESPACE::GraphProto graph_proto;
  GraphViewerToProto(graph_view, graph_proto, false, false);
  // LOGS(session.logger, INFO) << "  full graph: " << graph_proto.DebugString();

  // Set up for subgraph import.
  torch_mlir_onnx::GraphInfo subgraph_info(graph_view, model_info, graph_proto);
  if (torch_mlir_onnx::failed(subgraph_info.Initialize())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, model_info.error_message());
  }

  // Reset whole-graph inputs and replace with subgraph inputs.
  subgraph_info.inputs().clear();
  for (const auto& node_arg : graph_view.GetInputs()) {
    const ONNX_NAMESPACE::ValueInfoProto& input_info = node_arg->ToProto();
    // LOGS(session.logger, INFO) << "  input: " << input_info.DebugString();
    subgraph_info.inputs().push_back(&input_info);
  }

  // And the same with outputs.
  subgraph_info.outputs().clear();
  for (const auto& node_arg : graph_view.GetOutputs()) {
    const ONNX_NAMESPACE::ValueInfoProto& output_info = node_arg->ToProto();
    // LOGS(session.logger, INFO) << "  output: " << output_info.DebugString();
    subgraph_info.outputs().push_back(&output_info);
  }

  // Now import it.
  torch_mlir_onnx::ContextCache cc(model_info, session.context);
  torch_mlir_onnx::NodeImporter imp(subgraph_info, cc, module_op);
  MlirOperation func_op;  // Not owned.
  if (torch_mlir_onnx::failed(imp.DefineFunction(func_name, &func_op))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to define entry function for graph: ",
                           model_info.error_message(), ConsumeDiagnostics());
  }

  if (torch_mlir_onnx::failed(imp.ImportAll())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to import nodes",
                           ": ", model_info.error_message(),
                           ConsumeDiagnostics());
  }

  // Verify the function at the point of import because we have better diagnostics.
  if (!mlirOperationVerify(func_op)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Imported ONNX IR failed to verify.", ConsumeDiagnostics(),
                           "\nUnverified MLIR module:\n", MlirOperationToString(func_op, /*generic=*/true));
  }

  return common::Status::OK();
}

common::Status CompilerInvocation::CompileAndOutputVMFB(iree_compiler_output_t* output, fs::path vmfb_path) {
  // Main compilation.
  if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "IREE compilation error.", ConsumeDiagnostics());
  }

  // Attach the compiled output to a file.
  ireeCompilerOutputOpenFile(vmfb_path.c_str(), &output);

  // Output.
  if (auto* err = ireeCompilerInvocationOutputVMBytecode(inv, output)) {
    return ErrorToStatus(err, "Failure emitting VM bytecode: ");
  }

  return common::Status::OK();
}

std::string CompilerInvocation::ConsumeDiagnostics() {
  if (diagnostics.empty()) {
    return std::string();
  }

  std::string accum("\nDiagnostics:");
  for (auto& diag : diagnostics) {
    accum.append("\n  ");
    diag.ToString(accum);
  }

  diagnostics.clear();
  return accum;
}

common::Status CompilerOutput::MapMemory(void** contents, uint64_t* size) {
  return ErrorToStatus(ireeCompilerOutputMapMemory(output, contents, size), "Failed to map compiler output memory: ");
}

void DiagnosticRecord::ToString(std::string& accum) {
  switch (severity) {
    case IREE_COMPILER_DIAGNOSTIC_SEVERITY_NOTE:
      accum.append("note: ");
      break;
    case IREE_COMPILER_DIAGNOSTIC_SEVERITY_WARNING:
      accum.append("warning: ");
      break;
    case IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR:
      accum.append("error: ");
      break;
    case IREE_COMPILER_DIAGNOSTIC_SEVERITY_REMARK:
      accum.append("remark: ");
      break;
    default:
      accum.append("<unknown severity>: ");
  }
  accum.append(message);
}

}  // namespace onnxruntime::iree_ep_jit
