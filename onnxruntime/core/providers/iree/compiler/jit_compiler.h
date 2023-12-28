// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Interface to the IREE compiler. This is not included in minimal builds
// (which require a pre-compilation step).

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "iree/compiler/embedding_api.h"
#include "iree/compiler/mlir_interop.h"

#include <string>
#include <string_view>

namespace onnxruntime::iree_ep_jit {

common::Status ErrorToStatus(iree_compiler_error_t* err, std::string message_prefix);

struct DiagnosticRecord {
  iree_compiler_diagnostic_severity_t severity;
  std::string message;

  void ToString(std::string& accum);
  std::string ToString() {
    std::string accum;
    ToString(accum);
    return accum;
  }
};

// Wraps an iree_compiler_output_t*, destroying it when going out of scope if not null.
struct CompilerOutput {
  ~CompilerOutput() {
    if (output) {
      ireeCompilerOutputDestroy(output);
    }
  }

  // Maps memory (requires to be created with ireeCompilerOutputOpenMembuffer).
  common::Status MapMemory(void** contents, uint64_t* size);

  // Releases ownership of the output, returning a callback that can be used to
  // destroy it at a later date.
  std::function<void()> Release() {
    iree_compiler_output_t* local_output = output;
    this->output = nullptr;
    return [local_output]() {
      if (local_output) {
        ireeCompilerOutputDestroy(local_output);
      }
    };
  }

  iree_compiler_output_t* output = nullptr;
};

// Wraps the IREE C compiler API, holding a session.
// This is managed separatedly from the invocation because it is possible to pool sessions across multiple invocations.
struct CompilerSession {
  CompilerSession(const onnxruntime::logging::Logger& logger);
  ~CompilerSession();

  // Initialize the session.
  common::Status Initialize();
  common::Status SetFlag(const char* flag);

  // Unowned logger.
  const onnxruntime::logging::Logger& logger;
  // Owned session.
  iree_compiler_session_t* session;
  // Un-owned context (it is owned by the session).
  MlirContext context;
};

// Invocation of a session.
struct CompilerInvocation {
  CompilerInvocation(CompilerSession& session, const char* module_name);
  ~CompilerInvocation();

  // Imports a subgraph as a public function.
  common::Status ImportSubgraph(const onnxruntime::GraphViewer& graph_view, const std::string& func_name);

  // Compile and output a VMFB.
  common::Status CompileAndOutputVMFB(iree_compiler_output_t* output);

  // If there are any diagnostics, clears them and returns a loggable string.
  std::string ConsumeDiagnostics();

  CompilerSession& session;
  iree_compiler_invocation_t* inv;
  MlirOperation module_op;
  std::vector<DiagnosticRecord> diagnostics;
};

}  // namespace onnxruntime::iree_ep_jit
