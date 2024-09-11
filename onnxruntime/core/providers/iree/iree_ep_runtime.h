// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"
#include "iree/runtime/api.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace onnxruntime::iree_ep_rt {

// Handles a failing IREE status.
common::Status HandleFailingIREEStatus(iree_status_t iree_status);

// Handles an iree_status_t, translating it to an ORT Status.
inline common::Status HandleIREEStatus(iree_status_t iree_status) {
  if (iree_status_is_ok(iree_status)) {
    return common::Status::OK();
  }
  return HandleFailingIREEStatus(iree_status);
}

// Wraps an iree_runtime_instance_t.
struct Instance {
  Instance();
  ~Instance();

  // Initializes the instance.
  // TODO: We should probably pass the options in here and use it to set up.
  iree_status_t Initialize(std::string device_str);

  // Instance globals.
  iree_runtime_instance_options_t options;
  iree_runtime_instance_t* instance = nullptr;

  // Device globals.
  // TODO: This is hoaky and we need a way to configure multiples.
  iree_hal_device_t* device = nullptr;
};

struct Session {
  Session(std::shared_ptr<Instance> instance);
  ~Session();

  // Failable session initialization.
  iree_status_t Initialize();

  // Append a user-compiled bytecode module buffer to the session, along with a dispose callback.
  // The dispose callback will be invoked when Session is destroyed regardless of success/failure
  // of this call.
  iree_status_t AppendBytecodeModule(fs::path vmfb_path, std::function<void()> dispose_callback);

  // Calls the entrypoint. This returns an ORT Status and normalizes any IREE statuses to that
  // because that can arise from ORT interactions.
  common::Status Call(const char* entrypoint_name, const OrtApi* ort_api, OrtKernelContext* ort_context);

  std::shared_ptr<Instance> instance;
  iree_runtime_session_options_t session_options;
  iree_runtime_session_t* session = nullptr;
  std::vector<std::function<void()>> dispose_callbacks;
};

}  // namespace onnxruntime::iree_ep_rt
