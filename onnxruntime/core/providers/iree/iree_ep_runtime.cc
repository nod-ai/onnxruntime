// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/iree/iree_ep_runtime.h"

#include "core/session/onnxruntime_cxx_api.h"
#include <iostream>

namespace onnxruntime::iree_ep_rt {

common::Status HandleFailingIREEStatus(iree_status_t iree_status) {
  iree_status_ignore(iree_status);
  if (iree_status_is_ok(iree_status)) {
    return common::Status::OK();
  }

  std::string buffer = iree::Status::ToString(iree_status);

  return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "IREE Runtime Error: ", std::move(buffer));
}

Instance::Instance() {
  iree_runtime_instance_options_initialize(&options);
  iree_runtime_instance_options_use_all_available_drivers(&options);
}

Instance::~Instance() {
  if (instance) {
    iree_runtime_instance_release(instance);
  }
  if (device) {
    iree_hal_device_release(device);
  }
}

iree_status_t Instance::Initialize(std::string device_str) {
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &options, iree_allocator_system(), &instance));

  // TODO: Need real device selection.
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view(device_str.c_str()), &device));

  return iree_ok_status();
}

Session::Session(std::shared_ptr<Instance> instance) : instance(std::move(instance)) {
  iree_runtime_session_options_initialize(&session_options);
}

Session::~Session() {
  if (session) {
    iree_runtime_session_release(session);
  }
  for (auto& cb : dispose_callbacks) {
    cb();
  }
}

iree_status_t Session::Initialize() {
  iree_status_t res_status = iree_runtime_session_create_with_device(
      instance->instance, &session_options, instance->device,
      iree_runtime_instance_host_allocator(instance->instance),
      &session);
  iree_vm_module_t* custom_module = NULL;
  iree_allocator_t host_allocator = iree_allocator_system();
  IREE_CHECK_OK(iree_custom_module_async_create(
      iree_runtime_instance_vm_instance(instance->instance), instance->device,
      host_allocator, &custom_module));
  IREE_CHECK_OK(iree_runtime_session_append_module(session, custom_module));
  iree_vm_module_release(custom_module);
  return res_status;
}

iree_status_t Session::AppendBytecodeModule(fs::path vmfb_path, std::function<void()> dispose_callback) {
  dispose_callbacks.push_back(std::move(dispose_callback));
  // TODO(Shukla-Gaurav): load from memory instead of file.
  // return iree_runtime_session_append_bytecode_module_from_memory(
  //     session, iree_make_const_byte_span(contents, size),
  //     iree_allocator_null());
  return iree_runtime_session_append_bytecode_module_from_file(
      session, vmfb_path.c_str());
}

namespace {

struct SynchronousCall {
  SynchronousCall(iree_runtime_session_t* session) : session(session) {}
  ~SynchronousCall() {
    if (initialized) {
      iree_runtime_call_deinitialize(&call);
    }
  }

  iree_status_t InitializeByName(const char* entrypoint_name) {
    IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
        session, iree_make_cstring_view(entrypoint_name), &call));
    initialized = true;
    return iree_ok_status();
  }

  iree_runtime_session_t* session;
  iree_runtime_call_t call;
  bool initialized = false;
};

struct HalBufferView {
  ~HalBufferView() {
    if (bv) {
      iree_hal_buffer_view_release(bv);
    }
  }
  iree_hal_buffer_view_t* bv = nullptr;
};

iree_hal_element_type_t ConvertOrtElementType(ONNXTensorElementDataType et) {
  switch (et) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return IREE_HAL_ELEMENT_TYPE_UINT_8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return IREE_HAL_ELEMENT_TYPE_SINT_8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return IREE_HAL_ELEMENT_TYPE_UINT_16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return IREE_HAL_ELEMENT_TYPE_SINT_16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return IREE_HAL_ELEMENT_TYPE_SINT_32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return IREE_HAL_ELEMENT_TYPE_SINT_64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return IREE_HAL_ELEMENT_TYPE_BOOL_8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return IREE_HAL_ELEMENT_TYPE_UINT_32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return IREE_HAL_ELEMENT_TYPE_UINT_64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
      // TODO: FP8 types.
    default:
      return IREE_HAL_ELEMENT_TYPE_NONE;
  }
}

}  // namespace

common::Status Session::Call(const char* entrypoint_name, const OrtApi* ort_api, OrtKernelContext* ort_context_c) {
  // TODO: This is far from the most efficient way to make a call. Synchronous and copying. We can do
  // better but this gets points for simplicity and lets us bootstrap the tests.
  iree_vm_list_t* inputs = NULL;
  iree_allocator_t host_allocator = iree_allocator_system();
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                    host_allocator, &inputs));
  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                    host_allocator, &outputs));
  Ort::KernelContext context(ort_context_c);
  SynchronousCall call(session);
  ORT_RETURN_IF_ERROR(HandleIREEStatus(call.InitializeByName(entrypoint_name)));

  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  // iree_allocator_t host_allocator =
  //     iree_runtime_session_host_allocator(session);

  std::vector<iree_hal_dim_t> dims;

  // Process inputs. We could be smarter about this in a lot of ways, including carrying
  // more state from compilation so we are doing less munging here.

  std::cout << "input count: " << context.GetInputCount() << "\n";
  // for (size_t i = 0; i < context.GetInputCount(); ++i) {
  auto input_tensor = context.GetInput(0);
  ORT_ENFORCE(input_tensor.IsTensor());

  // The device type is rather... sparse... CPU, GPU and FPGA. Not sure how that
  // is useful for anything.
  auto ort_device_type = input_tensor.GetTensorMemoryInfo().GetDeviceType();
  ORT_ENFORCE(ort_device_type == OrtMemoryInfoDeviceType_CPU);

  const auto& tensor_type = input_tensor.GetTensorTypeAndShapeInfo();
  auto element_type = ConvertOrtElementType(tensor_type.GetElementType());
  ORT_ENFORCE(element_type != IREE_HAL_ELEMENT_TYPE_NONE, "Unsupported element type ",
              static_cast<int>(tensor_type.GetElementType()));
  ORT_ENFORCE(iree_hal_element_is_byte_aligned(element_type));
  size_t element_size_bytes = iree_hal_element_dense_byte_count(element_type);

  // Yes, that's right, returned as an std::vector by value :(
  // And of a different type than we expect.
  std::vector<int64_t> shape = tensor_type.GetShape();
  dims.resize(shape.size());
  std::copy(shape.begin(), shape.end(), dims.begin());

  // No convenient way to get the byte size of the raw data.
  size_t element_count = tensor_type.GetElementCount();
  const void* raw_data = input_tensor.GetTensorRawData();

  HalBufferView arg;
  iree_hal_buffer_params_t buffer_params;
  memset(&buffer_params, 0, sizeof(buffer_params));
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  ORT_RETURN_IF_ERROR(HandleIREEStatus(iree_hal_buffer_view_allocate_buffer_copy(
      device, device_allocator,
      // Shape rank and dimensions:
      dims.size(), dims.data(),
      // Element type:
      element_type,
      // Encoding type:
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      buffer_params,
      // The actual heap buffer to wrap or clone and its allocator:
      iree_make_const_byte_span(raw_data, element_count * element_size_bytes),
      // Buffer view + storage are returned and owned by the caller:
      &arg.bv)));

  iree_vm_ref_t input_view_ref = iree_hal_buffer_view_move_ref(arg.bv);
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &input_view_ref));

  iree_hal_semaphore_t* semaphore = NULL;
  IREE_CHECK_OK(iree_hal_semaphore_create(
      device, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  iree_hal_fence_t* fence_t1 = NULL;
  IREE_CHECK_OK(
      iree_hal_fence_create_at(semaphore, 1ull, host_allocator, &fence_t1));
  iree_hal_fence_t* fence_t2 = NULL;
  IREE_CHECK_OK(
      iree_hal_fence_create_at(semaphore, 2ull, host_allocator, &fence_t2));
  iree_hal_semaphore_release(semaphore);
  std::cout << "\n semaphore released";
  iree_vm_ref_t fence_t1_ref = iree_hal_fence_retain_ref(fence_t1);
  std::cout << "\n semaphore released1";
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &fence_t1_ref));
  std::cout << "\n semaphore released2";
  iree_vm_ref_t fence_t2_ref = iree_hal_fence_retain_ref(fence_t2);
  std::cout << "\n semaphore released3";
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &fence_t2_ref));
  std::cout << "\n semaphore released4";
  IREE_CHECK_OK(iree_hal_fence_signal(fence_t1));
  std::cout << "\n T=1 reached";
  // Add it to the call.
  iree_string_view_t entry_point = iree_make_cstring_view(entrypoint_name);
  IREE_CHECK_OK(
      iree_runtime_session_call_by_name(session, entry_point, inputs, outputs));
  // We could go do other things now while the async work progresses. Here we
  // just immediately wait.
  IREE_CHECK_OK(iree_hal_fence_wait(fence_t2, iree_infinite_timeout()));
  std::cout << "\n T=2 reached";
  // iree_status_t status = iree_runtime_call_inputs_push_back_buffer_view(&call.call, arg.bv);
  // ORT_RETURN_IF_ERROR(HandleIREEStatus(status));
  // }
  // Read back the tensor<?xi32> result:

  // Invoke.
  // ORT_RETURN_IF_ERROR(HandleIREEStatus(iree_runtime_call_invoke(&call.call, [>flags=<]0)));

  // Marshal the outputs.
  // TODO: Accessing the ORT output requires the shape and then we could get zero copy
  // access to an ORT managed buffer of some kind that can be used for zero copy. More
  // scaffolding is needed to exploit that, and we need to switch to the async calling
  // convention, which allows passing in slabs of result buffers. Further, that would
  // run the host-side computation (which would compute output metadata) inline.
  // For static cases, we could also side-load the shape from the compile time.
  // std::vector<int64_t> shape;
  std::cout << "output count: " << context.GetOutputCount() << "\n";
  // for (size_t i = 0; i < context.GetOutputCount(); ++i) {
  HalBufferView ret;
  ret.bv = iree_vm_list_get_buffer_view_assign(outputs, 0);
  // ORT_RETURN_IF_ERROR(HandleIREEStatus(
  //     iree_runtime_call_outputs_pop_front_buffer_view(&call.call, &ret.bv)));
  size_t ret_rank = iree_hal_buffer_view_shape_rank(ret.bv);
  const iree_hal_dim_t* ret_dims = iree_hal_buffer_view_shape_dims(ret.bv);
  shape.clear();
  shape.resize(ret_rank);
  std::copy(ret_dims, ret_dims + ret_rank, shape.begin());
  auto output_tensor = context.GetOutput(0, shape.data(), shape.size());
  ORT_ENFORCE(output_tensor.IsTensor());

  iree_hal_buffer_t* ret_buffer = iree_hal_buffer_view_buffer(ret.bv);
  // TODO: Synchronous mapping read, like everything in this function, is not a
  // great idea. It isn't supported on all device types and will need a scrub.
  iree_string_view_t device_val = iree_hal_device_id(device);
  auto device_str = std::string(device_val.data, device_val.size);
  if (device_str == "hip") {
    ORT_RETURN_IF_ERROR(HandleIREEStatus(iree_hal_device_transfer_d2h(
        iree_runtime_session_device(session),
        ret_buffer, 0, output_tensor.GetTensorMutableRawData(),
        iree_hal_buffer_view_byte_length(ret.bv), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout())));
    return common::Status::OK();
  }
    ORT_RETURN_IF_ERROR(HandleIREEStatus(iree_hal_buffer_map_read(ret_buffer, /*source_offset=*/0,
                                                                  output_tensor.GetTensorMutableRawData(),
                                                                  iree_hal_buffer_view_byte_length(ret.bv))));
    // }

    iree_vm_list_release(inputs);
    iree_vm_list_release(outputs);
    iree_hal_fence_release(fence_t1);
    iree_hal_fence_release(fence_t2);
    return common::Status::OK();
}

}  // namespace onnxruntime::iree_ep_rt
