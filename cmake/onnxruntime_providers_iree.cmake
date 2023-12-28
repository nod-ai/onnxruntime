# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# In a minimal build, we do not include the compiler.
# TODO: Parameterize this to exclude.
file(GLOB onnxruntime_providers_iree_jit_compiler_srcs
  "${ONNXRUNTIME_ROOT}/core/providers/iree/compiler/torch-mlir-import-onnx/*.cpp"
  "${ONNXRUNTIME_ROOT}/core/providers/iree/compiler/torch-mlir-import-onnx/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/iree/compiler/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/iree/compiler/*.h"
)

file(GLOB onnxruntime_providers_iree_cc_srcs
  "${ONNXRUNTIME_ROOT}/core/providers/iree/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/iree/*.h"
  ${onnxruntime_providers_iree_jit_compiler_srcs}
)

source_group(TREE "${ONNXRUNTIME_ROOT}/core" FILES ${onnxruntime_providers_iree_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_iree ${onnxruntime_providers_iree_cc_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_iree
  onnxruntime_common
  onnxruntime_framework onnx onnx_proto
  # Seems to be a dependency of ONNX itself but not transitively propagated.
  flatbuffers::flatbuffers Boost::mp11
)
target_link_libraries(onnxruntime_providers_iree PRIVATE
  onnx
  protobuf::libprotobuf
)

target_compile_options(
  onnxruntime_providers_iree PRIVATE
  # IREE's runtime headers are written in the C style and need some tweaks.
  -Wno-missing-field-initializers
  # We're not going to fix this.
  -Wno-unused-function
)

# Torch-mlir warnings disable warnings on external sources.
# TODO: Fix these at the source.
set_source_files_properties(
  ${onnxruntime_providers_iree_jit_compiler_srcs}
  PROPERTIES
    COMPILE_FLAGS
      "-Wno-unused-parameter -Wno-shorten-64-to-32"
)

# TODO: Dev package support in IREE is new so we are being a bit pedantic about config.
# This could be relaxes significantly.
set(ONNXRUNTIME_IREE_HOME "" CACHE STRING "Path to the IREE development package")
if(NOT ONNXRUNTIME_IREE_HOME)
  message(SEND_ERROR "Expected -DONNXRUNTIME_IREE_HOME to be set to the development package directory (see README)")
endif()
set(IREECompiler_DIR "${ONNXRUNTIME_IREE_HOME}/lib/cmake/IREE")
set(IREERuntime_DIR "${ONNXRUNTIME_IREE_HOME}/lib/cmake/IREE")

if(NOT EXISTS "${IREECompiler_DIR}/IREECompilerConfig.cmake")
  message(WARNING "Did not find IREECompilerConfig.cmake under ${IREECompiler_DIR}. Check -DONNXRUNTIME_IREE_HOME.")
endif()
if(NOT EXISTS "${IREERuntime_DIR}/IREERuntimeConfig.cmake")
  message(WARNING "Did not find IREERuntimeConfig.cmake under ${IREECompiler_DIR}. Check -DONNXRUNTIME_IREE_HOME.")
endif()

find_package(IREECompiler REQUIRED)
find_package(IREERuntime REQUIRED)

# Compiler deps. See README.md for temporary steps on how to make a dev package.
# TODO: Depend on real dev packages for iree-compiler and iree-runtime.
# TODO: Also figure out why the shared library libIREECompiler.so dep needs to be public.
# TODO: Once the dev package CMake integration is set up, depend via normal CMake libraries.
# TODO: Copy the IREECompiler runtime library to the appropriate place as part of installation.
# target_include_directories(
#   onnxruntime_providers_iree PRIVATE
#   ${ONNXRUNTIME_IREE_HOME}/include
# )
# target_link_directories(onnxruntime_providers_iree PUBLIC
#   ${ONNXRUNTIME_IREE_HOME}/lib
# )
target_link_libraries(onnxruntime_providers_iree PUBLIC
  iree_compiler_API_SharedImpl
  iree_runtime_unified
)

set_target_properties(onnxruntime_providers_iree PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_providers_iree PROPERTIES LINKER_LANGUAGE CXX)

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_iree
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
