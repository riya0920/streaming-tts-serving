cmake_minimum_required(VERSION 3.18)
project(triton_tts_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Triton backend SDK sources, cloned by scripts/provision.sh at the tag matching the
# running server. The ABI is not stable across releases, so a backend built against the
# wrong tag loads and then misbehaves rather than failing cleanly — pin it.
set(TRITON_SRC_DIR "/opt/tts/tools/triton-src" CACHE PATH "Triton backend/core/common sources")
foreach(repo backend core common)
  if(NOT EXISTS "${TRITON_SRC_DIR}/${repo}/include")
    message(FATAL_ERROR
      "missing ${TRITON_SRC_DIR}/${repo}/include — run scripts/provision.sh, or pass "
      "-DTRITON_SRC_DIR=<path>")
  endif()
endforeach()

find_package(CUDAToolkit REQUIRED)

# TensorRT ships in the NGC image without a CMake config package, so locate it directly.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT not found (NvInfer.h / libnvinfer)")
endif()
message(STATUS "TensorRT: ${TENSORRT_LIBRARY}")

# The Triton backend SDK ships headers AND implementations. Including only the headers
# links fine but fails at load time with
#   undefined symbol: _ZTIN6triton7backend20BackendModelInstanceE
# (typeinfo for BackendModelInstance) — because the vtables live in these .cc files.
# They are compiled into the backend rather than linked from a prebuilt library, since
# the image ships no such library.
# Minimal set on purpose. backend_input_collector.cc and backend_output_responder.cc
# reference CUDA kernels (RunGatherKernel) that live in the SDK's kernel.cu, which would
# drag the CUDA language and a separate compile into this build. This backend reads its
# input buffer and writes its output buffers through the raw C API, so neither helper is
# needed — including them only imports their unresolved kernel symbols.
set(TRITON_BACKEND_SRCS
  ${TRITON_SRC_DIR}/backend/src/backend_common.cc
  ${TRITON_SRC_DIR}/backend/src/backend_memory.cc
  ${TRITON_SRC_DIR}/backend/src/backend_model.cc
  ${TRITON_SRC_DIR}/backend/src/backend_model_instance.cc
)
foreach(src ${TRITON_BACKEND_SRCS})
  if(NOT EXISTS ${src})
    message(FATAL_ERROR "missing backend SDK source: ${src}")
  endif()
endforeach()

add_library(triton_tts_stream SHARED src/tts_stream.cc ${TRITON_BACKEND_SRCS})

target_include_directories(triton_tts_stream PRIVATE
  ${TRITON_SRC_DIR}/backend/include
  ${TRITON_SRC_DIR}/core/include
  ${TRITON_SRC_DIR}/common/include
  ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(triton_tts_stream PRIVATE
  ${TENSORRT_LIBRARY}
  CUDA::cudart
)

# Without this, Triton's headers fall back to GPU-free stand-ins — cudaStream_t becomes
# void* — and any real CUDA call against them fails to compile with a confusing
# void*-to-CUstream_st* conversion error.
target_compile_definitions(triton_tts_stream PRIVATE TRITON_ENABLE_GPU=1)

# Triton resolves TRITONBACKEND_* and TRITONSERVER_* from the server process at load
# time, so they are deliberately left undefined here.
target_link_options(triton_tts_stream PRIVATE "LINKER:--allow-shlib-undefined")

# Triton looks for lib<backend-name>.so, so the output must be libtriton_tts_stream.so
# for a backend named tts_stream.
set_target_properties(triton_tts_stream PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME "triton_tts_stream"
)

target_compile_options(triton_tts_stream PRIVATE -Wall -Wextra -O3)
