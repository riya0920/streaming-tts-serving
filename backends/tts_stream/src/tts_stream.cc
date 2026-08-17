// tts_stream — a Triton decoupled backend that turns VITS latents into a stream of
// audio chunks.
//
// This is the hot loop. It runs many times per second per session, across every
// session, and it is in C++ for a reason that is about variance rather than mean: a
// Python loop that costs 0.4 ms on average but occasionally 9 ms — because a GC pass or
// GIL contention landed badly — is fine for the p50 and fatal for the p99. The p99 is
// the number this project exists to defend.
//
// What it does per request:
//   1. take latents [1, C, T] produced by vits_frontend
//   2. plan chunks using M3's progressive sizing (small first chunk for TTFA, growing)
//   3. for each chunk, decode [a-P, b+P) through the TensorRT engine, where P is the
//      measured 13-frame receptive field
//   4. trim the contaminated context back off, convert float32 to 16-bit PCM
//   5. send that chunk as its own response, immediately, while later chunks decode
//
// Decoupled mode is what makes step 5 possible: one request, many responses over time.
// Without it, streaming would have to be bolted on top of a request/response server.
//
// No crossfade. M3 measured chunk seams against a single-pass decode at step-ratio
// 10.25 vs 10.26 — indistinguishable — once overlap is at or above the receptive field.
// The blend that the original design treated as essential turned out to be solving a
// problem the overlap had already solved.

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace tts_stream {

// Prefixed rather than named RETURN_IF_CUDA_ERROR: Triton's backend_common.h already
// defines a three-argument macro by that name, and the redefinition produces errors
// pointing at the CUDA call rather than at the collision.
#define TTS_CUDA_CHECK(X, MSG)                                                  \
  do {                                                                          \
    cudaError_t err__ = (X);                                                    \
    if (err__ != cudaSuccess) {                                                 \
      return TRITONSERVER_ErrorNew(                                             \
          TRITONSERVER_ERROR_INTERNAL,                                          \
          (std::string(MSG) + ": " + cudaGetErrorString(err__)).c_str());       \
    }                                                                           \
  } while (false)

// TensorRT logs through its own interface; forward it into Triton's log so engine
// problems show up in the same place as everything else.
class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("[TRT] ") + msg).c_str());
    } else if (severity == Severity::kWARNING) {
      LOG_MESSAGE(TRITONSERVER_LOG_WARN, (std::string("[TRT] ") + msg).c_str());
    }
  }
};
static TrtLogger g_trt_logger;

//==============================================================================
// ModelState — per-model config and the shared TensorRT engine.
//==============================================================================
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* m, ModelState** state);
  virtual ~ModelState() = default;

  nvinfer1::ICudaEngine* Engine() const { return engine_.get(); }

  int64_t FirstChunkFrames() const { return first_chunk_frames_; }
  int64_t MaxChunkFrames() const { return max_chunk_frames_; }
  double ChunkGrowth() const { return chunk_growth_; }
  int64_t OverlapFrames() const { return overlap_frames_; }
  int64_t Hop() const { return hop_; }
  int64_t Channels() const { return channels_; }
  const std::string& InputTensor() const { return input_tensor_; }
  const std::string& OutputTensor() const { return output_tensor_; }

 private:
  explicit ModelState(TRITONBACKEND_Model* m) : BackendModel(m) {}
  TRITONSERVER_Error* LoadEngine();
  TRITONSERVER_Error* ReadConfig();

  std::string engine_path_;
  std::string input_tensor_{"latents"};
  std::string output_tensor_{"waveform"};

  // Defaults come from measurement, not taste — see docs/PROFILE.md and
  // results/m3_sweep.json. overlap_frames in particular is the decoder's measured
  // receptive field; below it, seams click.
  int64_t first_chunk_frames_{12};
  int64_t max_chunk_frames_{50};
  double chunk_growth_{2.0};
  int64_t overlap_frames_{13};
  int64_t hop_{256};
  int64_t channels_{192};

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* m, ModelState** state)
{
  std::unique_ptr<ModelState> s(new ModelState(m));
  RETURN_IF_ERROR(s->ReadConfig());
  RETURN_IF_ERROR(s->LoadEngine());
  *state = s.release();
  return nullptr;
}

TRITONSERVER_Error*
ModelState::ReadConfig()
{
  common::TritonJson::Value params;
  if (!ModelConfig().Find("parameters", &params)) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                 "tts_stream requires a 'parameters' section");
  }

  auto get_str = [&](const char* key, std::string* out) -> bool {
    common::TritonJson::Value v;
    if (!params.Find(key, &v)) return false;
    std::string s;
    if (v.MemberAsString("string_value", &s) != nullptr) return false;
    *out = s;
    return true;
  };
  auto get_i64 = [&](const char* key, int64_t* out) {
    std::string s;
    if (get_str(key, &s) && !s.empty()) *out = std::stoll(s);
  };
  auto get_dbl = [&](const char* key, double* out) {
    std::string s;
    if (get_str(key, &s) && !s.empty()) *out = std::stod(s);
  };

  if (!get_str("engine_path", &engine_path_)) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                 "tts_stream requires parameter 'engine_path'");
  }
  get_str("input_tensor", &input_tensor_);
  get_str("output_tensor", &output_tensor_);
  get_i64("first_chunk_frames", &first_chunk_frames_);
  get_i64("max_chunk_frames", &max_chunk_frames_);
  get_i64("overlap_frames", &overlap_frames_);
  get_i64("hop", &hop_);
  get_i64("channels", &channels_);
  get_dbl("chunk_growth", &chunk_growth_);

  if (first_chunk_frames_ < 1 || max_chunk_frames_ < first_chunk_frames_) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                 "require 1 <= first_chunk_frames <= max_chunk_frames");
  }
  if (overlap_frames_ < 0) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                 "overlap_frames must be >= 0");
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::LoadEngine()
{
  std::ifstream f(engine_path_, std::ios::binary | std::ios::ate);
  if (!f) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        (std::string("cannot open engine: ") + engine_path_).c_str());
  }
  const std::streamsize size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> blob(static_cast<size_t>(size));
  if (!f.read(blob.data(), size)) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "engine read failed");
  }

  runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
  if (!runtime_) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "createInferRuntime failed");
  }
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) {
    // Overwhelmingly the cause here is a TensorRT version mismatch between the machine
    // that built the plan and this one. Plans are not portable across TRT versions or
    // GPU architectures.
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "deserializeCudaEngine failed — was this plan built with this TensorRT version "
        "and for this GPU architecture?");
  }

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("tts_stream loaded engine ") + engine_path_ +
               " (first=" + std::to_string(first_chunk_frames_) +
               " max=" + std::to_string(max_chunk_frames_) +
               " overlap=" + std::to_string(overlap_frames_) +
               " hop=" + std::to_string(hop_) + ")")
                  .c_str());
  return nullptr;
}

//==============================================================================
// ModelInstanceState — per-instance execution context, stream, and buffers.
//
// Buffers are allocated once at the maximum chunk shape and reused for every chunk of
// every request. Allocating per chunk would put a cudaMalloc — which synchronizes — in
// the middle of the hot loop, and the resulting jitter lands directly on the p99.
//==============================================================================
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(ModelState* ms, TRITONBACKEND_ModelInstance* mi,
                                    ModelInstanceState** state);
  ~ModelInstanceState();

  void ProcessRequests(TRITONBACKEND_Request** requests, uint32_t count);

 private:
  ModelInstanceState(ModelState* ms, TRITONBACKEND_ModelInstance* mi)
      : BackendModelInstance(ms, mi), model_state_(ms) {}
  TRITONSERVER_Error* Init();

  // Decode one window of latents and return the trimmed, PCM-converted chunk body.
  TRITONSERVER_Error* DecodeWindow(const float* latents_host, int64_t total_frames,
                                   int64_t lo, int64_t hi, int64_t keep_from,
                                   int64_t keep_frames, std::vector<int16_t>* pcm);

  ModelState* model_state_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  // Explicitly the global CUDA type. Triton's backend_common.h declares its own
  // `cudaStream_t` as void* when TRITON_ENABLE_GPU is unset, and since this class lives
  // in namespace triton::backend, an unqualified name binds to that one instead — which
  // compiles as far as the declaration and then fails at the first CUDA call.
  ::cudaStream_t stream_{nullptr};

  float* d_in_{nullptr};
  float* d_out_{nullptr};
  std::vector<float> h_out_;
  int64_t max_window_frames_{0};
};

TRITONSERVER_Error*
ModelInstanceState::Create(ModelState* ms, TRITONBACKEND_ModelInstance* mi,
                           ModelInstanceState** state)
{
  std::unique_ptr<ModelInstanceState> s(new ModelInstanceState(ms, mi));
  RETURN_IF_ERROR(s->Init());
  *state = s.release();
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::Init()
{
  ctx_.reset(model_state_->Engine()->createExecutionContext());
  if (!ctx_) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                 "createExecutionContext failed");
  }
  TTS_CUDA_CHECK(cudaSetDevice(DeviceId()), "cudaSetDevice");
  TTS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                       "cudaStreamCreate");

  max_window_frames_ =
      model_state_->MaxChunkFrames() + 2 * model_state_->OverlapFrames();
  const size_t in_elems =
      static_cast<size_t>(model_state_->Channels()) * max_window_frames_;
  const size_t out_elems =
      static_cast<size_t>(max_window_frames_) * model_state_->Hop();

  TTS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in_), in_elems * sizeof(float)), "cudaMalloc in");
  TTS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_), out_elems * sizeof(float)), "cudaMalloc out");
  h_out_.resize(out_elems);

  // Warm every window shape the chunk planner can produce. M3 measured the first decode
  // of a new tensor shape at ~66 ms against ~4.9 ms steady-state, because TensorRT and
  // cuDNN pick and cache an algorithm per shape. Paying that on a live request would put
  // a 60 ms spike straight into the p99 of whichever unlucky session hit it first.
  std::vector<float> zeros(in_elems, 0.0f);
  TTS_CUDA_CHECK(
      cudaMemcpyAsync(d_in_, zeros.data(), in_elems * sizeof(float),
                      cudaMemcpyHostToDevice, stream_), "warm H2D");
  int64_t size = model_state_->FirstChunkFrames();
  std::vector<int64_t> shapes;
  while (true) {
    shapes.push_back(size + 2 * model_state_->OverlapFrames());
    if (size >= model_state_->MaxChunkFrames()) break;
    size = std::min<int64_t>(model_state_->MaxChunkFrames(),
                             std::max<int64_t>(size + 1,
                                               static_cast<int64_t>(size * model_state_->ChunkGrowth())));
  }
  for (int64_t frames : shapes) {
    nvinfer1::Dims3 dims(1, static_cast<int32_t>(model_state_->Channels()),
                         static_cast<int32_t>(frames));
    if (!ctx_->setInputShape(model_state_->InputTensor().c_str(), dims)) continue;
    ctx_->setTensorAddress(model_state_->InputTensor().c_str(), d_in_);
    ctx_->setTensorAddress(model_state_->OutputTensor().c_str(), d_out_);
    ctx_->enqueueV3(stream_);
  }
  TTS_CUDA_CHECK(cudaStreamSynchronize(stream_), "warmup sync");

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("tts_stream instance ready on device ") +
               std::to_string(DeviceId()) + ", warmed " +
               std::to_string(shapes.size()) + " shapes")
                  .c_str());
  return nullptr;
}

ModelInstanceState::~ModelInstanceState()
{
  if (d_in_) cudaFree(d_in_);
  if (d_out_) cudaFree(d_out_);
  if (stream_) cudaStreamDestroy(stream_);
}

TRITONSERVER_Error*
ModelInstanceState::DecodeWindow(const float* latents_host, int64_t total_frames,
                                 int64_t lo, int64_t hi, int64_t keep_from,
                                 int64_t keep_frames, std::vector<int16_t>* pcm)
{
  const int64_t C = model_state_->Channels();
  const int64_t hop = model_state_->Hop();
  const int64_t window = hi - lo;

  // Latents arrive as [1, C, T] contiguous, so a frame slice is strided: each channel
  // is a separate contiguous run of T floats. Copy channel by channel.
  for (int64_t c = 0; c < C; ++c) {
    TTS_CUDA_CHECK(
        cudaMemcpyAsync(d_in_ + c * window,
                        latents_host + c * total_frames + lo,
                        window * sizeof(float), cudaMemcpyHostToDevice, stream_),
        "H2D latents");
  }

  nvinfer1::Dims3 dims(1, static_cast<int32_t>(C), static_cast<int32_t>(window));
  if (!ctx_->setInputShape(model_state_->InputTensor().c_str(), dims)) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                 "setInputShape failed — window outside engine profile");
  }
  ctx_->setTensorAddress(model_state_->InputTensor().c_str(), d_in_);
  ctx_->setTensorAddress(model_state_->OutputTensor().c_str(), d_out_);
  if (!ctx_->enqueueV3(stream_)) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "enqueueV3 failed");
  }

  const int64_t out_samples = window * hop;
  TTS_CUDA_CHECK(
      cudaMemcpyAsync(h_out_.data(), d_out_, out_samples * sizeof(float),
                      cudaMemcpyDeviceToHost, stream_), "D2H waveform");
  TTS_CUDA_CHECK(cudaStreamSynchronize(stream_), "stream sync");

  // Drop the context we decoded only to give the convolutions their receptive field,
  // then convert what remains to 16-bit PCM.
  const int64_t start = (keep_from - lo) * hop;
  const int64_t count = keep_frames * hop;
  pcm->resize(static_cast<size_t>(count));
  const float* src = h_out_.data() + start;
  for (int64_t i = 0; i < count; ++i) {
    float v = src[i];
    v = std::max(-1.0f, std::min(1.0f, v));
    (*pcm)[static_cast<size_t>(i)] = static_cast<int16_t>(v * 32767.0f);
  }
  return nullptr;
}

void
ModelInstanceState::ProcessRequests(TRITONBACKEND_Request** requests, uint32_t count)
{
  for (uint32_t r = 0; r < count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // In decoupled mode responses come from a factory, not from the request, because
    // the request is released long before the last chunk is sent.
    TRITONBACKEND_ResponseFactory* factory = nullptr;
    if (TRITONBACKEND_ResponseFactoryNew(&factory, request) != nullptr) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "ResponseFactoryNew failed");
      TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
      continue;
    }

    auto fail = [&](const std::string& msg) {
      TRITONBACKEND_Response* resp = nullptr;
      if (TRITONBACKEND_ResponseNewFromFactory(&resp, factory) == nullptr) {
        TRITONSERVER_Error* err =
            TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, msg.c_str());
        TRITONBACKEND_ResponseSend(resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
        TRITONSERVER_ErrorDelete(err);
      }
      TRITONBACKEND_ResponseFactoryDelete(factory);
      TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
    };

    TRITONBACKEND_Input* input = nullptr;
    if (TRITONBACKEND_RequestInput(request, "LATENTS", &input) != nullptr) {
      fail("missing input LATENTS");
      continue;
    }

    const int64_t* shape = nullptr;
    uint32_t dims_count = 0;
    TRITONSERVER_DataType dtype;
    uint64_t byte_size = 0;
    uint32_t buffer_count = 0;
    if (TRITONBACKEND_InputProperties(input, nullptr, &dtype, &shape, &dims_count,
                                      &byte_size, &buffer_count) != nullptr) {
      fail("InputProperties failed");
      continue;
    }
    if (dtype != TRITONSERVER_TYPE_FP32 || dims_count != 3) {
      fail("LATENTS must be FP32 with shape [1, C, T]");
      continue;
    }
    const int64_t total_frames = shape[2];
    if (shape[1] != model_state_->Channels() || total_frames < 1) {
      fail("LATENTS channel count does not match the configured engine");
      continue;
    }

    const void* buffer = nullptr;
    uint64_t buf_bytes = 0;
    TRITONSERVER_MemoryType mem_type = TRITONSERVER_MEMORY_CPU;
    int64_t mem_id = 0;
    if (TRITONBACKEND_InputBuffer(input, 0, &buffer, &buf_bytes, &mem_type, &mem_id)
        != nullptr) {
      fail("InputBuffer failed");
      continue;
    }
    const float* latents = static_cast<const float*>(buffer);

    // ---- plan chunks: progressive sizing, per M3 -------------------------------
    // Only the first chunk sets time-to-first-audio. Later chunks grow so the overlap
    // tax (68% at a 12-frame chunk, 34% at 50) is amortized while the listener is
    // already hearing audio.
    std::vector<std::pair<int64_t, int64_t>> chunks;
    {
      int64_t a = 0, size = model_state_->FirstChunkFrames();
      while (a < total_frames) {
        const int64_t b = std::min(a + size, total_frames);
        chunks.emplace_back(a, b);
        a = b;
        size = std::min<int64_t>(model_state_->MaxChunkFrames(),
                                 std::max<int64_t>(size + 1,
                                     static_cast<int64_t>(size * model_state_->ChunkGrowth())));
      }
    }

    const int64_t P = model_state_->OverlapFrames();
    bool failed = false;
    std::vector<int16_t> pcm;

    for (size_t i = 0; i < chunks.size(); ++i) {
      const int64_t a = chunks[i].first;
      const int64_t b = chunks[i].second;
      const int64_t lo = std::max<int64_t>(0, a - P);
      const int64_t hi = std::min<int64_t>(total_frames, b + P);
      const bool is_final = (i + 1 == chunks.size());

      TRITONSERVER_Error* err =
          DecodeWindow(latents, total_frames, lo, hi, a, b - a, &pcm);
      if (err != nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        fail("decode failed mid-stream");
        failed = true;
        break;
      }

      TRITONBACKEND_Response* response = nullptr;
      if (TRITONBACKEND_ResponseNewFromFactory(&response, factory) != nullptr) {
        fail("ResponseNewFromFactory failed");
        failed = true;
        break;
      }

      const int64_t out_shape[2] = {1, static_cast<int64_t>(pcm.size())};
      TRITONBACKEND_Output* output = nullptr;
      if (TRITONBACKEND_ResponseOutput(response, &output, "AUDIO_CHUNK",
                                       TRITONSERVER_TYPE_INT16, out_shape, 2) != nullptr) {
        fail("ResponseOutput failed");
        failed = true;
        break;
      }
      void* out_buffer = nullptr;
      TRITONSERVER_MemoryType out_mem = TRITONSERVER_MEMORY_CPU;
      int64_t out_mem_id = 0;
      const uint64_t out_bytes = pcm.size() * sizeof(int16_t);
      if (TRITONBACKEND_OutputBuffer(output, &out_buffer, out_bytes, &out_mem,
                                     &out_mem_id) != nullptr) {
        fail("OutputBuffer failed");
        failed = true;
        break;
      }
      std::memcpy(out_buffer, pcm.data(), out_bytes);

      // Index and finality travel with the chunk so the gateway can order and terminate
      // the stream without inferring either from timing.
      const int64_t scalar_shape[2] = {1, 1};
      TRITONBACKEND_Output* idx_out = nullptr;
      if (TRITONBACKEND_ResponseOutput(response, &idx_out, "CHUNK_INDEX",
                                       TRITONSERVER_TYPE_INT32, scalar_shape, 2)
          == nullptr) {
        void* p = nullptr;
        TRITONSERVER_MemoryType mt = TRITONSERVER_MEMORY_CPU;
        int64_t mi = 0;
        if (TRITONBACKEND_OutputBuffer(idx_out, &p, sizeof(int32_t), &mt, &mi) == nullptr) {
          const int32_t v = static_cast<int32_t>(i);
          std::memcpy(p, &v, sizeof(v));
        }
      }
      TRITONBACKEND_Output* fin_out = nullptr;
      if (TRITONBACKEND_ResponseOutput(response, &fin_out, "IS_FINAL",
                                       TRITONSERVER_TYPE_BOOL, scalar_shape, 2)
          == nullptr) {
        void* p = nullptr;
        TRITONSERVER_MemoryType mt = TRITONSERVER_MEMORY_CPU;
        int64_t mi = 0;
        if (TRITONBACKEND_OutputBuffer(fin_out, &p, sizeof(bool), &mt, &mi) == nullptr) {
          const bool v = is_final;
          std::memcpy(p, &v, sizeof(v));
        }
      }

      // Send now. The whole point is that the listener hears chunk 0 while chunk 1 is
      // still decoding — holding responses until the end would make this an ordinary
      // request/response model with extra steps.
      const uint32_t flags = is_final ? TRITONSERVER_RESPONSE_COMPLETE_FINAL : 0;
      TRITONBACKEND_ResponseSend(response, flags, nullptr);
    }

    if (!failed) {
      TRITONBACKEND_ResponseFactoryDelete(factory);
      TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
    }
  }
}

//==============================================================================
// Backend entry points
//==============================================================================
extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* name = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &name));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  uint32_t api_major = 0, api_minor = 0;
  RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_major, &api_minor));
  if ((api_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  ModelState* state = nullptr;
  RETURN_IF_ERROR(ModelState::Create(model, &state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(state)));
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  delete reinterpret_cast<ModelState*>(vstate);
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  TRITONBACKEND_Model* model = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));

  ModelInstanceState* istate = nullptr;
  RETURN_IF_ERROR(ModelInstanceState::Create(
      reinterpret_cast<ModelState*>(vstate), instance, &istate));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(istate)));
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  delete reinterpret_cast<ModelInstanceState*>(vstate);
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance,
                                   TRITONBACKEND_Request** requests, const uint32_t count)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  reinterpret_cast<ModelInstanceState*>(vstate)->ProcessRequests(requests, count);
  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::tts_stream
