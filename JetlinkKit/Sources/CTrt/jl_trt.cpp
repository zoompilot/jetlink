// jl_trt.h over TensorRT and the CUDA driver API, both opened at run time, so
// a build needs headers only and a machine without TensorRT still starts and
// can say why.
//
// From TensorRT only the extern "C" factories and version getters are looked
// up; every method called on what they return is an inline forwarder in the
// headers to a virtual inside the library, so nothing links against
// libnvinfer. CUDA is the driver API alone, whose handles are the runtime's
// too (a stream here is the cudaStream_t enqueueV3 takes), each entry point
// asked of cuGetProcAddress at the version its signature here matches: a shim
// built on CUDA 12 headers then gets the same functions from a CUDA 13 driver.
//
// One source for TensorRT 10 (Jetson, the 10.3 headers) and 11 (PCs). The
// libraries stay loaded for the life of the process: TensorRT's process-wide
// logger points into this one.
#include "jl_trt.h"
#include "jl_trt_util.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <dlfcn.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <mutex>

#define JL_STR2(x) #x
#define JL_STR(x) JL_STR2(x)

// Every driver entry point the shim uses: its name after "cu", the CUDA
// version whose signature this is (cudaTypedefs.h's PFN_cu<name>_v<version>),
// and the parameters. One table fills the struct and resolves it.
#define JL_DRIVER(X)                                                               \
  X(Init, 2000, (unsigned))                                                        \
  X(DriverGetVersion, 2020, (int *))                                               \
  X(DeviceGet, 2000, (CUdevice *, int))                                            \
  X(DeviceGetName, 2000, (char *, int, CUdevice))                                  \
  X(DeviceGetAttribute, 2000, (int *, CUdevice_attribute, CUdevice))               \
  X(DevicePrimaryCtxRetain, 7000, (CUcontext *, CUdevice))                         \
  X(DevicePrimaryCtxRelease, 11000, (CUdevice))                                    \
  X(CtxSetCurrent, 4000, (CUcontext))                                              \
  X(CtxSynchronize, 2000, ())                                                      \
  X(MemGetInfo, 3020, (size_t *, size_t *))                                        \
  X(MemAlloc, 3020, (CUdeviceptr *, size_t))                                       \
  X(MemFree, 3020, (CUdeviceptr))                                                  \
  X(MemHostAlloc, 2020, (void **, size_t, unsigned))                               \
  X(MemFreeHost, 2000, (void *))                                                   \
  X(MemcpyHtoDAsync, 3020, (CUdeviceptr, const void *, size_t, CUstream))          \
  X(MemcpyDtoHAsync, 3020, (void *, CUdeviceptr, size_t, CUstream))                \
  X(MemcpyDtoDAsync, 3020, (CUdeviceptr, CUdeviceptr, size_t, CUstream))           \
  X(MemsetD8Async, 3020, (CUdeviceptr, unsigned char, size_t, CUstream))           \
  X(StreamCreate, 2000, (CUstream *, unsigned))                                    \
  X(StreamSynchronize, 2000, (CUstream))                                           \
  X(StreamIsCapturing, 10000, (CUstream, CUstreamCaptureStatus *))                 \
  X(StreamDestroy, 4000, (CUstream))                                               \
  X(EventCreate, 2000, (CUevent *, unsigned))                                      \
  X(EventRecordWithFlags, 11010, (CUevent, CUstream, unsigned))                    \
  X(EventSynchronize, 2000, (CUevent))                                             \
  X(EventElapsedTime, 2000, (float *, CUevent, CUevent))                           \
  X(EventDestroy, 4000, (CUevent))                                                 \
  X(StreamBeginCapture, 10010, (CUstream, CUstreamCaptureMode))                    \
  X(StreamEndCapture, 10000, (CUstream, CUgraph *))                                \
  X(GraphInstantiateWithFlags, 11040, (CUgraphExec *, CUgraph, unsigned long long)) \
  X(GraphLaunch, 10000, (CUgraphExec, CUstream))                                   \
  X(GraphDestroy, 10000, (CUgraph))                                                \
  X(GraphExecDestroy, 10000, (CUgraphExec))                                        \
  X(GetErrorName, 6000, (CUresult, const char **))                                 \
  X(GetErrorString, 6000, (CUresult, const char **))

namespace {

struct Cuda {
#define JL_MEMBER(name, version, params) CUresult(*name) params;
  JL_DRIVER(JL_MEMBER)
#undef JL_MEMBER
};

// What TensorRT last logged as an error on this thread: the only reason it
// gives when a call returns NULL or false. Cleared before each such call.
thread_local char last_error[1024];

// Errors that leave the context unusable: every later call in it fails the
// same way, and only a new process gets a working one back (D15). Unknown is
// here too, since nothing says the context survived it.
bool is_sticky(CUresult r) {
  static const CUresult sticky[] = {
      CUDA_ERROR_ILLEGAL_ADDRESS,    CUDA_ERROR_LAUNCH_TIMEOUT,       CUDA_ERROR_HARDWARE_STACK_ERROR,
      CUDA_ERROR_ILLEGAL_INSTRUCTION, CUDA_ERROR_MISALIGNED_ADDRESS,  CUDA_ERROR_INVALID_ADDRESS_SPACE,
      CUDA_ERROR_INVALID_PC,         CUDA_ERROR_LAUNCH_FAILED,        CUDA_ERROR_ASSERT,
      CUDA_ERROR_ECC_UNCORRECTABLE,  CUDA_ERROR_NVLINK_UNCORRECTABLE, CUDA_ERROR_EXTERNAL_DEVICE,
      CUDA_ERROR_CONTEXT_IS_DESTROYED, CUDA_ERROR_DEVICE_UNAVAILABLE, CUDA_ERROR_DEINITIALIZED,
      CUDA_ERROR_UNKNOWN,
  };
  for (CUresult s : sticky) {
    if (r == s) {
      return true;
    }
  }
  return false;
}

// TensorRT keeps the first logger it is given for the whole process, so there
// is one, and jl_trt_set_logger changes where it forwards.
class Logger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    int level = static_cast<int>(severity);
    if (level <= JL_TRT_LOG_ERROR) {
      snprintf(last_error, sizeof last_error, "%s", msg);
    }
    std::lock_guard<std::mutex> hold(lock);
    if (fn != nullptr && level <= min) {
      fn(ctx, level, msg);
    }
  }

  std::mutex lock;
  jl_trt_log_fn fn = nullptr;
  void *ctx = nullptr;
  int min = JL_TRT_LOG_WARNING;
};

Logger logger;

// What TensorRT's factories take: the ILogger itself, as their inline
// wrappers pass it.
void *trt_logger() {
  return static_cast<nvinfer1::ILogger *>(&logger);
}

// IProgressMonitor's calls, forwarded as they come; TensorRT may make them
// from several threads, and Swift hears one at a time.
class Monitor final : public nvinfer1::IProgressMonitor {
public:
  void phaseStart(const char *phase, const char *parent, int32_t steps) noexcept override {
    emit(JL_TRT_PHASE_START, phase, parent, steps);
  }

  bool stepComplete(const char *phase, int32_t step) noexcept override {
    emit(JL_TRT_PHASE_STEP, phase, nullptr, step);
    // false ends the build (jl_trt_build_cancel)
    return !cancelled.load();
  }

  void phaseFinish(const char *phase) noexcept override {
    emit(JL_TRT_PHASE_FINISH, phase, nullptr, 0);
  }

  void emit(int event, const char *phase, const char *parent, int value) {
    std::lock_guard<std::mutex> hold(lock);
    if (fn != nullptr) {
      fn(ctx, event, phase, parent, value);
    }
  }

  std::mutex lock;
  jl_trt_progress_fn fn = nullptr;
  void *ctx = nullptr;
  std::atomic<bool> cancelled{false};
};

// The context each thread last made current, and for which open: a context
// released and retained again can come back at the same address.
struct Current {
  CUcontext ctx;
  uint64_t generation;
};
thread_local Current current = {nullptr, 0};
std::atomic<uint64_t> generations{0};

const char *or_empty(const char *s) {
  return s != nullptr ? s : "";
}

} // namespace

struct jl_trt {
  Cuda cu = {};
  CUdevice device = 0;
  CUcontext ctx = nullptr;
  uint64_t generation = 0;
  void *(*create_runtime)(void *, int32_t) = nullptr;
  void *(*create_builder)(void *, int32_t) = nullptr;
  void *(*create_parser)(void *, void *, int) = nullptr;
  std::atomic<int> sticky{0};
  std::mutex sticky_lock;
  char sticky_message[512] = {0};
  char device_name[256] = {0};
  char library[4096] = {0};
  jl_trt_info info = {};
};

struct jl_trt_engine {
  jl_trt *trt;
  nvinfer1::IRuntime *runtime;
  nvinfer1::ICudaEngine *engine;
};

struct jl_trt_context {
  jl_trt *trt;
  nvinfer1::IExecutionContext *context;
};

struct jl_trt_build {
  jl_trt *trt;
  nvinfer1::IBuilder *builder = nullptr;
  nvinfer1::INetworkDefinition *network = nullptr;
  nvonnxparser::IParser *parser = nullptr;
  nvinfer1::IBuilderConfig *config = nullptr;
  nvinfer1::ITimingCache *cache = nullptr;
  Monitor monitor;
};

namespace {

// The CUDA handles are ours with the types taken off.
CUstream cu(jl_trt_stream *s) {
  return reinterpret_cast<CUstream>(s);
}

CUevent cu(jl_trt_event *e) {
  return reinterpret_cast<CUevent>(e);
}

// The one CUDA error path: "<call>: <CUDA_ERROR_NAME>: <description>", and
// the latch when the error is sticky.
int cuda_fail(jl_trt *t, const char *call, CUresult r, char *err, size_t errlen) {
  const char *name = "CUDA_ERROR_UNKNOWN", *text = "unknown error";
  if (t->cu.GetErrorName != nullptr) {
    t->cu.GetErrorName(r, &name);
    t->cu.GetErrorString(r, &text);
  }
  if (!is_sticky(r)) {
    return say(err, errlen, JL_TRT_CUDA_ERROR, "%s: %s: %s", call, name, text);
  }
  std::lock_guard<std::mutex> hold(t->sticky_lock);
  if (t->sticky.load() == 0) {
    snprintf(t->sticky_message, sizeof t->sticky_message, "%s: %s: %s", call, name, text);
    t->sticky.store(1, std::memory_order_release);
  }
  return say(err, errlen, JL_TRT_CUDA_STICKY, "%s: %s: %s", call, name, text);
}

// Every call that reaches CUDA or TensorRT starts here: fail at once once the
// context is broken, else make it current on this thread if it is not yet.
int enter(jl_trt *t, char *err, size_t errlen) {
  if (t->sticky.load(std::memory_order_acquire) != 0) {
    std::lock_guard<std::mutex> hold(t->sticky_lock);
    return say(err, errlen, JL_TRT_CUDA_STICKY, "%s (latched)", t->sticky_message);
  }
  if (current.ctx != t->ctx || current.generation != t->generation) {
    CUresult r = t->cu.CtxSetCurrent(t->ctx);
    if (r != CUDA_SUCCESS) {
      return cuda_fail(t, "cuCtxSetCurrent", r, err, errlen);
    }
    current = {t->ctx, t->generation};
  }
  last_error[0] = '\0';
  return JL_TRT_OK;
}

// For the calls that return nothing: current if it can be, never failing.
void enter_quietly(jl_trt *t) {
  char ignored[1];
  enter(t, ignored, 0);
}

// enter, then one driver call: what most of the API is.
template <typename F> int call(jl_trt *t, const char *name, char *err, size_t errlen, F f) {
  int rc = enter(t, err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  CUresult r = f();
  return r == CUDA_SUCCESS ? JL_TRT_OK : cuda_fail(t, name, r, err, errlen);
}

// TensorRT returned NULL or false. A sticky CUDA error can sit behind that,
// and must not pass for a bad plan (D21 would delete a good one), so the
// context is asked, unless `stream` is capturing: a synchronize would end
// the capture, which fails anyway. Otherwise err is what TensorRT logged.
int refused(jl_trt *t, const char *name, const char *fallback, CUstream stream, char *err, size_t errlen) {
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream == nullptr || (t->cu.StreamIsCapturing(stream, &status) == CUDA_SUCCESS &&
                            status == CU_STREAM_CAPTURE_STATUS_NONE)) {
    CUresult r = t->cu.CtxSynchronize();
    if (is_sticky(r)) {
      return cuda_fail(t, name, r, err, errlen);
    }
  }
  return say(err, errlen, JL_TRT_ERROR, "%s: %s", name, last_error[0] ? last_error : fallback);
}

#define JL_ENTER(t)                    \
  do {                                 \
    int rc_ = enter((t), err, errlen); \
    if (rc_ != JL_TRT_OK) {            \
      return rc_;                      \
    }                                  \
  } while (0)

int open_cuda(jl_trt *t, int device, char *err, size_t errlen) {
  void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "no CUDA driver: %s", dlerror());
  }
  // the _v2 form exists from CUDA 12.0, the first from 11.3
  typedef CUresult (*V2)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *);
  typedef CUresult (*V1)(const char *, void **, int, cuuint64_t);
  auto v2 = reinterpret_cast<V2>(dlsym(lib, "cuGetProcAddress_v2"));
  auto v1 = reinterpret_cast<V1>(dlsym(lib, "cuGetProcAddress"));
  if (v2 == nullptr && v1 == nullptr) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "this CUDA driver is older than 11.3: it has no cuGetProcAddress");
  }
  auto resolve = [&](const char *name, int version, void **fn) {
    CUresult r = v2 != nullptr ? v2(name, fn, version, CU_GET_PROC_ADDRESS_LEGACY_STREAM, nullptr)
                               : v1(name, fn, version, CU_GET_PROC_ADDRESS_LEGACY_STREAM);
    return r == CUDA_SUCCESS && *fn != nullptr;
  };
#define JL_RESOLVE(name, version, params)                                                                 \
  if (!resolve("cu" #name, version, reinterpret_cast<void **>(&t->cu.name))) {                            \
    return say(err, errlen, JL_TRT_UNAVAILABLE, "this CUDA driver has no cu%s (version %d)", #name, version); \
  }
  JL_DRIVER(JL_RESOLVE)
#undef JL_RESOLVE
  CUresult r = t->cu.Init(0);
  if (r == CUDA_SUCCESS) {
    r = t->cu.DeviceGet(&t->device, device);
  }
  if (r != CUDA_SUCCESS) {
    cuda_fail(t, r == CUDA_ERROR_INVALID_DEVICE ? "cuDeviceGet" : "cuInit", r, err, errlen);
    return JL_TRT_UNAVAILABLE;
  }
  jl_trt_info &info = t->info;
  CUresult setup[] = {
      t->cu.DriverGetVersion(&info.cuda_driver),
      t->cu.DeviceGetName(t->device_name, static_cast<int>(sizeof t->device_name), t->device),
      t->cu.DeviceGetAttribute(&info.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, t->device),
      t->cu.DeviceGetAttribute(&info.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, t->device),
      t->cu.DevicePrimaryCtxRetain(&t->ctx, t->device),
  };
  for (CUresult s : setup) {
    if (s != CUDA_SUCCESS) {
      return cuda_fail(t, "cuDevice", s, err, errlen);
    }
  }
  t->generation = ++generations;
  return enter(t, err, errlen);
}

template <typename T> bool resolve(void *lib, const char *name, T *out) {
  *out = reinterpret_cast<T>(dlsym(lib, name));
  return *out != nullptr;
}

// A TensorRT library by name, from `dir` when there is one, else from the
// loader path. The ones it needs itself resolve beside it: NVIDIA's copies
// carry an $ORIGIN RPATH.
void *open_library(const char *dir, const char *name) {
  char path[4096];
  if (dir != nullptr) {
    snprintf(path, sizeof path, "%s/%s", dir, name);
  }
  return dlopen(dir != nullptr ? path : name, RTLD_NOW | RTLD_LOCAL);
}

int open_tensorrt(jl_trt *t, const char *dir, char *err, size_t errlen) {
  const char *name = "libnvinfer.so." JL_STR(NV_TENSORRT_MAJOR);
  void *lib = open_library(dir, name);
  if (lib == nullptr) {
    return dir != nullptr ? say(err, errlen, JL_TRT_UNAVAILABLE, "no TensorRT %d in %s: %s", NV_TENSORRT_MAJOR, dir, dlerror())
                          : say(err, errlen, JL_TRT_UNAVAILABLE, "no TensorRT %d: %s", NV_TENSORRT_MAJOR, dlerror());
  }
  int32_t (*version[4])() = {};
  if (!resolve(lib, "createInferRuntime_INTERNAL", &t->create_runtime) ||
      !resolve(lib, "createInferBuilder_INTERNAL", &t->create_builder) ||
      !resolve(lib, "getInferLibMajorVersion", &version[0]) || !resolve(lib, "getInferLibMinorVersion", &version[1]) ||
      !resolve(lib, "getInferLibPatchVersion", &version[2]) || !resolve(lib, "getInferLibBuildVersion", &version[3])) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "%s lacks an entry point jetlink needs: %s", name, dlerror());
  }
  Dl_info where = {};
  if (dladdr(reinterpret_cast<void *>(t->create_runtime), &where) != 0 && where.dli_fname != nullptr) {
    snprintf(t->library, sizeof t->library, "%s", where.dli_fname);
  }
  jl_trt_info &info = t->info;
  info.major = version[0]();
  info.minor = version[1]();
  info.patch = version[2]();
  info.build = version[3]();
  // A newer minor runs what older headers compiled; an older one may lack
  // what they call. The build number is a respin of the same release.
  if (info.major != NV_TENSORRT_MAJOR ||
      NV_TENSORRT_VERSION_INT(info.major, info.minor, info.patch) < static_cast<long>(NV_TENSORRT_VERSION)) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "TensorRT %d.%d.%d.%d is loaded; this build needs %d.x from %d.%d.%d on",
               info.major, info.minor, info.patch, info.build, NV_TENSORRT_MAJOR, NV_TENSORRT_MAJOR,
               NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
  }
  // Needed only to build, so a missing parser is jl_trt_build_create's error.
  void *parser = open_library(dir, "libnvonnxparser.so." JL_STR(NV_TENSORRT_MAJOR));
  if (parser != nullptr) {
    resolve(parser, "createNvOnnxParser_INTERNAL", &t->create_parser);
  }
  // Python registered the plugins whenever the library had them, so a plan
  // with a plugin layer must still load here. Absent is fine: jetlink's
  // models use none.
  void *plugins = open_library(dir, "libnvinfer_plugin.so." JL_STR(NV_TENSORRT_MAJOR));
  bool (*init_plugins)(void *, const char *) = nullptr;
  if (plugins != nullptr && resolve(plugins, "initLibNvInferPlugins", &init_plugins)) {
    info.plugins = init_plugins(trt_logger(), "") ? 1 : 0;
  }
  return JL_TRT_OK;
}

// TensorRT's data types, by their DataType value, as ONNX numbers them.
int onnx_type(nvinfer1::DataType type) {
  static const int onnx[] = {JL_TRT_FLOAT, JL_TRT_FLOAT16, JL_TRT_INT8, JL_TRT_INT32, JL_TRT_BOOL,
                             JL_TRT_UINT8, 0 /* FP8 */,    0 /* BF16 */, JL_TRT_INT64};
  size_t i = static_cast<size_t>(type);
  return i < sizeof onnx / sizeof onnx[0] ? onnx[i] : 0;
}

// onnx-tensorrt's errorCodeStr, which the headers do not carry
const char *parser_code(nvonnxparser::ErrorCode code) {
  static const char *const names[] = {
      "SUCCESS",           "INTERNAL_ERROR",        "MEM_ALLOC_FAILED",       "MODEL_DESERIALIZE_FAILED",
      "INVALID_VALUE",     "INVALID_GRAPH",         "INVALID_NODE",           "UNSUPPORTED_GRAPH",
      "UNSUPPORTED_NODE",  "UNSUPPORTED_NODE_ATTR", "UNSUPPORTED_NODE_INPUT", "UNSUPPORTED_NODE_DATATYPE",
      "UNSUPPORTED_NODE_DYNAMIC", "UNSUPPORTED_NODE_SHAPE", "REFIT_FAILED",
  };
  size_t i = static_cast<size_t>(code);
  return i < sizeof names / sizeof names[0] ? names[i] : "UNKNOWN";
}

} // namespace

extern "C" {

// --- library ---------------------------------------------------------------------

int jl_trt_open(int device, const char *lib_dir, jl_trt **out, char *err, size_t errlen) {
  *out = nullptr;
  jl_trt *t = new (std::nothrow) jl_trt();
  if (t == nullptr) {
    return say(err, errlen, JL_TRT_ERROR, "out of memory");
  }
  t->info.strongly_typed = NV_TENSORRT_MAJOR >= 11 ? 1 : 0;
  int rc = open_cuda(t, device, err, errlen);
  if (rc == JL_TRT_OK) {
    rc = open_tensorrt(t, lib_dir, err, errlen);
  }
  if (rc != JL_TRT_OK) {
    jl_trt_close(t);
    return rc;
  }
  t->info.device_name = t->device_name;
  t->info.library = t->library;
  *out = t;
  return JL_TRT_OK;
}

void jl_trt_close(jl_trt *t) {
  if (t == nullptr) {
    return;
  }
  if (t->ctx != nullptr) {
    t->cu.DevicePrimaryCtxRelease(t->device);
  }
  if (current.ctx == t->ctx) {
    current = {nullptr, 0};
  }
  delete t;
}

void jl_trt_get_info(const jl_trt *t, jl_trt_info *out) {
  *out = t->info;
}

int jl_trt_mem_info(jl_trt *t, size_t *free_bytes, size_t *total_bytes, char *err, size_t errlen) {
  return call(t, "cuMemGetInfo", err, errlen, [&] { return t->cu.MemGetInfo(free_bytes, total_bytes); });
}

void jl_trt_set_logger(jl_trt *t, int min_severity, jl_trt_log_fn fn, void *ctx) {
  (void)t;
  std::lock_guard<std::mutex> hold(logger.lock);
  logger.fn = fn;
  logger.ctx = ctx;
  logger.min = min_severity;
}

int jl_trt_sticky(const jl_trt *t) {
  return t->sticky.load(std::memory_order_acquire);
}

// --- memory and copies -------------------------------------------------------------

int jl_trt_mem_alloc(jl_trt *t, size_t size, jl_trt_dptr *out, char *err, size_t errlen) {
  CUdeviceptr p = 0;
  int rc = call(t, "cuMemAlloc", err, errlen, [&] { return t->cu.MemAlloc(&p, size); });
  *out = static_cast<jl_trt_dptr>(p);
  return rc;
}

void jl_trt_mem_free(jl_trt *t, jl_trt_dptr ptr) {
  if (ptr != 0) {
    enter_quietly(t);
    t->cu.MemFree(static_cast<CUdeviceptr>(ptr));
  }
}

int jl_trt_host_alloc(jl_trt *t, size_t size, void **out, char *err, size_t errlen) {
  *out = nullptr;
  // cudaHostAllocDefault, as Python allocated them
  return call(t, "cuMemHostAlloc", err, errlen, [&] { return t->cu.MemHostAlloc(out, size, 0); });
}

void jl_trt_host_free(jl_trt *t, void *ptr) {
  if (ptr != nullptr) {
    enter_quietly(t);
    t->cu.MemFreeHost(ptr);
  }
}

int jl_trt_copy_h2d(jl_trt *t, jl_trt_dptr dst, const void *src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  return call(t, "cuMemcpyHtoDAsync", err, errlen, [&] { return t->cu.MemcpyHtoDAsync(dst, src, size, cu(stream)); });
}

int jl_trt_copy_d2h(jl_trt *t, void *dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  return call(t, "cuMemcpyDtoHAsync", err, errlen, [&] { return t->cu.MemcpyDtoHAsync(dst, src, size, cu(stream)); });
}

int jl_trt_copy_d2d(jl_trt *t, jl_trt_dptr dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  return call(t, "cuMemcpyDtoDAsync", err, errlen, [&] { return t->cu.MemcpyDtoDAsync(dst, src, size, cu(stream)); });
}

int jl_trt_memset(jl_trt *t, jl_trt_dptr dst, uint8_t value, size_t size, jl_trt_stream *stream, char *err,
                  size_t errlen) {
  return call(t, "cuMemsetD8Async", err, errlen, [&] { return t->cu.MemsetD8Async(dst, value, size, cu(stream)); });
}

// --- streams and events ----------------------------------------------------------------

int jl_trt_stream_create(jl_trt *t, jl_trt_stream **out, char *err, size_t errlen) {
  CUstream s = nullptr;
  // CU_STREAM_DEFAULT, as cudaStreamCreate
  int rc = call(t, "cuStreamCreate", err, errlen, [&] { return t->cu.StreamCreate(&s, 0); });
  *out = reinterpret_cast<jl_trt_stream *>(s);
  return rc;
}

int jl_trt_stream_sync(jl_trt *t, jl_trt_stream *stream, char *err, size_t errlen) {
  return call(t, "cuStreamSynchronize", err, errlen, [&] { return t->cu.StreamSynchronize(cu(stream)); });
}

void jl_trt_stream_destroy(jl_trt *t, jl_trt_stream *stream) {
  if (stream != nullptr) {
    enter_quietly(t);
    t->cu.StreamDestroy(cu(stream));
  }
}

int jl_trt_event_create(jl_trt *t, unsigned flags, jl_trt_event **out, char *err, size_t errlen) {
  CUevent e = nullptr;
  int rc = call(t, "cuEventCreate", err, errlen, [&] { return t->cu.EventCreate(&e, flags); });
  *out = reinterpret_cast<jl_trt_event *>(e);
  return rc;
}

int jl_trt_event_record(jl_trt *t, jl_trt_event *event, jl_trt_stream *stream, unsigned flags, char *err,
                        size_t errlen) {
  return call(t, "cuEventRecordWithFlags", err, errlen,
              [&] { return t->cu.EventRecordWithFlags(cu(event), cu(stream), flags); });
}

int jl_trt_event_sync(jl_trt *t, jl_trt_event *event, char *err, size_t errlen) {
  return call(t, "cuEventSynchronize", err, errlen, [&] { return t->cu.EventSynchronize(cu(event)); });
}

int jl_trt_event_elapsed(jl_trt *t, jl_trt_event *start, jl_trt_event *end, float *ms, char *err, size_t errlen) {
  return call(t, "cuEventElapsedTime", err, errlen, [&] { return t->cu.EventElapsedTime(ms, cu(start), cu(end)); });
}

void jl_trt_event_destroy(jl_trt *t, jl_trt_event *event) {
  if (event != nullptr) {
    enter_quietly(t);
    t->cu.EventDestroy(cu(event));
  }
}

// --- graphs ----------------------------------------------------------------------------

int jl_trt_capture_begin(jl_trt *t, jl_trt_stream *stream, char *err, size_t errlen) {
  return call(t, "cuStreamBeginCapture", err, errlen,
              [&] { return t->cu.StreamBeginCapture(cu(stream), CU_STREAM_CAPTURE_MODE_THREAD_LOCAL); });
}

int jl_trt_capture_end(jl_trt *t, jl_trt_stream *stream, jl_trt_graph **out, char *err, size_t errlen) {
  CUgraph g = nullptr;
  int rc = call(t, "cuStreamEndCapture", err, errlen, [&] { return t->cu.StreamEndCapture(cu(stream), &g); });
  if (rc != JL_TRT_OK && g != nullptr) {
    // an invalidated capture can still hand back a partial graph
    t->cu.GraphDestroy(g);
    g = nullptr;
  }
  *out = reinterpret_cast<jl_trt_graph *>(g);
  return rc;
}

int jl_trt_graph_instantiate(jl_trt *t, jl_trt_graph *graph, jl_trt_graph_exec **out, char *err, size_t errlen) {
  CUgraphExec x = nullptr;
  int rc = call(t, "cuGraphInstantiateWithFlags", err, errlen,
                [&] { return t->cu.GraphInstantiateWithFlags(&x, reinterpret_cast<CUgraph>(graph), 0); });
  *out = reinterpret_cast<jl_trt_graph_exec *>(x);
  return rc;
}

int jl_trt_graph_launch(jl_trt *t, jl_trt_graph_exec *exec, jl_trt_stream *stream, char *err, size_t errlen) {
  return call(t, "cuGraphLaunch", err, errlen,
              [&] { return t->cu.GraphLaunch(reinterpret_cast<CUgraphExec>(exec), cu(stream)); });
}

void jl_trt_graph_destroy(jl_trt *t, jl_trt_graph *graph) {
  if (graph != nullptr) {
    enter_quietly(t);
    t->cu.GraphDestroy(reinterpret_cast<CUgraph>(graph));
  }
}

void jl_trt_graph_exec_destroy(jl_trt *t, jl_trt_graph_exec *exec) {
  if (exec != nullptr) {
    enter_quietly(t);
    t->cu.GraphExecDestroy(reinterpret_cast<CUgraphExec>(exec));
  }
}

// --- runtime -------------------------------------------------------------------------------

int jl_trt_engine_deserialize(jl_trt *t, const void *plan, size_t size, jl_trt_engine **out, char *err,
                              size_t errlen) {
  *out = nullptr;
  JL_ENTER(t);
  auto *runtime = static_cast<nvinfer1::IRuntime *>(t->create_runtime(trt_logger(), NV_TENSORRT_VERSION));
  if (runtime == nullptr) {
    // No plan was read: a driver or a libnvinfer this process cannot run,
    // which would otherwise delete every plan asked for.
    int rc = refused(t, "createInferRuntime", "TensorRT returned no runtime", nullptr, err, errlen);
    return rc == JL_TRT_ERROR ? JL_TRT_CUDA_ERROR : rc;
  }
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(plan, size);
  if (engine == nullptr) {
    int rc = refused(t, "deserializeCudaEngine", "TensorRT could not deserialize the plan", nullptr, err, errlen);
    // A plan that does not fit in the device memory left is still a good
    // plan: read before the runtime goes, which may log again.
    bool memory = rc == JL_TRT_ERROR && is_allocation_failure(last_error);
    delete runtime;
    return memory ? JL_TRT_CUDA_ERROR : rc;
  }
  *out = new jl_trt_engine{t, runtime, engine};
  return JL_TRT_OK;
}

void jl_trt_engine_destroy(jl_trt_engine *e) {
  if (e != nullptr) {
    enter_quietly(e->trt);
    delete e->engine;
    delete e->runtime;
    delete e;
  }
}

int jl_trt_engine_io_count(const jl_trt_engine *e) {
  return e->engine->getNbIOTensors();
}

int jl_trt_engine_io(const jl_trt_engine *e, int index, const char **name, int *is_input, int *type, int64_t *dims,
                     int *rank, char *err, size_t errlen) {
  int count = e->engine->getNbIOTensors();
  const char *n = index >= 0 && index < count ? e->engine->getIOTensorName(index) : nullptr;
  nvinfer1::Dims shape = n != nullptr ? e->engine->getTensorShape(n) : nvinfer1::Dims{};
  if (n == nullptr || shape.nbDims < 0 || shape.nbDims > JL_TRT_MAX_DIMS) {
    return say(err, errlen, JL_TRT_ERROR, "no IO tensor %d with usable dims; the engine has %d", index, count);
  }
  *name = n;
  *is_input = e->engine->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT;
  *type = onnx_type(e->engine->getTensorDataType(n));
  *rank = shape.nbDims;
  memcpy(dims, shape.d, static_cast<size_t>(shape.nbDims) * sizeof(int64_t));
  return JL_TRT_OK;
}

int jl_trt_context_create(jl_trt_engine *e, jl_trt_context **out, char *err, size_t errlen) {
  *out = nullptr;
  JL_ENTER(e->trt);
  nvinfer1::IExecutionContext *context = e->engine->createExecutionContext();
  if (context == nullptr) {
    return refused(e->trt, "createExecutionContext", "TensorRT returned no execution context", nullptr, err, errlen);
  }
  *out = new jl_trt_context{e->trt, context};
  return JL_TRT_OK;
}

void jl_trt_context_destroy(jl_trt_context *c) {
  if (c != nullptr) {
    enter_quietly(c->trt);
    delete c->context;
    delete c;
  }
}

int jl_trt_context_set_address(jl_trt_context *c, const char *name, jl_trt_dptr address, char *err, size_t errlen) {
  last_error[0] = '\0';
  if (c->context->setTensorAddress(name, reinterpret_cast<void *>(static_cast<uintptr_t>(address)))) {
    return JL_TRT_OK;
  }
  return say(err, errlen, JL_TRT_ERROR, "setTensorAddress: %s", last_error[0] ? last_error : "TensorRT refused it");
}

int jl_trt_context_enqueue(jl_trt_context *c, jl_trt_stream *stream, char *err, size_t errlen) {
  JL_ENTER(c->trt);
  if (c->context->enqueueV3(reinterpret_cast<cudaStream_t>(stream))) {
    return JL_TRT_OK;
  }
  return refused(c->trt, "enqueueV3", "TensorRT refused to enqueue", cu(stream), err, errlen);
}

// --- builder -----------------------------------------------------------------------------

int jl_trt_build_create(jl_trt *t, jl_trt_build **out, char *err, size_t errlen) {
  *out = nullptr;
  if (t->create_parser == nullptr) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "no ONNX parser: libnvonnxparser.so.%d did not load", NV_TENSORRT_MAJOR);
  }
  JL_ENTER(t);
  jl_trt_build *b = new jl_trt_build();
  b->trt = t;
  b->builder = static_cast<nvinfer1::IBuilder *>(t->create_builder(trt_logger(), NV_TENSORRT_VERSION));
  // 0: weakly typed on TensorRT 10 (TrtBackend says why); 11 ignores it
  b->network = b->builder != nullptr ? b->builder->createNetworkV2(0) : nullptr;
  b->parser = b->network != nullptr
                  ? static_cast<nvonnxparser::IParser *>(t->create_parser(b->network, trt_logger(), NV_ONNX_PARSER_VERSION))
                  : nullptr;
  b->config = b->parser != nullptr ? b->builder->createBuilderConfig() : nullptr;
  if (b->config == nullptr) {
    jl_trt_build_destroy(b);
    return refused(t, "createInferBuilder", "TensorRT returned no builder, network, parser or config", nullptr, err,
                   errlen);
  }
  *out = b;
  return JL_TRT_OK;
}

void jl_trt_build_destroy(jl_trt_build *b) {
  if (b == nullptr) {
    return;
  }
  enter_quietly(b->trt);
  // the reverse of creation: the config refers to the cache, the network to
  // the parser's weights
  delete b->config;
  delete b->cache;
  delete b->parser;
  delete b->network;
  delete b->builder;
  delete b;
}

int jl_trt_build_parse(jl_trt_build *b, const char *onnx_path, char *err, size_t errlen) {
  JL_ENTER(b->trt);
  if (b->parser->parseFromFile(onnx_path, 0)) {
    return JL_TRT_OK;
  }
  // one per line, as str(ParserError) prints them in TensorRT's Python
  say(err, errlen, JL_TRT_ERROR, "%s", last_error[0] ? last_error : "the ONNX parser failed without an error");
  for (int i = 0, used = 0; i < b->parser->getNbErrors() && err != nullptr && static_cast<size_t>(used) + 1 < errlen;
       i++) {
    const nvonnxparser::IParserError *e = b->parser->getError(i);
    bool whole = e->code() == nvonnxparser::ErrorCode::kMODEL_DESERIALIZE_FAILED ||
                 e->code() == nvonnxparser::ErrorCode::kREFIT_FAILED;
    char node[512] = "";
    if (!whole) {
      snprintf(node, sizeof node, "In node %d with name: %s and operator: %s ", e->node(), or_empty(e->nodeName()),
               or_empty(e->nodeOperator()));
    }
    int w = snprintf(err + used, errlen - used, "%s%s(%s): %s: %s", i > 0 ? "\n" : "", node, or_empty(e->func()),
                     parser_code(e->code()), or_empty(e->desc()));
    used += w > 0 ? w : 0;
  }
  return JL_TRT_ERROR;
}

int jl_trt_build_layers(const jl_trt_build *b) {
  return b->network->getNbLayers();
}

int jl_trt_build_set_fp16(jl_trt_build *b, char *err, size_t errlen) {
#if NV_TENSORRT_MAJOR >= 11
  (void)b;
  return say(err, errlen, JL_TRT_ERROR, "TensorRT %d has no FP16 flag; precision follows the ONNX", NV_TENSORRT_MAJOR);
#else
  (void)err;
  (void)errlen;
  b->config->setFlag(nvinfer1::BuilderFlag::kFP16);
  return JL_TRT_OK;
#endif
}

void jl_trt_build_set_optimization_level(jl_trt_build *b, int level) {
  b->config->setBuilderOptimizationLevel(level);
}

void jl_trt_build_set_workspace(jl_trt_build *b, size_t bytes) {
  b->config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, bytes);
}

void jl_trt_build_cancel(jl_trt_build *b) {
  b->monitor.cancelled.store(true);
}

void jl_trt_build_set_progress(jl_trt_build *b, jl_trt_progress_fn fn, void *ctx) {
  {
    std::lock_guard<std::mutex> hold(b->monitor.lock);
    b->monitor.fn = fn;
    b->monitor.ctx = ctx;
  }
  b->config->setProgressMonitor(fn != nullptr ? &b->monitor : nullptr);
}

int jl_trt_build_set_timing_cache(jl_trt_build *b, const void *data, size_t size, char *err, size_t errlen) {
  JL_ENTER(b->trt);
  nvinfer1::ITimingCache *cache = b->config->createTimingCache(size > 0 ? data : nullptr, size);
  if (cache == nullptr || !b->config->setTimingCache(*cache, false)) {
    delete cache;
    return say(err, errlen, JL_TRT_ERROR, "timing cache: %s",
               last_error[0] ? last_error : "it does not match this TensorRT and GPU");
  }
  // the config holds the new one now
  delete b->cache;
  b->cache = cache;
  return JL_TRT_OK;
}

int jl_trt_build_write_plan(jl_trt_build *b, const char *path, char *err, size_t errlen) {
  JL_ENTER(b->trt);
  nvinfer1::IHostMemory *plan = b->builder->buildSerializedNetwork(*b->network, *b->config);
  if (plan == nullptr) {
    return refused(b->trt, "buildSerializedNetwork", "TensorRT returned no engine", nullptr, err, errlen);
  }
  bool ok = write_file(path, plan->data(), plan->size());
  delete plan;
  return ok ? JL_TRT_OK : say(err, errlen, JL_TRT_ERROR, "cannot write %s: %s", path, strerror(errno));
}

int jl_trt_build_write_timing_cache(jl_trt_build *b, const char *path, char *err, size_t errlen) {
  if (b->cache == nullptr) {
    return say(err, errlen, JL_TRT_ERROR, "no timing cache is attached");
  }
  JL_ENTER(b->trt);
  nvinfer1::IHostMemory *blob = b->cache->serialize();
  bool ok = blob != nullptr && write_file(path, blob->data(), blob->size());
  delete blob;
  return ok ? JL_TRT_OK : say(err, errlen, JL_TRT_ERROR, "cannot write the timing cache to %s: %s", path, strerror(errno));
}

} // extern "C"
