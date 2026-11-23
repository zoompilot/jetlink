// TensorRT, and the CUDA it runs on, as plain C calls for Swift, which can
// neither call TensorRT's C++ API without C++ interop nor subclass the ILogger
// and IProgressMonitor TensorRT calls back into. The calls are the ones the
// Python server made, one for one; what to do with them stays in Swift, so the
// fake (host memory, no GPU) exercises every Swift path.
//
// A call that can fail returns JL_TRT_OK (0) or one of the codes below, and
// writes why into err: errlen bytes at most, always NUL terminated; err may be
// NULL. A CUDA failure reads "<call>: <CUDA_ERROR_NAME>: <description>". When
// TensorRT fails without a reason (a NULL engine, a false), the shim asks the
// context whether a sticky CUDA error is behind it and returns that as
// JL_TRT_CUDA_STICKY, since a failed deserialize must not pass for a bad plan;
// else err holds the last error TensorRT logged on the calling thread. Frees
// and destroys return nothing: close paths cannot act on an error.
//
// CUDA contexts are the shim's business: jl_trt_open retains the device's
// primary context, and every call makes it current on the calling thread when
// it is not, so Swift may call from any thread. Nothing takes a lock on the
// way to CUDA: a handle is used by one thread at a time, different handles on
// different threads at once (a build beside a loaded engine). Destroy
// everything made from a jl_trt before jl_trt_close, and a context before its
// engine.
#ifndef JL_TRT_H
#define JL_TRT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  JL_TRT_OK = 0,
  // TensorRT, the ONNX parser or a file said no.
  JL_TRT_ERROR = 1,
  // A CUDA call failed and the context still works: out of memory, a bad
  // argument, a capture that was invalidated. Also a deserialize that failed
  // for want of memory or of a runtime.
  JL_TRT_CUDA_ERROR = 2,
  // A CUDA error that breaks the context for good (CUDA's sticky errors: an
  // illegal address, a failed launch, a hardware exception) or a device that
  // is gone. Only a new process recovers (D15). It latches, atomically, since
  // a build thread and the session thread share it: from then on every call
  // on the same jl_trt returns it without reaching CUDA, and jl_trt_sticky
  // says so.
  JL_TRT_CUDA_STICKY = 3,
  // jl_trt_open only: no CUDA driver or no such device, or no TensorRT of the
  // major this shim was compiled for, or one older than its headers
  // (major.minor.patch; the build number is not compared).
  JL_TRT_UNAVAILABLE = 4,
};

// TensorRT's data types, as ONNX numbers them, so they read the same as
// jl_ort.h's and ElementType's. Any other TensorRT type reads 0: jetlink does
// not stage it.
enum {
  JL_TRT_FLOAT = 1,
  JL_TRT_UINT8 = 2,
  JL_TRT_INT8 = 3,
  JL_TRT_INT32 = 6,
  JL_TRT_INT64 = 7,
  JL_TRT_BOOL = 9,
  JL_TRT_FLOAT16 = 10,
};

// ILogger's severities: lower is worse.
enum {
  JL_TRT_LOG_INTERNAL_ERROR = 0,
  JL_TRT_LOG_ERROR = 1,
  JL_TRT_LOG_WARNING = 2,
  JL_TRT_LOG_INFO = 3,
  JL_TRT_LOG_VERBOSE = 4,
};

// Room for an IO tensor's dims (nvinfer1::Dims::MAX_DIMS).
#define JL_TRT_MAX_DIMS 8

// The libraries, the device and its retained primary context.
typedef struct jl_trt jl_trt;
// An IRuntime and the ICudaEngine it deserialized.
typedef struct jl_trt_engine jl_trt_engine;
typedef struct jl_trt_context jl_trt_context;
// An IBuilder with its network, ONNX parser, config, progress monitor and
// timing cache: one handle, because TensorRT wants them destroyed together
// and in order.
typedef struct jl_trt_build jl_trt_build;
typedef struct jl_trt_stream jl_trt_stream;
typedef struct jl_trt_event jl_trt_event;
typedef struct jl_trt_graph jl_trt_graph;
typedef struct jl_trt_graph_exec jl_trt_graph_exec;

// A device address (CUdeviceptr): an integer, so Swift cannot mistake it for
// memory it may touch.
typedef uint64_t jl_trt_dptr;

// A TensorRT log line at the logger's severity or worse. message lasts for
// the call only.
typedef void (*jl_trt_log_fn)(void *ctx, int severity, const char *message);

// IProgressMonitor's calls, as they come: which phase started (value is its
// step count, parent NULL for a root phase), took a step (value is the step)
// or finished (value 0, parent NULL). The build cannot be stopped from here.
enum {
  JL_TRT_PHASE_START = 0,
  JL_TRT_PHASE_STEP = 1,
  JL_TRT_PHASE_FINISH = 2,
};
typedef void (*jl_trt_progress_fn)(void *ctx, int event, const char *phase, const char *parent, int value);

// Both callbacks may run on TensorRT's own threads; the shim makes each one's
// calls one at a time.

typedef struct {
  // The libnvinfer loaded. Plans and timing caches are valid for this exact
  // build only.
  int major, minor, patch, build;
  // 1 on TensorRT 11, which has no FP16 flag.
  int strongly_typed;
  // cuDriverGetVersion: 12060 for CUDA 12.6.
  int cuda_driver;
  // 1 when libnvinfer_plugin was found and its plugins registered.
  int plugins;
  // The device's name ("Orin") and compute capability (8, 7), the device
  // part of the cache tag. device_name lasts until jl_trt_close.
  const char *device_name;
  int cc_major, cc_minor;
  // The libnvinfer loaded, as a path ("" when unknown); lasts until
  // jl_trt_close.
  const char *library;
} jl_trt_info;

// Opens CUDA and TensorRT and retains `device`'s primary context. TensorRT's
// libraries (libnvinfer, libnvonnxparser, libnvinfer_plugin, of the shim's
// major) come from `lib_dir` when it is not NULL, a PC's self-contained copy,
// else from the loader path, a Jetson's JetPack; libcuda.so.1 is the host's
// driver, always from the loader path. libnvonnxparser is needed by
// jl_trt_build_create alone, so a machine without it still loads plans;
// libnvinfer_plugin is registered when present and never required.
int jl_trt_open(int device, const char *lib_dir, jl_trt **out, char *err, size_t errlen);
void jl_trt_close(jl_trt *trt);

void jl_trt_get_info(const jl_trt *trt, jl_trt_info *out);

// Free and total device memory.
int jl_trt_mem_info(jl_trt *trt, size_t *free_bytes, size_t *total_bytes, char *err, size_t errlen);

// Where TensorRT's log lines go: those at min_severity or worse. TensorRT
// keeps one logger per process, so the shim hands it one forwarder and the
// logger set last receives every line. Set it right after jl_trt_open, before
// any engine or build; fn NULL drops them.
void jl_trt_set_logger(jl_trt *trt, int min_severity, jl_trt_log_fn fn, void *ctx);

// Nonzero once a call on trt returned JL_TRT_CUDA_STICKY.
int jl_trt_sticky(const jl_trt *trt);

// --- memory ---------------------------------------------------------------

// Neither allocation is zeroed.
int jl_trt_mem_alloc(jl_trt *trt, size_t size, jl_trt_dptr *out, char *err, size_t errlen);
void jl_trt_mem_free(jl_trt *trt, jl_trt_dptr ptr);

// Page-locked (cudaHostAllocDefault), so the copies below are asynchronous
// and a graph can replay them.
int jl_trt_host_alloc(jl_trt *trt, size_t size, void **out, char *err, size_t errlen);
void jl_trt_host_free(jl_trt *trt, void *ptr);

// Stream-ordered copies and a byte memset. Host memory comes from
// jl_trt_host_alloc.
int jl_trt_copy_h2d(jl_trt *trt, jl_trt_dptr dst, const void *src, size_t size, jl_trt_stream *stream,
                    char *err, size_t errlen);
int jl_trt_copy_d2h(jl_trt *trt, void *dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen);
int jl_trt_copy_d2d(jl_trt *trt, jl_trt_dptr dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream,
                    char *err, size_t errlen);
int jl_trt_memset(jl_trt *trt, jl_trt_dptr dst, uint8_t value, size_t size, jl_trt_stream *stream, char *err,
                  size_t errlen);

// --- streams and events -----------------------------------------------------

int jl_trt_stream_create(jl_trt *trt, jl_trt_stream **out, char *err, size_t errlen);
int jl_trt_stream_sync(jl_trt *trt, jl_trt_stream *stream, char *err, size_t errlen);
void jl_trt_stream_destroy(jl_trt *trt, jl_trt_stream *stream);

// Event flags, CUDA's values. Leave DISABLE_TIMING off for a pair that
// jl_trt_event_elapsed times.
enum {
  JL_TRT_EVENT_BLOCKING_SYNC = 0x1,
  JL_TRT_EVENT_DISABLE_TIMING = 0x2,
};
// Record flag: inside a capture, becomes an event node the host can wait on
// when the graph replays, rather than a dependency between streams.
enum {
  JL_TRT_RECORD_EXTERNAL = 0x1,
};
int jl_trt_event_create(jl_trt *trt, unsigned flags, jl_trt_event **out, char *err, size_t errlen);
int jl_trt_event_record(jl_trt *trt, jl_trt_event *event, jl_trt_stream *stream, unsigned flags, char *err,
                        size_t errlen);
int jl_trt_event_sync(jl_trt *trt, jl_trt_event *event, char *err, size_t errlen);
// Milliseconds between two recorded timing events.
int jl_trt_event_elapsed(jl_trt *trt, jl_trt_event *start, jl_trt_event *end, float *ms, char *err,
                         size_t errlen);
void jl_trt_event_destroy(jl_trt *trt, jl_trt_event *event);

// --- graphs -----------------------------------------------------------------

// Captures what is queued on `stream` between begin and end into a graph, in
// thread-local mode (trtexec's: global mode would stop a build on another
// thread from allocating). Begin, the calls captured and end run on one
// thread, with nothing that synchronizes in between. Enqueue once before the
// first capture, as TensorRT requires. After a failure inside a capture, still
// call end: it ends the capture, returns the error and leaves the stream
// usable.
int jl_trt_capture_begin(jl_trt *trt, jl_trt_stream *stream, char *err, size_t errlen);
int jl_trt_capture_end(jl_trt *trt, jl_trt_stream *stream, jl_trt_graph **out, char *err, size_t errlen);
int jl_trt_graph_instantiate(jl_trt *trt, jl_trt_graph *graph, jl_trt_graph_exec **out, char *err,
                             size_t errlen);
// Replays the graph on stream. The addresses it captured are baked in: every
// buffer stays where it is, and allocated, for as long as it is launched.
int jl_trt_graph_launch(jl_trt *trt, jl_trt_graph_exec *exec, jl_trt_stream *stream, char *err, size_t errlen);
void jl_trt_graph_destroy(jl_trt *trt, jl_trt_graph *graph);
void jl_trt_graph_exec_destroy(jl_trt *trt, jl_trt_graph_exec *exec);

// --- runtime ----------------------------------------------------------------

// Over the caller's bytes, which TensorRT does not keep past the call. A plan
// from another TensorRT build is JL_TRT_ERROR, and only a refusal of the plan
// is: no runtime, or a refusal TensorRT logged as memory it could not
// allocate, is JL_TRT_CUDA_ERROR, since the plan may be good (D21 deletes a
// plan on JL_TRT_ERROR).
int jl_trt_engine_deserialize(jl_trt *trt, const void *plan, size_t size, jl_trt_engine **out, char *err,
                              size_t errlen);
void jl_trt_engine_destroy(jl_trt_engine *engine);

int jl_trt_engine_io_count(const jl_trt_engine *engine);
// One IO tensor: its name (the engine's, valid until it is destroyed),
// whether it is an input, its type (JL_TRT_*, 0 for one jetlink does not
// stage) and its dims (dims has room for JL_TRT_MAX_DIMS; a dynamic one is
// -1, which the caller refuses).
int jl_trt_engine_io(const jl_trt_engine *engine, int index, const char **name, int *is_input, int *type,
                     int64_t *dims, int *rank, char *err, size_t errlen);

// Its scratch memory is allocated now (kSTATIC).
int jl_trt_context_create(jl_trt_engine *engine, jl_trt_context **out, char *err, size_t errlen);
void jl_trt_context_destroy(jl_trt_context *context);
int jl_trt_context_set_address(jl_trt_context *context, const char *name, jl_trt_dptr address, char *err,
                               size_t errlen);
// A refusal asks the context for a sticky error as above, unless the stream
// is capturing: that would end the capture.
int jl_trt_context_enqueue(jl_trt_context *context, jl_trt_stream *stream, char *err, size_t errlen);

// --- builder ----------------------------------------------------------------

// An IBuilder, its network and an ONNX parser over it. JL_TRT_UNAVAILABLE
// without libnvonnxparser.
int jl_trt_build_create(jl_trt *trt, jl_trt_build **out, char *err, size_t errlen);
void jl_trt_build_destroy(jl_trt_build *build);

// A path rather than bytes, so weights stored beside the file are found. On
// failure err holds every parser error, one per line: give it a few KB.
int jl_trt_build_parse(jl_trt_build *build, const char *onnx_path, char *err, size_t errlen);
int jl_trt_build_layers(const jl_trt_build *build);

// JL_TRT_ERROR on TensorRT 11.
int jl_trt_build_set_fp16(jl_trt_build *build, char *err, size_t errlen);
void jl_trt_build_set_optimization_level(jl_trt_build *build, int level);
// The workspace pool's limit: a ceiling the tactics must fit under, not an
// allocation.
void jl_trt_build_set_workspace(jl_trt_build *build, size_t bytes);
void jl_trt_build_set_progress(jl_trt_build *build, jl_trt_progress_fn fn, void *ctx);
// Ends a build in progress at its next step, from any thread (the progress
// callback's included): jl_trt_build_write_plan then fails. Needs a progress
// monitor set, as TensorRT asks only it whether to go on.
void jl_trt_build_cancel(jl_trt_build *build);

// A timing cache from an earlier build's bytes (NULL and 0 for an empty one),
// attached with mismatches not ignored; TensorRT copies the bytes. On
// failure (another build's cache, a truncated one) nothing is attached.
int jl_trt_build_set_timing_cache(jl_trt_build *build, const void *data, size_t size, char *err,
                                  size_t errlen);

// Builds and writes the plan to path. Blocks for the whole build (minutes for
// a big model) while progress arrives.
int jl_trt_build_write_plan(jl_trt_build *build, const char *path, char *err, size_t errlen);
// The attached timing cache, which the build added to.
int jl_trt_build_write_timing_cache(jl_trt_build *build, const char *path, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
