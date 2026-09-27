// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The few onnxruntime calls Jetlink makes, as plain C that Swift can call.
//
// onnxruntime's C API is a table of function pointers returning status
// objects that must be released; reaching it from Swift directly means an
// unsafe dance at every call. This keeps that dance in one place and hands
// Swift a session that owns its inputs and outputs: the caller binds a buffer
// to every input and output once, then `jl_ort_run` reads and writes those
// buffers in place, which is what the frame path wants.
//
// Every function that can fail returns 0 on success and nonzero on failure,
// with onnxruntime's message copied into `err` (always NUL terminated when
// errlen > 0).

#ifndef JL_ORT_H
#define JL_ORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JL_ORT_MAX_RANK 8
#define JL_ORT_MAX_NAME 256

// onnxruntime's ONNXTensorElementDataType values, which are ONNX's
// TensorProto.DataType values.
enum {
  JL_ORT_FLOAT = 1,
  JL_ORT_UINT8 = 2,
  JL_ORT_INT32 = 6,
  JL_ORT_INT64 = 7,
  JL_ORT_BOOL = 9,
  JL_ORT_FLOAT16 = 10,
};

typedef struct {
  char name[JL_ORT_MAX_NAME];
  int32_t elem_type;
  int32_t rank;
  int64_t dims[JL_ORT_MAX_RANK];  // -1 for a dimension the model leaves open
} jl_ort_io;

typedef struct jl_ort_session jl_ort_session;

// The onnxruntime version the binary was built from, e.g. "1.29.0".
const char *jl_ort_version(void);

// Creates the process-wide environment on first call; later calls are no-ops.
// `log_severity` is onnxruntime's: 0 verbose .. 4 fatal. Telemetry is off.
int jl_ort_init(int log_severity, char *err, size_t errlen);

// A session over the model at `model_path`. With `use_coreml` the CoreML
// execution provider is appended with the given options (the same keys the
// Python backend passes: ModelFormat, MLComputeUnits, ModelCacheDirectory),
// and the CPU provider takes whatever CoreML does not. `intra_threads` is the
// CPU provider's thread count; 1 runs CPU nodes on the calling thread with no
// spinning pool.
jl_ort_session *jl_ort_session_create(const char *model_path, int use_coreml,
                                      const char *const *ep_keys, const char *const *ep_values,
                                      size_t ep_count, int intra_threads, char *err, size_t errlen);

size_t jl_ort_input_count(const jl_ort_session *s);
size_t jl_ort_output_count(const jl_ort_session *s);
const jl_ort_io *jl_ort_input(const jl_ort_session *s, size_t index);
const jl_ort_io *jl_ort_output(const jl_ort_session *s, size_t index);

// The execution providers in use, comma separated, e.g.
// "CoreMLExecutionProvider,CPUExecutionProvider".
const char *jl_ort_providers(const jl_ort_session *s);

// Lays a tensor of the input's (or output's) declared shape and type over
// `data`, which must stay valid and hold at least `nbytes` until the session
// is released or the slot is bound again.
int jl_ort_bind_input(jl_ort_session *s, size_t index, void *data, size_t nbytes, char *err, size_t errlen);
int jl_ort_bind_output(jl_ort_session *s, size_t index, void *data, size_t nbytes, char *err, size_t errlen);

// Runs the model once over the bound buffers. Every input and output must be
// bound. Not reentrant for one session.
int jl_ort_run(jl_ort_session *s, char *err, size_t errlen);

void jl_ort_session_release(jl_ort_session *s);

#ifdef __cplusplus
}
#endif

#endif
