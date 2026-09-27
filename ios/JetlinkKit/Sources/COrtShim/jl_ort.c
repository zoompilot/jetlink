// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

#include "jl_ort.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime/onnxruntime_c_api.h>

struct jl_ort_session {
  OrtSession *session;
  OrtMemoryInfo *cpu;
  size_t n_in, n_out;
  jl_ort_io *in, *out;
  const char **in_names, **out_names;   // point into in[i].name / out[i].name
  OrtValue **in_values, **out_values;   // NULL until bound
  char providers[128];
};

static const OrtApi *g_api;
static OrtEnv *g_env;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void set_err(char *err, size_t errlen, const char *msg) {
  if (err == NULL || errlen == 0) return;
  snprintf(err, errlen, "%s", msg ? msg : "unknown onnxruntime error");
}

// Copies a failed status's message into err and releases it. Returns 0 for OK.
static int take(OrtStatus *status, char *err, size_t errlen) {
  if (status == NULL) return 0;
  set_err(err, errlen, g_api->GetErrorMessage(status));
  g_api->ReleaseStatus(status);
  return 1;
}

#define TRY(call) do { if (take((call), err, errlen)) goto fail; } while (0)

const char *jl_ort_version(void) {
  return OrtGetApiBase()->GetVersionString();
}

int jl_ort_init(int log_severity, char *err, size_t errlen) {
  pthread_mutex_lock(&g_lock);
  int rc = 0;
  if (g_env == NULL) {
    if (g_api == NULL) g_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (g_api == NULL) {
      set_err(err, errlen, "this onnxruntime does not provide the API version jetlink was built against");
      rc = 1;
    } else if (take(g_api->CreateEnv((OrtLoggingLevel)log_severity, "jetlink", &g_env), err, errlen)) {
      g_env = NULL;
      rc = 1;
    } else {
      // A car has no business uploading usage events; see quiet() in the
      // Python backend. Failure here is not worth refusing to run over.
      OrtStatus *st = g_api->DisableTelemetryEvents(g_env);
      if (st != NULL) g_api->ReleaseStatus(st);
    }
  }
  pthread_mutex_unlock(&g_lock);
  return rc;
}

static int describe(OrtTypeInfo *info, jl_ort_io *io, char *err, size_t errlen) {
  const OrtTensorTypeAndShapeInfo *tensor = NULL;
  ONNXTensorElementDataType type;
  size_t rank = 0;
  if (take(g_api->CastTypeInfoToTensorInfo(info, &tensor), err, errlen)) return 1;
  if (tensor == NULL) {
    set_err(err, errlen, "a model input or output is not a tensor");
    return 1;
  }
  if (take(g_api->GetTensorElementType(tensor, &type), err, errlen)) return 1;
  if (take(g_api->GetDimensionsCount(tensor, &rank), err, errlen)) return 1;
  if (rank > JL_ORT_MAX_RANK) {
    set_err(err, errlen, "a model input or output has more dimensions than jetlink supports");
    return 1;
  }
  io->elem_type = (int32_t)type;
  io->rank = (int32_t)rank;
  if (rank && take(g_api->GetDimensions(tensor, io->dims, rank), err, errlen)) return 1;
  return 0;
}

static int copy_name(OrtAllocator *alloc, char *name, jl_ort_io *io, char *err, size_t errlen) {
  size_t n = strlen(name);
  if (n >= sizeof io->name) {
    set_err(err, errlen, "a model input or output name is too long");
    alloc->Free(alloc, name);
    return 1;
  }
  memcpy(io->name, name, n + 1);
  alloc->Free(alloc, name);
  return 0;
}

jl_ort_session *jl_ort_session_create(const char *model_path, int use_coreml,
                                      const char *const *ep_keys, const char *const *ep_values,
                                      size_t ep_count, int intra_threads, char *err, size_t errlen) {
  if (g_env == NULL) {
    set_err(err, errlen, "jl_ort_init was not called");
    return NULL;
  }
  OrtSessionOptions *so = NULL;
  OrtAllocator *alloc = NULL;
  OrtTypeInfo *info = NULL;
  jl_ort_session *s = calloc(1, sizeof *s);
  if (s == NULL) {
    set_err(err, errlen, "out of memory");
    return NULL;
  }

  TRY(g_api->CreateSessionOptions(&so));
  if (intra_threads > 0) {
    TRY(g_api->SetIntraOpNumThreads(so, intra_threads));
  }
  TRY(g_api->SetInterOpNumThreads(so, 1));
  // Threads that spin between frames burn a phone's battery for nothing: the
  // CPU's share of this model is two Expand nodes.
  TRY(g_api->AddSessionConfigEntry(so, "session.intra_op.allow_spinning", "0"));
  TRY(g_api->AddSessionConfigEntry(so, "session.inter_op.allow_spinning", "0"));
  if (use_coreml) {
    TRY(g_api->SessionOptionsAppendExecutionProvider(so, "CoreML", ep_keys, ep_values, ep_count));
    snprintf(s->providers, sizeof s->providers, "CoreMLExecutionProvider,CPUExecutionProvider");
  } else {
    snprintf(s->providers, sizeof s->providers, "CPUExecutionProvider");
  }
  TRY(g_api->CreateSession(g_env, model_path, so, &s->session));
  g_api->ReleaseSessionOptions(so);
  so = NULL;

  TRY(g_api->GetAllocatorWithDefaultOptions(&alloc));
  TRY(g_api->SessionGetInputCount(s->session, &s->n_in));
  TRY(g_api->SessionGetOutputCount(s->session, &s->n_out));
  s->in = calloc(s->n_in ? s->n_in : 1, sizeof *s->in);
  s->out = calloc(s->n_out ? s->n_out : 1, sizeof *s->out);
  s->in_names = calloc(s->n_in ? s->n_in : 1, sizeof *s->in_names);
  s->out_names = calloc(s->n_out ? s->n_out : 1, sizeof *s->out_names);
  s->in_values = calloc(s->n_in ? s->n_in : 1, sizeof *s->in_values);
  s->out_values = calloc(s->n_out ? s->n_out : 1, sizeof *s->out_values);
  if (!s->in || !s->out || !s->in_names || !s->out_names || !s->in_values || !s->out_values) {
    set_err(err, errlen, "out of memory");
    goto fail;
  }
  for (size_t i = 0; i < s->n_in; i++) {
    char *name = NULL;
    TRY(g_api->SessionGetInputName(s->session, i, alloc, &name));
    if (copy_name(alloc, name, &s->in[i], err, errlen)) goto fail;
    s->in_names[i] = s->in[i].name;
    TRY(g_api->SessionGetInputTypeInfo(s->session, i, &info));
    if (describe(info, &s->in[i], err, errlen)) goto fail;
    g_api->ReleaseTypeInfo(info);
    info = NULL;
  }
  for (size_t i = 0; i < s->n_out; i++) {
    char *name = NULL;
    TRY(g_api->SessionGetOutputName(s->session, i, alloc, &name));
    if (copy_name(alloc, name, &s->out[i], err, errlen)) goto fail;
    s->out_names[i] = s->out[i].name;
    TRY(g_api->SessionGetOutputTypeInfo(s->session, i, &info));
    if (describe(info, &s->out[i], err, errlen)) goto fail;
    g_api->ReleaseTypeInfo(info);
    info = NULL;
  }
  TRY(g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s->cpu));
  return s;

fail:
  if (info != NULL) g_api->ReleaseTypeInfo(info);
  if (so != NULL) g_api->ReleaseSessionOptions(so);
  jl_ort_session_release(s);
  return NULL;
}

size_t jl_ort_input_count(const jl_ort_session *s) { return s->n_in; }
size_t jl_ort_output_count(const jl_ort_session *s) { return s->n_out; }
const jl_ort_io *jl_ort_input(const jl_ort_session *s, size_t i) { return i < s->n_in ? &s->in[i] : NULL; }
const jl_ort_io *jl_ort_output(const jl_ort_session *s, size_t i) { return i < s->n_out ? &s->out[i] : NULL; }
const char *jl_ort_providers(const jl_ort_session *s) { return s->providers; }

static size_t elem_size(int32_t type) {
  switch (type) {
    case JL_ORT_FLOAT: case JL_ORT_INT32: return 4;
    case JL_ORT_INT64: return 8;
    case JL_ORT_FLOAT16: return 2;
    case JL_ORT_UINT8: case JL_ORT_BOOL: return 1;
    default: return 0;
  }
}

static int bind(jl_ort_session *s, const jl_ort_io *io, OrtValue **slot, void *data, size_t nbytes,
                char *err, size_t errlen) {
  size_t size = elem_size(io->elem_type);
  if (size == 0) {
    set_err(err, errlen, "unsupported tensor element type");
    return 1;
  }
  size_t count = 1;
  for (int32_t d = 0; d < io->rank; d++) {
    if (io->dims[d] <= 0) {
      set_err(err, errlen, "a tensor has a dynamic dimension; jetlink needs fixed shapes");
      return 1;
    }
    count *= (size_t)io->dims[d];
  }
  if (nbytes < count * size) {
    set_err(err, errlen, "the buffer is smaller than the tensor");
    return 1;
  }
  OrtValue *value = NULL;
  if (take(g_api->CreateTensorWithDataAsOrtValue(s->cpu, data, count * size, io->dims, (size_t)io->rank,
                                                  (ONNXTensorElementDataType)io->elem_type, &value),
           err, errlen)) {
    return 1;
  }
  if (*slot != NULL) g_api->ReleaseValue(*slot);
  *slot = value;
  return 0;
}

int jl_ort_bind_input(jl_ort_session *s, size_t i, void *data, size_t nbytes, char *err, size_t errlen) {
  if (i >= s->n_in) {
    set_err(err, errlen, "no such input");
    return 1;
  }
  return bind(s, &s->in[i], &s->in_values[i], data, nbytes, err, errlen);
}

int jl_ort_bind_output(jl_ort_session *s, size_t i, void *data, size_t nbytes, char *err, size_t errlen) {
  if (i >= s->n_out) {
    set_err(err, errlen, "no such output");
    return 1;
  }
  return bind(s, &s->out[i], &s->out_values[i], data, nbytes, err, errlen);
}

int jl_ort_run(jl_ort_session *s, char *err, size_t errlen) {
  for (size_t i = 0; i < s->n_in; i++) {
    if (s->in_values[i] == NULL) {
      set_err(err, errlen, "an input is not bound");
      return 1;
    }
  }
  for (size_t i = 0; i < s->n_out; i++) {
    if (s->out_values[i] == NULL) {
      set_err(err, errlen, "an output is not bound");
      return 1;
    }
  }
  // Pre-created outputs: onnxruntime writes into the bound buffers rather
  // than allocating a result per frame.
  return take(g_api->Run(s->session, NULL, s->in_names, (const OrtValue *const *)s->in_values, s->n_in,
                         s->out_names, s->n_out, s->out_values),
              err, errlen);
}

void jl_ort_session_release(jl_ort_session *s) {
  if (s == NULL) return;
  if (s->in_values) {
    for (size_t i = 0; i < s->n_in; i++) {
      if (s->in_values[i]) g_api->ReleaseValue(s->in_values[i]);
    }
  }
  if (s->out_values) {
    for (size_t i = 0; i < s->n_out; i++) {
      if (s->out_values[i]) g_api->ReleaseValue(s->out_values[i]);
    }
  }
  if (s->cpu) g_api->ReleaseMemoryInfo(s->cpu);
  if (s->session) g_api->ReleaseSession(s->session);
  free(s->in);
  free(s->out);
  free(s->in_names);
  free(s->out_names);
  free(s->in_values);
  free(s->out_values);
  free(s);
}
