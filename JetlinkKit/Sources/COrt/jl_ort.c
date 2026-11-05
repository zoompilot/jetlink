#include "jl_ort.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <onnxruntime/onnxruntime_c_api.h>

#ifdef JL_ORT_DLOPEN
#include <dlfcn.h>
#endif

struct jl_env {
  OrtEnv *env;
};

struct jl_session {
  OrtSession *session;
};

struct jl_binding {
  OrtSession *session;
  OrtIoBinding *binding;
  OrtValue **values;
  size_t n_values;
};

struct jl_shared_allocator {
  OrtSession *session;
  OrtAllocator *allocator;
  OrtMemoryInfo *memory;
  size_t total_bytes;
};

#ifdef JL_ORT_DLOPEN
// NULL when the library or its entry point is missing; jl_env_create says why.
static const OrtApiBase *base(void) {
  static const OrtApiBase *cached;
  if (cached == NULL) {
    void *library = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (library != NULL) {
      const OrtApiBase *(*get)(void) = (const OrtApiBase *(*)(void))dlsym(library, "OrtGetApiBase");
      if (get != NULL) {
        cached = get();
      }
    }
  }
  return cached;
}
#else
static const OrtApiBase *base(void) {
  return OrtGetApiBase();
}
#endif

static const OrtApi *api(void) {
  static const OrtApi *cached;
  if (cached == NULL && base() != NULL) {
    cached = base()->GetApi(ORT_API_VERSION);
  }
  return cached;
}

static char *copy(const char *s) {
  size_t n = strlen(s) + 1;
  char *out = malloc(n);
  if (out != NULL) {
    memcpy(out, s, n);
  }
  return out;
}

// NULL for no status, else the message, and the status released.
static char *take(OrtStatus *status) {
  if (status == NULL) {
    return NULL;
  }
  char *message = copy(api()->GetErrorMessage(status));
  api()->ReleaseStatus(status);
  return message != NULL ? message : copy("onnxruntime failed");
}

#define TRY(call)                    \
  do {                               \
    char *error_ = take(call);       \
    if (error_ != NULL) {            \
      return error_;                 \
    }                                \
  } while (0)

const char *jl_version(void) {
  return base() != NULL ? base()->GetVersionString() : "";
}

void jl_free(void *p) {
  free(p);
}

char *jl_env_create(int log_severity, jl_env **out) {
  *out = NULL;
#ifdef JL_ORT_DLOPEN
  if (base() == NULL) {
    const char *why = dlerror();
    return copy(why != NULL ? why : "cannot open libonnxruntime.so");
  }
#endif
  if (api() == NULL) {
    return copy("this onnxruntime does not provide the C API version jetlink was built against");
  }
  jl_env *env = calloc(1, sizeof(jl_env));
  if (env == NULL) {
    return copy("out of memory");
  }
  // The official Apple build carries Microsoft's events SDK and uploads to
  // mobile.events.data.microsoft.com (seen from the iPhone simulator). The
  // switch below is read when the environment is created, which is before
  // DisableTelemetryEvents can be called, and events logged in between were
  // still uploaded. A car has no business sending any.
  setenv("ORT_DISABLE_TELEMETRY", "1", 1);
  char *error = take(api()->CreateEnv((OrtLoggingLevel)log_severity, "jetlink", &env->env));
  if (error != NULL) {
    free(env);
    return error;
  }
  free(take(api()->DisableTelemetryEvents(env->env)));
  *out = env;
  return NULL;
}

void jl_env_release(jl_env *env) {
  if (env == NULL) {
    return;
  }
  api()->ReleaseEnv(env->env);
  free(env);
}

char *jl_session_create(jl_env *env, const char *model_path, const char *provider,
                        const jl_option *options, size_t n_options,
                        const jl_option *config, size_t n_config, int threads,
                        jl_session **out) {
  *out = NULL;
  OrtSessionOptions *so = NULL;
  TRY(api()->CreateSessionOptions(&so));
  char *error = take(api()->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
  if (error == NULL) {
    // errors only: the CoreML partitioner is chatty
    error = take(api()->SetSessionLogSeverityLevel(so, 3));
  }
  // One thread each way, and none that spin between frames: the CPU's share
  // of an accelerator's session is a couple of nodes, and a pool spinning for
  // the next frame burns a phone's battery and fights the frame path for cores.
  if (error == NULL) {
    error = take(api()->SetIntraOpNumThreads(so, threads > 0 ? threads : 1));
  }
  if (error == NULL) {
    error = take(api()->SetInterOpNumThreads(so, 1));
  }
  if (error == NULL) {
    error = take(api()->AddSessionConfigEntry(so, "session.intra_op.allow_spinning", "0"));
  }
  if (error == NULL) {
    error = take(api()->AddSessionConfigEntry(so, "session.inter_op.allow_spinning", "0"));
  }
  for (size_t i = 0; error == NULL && i < n_config; i++) {
    error = take(api()->AddSessionConfigEntry(so, config[i].key, config[i].value));
  }
  if (error == NULL && provider != NULL) {
    const char **keys = calloc(n_options + 1, sizeof(char *));
    const char **values = calloc(n_options + 1, sizeof(char *));
    if (keys == NULL || values == NULL) {
      error = copy("out of memory");
    } else {
      for (size_t i = 0; i < n_options; i++) {
        keys[i] = options[i].key;
        values[i] = options[i].value;
      }
      error = take(api()->SessionOptionsAppendExecutionProvider(so, provider, keys, values, n_options));
    }
    free(keys);
    free(values);
  }
  jl_session *session = NULL;
  if (error == NULL) {
    session = calloc(1, sizeof(jl_session));
    if (session == NULL) {
      error = copy("out of memory");
    }
  }
  if (error == NULL) {
    error = take(api()->CreateSession(env->env, model_path, so, &session->session));
  }
  api()->ReleaseSessionOptions(so);
  if (error != NULL) {
    free(session);
    return error;
  }
  *out = session;
  return NULL;
}

void jl_session_release(jl_session *session) {
  if (session == NULL) {
    return;
  }
  api()->ReleaseSession(session->session);
  free(session);
}

size_t jl_session_io_count(const jl_session *session, int output) {
  size_t n = 0;
  OrtStatus *status = output ? api()->SessionGetOutputCount(session->session, &n)
                             : api()->SessionGetInputCount(session->session, &n);
  free(take(status));
  return n;
}

char *jl_session_io_info(const jl_session *session, int output, size_t index,
                         char *name, size_t name_cap, int *elem_type,
                         int64_t *dims, size_t dims_cap, size_t *rank) {
  OrtAllocator *allocator = NULL;
  TRY(api()->GetAllocatorWithDefaultOptions(&allocator));
  char *raw = NULL;
  TRY(output ? api()->SessionGetOutputName(session->session, index, allocator, &raw)
             : api()->SessionGetInputName(session->session, index, allocator, &raw));
  if (name_cap > 0) {
    strncpy(name, raw, name_cap - 1);
    name[name_cap - 1] = '\0';
  }
  free(take(api()->AllocatorFree(allocator, raw)));

  OrtTypeInfo *info = NULL;
  TRY(output ? api()->SessionGetOutputTypeInfo(session->session, index, &info)
             : api()->SessionGetInputTypeInfo(session->session, index, &info));
  const OrtTensorTypeAndShapeInfo *tensor = NULL;
  char *error = take(api()->CastTypeInfoToTensorInfo(info, &tensor));
  if (error == NULL && tensor == NULL) {
    error = copy("not a tensor");
  }
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t n = 0;
  if (error == NULL) {
    error = take(api()->GetTensorElementType(tensor, &type));
  }
  if (error == NULL) {
    error = take(api()->GetDimensionsCount(tensor, &n));
  }
  if (error == NULL) {
    int64_t *all = calloc(n > 0 ? n : 1, sizeof(int64_t));
    if (all == NULL) {
      error = copy("out of memory");
    } else {
      error = take(api()->GetDimensions(tensor, all, n));
      for (size_t i = 0; error == NULL && i < n && i < dims_cap; i++) {
        dims[i] = all[i];
      }
      free(all);
    }
  }
  api()->ReleaseTypeInfo(info);
  if (error != NULL) {
    return error;
  }
  *elem_type = (int)type;
  *rank = n;
  return NULL;
}

void jl_binding_release(jl_binding *binding) {
  if (binding == NULL) {
    return;
  }
  if (binding->binding != NULL) {
    api()->ReleaseIoBinding(binding->binding);
  }
  for (size_t i = 0; i < binding->n_values; i++) {
    if (binding->values[i] != NULL) {
      api()->ReleaseValue(binding->values[i]);
    }
  }
  free(binding->values);
  free(binding);
}

void jl_shared_allocator_release(jl_shared_allocator *allocator) {
  if (allocator == NULL) return;
  if (allocator->allocator != NULL) api()->ReleaseAllocator(allocator->allocator);
  if (allocator->memory != NULL) api()->ReleaseMemoryInfo(allocator->memory);
  free(allocator);
}

char *jl_shared_allocator_create(jl_session *session, jl_shared_allocator **out) {
  if (session == NULL || out == NULL) return copy("shared allocator needs a session and output");
  *out = NULL;
  jl_shared_allocator *allocator = calloc(1, sizeof(*allocator));
  if (allocator == NULL) return copy("out of memory");
  allocator->session = session->session;
  char *error = take(api()->CreateMemoryInfo("QnnHtpShared", OrtDeviceAllocator, 0,
                                             OrtMemTypeDefault, &allocator->memory));
  if (error == NULL) error = take(api()->CreateAllocator(session->session, allocator->memory, &allocator->allocator));
  const OrtMemoryInfo *actual = NULL;
  if (error == NULL) error = take(api()->AllocatorGetInfo(allocator->allocator, &actual));
  int different = 1;
  if (error == NULL) error = take(api()->CompareMemoryInfo(allocator->memory, actual, &different));
  if (error == NULL && different != 0) error = copy("session did not return QNN shared memory");
  if (error != NULL) {
    jl_shared_allocator_release(allocator);
    return error;
  }
  *out = allocator;
  return NULL;
}

char *jl_shared_allocator_alloc(jl_shared_allocator *allocator, size_t bytes, void **out) {
  if (allocator == NULL || out == NULL) return copy("invalid shared allocation");
  *out = NULL;
  if (bytes > SIZE_MAX - allocator->total_bytes) return copy("shared allocation byte count overflow");
  char *error = take(api()->AllocatorAlloc(allocator->allocator, bytes, out));
  if (error == NULL && *out == NULL) error = copy("QNN shared allocator returned no memory");
  if (error == NULL) allocator->total_bytes += bytes;
  return error;
}

void jl_shared_allocator_free(jl_shared_allocator *allocator, void *data) {
  if (allocator == NULL || data == NULL) return;
  char *error = take(api()->AllocatorFree(allocator->allocator, data));
  if (error != NULL) {
    fprintf(stderr, "Jetlink QNN shared memory free: %s\n", error);
    free(error);
  }
}

size_t jl_shared_allocator_total_bytes(const jl_shared_allocator *allocator) {
  return allocator == NULL ? 0 : allocator->total_bytes;
}

static char *wrap(const OrtMemoryInfo *memory, const jl_tensor *t, OrtValue **out) {
  return take(api()->CreateTensorWithDataAsOrtValue(memory, t->data, t->nbytes, t->dims, t->rank,
                                                    (ONNXTensorElementDataType)t->elem_type, out));
}

char *jl_binding_create(jl_session *session, const jl_tensor *inputs, size_t n_inputs,
                        const jl_tensor *outputs, size_t n_outputs,
                        const jl_shared_allocator *shared, jl_binding **out) {
  *out = NULL;
  if (shared != NULL && shared->session != session->session) {
    // ORT's QNN allocator tracks RPC allocations globally. A second HTP
    // context can register the same host-accessible allocation. Verify that
    // this target session really exposes QnnHtpShared; CPU/GPU may not claim it.
    jl_shared_allocator *compatible = NULL;
    char *error = jl_shared_allocator_create(session, &compatible);
    if (error != NULL) return error;
    jl_shared_allocator_release(compatible);
  }
  jl_binding *binding = calloc(1, sizeof(jl_binding));
  if (binding == NULL) {
    return copy("out of memory");
  }
  binding->session = session->session;
  binding->values = calloc(n_inputs + n_outputs + 1, sizeof(OrtValue *));
  if (binding->values == NULL) {
    free(binding);
    return copy("out of memory");
  }
  OrtMemoryInfo *cpu_memory = NULL;
  const OrtMemoryInfo *memory = shared != NULL ? shared->memory : NULL;
  char *error = NULL;
  if (memory == NULL) {
    error = take(api()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory));
    memory = cpu_memory;
  }
  if (error == NULL) {
    error = take(api()->CreateIoBinding(session->session, &binding->binding));
  }
  for (size_t i = 0; error == NULL && i < n_inputs; i++) {
    OrtValue **value = &binding->values[binding->n_values++];
    error = wrap(memory, &inputs[i], value);
    if (error == NULL) {
      error = take(api()->BindInput(binding->binding, inputs[i].name, *value));
    }
  }
  for (size_t i = 0; error == NULL && i < n_outputs; i++) {
    OrtValue **value = &binding->values[binding->n_values++];
    error = wrap(memory, &outputs[i], value);
    if (error == NULL) {
      error = take(api()->BindOutput(binding->binding, outputs[i].name, *value));
    }
  }
  if (cpu_memory != NULL) {
    api()->ReleaseMemoryInfo(cpu_memory);
  }
  if (error != NULL) {
    jl_binding_release(binding);
    return error;
  }
  *out = binding;
  return NULL;
}

char *jl_binding_run(jl_binding *binding) {
  return take(api()->RunWithBinding(binding->session, NULL, binding->binding));
}
