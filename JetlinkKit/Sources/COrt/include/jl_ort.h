// A few plain C calls over onnxruntime's C API, so Swift never walks OrtApi's
// table of function pointers. Only what the server needs: sessions with an
// execution provider (CoreML on Apple, QNN on Android), their inputs and
// outputs, and runs over caller-owned memory.
//
// Built with JL_ORT_DLOPEN (Android), the runtime is opened at run time from
// libonnxruntime.so, the app's copy from the onnxruntime AAR, so the build
// needs its headers only.
//
// Every call that can fail returns NULL on success, or a message the caller
// frees with jl_free.
#ifndef JL_ORT_H
#define JL_ORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ONNXTensorElementDataType, as ONNX numbers them.
enum {
  JL_FLOAT = 1,
  JL_UINT8 = 2,
  JL_INT8 = 3,
  JL_UINT16 = 4,
  JL_INT16 = 5,
  JL_INT32 = 6,
  JL_INT64 = 7,
  JL_BOOL = 9,
  JL_FLOAT16 = 10,
  JL_DOUBLE = 11,
};

typedef struct jl_env jl_env;
typedef struct jl_session jl_session;
typedef struct jl_binding jl_binding;
typedef struct jl_shared_allocator jl_shared_allocator;

typedef struct {
  const char *key;
  const char *value;
} jl_option;

typedef struct {
  const char *name;
  int elem_type;
  const int64_t *dims;
  size_t rank;
  void *data;
  size_t nbytes;
} jl_tensor;

// The runtime's version string, "1.29.0", or "" when it cannot be opened.
const char *jl_version(void);

char *jl_env_create(int log_severity, jl_env **out);
void jl_env_release(jl_env *env);

// provider is "CoreML" or "QNN" with its options, or NULL for the CPU alone.
// config holds session config entries ("ep.context_enable" and the like),
// added after jetlink's own. threads is the intra-op pool: 1 where an
// accelerator does the work.
char *jl_session_create(jl_env *env, const char *model_path, const char *provider,
                        const jl_option *options, size_t n_options,
                        const jl_option *config, size_t n_config, int threads,
                        jl_session **out);
void jl_session_release(jl_session *session);

// output 0 counts inputs, 1 outputs.
size_t jl_session_io_count(const jl_session *session, int output);

// One input's or output's name (NUL terminated, truncated to name_cap), element
// type and shape. *rank is the true rank; at most dims_cap dims are written. A
// symbolic dimension is -1.
char *jl_session_io_info(const jl_session *session, int output, size_t index,
                         char *name, size_t name_cap, int *elem_type,
                         int64_t *dims, size_t dims_cap, size_t *rank);

// QNN's host-accessible RPC memory. The session must outlive this allocator,
// and every allocation and binding must be freed before releasing it.
char *jl_shared_allocator_create(jl_session *session, jl_shared_allocator **out);
char *jl_shared_allocator_alloc(jl_shared_allocator *allocator, size_t bytes, void **out);
void jl_shared_allocator_free(jl_shared_allocator *allocator, void *data);
void jl_shared_allocator_release(jl_shared_allocator *allocator);
size_t jl_shared_allocator_total_bytes(const jl_shared_allocator *allocator);

// Binds caller-owned buffers to a session's inputs and outputs once, for
// jl_binding_run to run with. The buffers must outlive the binding.
char *jl_binding_create(jl_session *session, const jl_tensor *inputs, size_t n_inputs,
                        const jl_tensor *outputs, size_t n_outputs,
                        const jl_shared_allocator *shared, jl_binding **out);
char *jl_binding_run(jl_binding *binding);
void jl_binding_release(jl_binding *binding);

void jl_free(void *p);

#ifdef __cplusplus
}
#endif

#endif
