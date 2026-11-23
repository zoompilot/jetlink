// jl_trt.h over host memory, for tests: jl_trt_fake.h says what it models and
// which rules it keeps. Everything runs under one lock per jl_trt, a recursive
// one, so a log or progress callback may call back in.

// strtok_r, strdup and recursive mutexes are POSIX, not C11
#define _XOPEN_SOURCE 700

#include "jl_trt.h"
#include "jl_trt_fake.h"
#include "jl_trt_util.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>

#define MAX_TENSORS 32
#define MAX_FAILS 16
// What each enqueue adds to the clock timing events read, and what
// jl_trt_mem_info reports.
#define ENQUEUE_MS 1.0
#define TOTAL_MEMORY ((size_t)8 << 30)

enum { OBJ_DEVICE, OBJ_HOST, OBJ_STREAM, OBJ_EVENT, OBJ_GRAPH, OBJ_EXEC, OBJ_ENGINE, OBJ_CONTEXT, OBJ_BUILD };

enum { OP_H2D, OP_D2H, OP_D2D, OP_MEMSET, OP_ENQUEUE, OP_RECORD };

typedef struct {
  int kind;
  uintptr_t dst, src;
  size_t size;
  uint8_t value;
  // the context an enqueue runs, or the event a record marks
  void *target;
} op;

typedef struct {
  op *ops;
  size_t n, cap;
} op_list;

typedef struct obj {
  int kind;
  void *ptr;
  size_t size;
  struct obj *next;
} obj;

typedef struct {
  char call[48];
  int nth, code;
  char message[256];
} fail;

struct jl_trt {
  pthread_mutex_t lock;
  jl_trt_fake_config config;
  char device_name[128];
  int sticky;
  char sticky_message[320];
  jl_trt_log_fn log;
  void *log_ctx;
  int log_min;
  fail fails[MAX_FAILS];
  int n_fails;
  jl_trt_fake_stats stats;
  obj *live;
  // one capture at a time, and the thread that began it
  jl_trt_stream *capturing;
  pthread_t capture_thread;
  double clock_ms;
  char *build_tensors;
  int build_layers;
};

// Streams, graphs and execs each start with their op list.
struct jl_trt_stream {
  op_list ops;
  int capturing, invalidated;
};

struct jl_trt_event {
  unsigned flags;
  int recorded;
  // recorded inside a capture without JL_TRT_RECORD_EXTERNAL: not the host's
  int internal;
  double stamp;
};

struct jl_trt_graph {
  op_list ops;
};

struct jl_trt_graph_exec {
  op_list ops;
};

typedef struct {
  char name[64];
  int is_input, type, rank;
  int64_t dims[JL_TRT_MAX_DIMS];
  // the input a next_state is fed from, or -1
  int from;
} tensor;

struct jl_trt_engine {
  jl_trt *trt;
  int n;
  tensor tensors[MAX_TENSORS];
};

struct jl_trt_context {
  // its own, since a misused context can outlive its engine
  jl_trt *trt;
  jl_trt_engine *engine;
  jl_trt_dptr address[MAX_TENSORS];
  int ran;
};

struct jl_trt_build {
  jl_trt *trt;
  int parsed, fp16, optimization_level;
  size_t workspace;
  jl_trt_progress_fn progress;
  void *progress_ctx;
  // -1: no timing cache attached; else the builds it holds
  int cache_builds;
  // jl_trt_build_cancel: the build ends at its next step, as TensorRT's does
  int cancelled;
};

static const char *const default_tensors = "input x float16 1 8\n"
                                           "input state float16 1 8\n"
                                           "output y float32 1 8\n"
                                           "output next_state float16 1 8 from state\n";

// --- errors, the lock, injected failures -------------------------------------

static int leave(jl_trt *t, int code) {
  pthread_mutex_unlock(&t->lock);
  return code;
}

// What TensorRT logs before it returns NULL or false (an error), or a
// warning; either goes to the logger and, as JL_TRT_ERROR, into err.
static int trt_log(jl_trt *t, int severity, char *err, size_t errlen, const char *fmt, ...) {
  char line[320];
  va_list args;
  va_start(args, fmt);
  vsnprintf(line, sizeof line, fmt, args);
  va_end(args);
  if (t->log != NULL && severity <= t->log_min) {
    t->log(t->log_ctx, severity, line);
  }
  return say(err, errlen, JL_TRT_ERROR, "%s", line);
}

// Latches the sticky flag with why, for a GPU fault or a rule broken where
// there is no error to return.
static void latch(jl_trt *t, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(t->sticky_message, sizeof t->sticky_message, fmt, args);
  va_end(args);
  t->sticky = 1;
}

// A GPU fault: what a graph replaying into freed memory gets.
static int fault(jl_trt *t, const char *call, const char *why, char *err, size_t errlen) {
  latch(t, "CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered (fake: %s)", why);
  return say(err, errlen, JL_TRT_CUDA_STICKY, "%s: %s", call, t->sticky_message);
}

// Takes the lock. Anything but JL_TRT_OK has released it again, with err
// written: the latched sticky error, or a failure injected for this call.
// Returning the latch first is also what the real shim's probe after a failed
// deserialize or build gives.
static int enter(jl_trt *t, const char *call, char *err, size_t errlen) {
  pthread_mutex_lock(&t->lock);
  if (t->sticky) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_STICKY, "%s: %s (latched)", call, t->sticky_message));
  }
  for (int i = 0; i < t->n_fails; i++) {
    fail *f = &t->fails[i];
    if (strcmp(f->call, call) != 0 || --f->nth > 0) {
      continue;
    }
    int code = f->code;
    say(err, errlen, code, "%s: %s", call, f->message);
    if (code == JL_TRT_CUDA_STICKY) {
      latch(t, "%s", f->message);
    }
    t->fails[i] = t->fails[--t->n_fails];
    return leave(t, code);
  }
  return JL_TRT_OK;
}

// enter, or return what it said
#define ENTER(t, call)                              \
  do {                                              \
    int entered_ = enter((t), (call), err, errlen); \
    if (entered_ != JL_TRT_OK) {                    \
      return entered_;                              \
    }                                               \
  } while (0)

// A call CUDA forbids on the thread that is capturing: it fails, and so does
// the capture.
static int unsafe(jl_trt *t, const char *call, char *err, size_t errlen) {
  if (t->capturing == NULL || !pthread_equal(t->capture_thread, pthread_self())) {
    return JL_TRT_OK;
  }
  t->capturing->invalidated = 1;
  return say(err, errlen, JL_TRT_CUDA_ERROR,
             "%s: CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: operation not permitted when stream is capturing", call);
}

static int bad_handle(const char *call, char *err, size_t errlen) {
  return say(err, errlen, JL_TRT_CUDA_ERROR, "%s: CUDA_ERROR_INVALID_HANDLE: invalid resource handle", call);
}

static int invalid(const char *call, const char *why, char *err, size_t errlen) {
  return say(err, errlen, JL_TRT_CUDA_ERROR, "%s: CUDA_ERROR_INVALID_VALUE: invalid argument (fake: %s)", call, why);
}

// --- live objects ----------------------------------------------------------

static void track(jl_trt *t, int kind, void *ptr, size_t size) {
  obj *o = calloc(1, sizeof(obj));
  if (o == NULL) {
    abort();
  }
  *o = (obj){kind, ptr, size, t->live};
  t->live = o;
}

// Whether [addr, addr + size) lies inside one live allocation of `kind`; a
// handle is live when its own address is, with no size.
static int inside(jl_trt *t, int kind, uintptr_t addr, size_t size) {
  for (obj *o = t->live; o != NULL; o = o->next) {
    uintptr_t base = (uintptr_t)o->ptr;
    if (o->kind == kind && addr >= base && size <= o->size && addr - base <= o->size - size) {
      return 1;
    }
  }
  return 0;
}

#define live(t, kind, ptr) inside((t), (kind), (uintptr_t)(ptr), 0)

static void push(op_list *list, op o) {
  if (list->n == list->cap) {
    size_t cap = list->cap ? list->cap * 2 : 16;
    op *ops = realloc(list->ops, cap * sizeof(op));
    if (ops == NULL) {
      abort();
    }
    list->ops = ops;
    list->cap = cap;
  }
  list->ops[list->n++] = o;
}

static void free_ops(op_list *list) {
  free(list->ops);
  *list = (op_list){NULL, 0, 0};
}

static void destroy_obj(obj *o) {
  if (o->kind == OBJ_STREAM || o->kind == OBJ_GRAPH || o->kind == OBJ_EXEC) {
    free_ops(o->ptr);
  }
  free(o->ptr);
  free(o);
}

// Frees a live object. Destroying one that is not live latches: the real
// stack would fail quietly or crash.
static void destroy(jl_trt *t, int kind, void *ptr, const char *what) {
  if (t == NULL || ptr == NULL) {
    return;
  }
  pthread_mutex_lock(&t->lock);
  for (obj **link = &t->live; *link != NULL; link = &(*link)->next) {
    if ((*link)->kind == kind && (*link)->ptr == ptr) {
      obj *o = *link;
      *link = o->next;
      if (t->capturing == ptr) {
        t->capturing = NULL;
      }
      destroy_obj(o);
      pthread_mutex_unlock(&t->lock);
      return;
    }
  }
  latch(t, "fake: destroying %s that is not live", what);
  pthread_mutex_unlock(&t->lock);
}

// --- element types -----------------------------------------------------------

// The two types jetlink's engines take and give.
static size_t type_size(int type) {
  return type == JL_TRT_FLOAT ? 4 : 2;
}

static float half_to_float(uint16_t h) {
  int exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
  float f = exp == 31 ? (mant ? NAN : INFINITY) : ldexpf((float)(exp ? mant | 0x400 : mant), (exp ? exp : 1) - 25);
  return (h & 0x8000u) ? -f : f;
}

static double load(int type, const void *base, size_t index) {
  const uint8_t *p = (const uint8_t *)base + index * type_size(type);
  float f;
  uint16_t h;
  if (type == JL_TRT_FLOAT) {
    memcpy(&f, p, sizeof f);
    return f;
  }
  memcpy(&h, p, sizeof h);
  return half_to_float(h);
}

static void store(int type, void *base, size_t index, double value) {
  float f = (float)value;
  uint16_t h = float_to_half(f);
  memcpy((uint8_t *)base + index * type_size(type), type == JL_TRT_FLOAT ? (const void *)&f : (const void *)&h, type_size(type));
}

static size_t tensor_count(const tensor *x) {
  size_t n = 1;
  for (int i = 0; i < x->rank; i++) {
    n *= x->dims[i] < 0 ? 0 : (size_t)x->dims[i];
  }
  return n;
}

// --- running the engine and the work on a stream ------------------------------

// Whether every tensor has an address.
static int check_bound(jl_trt *t, jl_trt_context *c, char *err, size_t errlen) {
  jl_trt_engine *e = c->engine;
  for (int i = 0; i < e->n; i++) {
    if (c->address[i] == 0) {
      return trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "IExecutionContext::enqueueV3: tensor %s has no address", e->tensors[i].name);
    }
  }
  return JL_TRT_OK;
}

static int run_engine(jl_trt *t, jl_trt_context *c, char *err, size_t errlen) {
  if (!live(t, OBJ_CONTEXT, c)) {
    return fault(t, "enqueueV3", "a graph ran a destroyed execution context", err, errlen);
  }
  int rc = check_bound(t, c, err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  jl_trt_engine *e = c->engine;
  for (int i = 0; i < e->n; i++) {
    const tensor *x = &e->tensors[i];
    if (!inside(t, OBJ_DEVICE, (uintptr_t)c->address[i], tensor_count(x) * type_size(x->type))) {
      return fault(t, "enqueueV3", "a tensor is not inside live device memory", err, errlen);
    }
  }
  for (int o = 0; o < e->n; o++) {
    const tensor *out = &e->tensors[o];
    for (size_t j = 0; !out->is_input && j < tensor_count(out); j++) {
      double v = 0;
      for (int i = 0; i < e->n; i++) {
        const tensor *in = &e->tensors[i];
        if (in->is_input && (out->from < 0 || out->from == i)) {
          v += load(in->type, (void *)(uintptr_t)c->address[i], j % tensor_count(in));
        }
      }
      store(out->type, (void *)(uintptr_t)c->address[o], j, out->from >= 0 ? v + 1 : v);
    }
  }
  c->ran = 1;
  t->clock_ms += ENQUEUE_MS;
  t->stats.enqueues++;
  return JL_TRT_OK;
}

static const char *const op_calls[] = {"cuMemcpyHtoDAsync", "cuMemcpyDtoHAsync", "cuMemcpyDtoDAsync",
                                       "cuMemsetD8Async",   "enqueueV3",         "cuEventRecordWithFlags"};

// Does one piece of work. `replay`: from a graph, whose addresses were baked
// in, so a bad one is a GPU fault rather than a refused call.
static int execute(jl_trt *t, const op *o, int replay, char *err, size_t errlen) {
  const char *call = op_calls[o->kind];
  if (o->kind == OP_ENQUEUE) {
    return run_engine(t, o->target, err, errlen);
  }
  if (o->kind == OP_RECORD) {
    jl_trt_event *event = o->target;
    if (!live(t, OBJ_EVENT, event)) {
      return fault(t, call, "a graph recorded a destroyed event", err, errlen);
    }
    *event = (jl_trt_event){event->flags, 1, 0, t->clock_ms};
    return JL_TRT_OK;
  }
  int dst_device = o->kind != OP_D2H, src_device = o->kind == OP_D2H || o->kind == OP_D2D;
  const char *why = NULL;
  if ((dst_device && !inside(t, OBJ_DEVICE, o->dst, o->size)) || (src_device && !inside(t, OBJ_DEVICE, o->src, o->size))) {
    why = "device memory that is not allocated";
  } else if ((o->kind == OP_H2D && !inside(t, OBJ_HOST, o->src, o->size)) ||
             (o->kind == OP_D2H && !inside(t, OBJ_HOST, o->dst, o->size))) {
    why = "host memory jl_trt_host_alloc did not return";
  }
  if (why != NULL) {
    return replay ? fault(t, call, why, err, errlen) : invalid(call, why, err, errlen);
  }
  if (o->kind == OP_MEMSET) {
    memset((void *)o->dst, o->value, o->size);
    t->stats.memsets++;
  } else {
    memmove((void *)o->dst, (const void *)o->src, o->size);
    (*(o->kind == OP_H2D ? &t->stats.h2d : o->kind == OP_D2H ? &t->stats.d2h : &t->stats.d2d))++;
  }
  return JL_TRT_OK;
}

// Queues work on a stream: recorded while it captures, done at once otherwise.
static int submit(jl_trt *t, jl_trt_stream *s, op o, char *err, size_t errlen) {
  if (!live(t, OBJ_STREAM, s)) {
    return bad_handle(op_calls[o.kind], err, errlen);
  }
  if (!s->capturing) {
    return execute(t, &o, 0, err, errlen);
  }
  if (s->invalidated) {
    return say(err, errlen, JL_TRT_CUDA_ERROR,
               "%s: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: operation failed due to a previous error during capture",
               op_calls[o.kind]);
  }
  push(&s->ops, o);
  return JL_TRT_OK;
}

// --- library --------------------------------------------------------------------

void jl_trt_fake_defaults(jl_trt_fake_config *c) {
  *c = (jl_trt_fake_config){
      .major = 10, .minor = 3, .build = 30, .cuda_driver = 12060, .plugins = 1, .device_name = "Orin", .cc_major = 8, .cc_minor = 7};
}

int jl_trt_open(int device, const char *lib_dir, jl_trt **out, char *err, size_t errlen) {
  (void)device;
  *out = NULL;
  // naming the directory, so a test sees it passed through
  return say(err, errlen, JL_TRT_UNAVAILABLE, "this is the fake shim, built without TensorRT%s%s", lib_dir ? "; asked for it in " : "",
             lib_dir ? lib_dir : "");
}

int jl_trt_fake_open(const jl_trt_fake_config *config, jl_trt **out, char *err, size_t errlen) {
  *out = NULL;
  jl_trt *t = calloc(1, sizeof(jl_trt));
  if (t == NULL) {
    return say(err, errlen, JL_TRT_ERROR, "out of memory");
  }
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&t->lock, &attr);
  pthread_mutexattr_destroy(&attr);
  if (config != NULL) {
    t->config = *config;
  } else {
    jl_trt_fake_defaults(&t->config);
  }
  snprintf(t->device_name, sizeof t->device_name, "%s", t->config.device_name ? t->config.device_name : "");
  t->config.device_name = t->device_name;
  t->build_layers = 3;
  *out = t;
  return JL_TRT_OK;
}

void jl_trt_close(jl_trt *t) {
  if (t == NULL) {
    return;
  }
  while (t->live != NULL) {
    obj *o = t->live;
    t->live = o->next;
    destroy_obj(o);
  }
  free(t->build_tensors);
  pthread_mutex_destroy(&t->lock);
  free(t);
}

void jl_trt_get_info(const jl_trt *t, jl_trt_info *out) {
  const jl_trt_fake_config *c = &t->config;
  *out = (jl_trt_info){c->major,  c->minor,       c->patch,    c->build, c->strongly_typed, c->cuda_driver,
                       c->plugins, t->device_name, c->cc_major, c->cc_minor, "the fake shim"};
}

int jl_trt_mem_info(jl_trt *t, size_t *free_bytes, size_t *total_bytes, char *err, size_t errlen) {
  ENTER(t, "mem_info");
  *free_bytes = *total_bytes = TOTAL_MEMORY;
  return leave(t, JL_TRT_OK);
}

void jl_trt_set_logger(jl_trt *t, int min_severity, jl_trt_log_fn fn, void *ctx) {
  pthread_mutex_lock(&t->lock);
  t->log = fn;
  t->log_ctx = ctx;
  t->log_min = min_severity;
  pthread_mutex_unlock(&t->lock);
}

int jl_trt_sticky(const jl_trt *t) {
  jl_trt *m = (jl_trt *)t;
  pthread_mutex_lock(&m->lock);
  int sticky = m->sticky;
  pthread_mutex_unlock(&m->lock);
  return sticky;
}

void jl_trt_fake_fail(jl_trt *t, const char *call, int nth, int code, const char *message) {
  pthread_mutex_lock(&t->lock);
  if (t->n_fails < MAX_FAILS) {
    fail *f = &t->fails[t->n_fails++];
    snprintf(f->call, sizeof f->call, "%s", call);
    f->nth = nth < 1 ? 1 : nth;
    f->code = code;
    snprintf(f->message, sizeof f->message, "%s", message ? message : "injected");
  }
  pthread_mutex_unlock(&t->lock);
}

void jl_trt_fake_set_build(jl_trt *t, const char *tensor_lines, int layers) {
  pthread_mutex_lock(&t->lock);
  free(t->build_tensors);
  t->build_tensors = tensor_lines ? strdup(tensor_lines) : NULL;
  t->build_layers = layers;
  pthread_mutex_unlock(&t->lock);
}

void jl_trt_fake_get_stats(jl_trt *t, jl_trt_fake_stats *out) {
  pthread_mutex_lock(&t->lock);
  *out = t->stats;
  int64_t *counts[] = {&out->device_allocs, &out->host_allocs, &out->streams, &out->events, &out->graphs,
                       &out->graph_execs,   &out->engines,     &out->contexts, &out->builds};
  for (obj *o = t->live; o != NULL; o = o->next) {
    (*counts[o->kind])++;
  }
  pthread_mutex_unlock(&t->lock);
}

// --- memory ------------------------------------------------------------------------

static int alloc(jl_trt *t, const char *call, int kind, size_t size, void **out, char *err, size_t errlen) {
  *out = NULL;
  ENTER(t, call);
  const char *cu = kind == OBJ_DEVICE ? "cuMemAlloc" : "cuMemHostAlloc";
  int rc = unsafe(t, cu, err, errlen);
  if (rc != JL_TRT_OK || size == 0) {
    return leave(t, rc != JL_TRT_OK ? rc : invalid(cu, "zero bytes", err, errlen));
  }
  void *p = malloc(size);
  if (p == NULL) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "%s: CUDA_ERROR_OUT_OF_MEMORY: out of memory", cu));
  }
  memset(p, 0xff, size);
  track(t, kind, p, size);
  *out = p;
  return leave(t, JL_TRT_OK);
}

int jl_trt_mem_alloc(jl_trt *t, size_t size, jl_trt_dptr *out, char *err, size_t errlen) {
  void *p;
  int rc = alloc(t, "mem_alloc", OBJ_DEVICE, size, &p, err, errlen);
  *out = (jl_trt_dptr)(uintptr_t)p;
  return rc;
}

void jl_trt_mem_free(jl_trt *t, jl_trt_dptr ptr) {
  destroy(t, OBJ_DEVICE, (void *)(uintptr_t)ptr, "device memory");
}

int jl_trt_host_alloc(jl_trt *t, size_t size, void **out, char *err, size_t errlen) {
  return alloc(t, "host_alloc", OBJ_HOST, size, out, err, errlen);
}

void jl_trt_host_free(jl_trt *t, void *ptr) {
  destroy(t, OBJ_HOST, ptr, "host memory");
}

static int queue(jl_trt *t, const char *call, jl_trt_stream *s, op o, char *err, size_t errlen) {
  ENTER(t, call);
  return leave(t, submit(t, s, o, err, errlen));
}

int jl_trt_copy_h2d(jl_trt *t, jl_trt_dptr dst, const void *src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  return queue(t, "copy_h2d", stream, (op){OP_H2D, (uintptr_t)dst, (uintptr_t)src, size, 0, NULL}, err, errlen);
}

int jl_trt_copy_d2h(jl_trt *t, void *dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  return queue(t, "copy_d2h", stream, (op){OP_D2H, (uintptr_t)dst, (uintptr_t)src, size, 0, NULL}, err, errlen);
}

int jl_trt_copy_d2d(jl_trt *t, jl_trt_dptr dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  return queue(t, "copy_d2d", stream, (op){OP_D2D, (uintptr_t)dst, (uintptr_t)src, size, 0, NULL}, err, errlen);
}

int jl_trt_memset(jl_trt *t, jl_trt_dptr dst, uint8_t value, size_t size, jl_trt_stream *stream, char *err,
                  size_t errlen) {
  return queue(t, "memset", stream, (op){OP_MEMSET, (uintptr_t)dst, 0, size, value, NULL}, err, errlen);
}

// --- streams and events ----------------------------------------------------------

// A new zeroed object of `kind`, tracked, under the lock.
static int make(jl_trt *t, int kind, size_t size, void **out, char *err, size_t errlen) {
  *out = calloc(1, size);
  if (*out == NULL) {
    return say(err, errlen, JL_TRT_CUDA_ERROR, "CUDA_ERROR_OUT_OF_MEMORY: out of memory");
  }
  track(t, kind, *out, 0);
  return JL_TRT_OK;
}

int jl_trt_stream_create(jl_trt *t, jl_trt_stream **out, char *err, size_t errlen) {
  ENTER(t, "stream_create");
  return leave(t, make(t, OBJ_STREAM, sizeof(jl_trt_stream), (void **)out, err, errlen));
}

int jl_trt_stream_sync(jl_trt *t, jl_trt_stream *s, char *err, size_t errlen) {
  ENTER(t, "stream_sync");
  int rc = unsafe(t, "cuStreamSynchronize", err, errlen);
  if (rc == JL_TRT_OK && !live(t, OBJ_STREAM, s)) {
    rc = bad_handle("cuStreamSynchronize", err, errlen);
  }
  return leave(t, rc);
}

void jl_trt_stream_destroy(jl_trt *t, jl_trt_stream *s) {
  destroy(t, OBJ_STREAM, s, "a stream");
}

int jl_trt_event_create(jl_trt *t, unsigned flags, jl_trt_event **out, char *err, size_t errlen) {
  ENTER(t, "event_create");
  int rc = make(t, OBJ_EVENT, sizeof(jl_trt_event), (void **)out, err, errlen);
  if (rc == JL_TRT_OK) {
    (*out)->flags = flags;
  }
  return leave(t, rc);
}

int jl_trt_event_record(jl_trt *t, jl_trt_event *event, jl_trt_stream *s, unsigned flags, char *err, size_t errlen) {
  ENTER(t, "event_record");
  if (!live(t, OBJ_EVENT, event)) {
    return leave(t, bad_handle("cuEventRecordWithFlags", err, errlen));
  }
  if (live(t, OBJ_STREAM, s) && s->capturing && !(flags & JL_TRT_RECORD_EXTERNAL)) {
    // a dependency inside the graph, not something the host can wait on
    event->internal = !s->invalidated;
    return leave(t, JL_TRT_OK);
  }
  return leave(t, submit(t, s, (op){OP_RECORD, 0, 0, 0, 0, event}, err, errlen));
}

// Everything here is finished by the time it returns, so a wait is a check.
static int host_wait(jl_trt *t, const char *call, jl_trt_event *event, char *err, size_t errlen) {
  int rc = unsafe(t, call, err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (!live(t, OBJ_EVENT, event)) {
    return bad_handle(call, err, errlen);
  }
  return event->internal ? invalid(call, "recorded inside a capture without JL_TRT_RECORD_EXTERNAL", err, errlen) : JL_TRT_OK;
}

int jl_trt_event_sync(jl_trt *t, jl_trt_event *event, char *err, size_t errlen) {
  ENTER(t, "event_sync");
  return leave(t, host_wait(t, "cuEventSynchronize", event, err, errlen));
}

int jl_trt_event_elapsed(jl_trt *t, jl_trt_event *start, jl_trt_event *end, float *ms, char *err, size_t errlen) {
  ENTER(t, "event_elapsed");
  if (!live(t, OBJ_EVENT, start) || !live(t, OBJ_EVENT, end) ||
      ((start->flags | end->flags) & JL_TRT_EVENT_DISABLE_TIMING) || !start->recorded || !end->recorded) {
    return leave(t, bad_handle("cuEventElapsedTime", err, errlen));
  }
  *ms = (float)(end->stamp - start->stamp);
  return leave(t, JL_TRT_OK);
}

void jl_trt_event_destroy(jl_trt *t, jl_trt_event *event) {
  destroy(t, OBJ_EVENT, event, "an event");
}

// --- graphs ------------------------------------------------------------------------

static const char *const illegal_state = "CUDA_ERROR_ILLEGAL_STATE: the operation cannot be performed in the present state";

int jl_trt_capture_begin(jl_trt *t, jl_trt_stream *s, char *err, size_t errlen) {
  ENTER(t, "capture_begin");
  if (!live(t, OBJ_STREAM, s) || t->capturing != NULL) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuStreamBeginCapture: %s", illegal_state));
  }
  s->capturing = 1;
  s->invalidated = 0;
  free_ops(&s->ops);
  t->capturing = s;
  t->capture_thread = pthread_self();
  return leave(t, JL_TRT_OK);
}

int jl_trt_capture_end(jl_trt *t, jl_trt_stream *s, jl_trt_graph **out, char *err, size_t errlen) {
  *out = NULL;
  ENTER(t, "capture_end");
  if (!live(t, OBJ_STREAM, s) || !s->capturing) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuStreamEndCapture: %s", illegal_state));
  }
  s->capturing = 0;
  t->capturing = NULL;
  int rc = s->invalidated ? say(err, errlen, JL_TRT_CUDA_ERROR,
                                "cuStreamEndCapture: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: operation failed due to a "
                                "previous error during capture")
                          : make(t, OBJ_GRAPH, sizeof(jl_trt_graph), (void **)out, err, errlen);
  if (rc == JL_TRT_OK) {
    (*out)->ops = s->ops;
    s->ops = (op_list){NULL, 0, 0};
  }
  free_ops(&s->ops);
  return leave(t, rc);
}

int jl_trt_graph_instantiate(jl_trt *t, jl_trt_graph *graph, jl_trt_graph_exec **out, char *err, size_t errlen) {
  *out = NULL;
  ENTER(t, "graph_instantiate");
  if (!live(t, OBJ_GRAPH, graph)) {
    return leave(t, invalid("cuGraphInstantiateWithFlags", "not a live graph", err, errlen));
  }
  int rc = make(t, OBJ_EXEC, sizeof(jl_trt_graph_exec), (void **)out, err, errlen);
  for (size_t i = 0; rc == JL_TRT_OK && i < graph->ops.n; i++) {
    push(&(*out)->ops, graph->ops.ops[i]);
  }
  return leave(t, rc);
}

int jl_trt_graph_launch(jl_trt *t, jl_trt_graph_exec *exec, jl_trt_stream *s, char *err, size_t errlen) {
  ENTER(t, "graph_launch");
  if (!live(t, OBJ_EXEC, exec) || !live(t, OBJ_STREAM, s)) {
    return leave(t, bad_handle("cuGraphLaunch", err, errlen));
  }
  t->stats.graph_launches++;
  int rc = JL_TRT_OK;
  for (size_t i = 0; rc == JL_TRT_OK && i < exec->ops.n; i++) {
    rc = s->capturing ? submit(t, s, exec->ops.ops[i], err, errlen) : execute(t, &exec->ops.ops[i], 1, err, errlen);
  }
  return leave(t, rc);
}

void jl_trt_graph_destroy(jl_trt *t, jl_trt_graph *graph) {
  destroy(t, OBJ_GRAPH, graph, "a graph");
}

void jl_trt_graph_exec_destroy(jl_trt *t, jl_trt_graph_exec *exec) {
  destroy(t, OBJ_EXEC, exec, "a graph exec");
}

// --- runtime -------------------------------------------------------------------------

static int parse_plan(jl_trt *t, char *text, jl_trt_engine *e, char *err, size_t errlen) {
  char *save = NULL;
  int line_no = 0, magic = 0;
  for (char *line = strtok_r(text, "\n", &save); line != NULL; line = strtok_r(NULL, "\n", &save)) {
    char *w[4 + JL_TRT_MAX_DIMS + 2], *wsave = NULL;
    int n = 0;
    line_no++;
    for (char *word = strtok_r(line, " \t\r", &wsave); word != NULL; word = strtok_r(NULL, " \t\r", &wsave)) {
      if (n == (int)(sizeof w / sizeof w[0])) {
        return trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: plan line %d has too many words", line_no);
      }
      w[n++] = word;
    }
    if (n == 0 || (magic && strcmp(w[0], "settings") == 0)) {
      continue;
    }
    if (!magic) {
      if (n != 2 || strcmp(w[0], "jl_trt_fake_plan") != 0 || strcmp(w[1], "1") != 0) {
        return trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: not a fake plan");
      }
      magic = 1;
      continue;
    }
    if (strcmp(w[0], "built") == 0 && n == 2) {
      char have[64];
      snprintf(have, sizeof have, "%d.%d.%d.%d", t->config.major, t->config.minor, t->config.patch, t->config.build);
      if (strcmp(w[1], have) != 0) {
        return trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: the plan was built by TensorRT %s, this is %s; rebuild it", w[1], have);
      }
      continue;
    }
    if ((strcmp(w[0], "input") != 0 && strcmp(w[0], "output") != 0) || n < 4 || e->n == MAX_TENSORS) {
      return trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: plan line %d is not understood", line_no);
    }
    tensor *x = &e->tensors[e->n];
    *x = (tensor){.is_input = w[0][0] == 'i', .type = strcmp(w[2], "float32") == 0   ? JL_TRT_FLOAT
                                                     : strcmp(w[2], "float16") == 0 ? JL_TRT_FLOAT16
                                                                                    : -1,
                  .from = -1};
    snprintf(x->name, sizeof x->name, "%s", w[1]);
    int k = 3;
    for (; k < n && strcmp(w[k], "from") != 0 && x->rank < JL_TRT_MAX_DIMS; k++) {
      x->dims[x->rank++] = strtoll(w[k], NULL, 10);
    }
    for (int i = 0; k + 2 == n && !x->is_input && i < e->n; i++) {
      x->from = e->tensors[i].is_input && strcmp(e->tensors[i].name, w[k + 1]) == 0 ? i : x->from;
    }
    if (x->type < 0 || (k < n && x->from < 0)) {
      return trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: plan line %d has an unknown type or `from`", line_no);
    }
    e->n++;
  }
  return magic ? JL_TRT_OK : trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: not a fake plan");
}

int jl_trt_engine_deserialize(jl_trt *t, const void *plan, size_t size, jl_trt_engine **out, char *err,
                              size_t errlen) {
  *out = NULL;
  int entered = enter(t, "engine_deserialize", err, errlen);
  if (entered != JL_TRT_OK) {
    // An injected refusal reads as the shim reads TensorRT's log: one for
    // want of memory is not the plan's.
    return entered == JL_TRT_ERROR && err != NULL && is_allocation_failure(err) ? JL_TRT_CUDA_ERROR : entered;
  }
  int rc = unsafe(t, "cuMemAlloc", err, errlen);
  char *text = rc == JL_TRT_OK ? strndup(plan, size) : NULL;
  jl_trt_engine *e = text != NULL ? calloc(1, sizeof(jl_trt_engine)) : NULL;
  if (e != NULL) {
    e->trt = t;
    rc = parse_plan(t, text, e, err, errlen);
  } else if (rc == JL_TRT_OK) {
    rc = say(err, errlen, JL_TRT_CUDA_ERROR, "out of memory");
  }
  free(text);
  if (rc != JL_TRT_OK) {
    free(e);
    return leave(t, rc);
  }
  track(t, OBJ_ENGINE, e, 0);
  *out = e;
  return leave(t, JL_TRT_OK);
}

void jl_trt_engine_destroy(jl_trt_engine *e) {
  if (e == NULL) {
    return;
  }
  jl_trt *t = e->trt;
  pthread_mutex_lock(&t->lock);
  for (obj *o = t->live; o != NULL; o = o->next) {
    if (o->kind == OBJ_CONTEXT && ((jl_trt_context *)o->ptr)->engine == e) {
      latch(t, "fake: an engine destroyed before its execution context");
    }
  }
  destroy(t, OBJ_ENGINE, e, "an engine");
  pthread_mutex_unlock(&t->lock);
}

int jl_trt_engine_io_count(const jl_trt_engine *e) {
  return e->n;
}

int jl_trt_engine_io(const jl_trt_engine *e, int index, const char **name, int *is_input, int *type, int64_t *dims,
                     int *rank, char *err, size_t errlen) {
  if (index < 0 || index >= e->n) {
    return say(err, errlen, JL_TRT_ERROR, "no IO tensor %d; the engine has %d", index, e->n);
  }
  const tensor *x = &e->tensors[index];
  *name = x->name;
  *is_input = x->is_input;
  *type = x->type;
  *rank = x->rank;
  memcpy(dims, x->dims, (size_t)x->rank * sizeof(int64_t));
  return JL_TRT_OK;
}

int jl_trt_context_create(jl_trt_engine *e, jl_trt_context **out, char *err, size_t errlen) {
  jl_trt *t = e->trt;
  *out = NULL;
  ENTER(t, "context_create");
  int rc = unsafe(t, "cuMemAlloc", err, errlen);
  if (rc == JL_TRT_OK) {
    rc = make(t, OBJ_CONTEXT, sizeof(jl_trt_context), (void **)out, err, errlen);
  }
  if (rc == JL_TRT_OK) {
    (*out)->trt = t;
    (*out)->engine = e;
  }
  return leave(t, rc);
}

void jl_trt_context_destroy(jl_trt_context *c) {
  if (c != NULL) {
    destroy(c->trt, OBJ_CONTEXT, c, "an execution context");
  }
}

int jl_trt_context_set_address(jl_trt_context *c, const char *name, jl_trt_dptr address, char *err, size_t errlen) {
  jl_trt *t = c->trt;
  ENTER(t, "context_set_address");
  for (int i = 0; i < c->engine->n; i++) {
    if (strcmp(c->engine->tensors[i].name, name) == 0) {
      c->address[i] = address;
      return leave(t, JL_TRT_OK);
    }
  }
  return leave(t, trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "IExecutionContext::setTensorAddress: no IO tensor named %s", name));
}

int jl_trt_context_enqueue(jl_trt_context *c, jl_trt_stream *s, char *err, size_t errlen) {
  jl_trt *t = c->trt;
  ENTER(t, "context_enqueue");
  int capturing = live(t, OBJ_STREAM, s) && s->capturing;
  int rc = check_bound(t, c, err, errlen);
  if (rc == JL_TRT_OK && capturing && !c->ran) {
    rc = trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: enqueue once before capturing, as TensorRT requires");
  }
  if (rc != JL_TRT_OK && capturing) {
    s->invalidated = 1;
  }
  return leave(t, rc != JL_TRT_OK ? rc : submit(t, s, (op){OP_ENQUEUE, 0, 0, 0, 0, c}, err, errlen));
}

// --- builder -------------------------------------------------------------------------

int jl_trt_build_create(jl_trt *t, jl_trt_build **out, char *err, size_t errlen) {
  *out = NULL;
  ENTER(t, "build_create");
  int rc = make(t, OBJ_BUILD, sizeof(jl_trt_build), (void **)out, err, errlen);
  if (rc == JL_TRT_OK) {
    **out = (jl_trt_build){.trt = t, .optimization_level = 3, .cache_builds = -1};
  }
  return leave(t, rc);
}

void jl_trt_build_destroy(jl_trt_build *b) {
  if (b != NULL) {
    destroy(b->trt, OBJ_BUILD, b, "a build");
  }
}

int jl_trt_build_parse(jl_trt_build *b, const char *onnx_path, char *err, size_t errlen) {
  ENTER(b->trt, "build_parse");
  FILE *f = fopen(onnx_path, "rb");
  if (f == NULL) {
    return leave(b->trt, say(err, errlen, JL_TRT_ERROR, "(parseFromFile): MODEL_DESERIALIZE_FAILED: fake: cannot open %s", onnx_path));
  }
  fclose(f);
  b->parsed = 1;
  return leave(b->trt, JL_TRT_OK);
}

int jl_trt_build_layers(const jl_trt_build *b) {
  return b->parsed ? b->trt->build_layers : 0;
}

int jl_trt_build_set_fp16(jl_trt_build *b, char *err, size_t errlen) {
  ENTER(b->trt, "build_set_fp16");
  if (b->trt->config.strongly_typed) {
    return leave(b->trt, say(err, errlen, JL_TRT_ERROR, "TensorRT %d has no FP16 flag; precision follows the ONNX",
                             b->trt->config.major));
  }
  b->fp16 = 1;
  return leave(b->trt, JL_TRT_OK);
}

void jl_trt_build_set_optimization_level(jl_trt_build *b, int level) {
  b->optimization_level = level;
}

void jl_trt_build_set_workspace(jl_trt_build *b, size_t bytes) {
  b->workspace = bytes;
}

void jl_trt_build_set_progress(jl_trt_build *b, jl_trt_progress_fn fn, void *ctx) {
  b->progress = fn;
  b->progress_ctx = ctx;
}

void jl_trt_build_cancel(jl_trt_build *b) {
  __atomic_store_n(&b->cancelled, 1, __ATOMIC_SEQ_CST);
}

int jl_trt_build_set_timing_cache(jl_trt_build *b, const void *data, size_t size, char *err, size_t errlen) {
  jl_trt *t = b->trt;
  ENTER(t, "build_set_timing_cache");
  b->cache_builds = data == NULL || size == 0 ? 0 : -1;
  if (b->cache_builds == 0) {
    return leave(t, JL_TRT_OK);
  }
  char text[128];
  size_t n = size < sizeof text - 1 ? size : sizeof text - 1;
  memcpy(text, data, n);
  text[n] = '\0';
  int v[4], builds;
  if (sscanf(text, "jl_trt_fake_timing %d.%d.%d.%d %d", &v[0], &v[1], &v[2], &v[3], &builds) != 5) {
    return leave(t, trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: the timing cache is truncated or not one"));
  }
  if (v[0] != t->config.major || v[1] != t->config.minor || v[2] != t->config.patch || v[3] != t->config.build) {
    return leave(t, trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: the timing cache is from TensorRT %d.%d.%d.%d", v[0], v[1], v[2], v[3]));
  }
  b->cache_builds = builds;
  return leave(t, JL_TRT_OK);
}

static void progress(jl_trt_build *b, int event, const char *phase, const char *parent, int value) {
  if (b->progress != NULL) {
    b->progress(b->progress_ctx, event, phase, parent, value);
  }
}

static int write_text(const char *path, const char *text, char *err, size_t errlen) {
  return write_file(path, text, strlen(text)) ? JL_TRT_OK : say(err, errlen, JL_TRT_ERROR, "cannot write %s", path);
}

int jl_trt_build_write_plan(jl_trt_build *b, const char *path, char *err, size_t errlen) {
  jl_trt *t = b->trt;
  ENTER(t, "build_write_plan");
  if (!b->parsed) {
    return leave(t, trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: nothing parsed to build"));
  }
  trt_log(t, JL_TRT_LOG_WARNING, NULL, 0, "fake: building a plan");
  int layers = t->build_layers > 0 ? t->build_layers : 1;
  progress(b, JL_TRT_PHASE_START, "fake build", NULL, layers);
  for (int i = 0; i < layers; i++) {
    progress(b, JL_TRT_PHASE_START, "fake tactics", "fake build", 2);
    progress(b, JL_TRT_PHASE_STEP, "fake tactics", NULL, 0);
    if (__atomic_load_n(&b->cancelled, __ATOMIC_SEQ_CST)) {
      return leave(t, trt_log(t, JL_TRT_LOG_ERROR, err, errlen, "fake: the build was cancelled"));
    }
    progress(b, JL_TRT_PHASE_STEP, "fake tactics", NULL, 1);
    progress(b, JL_TRT_PHASE_FINISH, "fake tactics", NULL, 0);
    progress(b, JL_TRT_PHASE_STEP, "fake build", NULL, i);
  }
  progress(b, JL_TRT_PHASE_FINISH, "fake build", NULL, 0);
  const char *tensors = t->build_tensors ? t->build_tensors : default_tensors;
  const char *cache = b->cache_builds < 0 ? "none" : b->cache_builds == 0 ? "cold" : "warm";
  size_t cap = strlen(tensors) + 256;
  char *text = malloc(cap);
  if (text == NULL) {
    return leave(t, say(err, errlen, JL_TRT_ERROR, "out of memory"));
  }
  snprintf(text, cap,
           "jl_trt_fake_plan 1\nbuilt %d.%d.%d.%d\nsettings fp16=%d optimization_level=%d workspace=%zu timing_cache=%s\n%s",
           t->config.major, t->config.minor, t->config.patch, t->config.build, b->fp16, b->optimization_level,
           b->workspace, cache, tensors);
  int rc = write_text(path, text, err, errlen);
  free(text);
  b->cache_builds += rc == JL_TRT_OK && b->cache_builds >= 0;
  return leave(t, rc);
}

int jl_trt_build_write_timing_cache(jl_trt_build *b, const char *path, char *err, size_t errlen) {
  jl_trt *t = b->trt;
  ENTER(t, "build_write_timing_cache");
  if (b->cache_builds < 0) {
    return leave(t, say(err, errlen, JL_TRT_ERROR, "no timing cache is attached"));
  }
  char text[128];
  snprintf(text, sizeof text, "jl_trt_fake_timing %d.%d.%d.%d %d\n", t->config.major, t->config.minor, t->config.patch,
           t->config.build, b->cache_builds);
  return leave(t, write_text(path, text, err, errlen));
}
