// Public QNN API adapter; build against the same SDK as the packaged backend.
// This is a numerical workaround, independent of shared I/O and scheduling.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "QnnInterface.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpGraph.h"

static pthread_once_t initialized = PTHREAD_ONCE_INIT;
static void *backend;
static QnnInterface_t wrapped;
static const QnnInterface_t *providers[] = {&wrapped};
static QnnGraph_CreateFn_t real_create;
static QnnContext_FreeFn_t real_context_free;
static Qnn_ErrorHandle_t init_status = QNN_COMMON_ERROR_GENERAL;

// Some backends consume graph configuration during finalization. Keep both
// the list and the custom configuration alive until contextFree succeeds.
typedef struct ConfigStorage {
  struct ConfigStorage *next;
  Qnn_ContextHandle_t context;
  QnnHtpGraph_CustomConfig_t custom;
  QnnGraph_Config_t config;
  const QnnGraph_Config_t *all[];
} ConfigStorage;
static ConfigStorage *retained;
static pthread_mutex_t retained_mutex = PTHREAD_MUTEX_INITIALIZER;

static Qnn_ErrorHandle_t context_free(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile) {
  Qnn_ErrorHandle_t status = real_context_free(context, profile);
  if (status != QNN_SUCCESS) return status;
  pthread_mutex_lock(&retained_mutex);
  ConfigStorage **link = &retained;
  while (*link != NULL) {
    ConfigStorage *entry = *link;
    if (entry->context == context) {
      *link = entry->next;
      free(entry);
    } else {
      link = &entry->next;
    }
  }
  pthread_mutex_unlock(&retained_mutex);
  return status;
}

static Qnn_ErrorHandle_t graph_create(Qnn_ContextHandle_t context, const char *name,
    const QnnGraph_Config_t **original, Qnn_GraphHandle_t *graph) {
  size_t count = 0;
  if (original != NULL) {
    while (count < 128 && original[count] != NULL) ++count;
    if (count == 128) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  }
  ConfigStorage *storage = calloc(1, sizeof(*storage) + (count + 2) * sizeof(*storage->all));
  if (storage == NULL) return QNN_COMMON_ERROR_MEM_ALLOC;
  storage->context = context;
  storage->custom.option = QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION;
  storage->custom.advancedActivationFusion = false;
  storage->config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  storage->config.customConfig = &storage->custom;
  for (size_t i = 0; i < count; ++i) storage->all[i] = original[i];
  storage->all[count] = &storage->config;
  Qnn_ErrorHandle_t status = real_create(context, name, storage->all, graph);
  if (status == QNN_SUCCESS) {
    pthread_mutex_lock(&retained_mutex);
    storage->next = retained;
    retained = storage;
    pthread_mutex_unlock(&retained_mutex);
  } else {
    free(storage);
  }
  return status;
}

static void initialize(void) {
  Dl_info location;
  if (dladdr((void *)&initialize, &location) == 0 || location.dli_fname == NULL) return;
  const char *slash = strrchr(location.dli_fname, '/');
  if (slash == NULL) return;
  char path[4096];
  if (snprintf(path, sizeof(path), "%.*s/libQnnHtp.so", (int)(slash - location.dli_fname), location.dli_fname) >= (int)sizeof(path)) return;
  backend = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (backend == NULL) {
    fprintf(stderr, "QNN no-fusion adapter: cannot open the sibling HTP backend: %s\n", dlerror());
    return;
  }
  typedef Qnn_ErrorHandle_t (*GetProviders)(const QnnInterface_t ***, uint32_t *);
  GetProviders get = (GetProviders)dlsym(backend, "QnnInterface_getProviders");
  if (get == NULL) return;
  const QnnInterface_t **list = NULL;
  uint32_t count = 0;
  if (get(&list, &count) != QNN_SUCCESS || list == NULL) return;
  for (uint32_t i = 0; i < count; ++i) {
    const QnnInterface_t *p = list[i];
    if (p == NULL || p->backendId != QNN_BACKEND_ID_HTP ||
        p->apiVersion.coreApiVersion.major != QNN_API_VERSION_MAJOR ||
        p->apiVersion.coreApiVersion.minor < QNN_API_VERSION_MINOR) continue;
    wrapped = *p;
    real_create = p->QNN_INTERFACE_VER_NAME.graphCreate;
    real_context_free = p->QNN_INTERFACE_VER_NAME.contextFree;
    if (real_create == NULL || real_context_free == NULL) return;
    wrapped.QNN_INTERFACE_VER_NAME.graphCreate = graph_create;
    wrapped.QNN_INTERFACE_VER_NAME.contextFree = context_free;
    init_status = QNN_SUCCESS;
    return;
  }
}

Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t ***out, uint32_t *count) {
  if (out == NULL || count == NULL) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  *out = NULL;
  *count = 0;
  pthread_once(&initialized, initialize);
  if (init_status != QNN_SUCCESS) return init_status;
  *out = providers;
  *count = 1;
  return QNN_SUCCESS;
}
