#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "QnnInterface.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpGraph.h"

#ifdef FAKE_BACKEND
static const QnnGraph_Config_t **saved[4];
static Qnn_ContextHandle_t contexts[4];
static size_t saved_count;
static bool first_free = true;
static void check(const QnnGraph_Config_t **configs) {
  assert(configs && configs[0] && configs[1] && !configs[2]);
  assert(configs[0]->option == QNN_GRAPH_CONFIG_OPTION_PRIORITY);
  assert(configs[1]->option == QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  const QnnHtpGraph_CustomConfig_t *c = configs[1]->customConfig;
  assert(c->option == QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION);
  assert(!c->advancedActivationFusion);
}
static Qnn_ErrorHandle_t create(Qnn_ContextHandle_t context, const char *name,
    const QnnGraph_Config_t **configs, Qnn_GraphHandle_t *graph) {
  check(configs);
  if (strcmp(name, "fail") == 0) return QNN_COMMON_ERROR_GENERAL;
  assert(saved_count < 4);
  saved[saved_count] = configs;
  contexts[saved_count++] = context;
  *graph = (void *)0x1;
  return QNN_SUCCESS;
}
static Qnn_ErrorHandle_t free_context(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile) {
  (void)profile;
  // Inspect configuration after graphCreate has returned, across contexts and
  // after a failed contextFree. The wrapper may release only after success.
  for (size_t i = 0; i < saved_count; ++i) if (saved[i]) check(saved[i]);
  if (first_free) { first_free = false; return QNN_COMMON_ERROR_GENERAL; }
  for (size_t i = 0; i < saved_count; ++i) if (contexts[i] == context) saved[i] = NULL;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t ***list, uint32_t *count) {
  static QnnInterface_t p = QNN_INTERFACE_INIT;
  static const QnnInterface_t *ps[] = {&p};
  p.backendId = QNN_BACKEND_ID_HTP;
  p.apiVersion.coreApiVersion.major = QNN_API_VERSION_MAJOR;
  p.apiVersion.coreApiVersion.minor = QNN_API_VERSION_MINOR;
  p.QNN_INTERFACE_VER_NAME.graphCreate = create;
  p.QNN_INTERFACE_VER_NAME.contextFree = free_context;
  *list = ps; *count = 1;
  return QNN_SUCCESS;
}
#else
#include <dlfcn.h>
int main(int argc, char **argv) {
  assert(argc == 2);
  void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) { puts(dlerror()); return 1; }
  typedef Qnn_ErrorHandle_t (*Get)(const QnnInterface_t ***, uint32_t *);
  Get get = (Get)dlsym(lib, "QnnInterface_getProviders");
  assert(get && get(NULL, NULL) == QNN_COMMON_ERROR_INVALID_ARGUMENT);
  const QnnInterface_t **p; uint32_t n;
  assert(get(&p, &n) == QNN_SUCCESS && n == 1);
  QnnGraph_Config_t priority = QNN_GRAPH_CONFIG_INIT;
  priority.option = QNN_GRAPH_CONFIG_OPTION_PRIORITY;
  const QnnGraph_Config_t *configs[] = {&priority, NULL};
  Qnn_GraphHandle_t graph = NULL;
  Qnn_ContextHandle_t a = (void *)0x10, b = (void *)0x20;
  assert(p[0]->QNN_INTERFACE_VER_NAME.graphCreate(a, "first", configs, &graph) == QNN_SUCCESS);
  assert(p[0]->QNN_INTERFACE_VER_NAME.graphCreate(a, "second", configs, &graph) == QNN_SUCCESS);
  assert(p[0]->QNN_INTERFACE_VER_NAME.graphCreate(b, "third", configs, &graph) == QNN_SUCCESS);
  assert(p[0]->QNN_INTERFACE_VER_NAME.graphCreate(a, "fail", configs, &graph) == QNN_COMMON_ERROR_GENERAL);
  assert(p[0]->QNN_INTERFACE_VER_NAME.contextFree(a, NULL) == QNN_COMMON_ERROR_GENERAL);
  assert(p[0]->QNN_INTERFACE_VER_NAME.contextFree(a, NULL) == QNN_SUCCESS);
  assert(p[0]->QNN_INTERFACE_VER_NAME.contextFree(b, NULL) == QNN_SUCCESS);
  puts("configuration forwarding, graph failure and context lifetime passed");
  return 0;
}
#endif
