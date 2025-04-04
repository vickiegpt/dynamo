#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status codes for our C API
typedef enum {
  NIXL_CAPI_SUCCESS = 0,
  NIXL_CAPI_ERROR_INVALID_PARAM = -1,
  NIXL_CAPI_ERROR_BACKEND = -2,
} nixl_capi_status_t;

struct nixl_capi_agent_s;
struct nixl_capi_params_s;
struct nixl_capi_mem_list_s;
struct nixl_capi_string_list_s;

// Opaque handle types for C++ objects
typedef struct nixl_capi_agent_s* nixl_capi_agent_t;
typedef struct nixl_capi_params_s* nixl_capi_params_t;
typedef struct nixl_capi_mem_list_s* nixl_capi_mem_list_t;
typedef struct nixl_capi_string_list_s* nixl_capi_string_list_t;

// Core API functions
nixl_capi_status_t nixl_capi_create_agent(const char* name, nixl_capi_agent_t* agent);

nixl_capi_status_t nixl_capi_destroy_agent(nixl_capi_agent_t agent);

// Plugin and parameter functions
nixl_capi_status_t nixl_capi_get_available_plugins(nixl_capi_agent_t agent, nixl_capi_string_list_t* plugins);
nixl_capi_status_t nixl_capi_destroy_string_list(nixl_capi_string_list_t list);
nixl_capi_status_t nixl_capi_string_list_size(nixl_capi_string_list_t list, size_t* size);
nixl_capi_status_t nixl_capi_string_list_get(nixl_capi_string_list_t list, size_t index, const char** str);

nixl_capi_status_t nixl_capi_get_plugin_params(
    nixl_capi_agent_t agent, const char* plugin_name, nixl_capi_mem_list_t* mems, nixl_capi_params_t* params);

nixl_capi_status_t nixl_capi_destroy_mem_list(nixl_capi_mem_list_t list);
nixl_capi_status_t nixl_capi_destroy_params(nixl_capi_params_t params);

#ifdef __cplusplus
}
#endif
