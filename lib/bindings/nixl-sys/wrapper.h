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

// Opaque handle types for C++ objects
typedef struct nixl_capi_agent_s* nixl_capi_agent_t;


// Core API functions
nixl_capi_status_t nixl_capi_create_agent(const char* name, nixl_capi_agent_t* agent);

nixl_capi_status_t nixl_capi_destroy_agent(nixl_capi_agent_t agent);

#ifdef __cplusplus
}
#endif
