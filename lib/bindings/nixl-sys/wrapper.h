#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
// C++ standard library includes first
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// NIXL includes after standard library
#include <backend/backend_aux.h>
#include <backend/backend_engine.h>
#include <nixl.h>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl_types.h>
#include <utils/common/nixl_time.h>
#include <utils/serdes/serdes.h>

// C++ type aliases
using nixl_xfer_dlist_t_cpp = nixl_xfer_dlist_t;
using nixl_reg_dlist_t_cpp = nixl_reg_dlist_t;
using nixlAgentConfig_cpp = nixlAgentConfig;
using nixlAgent_cpp = nixlAgent;
using nixlBackendH_cpp = nixlBackendH;
using nixlXferReqH_cpp = nixlXferReqH;
using nixlDlistH_cpp = nixlDlistH;
using nixlBlobDesc_cpp = nixlBlobDesc;
using nixlBasicDesc_cpp = nixlBasicDesc;
using nixl_b_params_t_cpp = nixl_b_params_t;
using nixl_mem_list_t_cpp = nixl_mem_list_t;
using nixl_opt_args_t_cpp = nixl_opt_args_t;
using nixl_notifs_t_cpp = nixl_notifs_t;

extern "C" {
#endif

#ifndef __cplusplus
// Define the NIXL enums only when not in C++ mode
typedef enum { DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG } nixl_mem_t;

typedef enum { NIXL_READ, NIXL_WRITE } nixl_xfer_op_t;

typedef enum {
  NIXL_IN_PROG,
  NIXL_SUCCESS,
  NIXL_ERR_NOT_POSTED,
  NIXL_ERR_INVALID_PARAM,
  NIXL_ERR_BACKEND,
  NIXL_ERR_NOT_FOUND,
  NIXL_ERR_MISMATCH,
  NIXL_ERR_NOT_ALLOWED,
  NIXL_ERR_REPOST_ACTIVE,
  NIXL_ERR_UNKNOWN,
  NIXL_ERR_NOT_SUPPORTED
} nixl_status_t;

// Forward declarations for C
struct nixl_xfer_dlist_t_;
struct nixl_reg_dlist_t_;
struct nixlAgentConfig_;
struct nixlAgent_;
struct nixlBackendH_;
struct nixlXferReqH_;
struct nixlDlistH_;
struct nixlBlobDesc_;
struct nixlBasicDesc_;
struct nixl_b_params_t_;
struct nixl_mem_list_t_;
struct nixl_opt_args_t_;
struct nixl_notifs_t_;

typedef struct nixl_xfer_dlist_t_ nixl_xfer_dlist_t;
typedef struct nixl_reg_dlist_t_ nixl_reg_dlist_t;
typedef struct nixlAgentConfig_ nixlAgentConfig;
typedef struct nixlAgent_ nixlAgent;
typedef struct nixlBackendH_ nixlBackendH;
typedef struct nixlXferReqH_ nixlXferReqH;
typedef struct nixlDlistH_ nixlDlistH;
typedef struct nixlBlobDesc_ nixlBlobDesc;
typedef struct nixlBasicDesc_ nixlBasicDesc;
typedef struct nixl_b_params_t_ nixl_b_params_t;
typedef struct nixl_mem_list_t_ nixl_mem_list_t;
typedef struct nixl_opt_args_t_ nixl_opt_args_t;
typedef struct nixl_notifs_t_ nixl_notifs_t;
typedef const char* nixl_backend_t;
#endif

// Core API functions
nixl_status_t nixl_create_agent(const char* name, nixlAgentConfig* config, nixlAgent** agent);
nixl_status_t nixl_destroy_agent(nixlAgent* agent);

// Memory management functions
nixl_status_t nixl_create_xfer_dlist(nixl_mem_t type, bool sorted, size_t init_size, nixl_xfer_dlist_t** list);
nixl_status_t nixl_destroy_xfer_dlist(nixl_xfer_dlist_t* list);

// Transfer operations
nixl_status_t nixl_post_xfer_req(nixlAgent* agent, nixlXferReqH* reqh);
nixl_status_t nixl_get_xfer_status(nixlAgent* agent, nixlXferReqH* reqh);
nixl_status_t nixl_release_xfer_req(nixlAgent* agent, nixlXferReqH* reqh);

// Plugin and backend management
nixl_status_t nixl_get_avail_plugins(nixlAgent* agent, nixl_backend_t** plugins, size_t* count);
nixl_status_t nixl_get_plugin_params(
    nixlAgent* agent, const char* plugin_name, nixl_mem_list_t* mems, nixl_b_params_t* params);
nixl_status_t nixl_create_backend(
    nixlAgent* agent, const char* plugin_name, nixl_b_params_t* params, nixlBackendH** backend);
nixl_status_t nixl_get_backend_params(
    nixlAgent* agent, nixlBackendH* backend, nixl_mem_list_t* mems, nixl_b_params_t* params);

// Memory registration
nixl_status_t nixl_register_mem(nixlAgent* agent, nixl_reg_dlist_t* dlist, nixl_opt_args_t* extra_params);
nixl_status_t nixl_deregister_mem(nixlAgent* agent, nixl_reg_dlist_t* dlist, nixl_opt_args_t* extra_params);

// Metadata management
nixl_status_t nixl_get_local_md(nixlAgent* agent, char** metadata);
nixl_status_t nixl_load_remote_md(nixlAgent* agent, const char* metadata, char** result);
nixl_status_t nixl_invalidate_remote_md(nixlAgent* agent, const char* remote_agent);

// Transfer request creation
nixl_status_t nixl_create_xfer_req(
    nixlAgent* agent, nixl_xfer_op_t op, nixl_xfer_dlist_t* src_descs, nixl_xfer_dlist_t* dst_descs,
    const char* remote_agent, nixlXferReqH** req_handle, nixl_opt_args_t* extra_params);

// Notification handling
nixl_status_t nixl_get_notifs(nixlAgent* agent, nixl_notifs_t* notifs);

#ifdef __cplusplus
}
#endif
