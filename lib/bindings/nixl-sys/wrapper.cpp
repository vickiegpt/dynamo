#include "wrapper.h"

#ifdef __cplusplus
#include <backend/backend_aux.h>
#include <backend/backend_engine.h>
#include <nixl.h>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl_types.h>
#include <utils/common/nixl_time.h>
#include <utils/serdes/serdes.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace std {
using ::std::exception;
using ::std::string;
}  // namespace std

extern "C" {
#endif

nixl_status_t
nixl_create_agent(const char* name, nixlAgentConfig* config, nixlAgent** agent)
{
  try {
    auto cpp_agent = new nixlAgent(std::string(name), *reinterpret_cast<nixlAgentConfig_cpp*>(config));
    *agent = reinterpret_cast<nixlAgent*>(cpp_agent);
    return NIXL_SUCCESS;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_destroy_agent(nixlAgent* agent)
{
  try {
    delete reinterpret_cast<nixlAgent_cpp*>(agent);
    return NIXL_SUCCESS;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_create_xfer_dlist(nixl_mem_t type, bool sorted, size_t init_size, nixl_xfer_dlist_t** list)
{
  try {
    auto cpp_list = new nixl_xfer_dlist_t_cpp(type, sorted, init_size);
    *list = reinterpret_cast<nixl_xfer_dlist_t*>(cpp_list);
    return NIXL_SUCCESS;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_destroy_xfer_dlist(nixl_xfer_dlist_t* list)
{
  try {
    delete reinterpret_cast<nixl_xfer_dlist_t_cpp*>(list);
    return NIXL_SUCCESS;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_post_xfer_req(nixlAgent* agent, nixlXferReqH* reqh)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_reqh = reinterpret_cast<nixlXferReqH_cpp*>(reqh);
    return cpp_agent->postXferReq(cpp_reqh);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_get_xfer_status(nixlAgent* agent, nixlXferReqH* reqh)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_reqh = reinterpret_cast<nixlXferReqH_cpp*>(reqh);
    return cpp_agent->getXferStatus(cpp_reqh);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_release_xfer_req(nixlAgent* agent, nixlXferReqH* reqh)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_reqh = reinterpret_cast<nixlXferReqH_cpp*>(reqh);
    return cpp_agent->releaseXferReq(cpp_reqh);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_get_avail_plugins(nixlAgent* agent, nixl_backend_t** plugins, size_t* count)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    std::vector<nixl_backend_t> cpp_plugins;
    auto status = cpp_agent->getAvailPlugins(cpp_plugins);
    if (status != NIXL_SUCCESS) {
      return status;
    }
    *count = cpp_plugins.size();
    *plugins = new nixl_backend_t[*count];
    for (size_t i = 0; i < *count; i++) {
      (*plugins)[i] = cpp_plugins[i];
    }
    return NIXL_SUCCESS;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_get_plugin_params(nixlAgent* agent, const char* plugin_name, nixl_mem_list_t* mems, nixl_b_params_t* params)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_mems = reinterpret_cast<nixl_mem_list_t_cpp*>(mems);
    auto cpp_params = reinterpret_cast<nixl_b_params_t_cpp*>(params);
    return cpp_agent->getPluginParams(plugin_name, *cpp_mems, *cpp_params);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_create_backend(nixlAgent* agent, const char* plugin_name, nixl_b_params_t* params, nixlBackendH** backend)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_params = reinterpret_cast<nixl_b_params_t_cpp*>(params);
    nixlBackendH_cpp* cpp_backend;
    auto status = cpp_agent->createBackend(plugin_name, *cpp_params, cpp_backend);
    if (status == NIXL_SUCCESS) {
      *backend = reinterpret_cast<nixlBackendH*>(cpp_backend);
    }
    return status;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_get_backend_params(nixlAgent* agent, nixlBackendH* backend, nixl_mem_list_t* mems, nixl_b_params_t* params)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_backend = reinterpret_cast<nixlBackendH_cpp*>(backend);
    auto cpp_mems = reinterpret_cast<nixl_mem_list_t_cpp*>(mems);
    auto cpp_params = reinterpret_cast<nixl_b_params_t_cpp*>(params);
    return cpp_agent->getBackendParams(cpp_backend, *cpp_mems, *cpp_params);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_register_mem(nixlAgent* agent, nixl_reg_dlist_t* dlist, nixl_opt_args_t* extra_params)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_dlist = reinterpret_cast<nixl_reg_dlist_t_cpp*>(dlist);
    auto cpp_extra = reinterpret_cast<nixl_opt_args_t_cpp*>(extra_params);
    return cpp_agent->registerMem(*cpp_dlist, cpp_extra);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_deregister_mem(nixlAgent* agent, nixl_reg_dlist_t* dlist, nixl_opt_args_t* extra_params)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_dlist = reinterpret_cast<nixl_reg_dlist_t_cpp*>(dlist);
    auto cpp_extra = reinterpret_cast<nixl_opt_args_t_cpp*>(extra_params);
    return cpp_agent->deregisterMem(*cpp_dlist, cpp_extra);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_get_local_md(nixlAgent* agent, char** metadata)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    std::string cpp_meta;
    auto status = cpp_agent->getLocalMD(cpp_meta);
    if (status == NIXL_SUCCESS) {
      *metadata = strdup(cpp_meta.c_str());
    }
    return status;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_load_remote_md(nixlAgent* agent, const char* metadata, char** result)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    std::string cpp_result;
    auto status = cpp_agent->loadRemoteMD(metadata, cpp_result);
    if (status == NIXL_SUCCESS) {
      *result = strdup(cpp_result.c_str());
    }
    return status;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_invalidate_remote_md(nixlAgent* agent, const char* remote_agent)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    return cpp_agent->invalidateRemoteMD(remote_agent);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_create_xfer_req(
    nixlAgent* agent, nixl_xfer_op_t op, nixl_xfer_dlist_t* src_descs, nixl_xfer_dlist_t* dst_descs,
    const char* remote_agent, nixlXferReqH** req_handle, nixl_opt_args_t* extra_params)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_src = reinterpret_cast<nixl_xfer_dlist_t_cpp*>(src_descs);
    auto cpp_dst = reinterpret_cast<nixl_xfer_dlist_t_cpp*>(dst_descs);
    auto cpp_extra = reinterpret_cast<nixl_opt_args_t_cpp*>(extra_params);
    nixlXferReqH_cpp* cpp_handle;
    auto status = cpp_agent->createXferReq(op, *cpp_src, *cpp_dst, remote_agent, cpp_handle, cpp_extra);
    if (status == NIXL_SUCCESS) {
      *req_handle = reinterpret_cast<nixlXferReqH*>(cpp_handle);
    }
    return status;
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

nixl_status_t
nixl_get_notifs(nixlAgent* agent, nixl_notifs_t* notifs)
{
  try {
    auto cpp_agent = reinterpret_cast<nixlAgent_cpp*>(agent);
    auto cpp_notifs = reinterpret_cast<nixl_notifs_t_cpp*>(notifs);
    return cpp_agent->getNotifs(*cpp_notifs);
  }
  catch (const std::exception& e) {
    return NIXL_ERR_BACKEND;
  }
}

#ifdef __cplusplus
}
#endif
