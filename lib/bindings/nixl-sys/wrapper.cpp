#include "wrapper.h"

#include <nixl.h>

#include <iterator>
#include <map>
#include <string>
#include <vector>

extern "C" {

// Internal struct definitions to match our opaque types
struct nixl_capi_agent_s {
  nixlAgent* agent;
};

struct nixl_capi_string_list_s {
  std::vector<std::string> strings;
};

struct nixl_capi_params_s {
  nixl_b_params_t params;
};

struct nixl_capi_mem_list_s {
  nixl_mem_list_t mems;
};

struct nixl_capi_backend_s {
  nixlBackendH* backend;
};

struct nixl_capi_opt_args_s {
  nixl_opt_args_t args;
};

struct nixl_capi_param_iter_s {
  nixl_b_params_t::iterator current;
  nixl_b_params_t::iterator end;
  std::string current_key;    // Keep string alive while iterator exists
  std::string current_value;  // Keep string alive while iterator exists
};

// Internal structs for descriptor lists
struct nixl_capi_xfer_dlist_s {
  nixl_xfer_dlist_t* dlist;
};

struct nixl_capi_reg_dlist_s {
  nixl_reg_dlist_t* dlist;
};

nixl_capi_status_t
nixl_capi_create_agent(const char* name, nixl_capi_agent_t* agent)
{
  if (!name || !agent) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    nixlAgentConfig nixl_config(true);  // Use progress thread
    std::string agent_name = name;
    auto inner = new nixlAgent(agent_name, nixl_config);

    auto agent_handle = new nixl_capi_agent_s;
    agent_handle->agent = inner;
    *agent = agent_handle;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_destroy_agent(nixl_capi_agent_t agent)
{
  if (!agent) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete agent->agent;
    delete agent;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_get_available_plugins(nixl_capi_agent_t agent, nixl_capi_string_list_t* plugins)
{
  if (!agent || !plugins) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    std::vector<nixl_backend_t> backend_plugins;
    nixl_status_t ret = agent->agent->getAvailPlugins(backend_plugins);

    if (ret != NIXL_SUCCESS) {
      return NIXL_CAPI_ERROR_BACKEND;
    }

    auto list = new nixl_capi_string_list_s;
    list->strings = std::move(backend_plugins);
    *plugins = list;

    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_destroy_string_list(nixl_capi_string_list_t list)
{
  if (!list) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete list;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_string_list_size(nixl_capi_string_list_t list, size_t* size)
{
  if (!list || !size) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *size = list->strings.size();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_string_list_get(nixl_capi_string_list_t list, size_t index, const char** str)
{
  if (!list || !str || index >= list->strings.size()) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *str = list->strings[index].c_str();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_get_plugin_params(
    nixl_capi_agent_t agent, const char* plugin_name, nixl_capi_mem_list_t* mems, nixl_capi_params_t* params)
{
  if (!agent || !plugin_name || !mems || !params) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    auto mem_list = new nixl_capi_mem_list_s;
    auto param_list = new nixl_capi_params_s;

    nixl_status_t ret = agent->agent->getPluginParams(plugin_name, mem_list->mems, param_list->params);

    if (ret != NIXL_SUCCESS) {
      delete mem_list;
      delete param_list;
      return NIXL_CAPI_ERROR_BACKEND;
    }

    *mems = mem_list;
    *params = param_list;

    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_destroy_mem_list(nixl_capi_mem_list_t list)
{
  if (!list) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete list;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_destroy_params(nixl_capi_params_t params)
{
  if (!params) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete params;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_create_backend(
    nixl_capi_agent_t agent, const char* plugin_name, nixl_capi_params_t params, nixl_capi_backend_t* backend)
{
  if (!agent || !plugin_name || !params || !backend) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    auto backend_handle = new nixl_capi_backend_s;
    nixl_status_t ret = agent->agent->createBackend(plugin_name, params->params, backend_handle->backend);

    if (ret != NIXL_SUCCESS) {
      delete backend_handle;
      return NIXL_CAPI_ERROR_BACKEND;
    }

    *backend = backend_handle;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_destroy_backend(nixl_capi_backend_t backend)
{
  if (!backend) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete backend;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_create_opt_args(nixl_capi_opt_args_t* args)
{
  if (!args) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    auto opt_args = new nixl_capi_opt_args_s;
    *args = opt_args;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_destroy_opt_args(nixl_capi_opt_args_t args)
{
  if (!args) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete args;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_opt_args_add_backend(nixl_capi_opt_args_t args, nixl_capi_backend_t backend)
{
  if (!args || !backend) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    args->args.backends.push_back(backend->backend);
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_params_is_empty(nixl_capi_params_t params, bool* is_empty)
{
  if (!params || !is_empty) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *is_empty = params->params.empty();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_params_create_iterator(nixl_capi_params_t params, nixl_capi_param_iter_t* iter)
{
  if (!params || !iter) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    auto param_iter = new nixl_capi_param_iter_s;
    param_iter->current = params->params.begin();
    param_iter->end = params->params.end();
    *iter = param_iter;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_params_iterator_next(nixl_capi_param_iter_t iter, const char** key, const char** value, bool* has_next)
{
  if (!iter || !key || !value || !has_next) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    if (iter->current == iter->end) {
      *has_next = false;
      return NIXL_CAPI_SUCCESS;
    }

    // Store the strings in the iterator to keep them alive
    iter->current_key = iter->current->first;
    iter->current_value = iter->current->second;

    *key = iter->current_key.c_str();
    *value = iter->current_value.c_str();

    ++iter->current;
    *has_next = (iter->current != iter->end);

    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_params_destroy_iterator(nixl_capi_param_iter_t iter)
{
  if (!iter) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete iter;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_mem_list_is_empty(nixl_capi_mem_list_t list, bool* is_empty)
{
  if (!list || !is_empty) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *is_empty = list->mems.empty();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_mem_list_size(nixl_capi_mem_list_t list, size_t* size)
{
  if (!list || !size) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *size = list->mems.size();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_mem_list_get(nixl_capi_mem_list_t list, size_t index, nixl_capi_mem_type_t* mem_type)
{
  if (!list || !mem_type || index >= list->mems.size()) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *mem_type = static_cast<nixl_capi_mem_type_t>(list->mems[index]);
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_mem_type_to_string(nixl_capi_mem_type_t mem_type, const char** str)
{
  if (!str) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    static const char* mem_type_strings[] = {
        "Unknown",
        "DRAM",
        "GPU",
    };

    if (mem_type < 0 || mem_type >= sizeof(mem_type_strings) / sizeof(mem_type_strings[0])) {
      return NIXL_CAPI_ERROR_INVALID_PARAM;
    }

    *str = mem_type_strings[mem_type];
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_get_backend_params(
    nixl_capi_agent_t agent, nixl_capi_backend_t backend, nixl_capi_mem_list_t* mems, nixl_capi_params_t* params)
{
  if (!agent || !backend || !mems || !params) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    auto mem_list = new nixl_capi_mem_list_s;
    auto param_list = new nixl_capi_params_s;

    nixl_status_t ret = agent->agent->getBackendParams(backend->backend, mem_list->mems, param_list->params);

    if (ret != NIXL_SUCCESS) {
      delete mem_list;
      delete param_list;
      return NIXL_CAPI_ERROR_BACKEND;
    }

    *mems = mem_list;
    *params = param_list;

    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

// Transfer descriptor list functions
nixl_capi_status_t
nixl_capi_create_xfer_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_xfer_dlist_t* dlist)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    auto d = new nixl_capi_xfer_dlist_s;
    d->dlist = new nixl_xfer_dlist_t(static_cast<nixl_mem_t>(mem_type));
    *dlist = d;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_destroy_xfer_dlist(nixl_capi_xfer_dlist_t dlist)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete dlist->dlist;
    delete dlist;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_xfer_dlist_add_desc(nixl_capi_xfer_dlist_t dlist, uintptr_t addr, size_t len, uint32_t dev_id)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    nixlBasicDesc desc(addr, len, dev_id);
    dlist->dlist->addDesc(desc);
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_xfer_dlist_len(nixl_capi_xfer_dlist_t dlist, size_t* len)
{
  if (!dlist || !len) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *len = dlist->dlist->descCount();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_xfer_dlist_has_overlaps(nixl_capi_xfer_dlist_t dlist, bool* has_overlaps)
{
  if (!dlist || !has_overlaps) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *has_overlaps = dlist->dlist->hasOverlaps();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_xfer_dlist_clear(nixl_capi_xfer_dlist_t dlist)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    dlist->dlist->clear();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_xfer_dlist_resize(nixl_capi_xfer_dlist_t dlist, size_t new_size)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    dlist->dlist->resize(new_size);
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

// Registration descriptor list functions
nixl_capi_status_t
nixl_capi_create_reg_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_reg_dlist_t* dlist)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    auto d = new nixl_capi_reg_dlist_s;
    d->dlist = new nixl_reg_dlist_t(static_cast<nixl_mem_t>(mem_type));
    *dlist = d;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_destroy_reg_dlist(nixl_capi_reg_dlist_t dlist)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    delete dlist->dlist;
    delete dlist;
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_reg_dlist_add_desc(nixl_capi_reg_dlist_t dlist, uintptr_t addr, size_t len, uint32_t dev_id)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    nixlBlobDesc desc(addr, len, dev_id, nixl_blob_t());  // Empty metadata
    dlist->dlist->addDesc(desc);
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_reg_dlist_len(nixl_capi_reg_dlist_t dlist, size_t* len)
{
  if (!dlist || !len) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *len = dlist->dlist->descCount();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_reg_dlist_has_overlaps(nixl_capi_reg_dlist_t dlist, bool* has_overlaps)
{
  if (!dlist || !has_overlaps) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    *has_overlaps = dlist->dlist->hasOverlaps();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_reg_dlist_clear(nixl_capi_reg_dlist_t dlist)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    dlist->dlist->clear();
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_reg_dlist_resize(nixl_capi_reg_dlist_t dlist, size_t new_size)
{
  if (!dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    dlist->dlist->resize(new_size);
    return NIXL_CAPI_SUCCESS;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

// Memory registration functions
nixl_capi_status_t
nixl_capi_register_mem(nixl_capi_agent_t agent, nixl_capi_reg_dlist_t dlist, nixl_capi_opt_args_t opt_args)
{
  if (!agent || !dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    nixl_status_t ret = agent->agent->registerMem(*dlist->dlist, opt_args ? &opt_args->args : nullptr);
    return ret == NIXL_SUCCESS ? NIXL_CAPI_SUCCESS : NIXL_CAPI_ERROR_BACKEND;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

nixl_capi_status_t
nixl_capi_deregister_mem(nixl_capi_agent_t agent, nixl_capi_reg_dlist_t dlist, nixl_capi_opt_args_t opt_args)
{
  if (!agent || !dlist) {
    return NIXL_CAPI_ERROR_INVALID_PARAM;
  }

  try {
    nixl_status_t ret = agent->agent->deregisterMem(*dlist->dlist, opt_args ? &opt_args->args : nullptr);
    return ret == NIXL_SUCCESS ? NIXL_CAPI_SUCCESS : NIXL_CAPI_ERROR_BACKEND;
  }
  catch (...) {
    return NIXL_CAPI_ERROR_BACKEND;
  }
}

}  // extern "C"
