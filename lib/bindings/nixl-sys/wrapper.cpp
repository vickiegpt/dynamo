#include "wrapper.h"

#include <nixl.h>

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

}  // extern "C"
