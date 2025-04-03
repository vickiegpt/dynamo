#include "wrapper.h"

#include <nixl.h>

#include <string>
#include <vector>

extern "C" {

// Internal struct definition to match our opaque type
struct nixl_capi_agent_s {
  nixlAgent* agent;
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

}  // extern "C"
