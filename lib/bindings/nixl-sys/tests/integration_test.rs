// #![allow(unused_variables)]

use nixl_sys::*;
// use std::ffi::CString;

const AGENT1_NAME: &str = "Agent001";
const AGENT2_NAME: &str = "Agent002";
// const NIXL_IN_PROG: i32 = 1;
// const NIXL_SUCCESS: i32 = 0;
// const NIXL_WRITE: i32 = 1;

fn print_params(params: &Params, mems: &MemList) {
    println!("Parameters:");
    if !params.is_empty().unwrap() {
        for param in params.iter().unwrap() {
            let param = param.unwrap();
            println!("  {} = {}", param.key, param.value);
        }
    } else {
        println!("  (empty)");
    }

    println!("Mems:");
    if !mems.is_empty().unwrap() {
        for mem_type in mems.iter() {
            println!("  {}", mem_type.unwrap());
        }
    } else {
        println!("  (empty)");
    }
}

#[test]
fn test_basic_agent_lifecycle() {
    // Create two agents like in the C++ example
    let agent1 = Agent::new(AGENT1_NAME).expect("Failed to create agent1");
    let agent2 = Agent::new(AGENT2_NAME).expect("Failed to create agent2");

    // Get available plugins
    let plugins = agent1
        .get_available_plugins()
        .expect("Failed to get plugins");
    println!("Available plugins:");
    for plugin in plugins.iter() {
        println!("{}", plugin.expect("Failed to get plugin name"));
    }

    // Get plugin parameters for both agents
    let (mems1, params1) = agent1
        .get_plugin_params("UCX")
        .expect("Failed to get UCX params for agent1");
    let (mems2, params2) = agent2
        .get_plugin_params("UCX")
        .expect("Failed to get UCX params for agent2");

    println!("Params before init:");
    print_params(&params1, &mems1);
    print_params(&params2, &mems2);

    // Create backends for both agents
    let backend1 = agent1
        .create_backend("UCX", &params1)
        .expect("Failed to create backend for agent1");
    let backend2 = agent2
        .create_backend("UCX", &params2)
        .expect("Failed to create backend for agent2");

    // Create and populate optional arguments
    let mut extra_params1 = OptArgs::new().expect("Failed to create extra params for agent1");
    let mut extra_params2 = OptArgs::new().expect("Failed to create extra params for agent2");

    extra_params1
        .add_backend(&backend1)
        .expect("Failed to add backend to extra params1");
    extra_params2
        .add_backend(&backend2)
        .expect("Failed to add backend to extra params2");

    // The rest of the example will be implemented as we add more bindings
}

// #[test]
// fn test_nixl_transfer() {
//     // Create agent configs
//     let mut config1 = AgentConfig::new();
//     let mut config2 = AgentConfig::new();

//     // Create agents
//     let mut agent1 = Agent::new(AGENT1_NAME, &mut config1).unwrap();
//     let mut agent2 = Agent::new(AGENT2_NAME, &mut config2).unwrap();

//     // Get available plugins and their parameters
//     let plugins = agent1.get_available_plugins().unwrap();
//     let mut params1 = agent1.get_plugin_params(&plugins[0]).unwrap();
//     let mut params2 = agent2.get_plugin_params(&plugins[0]).unwrap();

//     // Create backends
//     let mut backend1 = agent1.create_backend(&mut params1).unwrap();
//     let mut backend2 = agent2.create_backend(&mut params2).unwrap();

//     // Create extra parameters with backends
//     let mut extra_params1 = OptArgs::new();
//     let mut extra_params2 = OptArgs::new();
//     extra_params1.add_backend(&mut backend1);
//     extra_params2.add_backend(&mut backend2);

//     // Register memory
//     let mut reg_desc1 = RegDlist::new();
//     let mut reg_desc2 = RegDlist::new();
//     let mut blob_desc1 = nixlBlobDesc::default();
//     let mut blob_desc2 = nixlBlobDesc::default();
//     // Set up blob descriptors...
//     reg_desc1.add_blob_desc(&mut blob_desc1);
//     reg_desc2.add_blob_desc(&mut blob_desc2);

//     let reg_handle1 = agent1
//         .register_memory(reg_desc1.inner_mut(), extra_params1.inner_mut())
//         .unwrap();
//     let reg_handle2 = agent2
//         .register_memory(reg_desc2.inner_mut(), extra_params2.inner_mut())
//         .unwrap();

//     // Exchange metadata
//     let meta1 = agent1.get_memory_metadata(reg_handle1).unwrap();
//     let meta2 = agent2.get_memory_metadata(reg_handle2).unwrap();

//     // Create transfer request
//     let mut src_desc = XferDlist::new();
//     let mut dst_desc = XferDlist::new();
//     let mut basic_desc1 = nixlBasicDesc::default();
//     let mut basic_desc2 = nixlBasicDesc::default();
//     // Set up basic descriptors...
//     src_desc.add_basic_desc(&mut basic_desc1);
//     dst_desc.add_basic_desc(&mut basic_desc2);

//     let mut extra_params = OptArgs::new();
//     extra_params.set_notification("Transfer in progress");

//     let xfer = agent1
//         .post_transfer(
//             NIXL_WRITE,
//             src_desc.inner_mut(),
//             dst_desc.inner_mut(),
//             extra_params.inner_mut(),
//         )
//         .unwrap();

//     // Wait for completion
//     let mut status = NIXL_IN_PROG;
//     while status == NIXL_IN_PROG {
//         status = agent1.check_transfer_status(xfer).unwrap();
//     }
//     assert_eq!(status, NIXL_SUCCESS);

//     // Check notifications
//     let notifs = Notifications::new();
//     let agent1_notifs = notifs.get_agent_notifications(AGENT1_NAME);
//     assert!(!agent1_notifs.is_empty());
//     println!("Agent 1 notifications: {:?}", agent1_notifs);

//     // Cleanup
//     agent1.unregister_memory(reg_handle1).unwrap();
//     agent2.unregister_memory(reg_handle2).unwrap();
// }
