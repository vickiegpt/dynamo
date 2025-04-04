// #![allow(unused_variables)]

use nixl_sys::*;

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
    const AGENT1_NAME: &str = "agent1";
    const AGENT2_NAME: &str = "agent2";

    // Create agents
    let agent1 = Agent::new(AGENT1_NAME).unwrap();
    let agent2 = Agent::new(AGENT2_NAME).unwrap();

    // Get available plugins
    let plugins1 = agent1.get_available_plugins().unwrap();
    let plugins2 = agent2.get_available_plugins().unwrap();

    println!("Available plugins for agent1:");
    for plugin in plugins1.iter() {
        println!("  {}", plugin.unwrap());
    }

    println!("Available plugins for agent2:");
    for plugin in plugins2.iter() {
        println!("  {}", plugin.unwrap());
    }

    // Get plugin parameters
    let plugin_name = plugins1.get(0).unwrap();
    let (mems1, params1) = agent1.get_plugin_params(&plugin_name).unwrap();
    let (mems2, params2) = agent2.get_plugin_params(&plugin_name).unwrap();

    println!("Initial params for agent1:");
    print_params(&params1, &mems1);
    println!("Initial params for agent2:");
    print_params(&params2, &mems2);

    // Create backends
    let backend1 = agent1.create_backend(&plugin_name, &params1).unwrap();
    let backend2 = agent2.create_backend(&plugin_name, &params2).unwrap();

    // Get backend parameters after initialization
    let (mems1_after, params1_after) = agent1.get_backend_params(&backend1).unwrap();
    let (mems2_after, params2_after) = agent2.get_backend_params(&backend2).unwrap();

    println!("Params after init for agent1:");
    print_params(&params1_after, &mems1_after);
    println!("Params after init for agent2:");
    print_params(&params2_after, &mems2_after);

    // Create optional arguments and add backends
    let mut extra_params1 = OptArgs::new().unwrap();
    let mut extra_params2 = OptArgs::new().unwrap();
    extra_params1.add_backend(&backend1).unwrap();
    extra_params2.add_backend(&backend2).unwrap();

    // Allocate and initialize memory regions
    let mut storage1 = SystemStorage::new(256).unwrap();
    let mut storage2 = SystemStorage::new(256).unwrap();

    // Initialize memory patterns
    storage1.memset(0xbb);
    storage2.memset(0x00);

    // Create registration descriptor lists
    let mut dlist1 = RegDescList::new(MemType::Dram).unwrap();
    let mut dlist2 = RegDescList::new(MemType::Dram).unwrap();

    // Add descriptors
    dlist1.add_storage_desc(&storage1).unwrap();
    dlist2.add_storage_desc(&storage2).unwrap();

    // Verify descriptor lists
    assert_eq!(dlist1.len().unwrap(), 1);
    assert_eq!(dlist2.len().unwrap(), 1);

    // Verify memory patterns
    assert!(storage1.as_slice().iter().all(|&x| x == 0xbb));
    assert!(storage2.as_slice().iter().all(|&x| x == 0x00));

    // More implementation will follow as we add more bindings
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
