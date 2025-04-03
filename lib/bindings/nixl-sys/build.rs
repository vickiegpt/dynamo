use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=nixl");
    println!("cargo:rustc-link-lib=serdes");
    println!("cargo:rustc-link-lib=ucx_utils");
    println!("cargo:rustc-link-lib=stream");
    println!("cargo:rustc-link-lib=nixl_build");

    // Link against C++ standard library
    println!("cargo:rustc-link-lib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");

    let nixl_include = "/opt/nvidia/nvda_nixl/include";

    // Build the C++ wrapper
    let mut builder = cc::Build::new();
    builder
        .cpp(true)
        .file("wrapper.cpp")
        // NIXL includes
        .include(nixl_include)
        .include(format!("{}/utils", nixl_include))
        .include(format!("{}/backend", nixl_include))
        // System C++ includes
        .include("/usr/include/c++/11")
        .include("/usr/include/x86_64-linux-gnu/c++/11")
        .include("/usr/include/c++/11/backward")
        .include("/usr/lib/gcc/x86_64-linux-gnu/11/include")
        .include("/usr/include/x86_64-linux-gnu")
        .include("/usr/include")
        // C++ flags
        .flag("-std=c++11")
        .flag("-fPIC")
        .compile("wrapper");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        // Enable C++ support
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++11")
        // NIXL includes
        .clang_arg(format!("-I{}", nixl_include))
        .clang_arg(format!("-I{}/utils", nixl_include))
        .clang_arg(format!("-I{}/backend", nixl_include))
        // System C++ includes
        .clang_arg("-I/usr/include/c++/11")
        .clang_arg("-I/usr/include/x86_64-linux-gnu/c++/11")
        .clang_arg("-I/usr/include/c++/11/backward")
        .clang_arg("-I/usr/lib/gcc/x86_64-linux-gnu/11/include")
        .clang_arg("-I/usr/include/x86_64-linux-gnu")
        .clang_arg("-I/usr/include")
        // Handle C++ types
        .size_t_is_usize(true)
        .opaque_type("std::.*")
        .opaque_type("nixl_notifs_t")
        .opaque_type("nixl_mem_list_t")
        .opaque_type("nixl_b_params_t")
        .opaque_type("nixl_opt_args_t")
        .allowlist_file("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
