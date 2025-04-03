use std::env;
use std::path::PathBuf;

fn main() {
    let nixl_root = "/opt/nvidia/nvda_nixl";
    let nixl_include = format!("{}/include", nixl_root);
    let nixl_lib = format!("{}/lib/x86_64-linux-gnu", nixl_root);

    // Tell cargo to look for shared libraries in the specified directories
    println!("cargo:rustc-link-search={}", nixl_lib);

    // Link against NIXL libraries in correct order
    println!("cargo:rustc-link-lib=dylib=nixl");
    println!("cargo:rustc-link-lib=dylib=nixl_build");
    println!("cargo:rustc-link-lib=dylib=serdes");

    // Build the C++ wrapper
    cc::Build::new()
        .cpp(true)
        .compiler("g++") // Ensure we're using the C++ compiler
        .file("wrapper.cpp")
        .include(&nixl_include)
        .flag("-std=c++17")
        .flag("-fPIC")
        // Change ABI flag if necessary to match your precompiled libraries:
        //        .flag("-D_GLIBCXX_USE_CXX11_ABI=0")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-unused-variable")
        .compile("wrapper");

    // Link against NIXL libraries in correct order
    println!("cargo:rustc-link-lib=dylib=nixl");
    println!("cargo:rustc-link-lib=dylib=nixl_build");
    println!("cargo:rustc-link-lib=dylib=serdes");

    // Link against C++ standard library
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");

    // Get the output path for bindings
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate bindings
    bindgen::Builder::default()
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
