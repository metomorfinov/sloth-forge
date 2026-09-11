use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=include/sloth_vulkan.h");
    println!("cargo:rerun-if-changed=rust/c_stub/stub.c");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let native_so = Path::new(&manifest_dir).join("libsloth_vulkan.so");
    let build_so = Path::new(&manifest_dir).join("build").join("libsloth_vulkan.so");

    if native_so.exists() {
        println!("cargo:rustc-link-search=native={}", manifest_dir);
        println!("cargo:rustc-link-lib=dylib=sloth_vulkan");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", manifest_dir);
    } else if build_so.exists() {
        let build_dir = Path::new(&manifest_dir).join("build");
        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!("cargo:rustc-link-lib=dylib=sloth_vulkan");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.display());
    } else {
        // Compile CPU-fallback stub so tests & compilation work reliably across all environments
        cc::Build::new()
            .file("rust/c_stub/stub.c")
            .include("include")
            .flag("-O3")
            .compile("sloth_vulkan_stub");
    }
}
