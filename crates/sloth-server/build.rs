use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();
    let vk_dir = root.join("crates").join("sloth-vulkan-sys");

    println!("cargo:rustc-link-search=native={}", vk_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", vk_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../crates/sloth-vulkan-sys");
}
