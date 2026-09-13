//! Сборка Vulkan-слоя.
//!
//! Раньше скрипт линковал готовую `libsloth_vulkan.so` из папки крейта, если она лежала рядом,
//! и никогда её не пересобирал: правки C++ и шейдеров молча не попадали в программу, а на
//! Windows (где файл называется `.dll`) всегда собиралась CPU-заглушка без видеокарты.
//!
//! Теперь C++-слой компилируется из исходников при каждом их изменении и встраивается в
//! программу статически. Режим задаёт переменная `SLOTH_VULKAN`:
//! - `auto` (по умолчанию): настоящий Vulkan, если найден загрузчик, иначе CPU-заглушка;
//! - `native`: только настоящий Vulkan, без загрузчика сборка падает с понятной ошибкой;
//! - `stub`: CPU-заглушка (для проверки заглушки на машине с видеокартой).

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const MODE_ENV: &str = "SLOTH_VULKAN";
const GLSLANG_ENV: &str = "GLSLANG_VALIDATOR";
const VULKAN_SDK_ENV: &str = "VULKAN_SDK";

const CPP_SOURCES: [&str; 4] = [
    "src/vk_context.cpp",
    "src/vk_buffer.cpp",
    "src/vk_pipeline.cpp",
    "src/sloth_vulkan_api.cpp",
];
const SHADERS: [&str; 5] = [
    "gemm_f32",
    "lora_forward",
    "lora_backward",
    "rmsnorm",
    "adamw",
];
const GENERATED_SPV_HEADER: &str = "spv_embedded_generated.h";
/// Папки, где дистрибутивы Linux держат `libvulkan.so`.
const UNIX_LIBRARY_DIRS: [&str; 5] = [
    "/usr/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/local/lib",
];

/// Как слинковать загрузчик Vulkan.
struct VulkanLoader {
    search_dir: Option<PathBuf>,
    lib_name: &'static str,
}

fn main() {
    for path in ["include", "src", "shaders", "rust/c_stub"] {
        println!("cargo:rerun-if-changed={path}");
    }
    for var in [MODE_ENV, GLSLANG_ENV, VULKAN_SDK_ENV, "LIBRARY_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let mode = env::var(MODE_ENV).unwrap_or_else(|_| "auto".to_string());
    match mode.as_str() {
        "stub" => build_stub(),
        "native" => match find_vulkan_loader() {
            Some(loader) => build_native(&loader),
            None => panic!(
                "{MODE_ENV}=native, но загрузчик Vulkan не найден: установите пакет с libvulkan \
                 (Linux) или Vulkan SDK с переменной {VULKAN_SDK_ENV} (Windows)"
            ),
        },
        "auto" => match find_vulkan_loader() {
            Some(loader) => build_native(&loader),
            None => {
                println!(
                    "cargo:warning=Загрузчик Vulkan не найден: собрана CPU-заглушка без видеокарты \
                     (установите libvulkan или Vulkan SDK)"
                );
                build_stub();
            }
        },
        other => panic!("{MODE_ENV}={other}: допустимые значения auto, native, stub"),
    }
}

fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").unwrap_or_default()
}

fn find_vulkan_loader() -> Option<VulkanLoader> {
    let sdk = env::var_os(VULKAN_SDK_ENV).map(PathBuf::from);
    if target_os() == "windows" {
        let lib_dir = sdk?.join("Lib");
        return lib_dir
            .join("vulkan-1.lib")
            .exists()
            .then_some(VulkanLoader {
                search_dir: Some(lib_dir),
                lib_name: "vulkan-1",
            });
    }

    let file_name = if target_os() == "macos" {
        "libvulkan.dylib"
    } else {
        // Без версии в имени: именно этот файл нужен компоновщику для `-lvulkan`
        "libvulkan.so"
    };
    let sdk_dirs = sdk.into_iter().map(|sdk| sdk.join("lib"));
    let library_path_dirs = env::var_os("LIBRARY_PATH")
        .map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default();
    let candidates: Vec<PathBuf> = sdk_dirs
        .chain(library_path_dirs)
        .chain(UNIX_LIBRARY_DIRS.iter().map(PathBuf::from))
        .collect();
    let dir = candidates
        .into_iter()
        .find(|dir| dir.join(file_name).exists())?;
    Some(VulkanLoader {
        search_dir: Some(dir),
        lib_name: "vulkan",
    })
}

fn build_native(loader: &VulkanLoader) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cargo задаёт OUT_DIR"));
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++20")
        .include("include")
        .include("src")
        .define("SLOTH_VK_STATIC", None)
        .warnings(true)
        .extra_warnings(true)
        .files(CPP_SOURCES);
    if let Some(generated_dir) = compile_shaders(&out_dir) {
        build
            .include(generated_dir)
            .define("SLOTH_VK_GENERATED_SPV", None);
    }
    build.compile("sloth_vulkan");

    if let Some(dir) = &loader.search_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=dylib={}", loader.lib_name);
    println!("cargo:rustc-env=SLOTH_VK_BACKEND=vulkan");
}

fn build_stub() {
    cc::Build::new()
        .file("rust/c_stub/stub.c")
        .include("include")
        .define("SLOTH_VK_STATIC", None)
        .warnings(true)
        .compile("sloth_vulkan_stub");
    println!("cargo:rustc-env=SLOTH_VK_BACKEND=stub");
}

fn glslang_command() -> Option<String> {
    let command = env::var(GLSLANG_ENV).unwrap_or_else(|_| "glslangValidator".to_string());
    let available = Command::new(&command)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    available.then_some(command)
}

/// Компилирует шейдеры в SPIR-V и пишет заголовок со встроенными копиями в `OUT_DIR`.
/// Возвращает папку с заголовком или `None`, если компилятора шейдеров нет.
fn compile_shaders(out_dir: &Path) -> Option<PathBuf> {
    let Some(glslang) = glslang_command() else {
        println!(
            "cargo:warning=glslangValidator не найден: используются сохранённые копии шейдеров \
             (src/spv_embedded.h), правки shaders/*.comp в сборку не попадут"
        );
        return None;
    };
    let spv_dir = out_dir.join("spv");
    fs::create_dir_all(&spv_dir).expect("папка для SPIR-V создана");

    let mut header = String::from(
        "// Сгенерировано build.rs из shaders/*.comp, не редактировать.\n\
         #ifndef SLOTH_SPV_EMBEDDED_GENERATED_H\n#define SLOTH_SPV_EMBEDDED_GENERATED_H\n\n\
         #include <cstddef>\n#include <cstdint>\n\nnamespace sloth {\n",
    );
    for shader in SHADERS {
        let source = Path::new("shaders").join(format!("{shader}.comp"));
        let output_path = spv_dir.join(format!("{shader}.spv"));
        let output = Command::new(&glslang)
            // -V без --target-env даёт SPIR-V 1.0: он загружается и на драйверах Vulkan 1.0
            .arg("-V")
            .arg(&source)
            .arg("-o")
            .arg(&output_path)
            .output()
            .expect("glslangValidator запускается");
        if !output.status.success() {
            panic!(
                "Шейдер {} не скомпилировался:\n{}{}",
                source.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let bytes = fs::read(&output_path).expect("SPIR-V прочитан");
        assert!(
            bytes.len().is_multiple_of(4),
            "размер SPIR-V {} не кратен 4",
            output_path.display()
        );
        append_word_table(&mut header, shader, &bytes);
    }
    header.push_str("} // namespace sloth\n#endif\n");

    let include_dir = out_dir.join("include");
    fs::create_dir_all(&include_dir).expect("папка заголовков создана");
    fs::write(include_dir.join(GENERATED_SPV_HEADER), header).expect("заголовок шейдеров записан");
    Some(include_dir)
}

/// Та же таблица, что пишет `make embed`: `spv_<имя>_size` в байтах и слова SPIR-V.
fn append_word_table(header: &mut String, shader: &str, bytes: &[u8]) {
    const WORDS_PER_LINE: usize = 8;
    let (chunks, _) = bytes.as_chunks::<4>();
    let words: Vec<u32> = chunks
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect();
    let _ = writeln!(
        header,
        "inline const size_t spv_{shader}_size = {};",
        bytes.len()
    );
    let _ = writeln!(header, "inline const uint32_t spv_{shader}_data[] = {{");
    for line in words.chunks(WORDS_PER_LINE) {
        let row: Vec<String> = line.iter().map(|word| format!("0x{word:08x}u")).collect();
        let _ = writeln!(header, "    {},", row.join(", "));
    }
    header.push_str("};\n\n");
}
