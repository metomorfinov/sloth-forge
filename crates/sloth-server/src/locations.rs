//! Где сервер ищет свои файлы: собранный фронтенд и папку моделей.
//!
//! Раньше пути были зашиты под одну машину (`/home/rivergod/...`) или считались
//! от папки, из которой запустили сервер. Теперь порядок поиска одинаков везде:
//! 1. переменная окружения (`STATIC_DIR`, `SLOTH_MODELS_DIR`);
//! 2. рядом с исполняемым файлом (установленная версия, например на Windows);
//! 3. корень проекта, из которого собран сервер (разработка через `cargo run`);
//! 4. текущая папка.

use std::path::{Path, PathBuf};

/// Переменная окружения с путём к собранному фронтенду.
pub const STATIC_DIR_ENV: &str = "STATIC_DIR";
/// Переменная окружения с путём к папке моделей.
pub const MODELS_DIR_ENV: &str = "SLOTH_MODELS_DIR";

/// Корень репозитория на момент сборки: `crates/sloth-server/../..`.
fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(Path::to_path_buf)
}

/// Кандидаты в порядке приоритета: рядом с exe, корень проекта, текущая папка.
fn candidates(relative: &Path) -> Vec<PathBuf> {
    let mut list = Vec::new();
    if let Some(dir) = exe_dir() {
        list.push(dir.join(relative));
    }
    list.push(project_root().join(relative));
    if let Ok(cwd) = std::env::current_dir() {
        list.push(cwd.join(relative));
    }
    list
}

/// Убирает `..` из пути, если он существует; иначе оставляет как есть.
fn normalize(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

/// Папка собранного фронтенда (`frontend/dist` с `index.html`) или `None`, если её нет.
pub fn find_static_dir() -> Option<PathBuf> {
    if let Some(custom) = std::env::var_os(STATIC_DIR_ENV) {
        return Some(PathBuf::from(custom));
    }
    candidates(Path::new("frontend").join("dist").as_path())
        .into_iter()
        .find(|dir| dir.join("index.html").is_file())
        .map(normalize)
}

/// Папка моделей. Если ни один кандидат не существует, возвращается `models`
/// рядом с исполняемым файлом: туда загрузчик и положит первую модель.
pub fn find_models_dir() -> PathBuf {
    if let Some(custom) = std::env::var_os(MODELS_DIR_ENV) {
        return PathBuf::from(custom);
    }
    candidates(Path::new("models"))
        .into_iter()
        .find(|dir| dir.is_dir())
        .map(normalize)
        .unwrap_or_else(|| {
            exe_dir()
                .map(|dir| dir.join("models"))
                .unwrap_or_else(|| PathBuf::from("models"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_root_contains_workspace_manifest() {
        assert!(
            project_root().join("Cargo.toml").is_file(),
            "корень проекта должен указывать на папку с workspace Cargo.toml"
        );
    }

    #[test]
    fn models_dir_is_found_in_project_without_env() {
        // В репозитории папка models/ есть всегда (в ней лежит README.txt)
        if std::env::var_os(MODELS_DIR_ENV).is_none() {
            assert!(find_models_dir().is_dir());
        }
    }
}
