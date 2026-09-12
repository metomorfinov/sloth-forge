//! Песочница путей.
//!
//! Любой путь, пришедший из HTTP-запроса, разрешается только внутри заранее
//! разрешённых корневых папок: папки моделей, добавленных пользователем
//! scan-folders или домашней папки (для обзора каталогов). Без этого запрос
//! к API мог удалить или прочитать любой файл на диске.

use std::path::{Component, Path, PathBuf};

/// Почему путь из запроса отклонён.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathRejection {
    /// Пустой путь, нулевой байт или объект не того типа (например, папка вместо файла).
    Invalid,
    /// По этому пути ничего нет ни в одном из корней.
    NotFound,
    /// Объект существует, но лежит вне разрешённых корней.
    OutsideRoots,
}

/// Разрешает существующий путь пользователя внутри одного из корней.
///
/// Относительный путь пробуется от каждого корня по очереди, абсолютный — как есть.
/// Результат проходит через `canonicalize`, поэтому `..` и символические ссылки,
/// ведущие наружу, не помогают выйти за пределы корня.
pub fn resolve_existing_within(
    roots: &[PathBuf],
    user_path: &str,
) -> Result<PathBuf, PathRejection> {
    let trimmed = user_path.trim();
    if trimmed.is_empty() || trimmed.contains('\0') {
        return Err(PathRejection::Invalid);
    }

    // Корни тоже приводим к каноническому виду, иначе сравнение префиксов ненадёжно.
    let canonical_roots: Vec<PathBuf> = roots
        .iter()
        .filter_map(|root| root.canonicalize().ok())
        .collect();
    if canonical_roots.is_empty() {
        return Err(PathRejection::NotFound);
    }

    let requested = Path::new(trimmed);
    let candidates: Vec<PathBuf> = if requested.is_absolute() {
        vec![requested.to_path_buf()]
    } else {
        canonical_roots
            .iter()
            .map(|root| root.join(requested))
            .collect()
    };

    let mut exists_outside = false;
    for candidate in candidates {
        let Ok(resolved) = candidate.canonicalize() else {
            continue;
        };
        if canonical_roots
            .iter()
            .any(|root| resolved.starts_with(root))
        {
            return Ok(resolved);
        }
        exists_outside = true;
    }

    Err(if exists_outside {
        PathRejection::OutsideRoots
    } else {
        PathRejection::NotFound
    })
}

/// То же, что [`resolve_existing_within`], но результат обязан быть обычным файлом.
pub fn resolve_existing_file_within(
    roots: &[PathBuf],
    user_path: &str,
) -> Result<PathBuf, PathRejection> {
    let resolved = resolve_existing_within(roots, user_path)?;
    if resolved.is_file() {
        Ok(resolved)
    } else {
        Err(PathRejection::Invalid)
    }
}

/// То же, что [`resolve_existing_within`], но результат обязан быть папкой.
pub fn resolve_existing_dir_within(
    roots: &[PathBuf],
    user_path: &str,
) -> Result<PathBuf, PathRejection> {
    let resolved = resolve_existing_within(roots, user_path)?;
    if resolved.is_dir() {
        Ok(resolved)
    } else {
        Err(PathRejection::Invalid)
    }
}

/// Проверяет, что строка — одно безопасное имя файла без разделителей папок и `..`.
/// Нужна там, где имя подставляется в путь (например, `{variant}.gguf`).
pub fn is_plain_file_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    ) && !name.contains(['/', '\\', '\0'])
}

/// Домашняя папка пользователя: `HOME` на Linux/macOS, `USERPROFILE` на Windows.
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn sandbox() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let base = tempfile::tempdir().expect("не удалось создать временную папку");
        let root = base.path().join("models");
        let outside = base.path().join("secret.txt");
        fs::create_dir_all(root.join("sub")).expect("не удалось создать models/sub");
        fs::write(root.join("model.gguf"), b"GGUF").expect("не удалось создать model.gguf");
        fs::write(&outside, b"secret").expect("не удалось создать secret.txt");
        (base, root, outside)
    }

    #[test]
    fn accepts_relative_file_inside_root() {
        let (_base, root, _) = sandbox();
        let resolved =
            resolve_existing_file_within(std::slice::from_ref(&root), "model.gguf").unwrap();
        assert_eq!(resolved, root.join("model.gguf").canonicalize().unwrap());
    }

    #[test]
    fn accepts_absolute_file_inside_root() {
        let (_base, root, _) = sandbox();
        let absolute = root.join("model.gguf");
        assert!(resolve_existing_file_within(&[root], absolute.to_str().unwrap()).is_ok());
    }

    #[test]
    fn rejects_absolute_path_outside_root() {
        let (_base, root, outside) = sandbox();
        let result = resolve_existing_within(&[root], outside.to_str().unwrap());
        assert_eq!(result, Err(PathRejection::OutsideRoots));
    }

    #[test]
    fn rejects_parent_traversal() {
        let (_base, root, _) = sandbox();
        let result = resolve_existing_within(&[root], "../secret.txt");
        assert_eq!(result, Err(PathRejection::OutsideRoots));
    }

    #[test]
    fn reports_missing_and_invalid() {
        let (_base, root, _) = sandbox();
        assert_eq!(
            resolve_existing_within(std::slice::from_ref(&root), "nope.gguf"),
            Err(PathRejection::NotFound)
        );
        assert_eq!(
            resolve_existing_within(std::slice::from_ref(&root), "   "),
            Err(PathRejection::Invalid)
        );
        assert_eq!(
            resolve_existing_file_within(&[root], "sub"),
            Err(PathRejection::Invalid)
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escaping_root() {
        let (_base, root, outside) = sandbox();
        std::os::unix::fs::symlink(&outside, root.join("link.gguf")).unwrap();
        let result = resolve_existing_within(&[root], "link.gguf");
        assert_eq!(result, Err(PathRejection::OutsideRoots));
    }

    #[test]
    fn plain_file_name_check() {
        assert!(is_plain_file_name("Q4_K_M.gguf"));
        assert!(!is_plain_file_name("../Q4_K_M.gguf"));
        assert!(!is_plain_file_name("sub/Q4_K_M.gguf"));
        assert!(!is_plain_file_name(".."));
        assert!(!is_plain_file_name(""));
    }
}
