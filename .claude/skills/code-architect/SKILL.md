---
name: code-architect
description: Стандарты архитектуры кода, оптимизации под Vulkan/Polaris, чистого Rust и строгой типизации TypeScript.
---

# Code Architect: Стандарты разработки

## 1. Rust (Backend & Core)
- **Обработка ошибок**: Используй `Result<T, E>` и явные ошибки с `thiserror` в библиотеках (`sloth-core`), либо структурированные ошибки в `sloth-server`. Никаких пустых `unwrap()` на путях выполнения пользователя.
- **Axum эндпоинты**:
  * Если эндпоинт может вызываться без тела запроса или с разным Content-Type, оборачивай пейлоад в `Option<Json<Value>>`, чтобы избежать автоматического HTTP 415.
  * Всегда возвращай структуру, ожидаемую TypeScript интерфейсом фронтенда.
- **Асинхронность**: Не блокируй Tokio runtime тяжелыми вычислениями — используй `tokio::task::spawn_blocking` для синхронных дисковых или математических операций.

## 2. Vulkan & C++20 (Compute Engine)
- **Polaris 10 (AMD RX 570)**: Архитектура GCN 4.0, Wavefront 64. Размер рабочей группы в шейдерах (`local_size_x = 64` или кратно 64) для 100% утилизации вычислительных блоков.
- **Управление памятью VRAM**:
  * `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` для весов и градиентов (быстрая VRAM 4GB).
  * Staging буферы (`HOST_VISIBLE | HOST_COHERENT`) для трансфера данных между хостом и GPU.
- **Барьеры памяти**: Явные `vkCmdPipelineBarrier` с `VK_ACCESS_SHADER_WRITE_BIT -> VK_ACCESS_SHADER_READ_BIT` между проходами GEMM/LoRA.

## 3. TypeScript & React
- **Никаких `any`**: Явно типизируй ответы бэкенда и состояния компонентов.
- **Обработка статусов сети**: При вызове API всегда отображай состояния: `loading`, `error`, `success`. Не глуши ошибки в пустых `catch {}`.
- **Сборка**: Перед завершением работы всегда проверяй чистоту сборки `npm run build`.
