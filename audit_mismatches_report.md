# SlothForge: Полный аудит интеграции фронтенда и бэкенда

Дата аудита: 12 сентября 2026 г.  
Коммит исправления: `c698655`  
Репозиторий: [github.com/metomorfinov/sloth-forge](https://github.com/metomorfinov/sloth-forge)

---

## 1. Расследование бага «DeepSeek весит дохуя, а пишет 5.0 GB»

### В чём заключалась проблема
Пользователь открыл модель **`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`** (284.3 миллиарда параметров, реальный размер весов квантования Q4 — более 155 ГБ):
* Внизу карточки красный бейдж корректно писал: `Over VRAM budget · ~153.3 GB` (фронтенд считал оценку из общего числа параметров `totalParams: 284.3B`).
* Однако в выпадающем списке вариантов квантования отображалось: **`Q4_K_M • GGUF 5.0 GB`**, и кнопка «Download» вела к 404-ошибке.

### Первопричина в бэкенде
В файле `crates/sloth-server/src/hub.rs` эндпоинт `GET /api/hub/gguf-variants` содержал заглушку со статическим сопоставлением подстрок в названии репозитория:
```rust
// СТАРЫЙ КОД В hub.rs:
let (q4_sz, q8_sz, rec) = if repo_lower.contains("70b") {
    (40_000_000_000u64, 75_000_000_000u64, false)
} else if repo_lower.contains("32b") || repo_lower.contains("30b") || repo_lower.contains("glm-5") {
    (20_000_000_000u64, 36_000_000_000u64, false)
} else if repo_lower.contains("14b") {
    (8_500_000_000u64, 15_000_000_000u64, false)
} else if repo_lower.contains("8b") || repo_lower.contains("7b") {
    (4_800_000_000u64, 8_500_000_000u64, false)
} else if repo_lower.contains("1b") || repo_lower.contains("0.5b") {
    (800_000_000u64, 1_500_000_000u64, true)
} else {
    // ЛЮБАЯ ДРУГАЯ МОДЕЛЬ (включая DeepSeek-V4-Flash-Vision-Exp)
    (5_000_000_000u64, 9_000_000_000u64, false) // <-- ВОТ ЗДЕСЬ ВОЗВРАЩАЛОСЬ 5.0 GB!
};
```
Так как в строке `DeepSeek-V4-Flash-Vision-Exp` нет подстроки «70b» или «32b», бэкенд отдавал фиктивные `5_000_000_000` байт (5.0 GB) и выдуманное имя файла `DeepSeek-V4-Flash-Vision-Exp-GGUF-Q4_K_M.gguf`.

### Второе узкое место: Шардированные модели
Модели такого гигантского размера разбиты на шарды (например, 5 файлов по ~49 ГБ внутри директории `UD-Q4_K_XL/`):
* `UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00001-of-00005.gguf` (5.3 MB)
* `UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00002-of-00005.gguf` (48.9 GB)
* `UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00003-of-00005.gguf` (48.9 GB)
* `UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00004-of-00005.gguf` (50.0 GB)
* `UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00005-of-00005.gguf` (7.17 GB)
**Сумма шардов: 155 095 288 672 байт (155.09 GB / 144.4 GiB).**

При попытке скачивания путь содержал вложенную папку `UD-Q4_K_XL/`, а бэкенд создавал только корневую папку `models/`, что приводило к ошибке файловой системы `No such file or directory`.

---

## 2. Реализованное исправление (Коммит `c698655`)

1. **Реальный опрос дерева файлов Hugging Face**:
   * Эндпоинт `GET /api/hub/gguf-variants` теперь динамически запрашивает бесплатный публичный API Hugging Face:
     `https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true`
   * Никакого HF-токена не требуется — всё парсится анонимно за миллисекунды.
2. **Агрегация шардов и расчёт реального веса**:
   * Все файлы одного квантования (например, 5 шардов `UD-Q4_K_XL`) автоматически группируются.
   * Их точные размеры суммируются в `size_bytes` и `download_size_bytes` (155 095 288 672 байт).
   * Количество шардов передаётся во фронтенд (`shard_count: 5`).
3. **Автоматическое определение мультимодальности (Vision)**:
   * Наличие проектора `mmproj-*.gguf` в репозитории автоматически выставляет `has_vision: true`.
4. **Кэширование в памяти**:
   * В `AppState` добавлен потокобезопасный LRU/TTL кэш на 10 минут (`gguf_variants_cache`), благодаря чему при повторном клике по модели карточка открывается мгновенно без повторного сетевого запроса к HF.
5. **Безопасное создание вложенных директорий**:
   * В `handle_download_start` добавлено рекурсивное создание родительских директорий (`tokio::fs::create_dir_all(parent)`), поэтому шарды в папках `UD-*` корректно скачиваются на диск.
6. **Надёжный офлайн-фоллбэк**:
   * Если сеть недоступна или HF временно не отвечает, фоллбэк учитывает архитектуры 284B, 70B, 32B, 14B, 7B, 3B, 1B и выставляет реальные масштабированные размеры вместо фиктивных 5 ГБ.

---

## 3. Сквозной аудит всех вкладок приложения

| Раздел / Модуль | Вызываемые API-эндпоинты | Статус бэкенда | Результат проверки |
|---|---|---|---|
| **Model Hub (Discover)** | `/api/hub/cached-models`, `/api/hub/cached-gguf`, `/api/hub/local`, `/api/hub/hidden-models`, `/api/hub/active-downloads`, `/api/hub/transport-status` | `200 OK` (Axum) | Список моделей рендерится без 404. Модели, превышающие 4 ГБ VRAM, получают бейдж `Over VRAM budget`. |
| **Model Hub (GGUF Variants)** | `/api/hub/gguf-variants?repo_id=...` | `200 OK` (Live HF + Cache) | Возвращает реальные имена шардов, форматы (`UD-Q4_K_XL`, `Q8_0`, `Q4_K_M` и т.д.) и точные байты (например, 155 ГБ для DeepSeek, 2.02 ГБ для Llama 3.2 3B). |
| **Model Hub (Download)** | `POST /api/hub/download`, `GET /api/hub/download-status`, `GET /api/hub/download-progress`, `POST /api/hub/download/cancel` | `200 OK` (Stream reqwest) | Скачивание работает без токена HF; создаются вложенные пути под шарды; реальный размер учитывается в шкале прогресса. |
| **Train (Дообучение)** | `POST /api/train/start`, `POST /api/train/stop`, `GET /api/train/status`, `GET /api/train/runs`, `GET /api/train/progress` | `200 OK` (LoRA + Vulkan) | Проверено: запуск шагов обучения, лосс-кривая (sparkline), сохранение истории запусков в `runs`. QLoRA под 4GB VRAM активен. |
| **Chat (Инференс)** | `POST /api/inference/load`, `POST /api/inference/unload`, `POST /api/inference/chat/completions`, `POST /api/inference/chat/count_tokens` | `200 OK` (SSE + JSON) | Проверено: одиночные ответы и потоковый SSE-стриминг чанка токенов на Vulkan-движке. |
| **System & Telemetry** | `GET /api/system`, `GET /api/health`, `WS /ws/telemetry` | `200 OK` (sysinfo 0.33 + VulkanContext) | Отдаются реальные значения: 6 ядер CPU, 15.5 GB RAM, 929 GB SSD, GPU AMD Radeon RX 570 Series 4.0 GB (3.65 GB свободно). |
| **Settings (Настройки)** | `/api/settings/personalization`, `/api/settings/vram-budget`, `/api/settings/upload-limit`, `/api/settings/last-local-model`, `/api/studio/install-source` | `200 OK` | Все настройки открываются без ошибок и уведомлений toast error. |

---

## 4. Результаты верификации

1. **Интеграционные тесты Rust**:
   ```
   cargo test -p sloth-server
   test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.05s
   ```
2. **Сборка фронтенда**:
   ```
   npm run build
   ✓ built in 2.81s (0 errors)
   ```
3. **Запрос к живому серверу по DeepSeek**:
   ```json
   {
     "default_variant": "UD-Q4_K_XL",
     "has_vision": true,
     "repo_id": "unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF",
     "variants_count": 10,
     "first_variant": {
       "display_label": "UD-Q4_K_XL",
       "download_size_bytes": 155095288672,
       "filename": "UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00001-of-00005.gguf",
       "files": [
         "UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00001-of-00005.gguf",
         "UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00002-of-00005.gguf",
         "UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00003-of-00005.gguf",
         "UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00004-of-00005.gguf",
         "UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00005-of-00005.gguf"
       ],
       "quant": "UD-Q4_K_XL",
       "shard_count": 5,
       "size_bytes": 155095288672
     }
   }
   ```
В интерфейсе теперь отображается реальный вес: **155.1 GB (144.4 GiB)** вместо 5.0 GB.
