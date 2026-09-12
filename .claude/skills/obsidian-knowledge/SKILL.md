---
name: obsidian-knowledge
description: Взаимодействие с Obsidian Vault пользователя (/home/rivergod/Documents/Obsidian Vault) для ведения графов знаний, таблиц и canvas-диаграмм.
---

# Obsidian Knowledge: Работа с базой знаний и графами

## 1. Контекст и расположение
- **Путь к хранилищу Obsidian Vault**: `/home/rivergod/Documents/Obsidian Vault`
- **Папка проекта**: `/home/rivergod/Documents/Obsidian Vault/SlothForge`
- **Инструменты доступа**:
  * MCP-сервер `obsidian` (seekstone): быстрый поиск по хранилищу и чтение заметок.
  * Прямая запись файлов: создание `.md` заметок и `.canvas` графов.

## 2. Ведение таблиц и заметок
- При принятии важных архитектурных решений или исправлении багов фиксируй данные в:
  `/home/rivergod/Documents/Obsidian Vault/SlothForge/SlothForge Hub.md`
- Используй Markdown-таблицы:
  * Таблица компонентов и путей
  * Таблица REST API эндпоинтов
  * Таблица локальных моделей и квантов
  * Журнал изменений и багов
- Используй Wiki-ссылки `[[Имя Заметки]]` для связывания сущностей между собой.

## 3. Obsidian Canvas (.canvas)
- Граф архитектуры хранится в:
  `/home/rivergod/Documents/Obsidian Vault/SlothForge/SlothForge Architecture.canvas`
- Формат — валидный JSON с массивами `nodes` и `edges`.
- Цветовая кодировка узлов:
  * `1` — Красный (ошибки, депрекейтед)
  * `2` — Оранжевый (Фронтенд / UI)
  * `3` — Желтый (Модели / Веса)
  * `4` — Зеленый (GPU / Vulkan Compute)
  * `5` — Голубой (Бэкенд / API Server)
  * `6` — Фиолетовый (ML Core / LoRA)
