# Smart Gallery Guide Bot

Умный аудиогид для картинной галереи в формате Telegram-бота.


## Промежуточный отчет
Промежуточный отчет в `pdf` формате можно найти в директории `docs/` или [нажмите сюда](docs/intermediate_report.pdf)

## Решаемая проблема
В современных музеях услуги гида и аудиогида обладают значительными ограничениями с точки зрения персонализации и гибкости предоставления информации. Традиционные групповые экскурсии, проводимые гидами, не позволяют посетителям адаптировать маршрут под свои интересы. Аудиогиды обеспечивают индивидуальный формат, однако объем выдаваемой информации строго ограничен заранее записанными материалами и не предусматривает интерактивного взаимодействия и возможности получения справок на уникальные, частные вопросы. Современный посетитель заинтересован в свободном изучении экспозиций, выборе темпа осмотра и возможности получения расширенной информации об объектах в реальном времени.

## Технологический стек
* **Vector DB**: ChromaDB (HNSW, cosine similarity) с гибридным поиском поверх BM25 + RRF
* **ML Models**:
  * Vision Encoder: google/siglip-base-patch16-224
  * Text Encoder: deepvk/USER-bge-m3
  * VLM: Qwen3-VL-8B-Instruct (через сервер инференса vLLM)
* **Messenger**: python-telegram-bot

## Установка
Сначала необходимо склонировать репозиторий:

```bash
git clone https://github.com/Ivan-Sidorov/smart_gallery_guide.git
```

Затем перейти в директорию проекта и установить зависимости:

```bash
# переходим в проект
cd smart_gallery_guide

# создаем окружение
python3 -m venv .venv
source .venv/bin/activate

# ставим зависимости (прод)
pip install -e .

# ставим зависимости (разработка: pytest, ruff)
pip install -e ".[dev]"
```

## Структура проекта

```
smart_gallery_guide/
├── bot/              # Telegram бот (хэндлеры, клавиатуры)
├── agent/            # Агент-оркестратор запросов
├── models/           # ML-компоненты: text/vision энкодеры, VLM-клиент
├── database/         # Обёртка над ChromaDB + гибридный поиск (BM25/RRF)
├── services/         # Внешние сервисы (web-search)
├── scripts/          # Утилиты индексации и бенчмаркинга
├── config/           # Конфигурация
├── tests/            # pytest smoke-тесты
└── data/             # Данные экспонатов (изображения, метаданные, FAQ)
```

## Использование

### Переменные окружения
На первом шаге необходимо задать переменные окружения. С полным списком можно ознакомиться в `env.example`.

### Запуск vLLM сервера
Перед использованием VLM в сервисе необходимо запустить vLLM сервер:

```bash
./scripts/start_vllm_server.sh
```

### Запуск сервиса
Далее необходимо запустить самого бота:

```bash
python3 -m bot.bot
```

## Детали

### Расширение базы экспонатов

Скрипт выгружает изображения и формирует метадату по шаблону `data/metadata/example_exhibit.json`.

Датасет на Hugging Face:
`https://huggingface.co/datasets/Artificio/WikiArt`

```bash
# Загрузит 500 картин в data/exhibits и создаст 500 JSON в data/metadata
python3 scripts/export_wikiart_exhibits.py --count 500
```

### Расширение метадаты через Perplexity (Sonar)

Скрипт читает `data/metadata/*.json`, берет `title` и `artist`, запрашивает информацию у Perplexity (модель `sonar`) со **structured output (JSON Schema)** и сохраняет результат в `data/metadata_expand/*.json` в формате:

```json
{
  "title": "",
  "artist": "",
  "year": "",
  "style": "",
  "genre": "",
  "description": "",
  "interesting_facts": []
}
```

Запуск генерации метадаты через Perplexity:

```bash
python3 scripts/expand_metadata_perplexity.py --limit 500
```

Объединение базовой метадаты с расщиренной:

```bash
python3 scripts/merge_metadata_expand.py --limit 500
```



## Конфигурация
Основные настройки задаются переменными окружения (см. `env.example`):

**Telegram**
* `TELEGRAM_BOT_TOKEN` — токен Telegram-бота.

**ChromaDB**
* `CHROMA_PERSIST_DIR` — каталог персистентного хранилища Chroma (по умолчанию `./chroma_store`).
* `CHROMA_COLLECTION_EXHIBITS`, `CHROMA_COLLECTION_TITLE`, `CHROMA_COLLECTION_DESC`, `CHROMA_COLLECTION_FAQ` — имена коллекций.

**vLLM (сервер инференса VLM)**
* `VLLM_API_BASE_URL` — URL OpenAI-совместимого vLLM API.
* `VLLM_VLM_MODEL` — идентификатор модели.
* `VLLM_VLM_MAX_TOKENS`, `VLLM_VLM_TEMPERATURE` — параметры генерации.
* `VLLM_API_KEY` — ключ доступа к API.
* `VLLM_SYSTEM_PROMPT` — системный промпт.

**Энкодеры**
* `VISION_ENCODER_MODEL` — идентификатор визуального энкодера.
* `TEXT_ENCODER_MODEL` — идентификатор текстового энкодера.

**Пороги ранжирования**
* `EXHIBIT_MATCH_THRESHOLD` — порог уверенного совпадения.
* `FAQ_RELEVANCE_THRESHOLD` — порог, при котором ответ из FAQ возвращается без VLM-fallback.
* `DISPLAY_SCORE_THRESHOLD` — минимальный score для показа результата пользователю.

**Веб-поиск (обогащение ответов VLM)**
* `WEB_SEARCH_ENABLED` — включение веб-поиска (`true`/`false`).
* `WEB_SEARCH_MAX_RESULTS` — максимум сниппетов в контексте.

**Perplexity (используется только скриптами расширения метаданных)**
* `PERPLEXITY_API_KEY`, `PERPLEXITY_BASE_URL`, `PERPLEXITY_MODEL`.

## Разработка

Запуск тестов:

```bash
pytest
```

Проверка стиля и линтер:

```bash
ruff check .
ruff format --check .
```
