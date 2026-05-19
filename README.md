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
├── api/              # FastAPI backend (REST + orchestration)
├── adapters/         # Мессенджер-адаптеры (сейчас Telegram)
├── workers/          # Фоновые воркеры (VLM queue worker)
├── core/             # ML-ядро: agent, encoders, vector DB, search
├── db/               # SQLAlchemy модели, репозитории, миграции
├── scripts/          # Утилиты индексации и бенчмаркинга
├── tests/            # pytest smoke-тесты
└── data/             # Данные экспонатов (изображения, метаданные, FAQ)
```

## Использование

### Переменные окружения
На первом шаге необходимо задать переменные окружения. С полным списком можно ознакомиться в `env.example`.

```bash
cp env.example .env
# отредактируйте .env при необходимости
```

### PostgreSQL локально (без Docker)

API, worker и скрипты загрузки данных требуют работающий PostgreSQL. При старте API проверяется подключение и наличие схемы (таблица `exhibits`).

Значения по умолчанию из `env.example`:

```text
DATABASE_URL=postgresql+asyncpg://smart_guide:smart_guide@localhost:5432/smart_guide
```

Убедитесь, что в `.env` указан тот же URL (или свой хост/порт/логин).

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
# or
sudo service postgresql start
```

#### Создание пользователя и базы

```bash
# Linux
sudo -u postgres psql
```

В интерактивной консоли `psql`:

```sql
CREATE USER smart_guide WITH PASSWORD 'smart_guide';
CREATE DATABASE smart_guide OWNER smart_guide;
\q
```

Проверка:

```bash
psql "postgresql://smart_guide:smart_guide@localhost:5432/smart_guide" -c "SELECT 1"
```

#### Миграции схемы

Из корня проекта, с активированным `.venv`:

```bash
source .venv/bin/activate
export DATABASE_URL=postgresql+asyncpg://smart_guide:smart_guide@localhost:5432/smart_guide
alembic upgrade head
```

После этого можно запускать API и worker (см. ниже). Если Postgres не запущен, при старте API будет `Connection refused` на порту `5432`.

### Запуск vLLM сервера (локально)
Перед использованием VLM в сервисе необходимо запустить vLLM сервер:

```bash
./scripts/start_vllm_server.sh
```

### Запуск сервиса локально
Перед запуском: PostgreSQL запущен, в `.env` задан `DATABASE_URL`, выполнен `alembic upgrade head` (см. [PostgreSQL локально](#postgresql-локально-без-docker)).

Для локального запуска новой архитектуры нужны три процесса:

```bash
# 1) API
uvicorn api.main:app --host 0.0.0.0 --port 8081

# 2) worker
python3 -m workers.vlm_worker

# 3) Telegram adapter
python3 -m adapters.telegram.app
```

### Запуск полного стека через Docker Compose

Compose поднимает:
- `postgres`
- `api` (FastAPI + Chroma persistent volume)
- `adapter-telegram`
- `vlm-worker`
- `vllm`
- `nginx` (reverse proxy на API)

Быстрый старт:

```bash
# заполнить env
cp env.example .env

# сборка и запуск
docker compose -f deploy/docker-compose.yml up -d --build

# опционально: второй воркер очереди
docker compose -f deploy/docker-compose.yml up -d --scale vlm-worker=2
```

Проверка готовности:

```bash
curl http://localhost:8081/readyz
curl http://localhost/healthz
```

`migrator` запускает `alembic upgrade head` автоматически перед стартом API.

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

### Загрузка в Postgres

1. Загрузка экспонатов из `data/exhibits` + `data/metadata` в таблицу `exhibits`:

```bash
python3 scripts/load_exhibits.py
```

2. Загрузка FAQ из `data/faq` в таблицу `faq_items`:

```bash
python3 scripts/load_faq.py
```

3. Синхронизация Chroma из Postgres:

```bash
python3 scripts/reindex_chroma.py
```

Для измененных экспонатов `load_exhibits.py` выставляет `needs_reindex=true`, а `reindex_chroma.py` пересчитывает эмбеддинги и обновляет индексы Chroma.
`load_faq.py` записывает FAQ в Postgres, после чего `reindex_chroma.py` переносит их в FAQ-индекс Chroma (по умолчанию для записей с `indexed_at IS NULL`).



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
* Системные промпты VLM — `core/vlm/prompts.py` (`DEFAULT_BASE_SYSTEM_PROMPT`, `DEFAULT_SEARCH_EVAL_SYSTEM_PROMPT`).

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

**PostgreSQL**
* `DATABASE_URL` — строка подключения SQLAlchemy/asyncpg (по умолчанию `localhost:5432`, БД `smart_guide`).
* `DATABASE_ECHO`, `DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW` — пул соединений.

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
