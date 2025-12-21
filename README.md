# Smart Gallery Guide Bot

Умный аудиогид для картинной галереи в формате Telegram-бота.


## Промежуточный отчет
Промежуточный отчет в `pdf` формате можно найти в директории `docs/` или [нажмите сюда](docs/intermediate_report.pdf)

## Решаемая проблема
В современных музеях услуги гида и аудиогида обладают значительными ограничениями с точки зрения персонализации и гибкости предоставления информации. Традиционные групповые экскурсии, проводимые гидами, не позволяют посетителям адаптировать маршрут под свои интересы. Аудиогиды обеспечивают индивидуальный формат, однако объем выдаваемой информации строго ограничен заранее записанными материалами и не предусматривает интерактивного взаимодействия и возможности получения справок на уникальные, частные вопросы. Современный посетитель заинтересован в свободном изучении экспозиций, выборе темпа осмотра и возможности получения расширенной информации об объектах в реальном времени.

## Технологический стек
* **Vector DB**: FAISS (IndexFlatIP)
* **ML Models**:
  * Vision Encoder: CLIP
  * VLM: Qwen3-VL-8B-Instruct

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

# ставим зависимости
pip install -e .
```

## Структура проекта

```
smart_gallery_guide/
├── bot/              # Telegram бот
├── agent/            # Агент обработки запросов
├── models/           # ML модели
├── database/         # Работа с векторной БД
├── scripts/          # Утилиты и скрипты
├── config/           # Конфигурация
└── data/             # Данные экспонатов
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
Основные настройки находятся в `.env` файле:
* `TELEGRAM_BOT_TOKEN` - токен тг бота
* `FAISS_STORAGE_DIR` - путь к каталогу с индексами FAISS (по умолчанию, `./faiss_store`)
* `VLLM_API_BASE_URL` - URL vllm API сервера
* `VLLM_VLM_MODEL` - название VLM модели
* `VLLM_VLM_MAX_TOKENS` - максимальное количество токенов в ответе
* `VLLM_VLM_TEMPERATURE` - температура для генерации
* `VLLM_API_KEY` - vllm API ключ
* Пороги для поиска и релевантности (`EXHIBIT_MATCH_THRESHOLD`, `FAQ_RELEVANCE_THRESHOLD`)
* `PERPLEXITY_API_KEY` - API ключ Perplexity (для `scripts/expand_metadata_perplexity.py`)
* `PERPLEXITY_BASE_URL` - base URL Perplexity API (по умолчанию, `https://api.perplexity.ai`)
* `PERPLEXITY_MODEL` - модель Perplexity (по умолчанию, `sonar`)
