# FAQ storage schema

FAQ files are stored separately from exhibit metadata in this directory:

- `data/faq/<exhibit_id>.json`

Each FAQ file is linked to an exhibit by `exhibit_id`.

## JSON schema (practical format)

```json
{
    "exhibit_id": "3031",
    "title": "Монета. 552-565 гг.",
    "artist": "",
    "questions": [
        {
            "question": "Почему эта монета важна для своей эпохи?",
            "answer": "Она отражает денежное обращение и политическую символику времени правления Юстиниана I."
        },
        {
            "question": "Что можно понять о правителе по изображению на монете?",
            "answer": "Иконография подчеркивает легитимность власти и связь императора с христианской идеей государства."
        },
        {
            "question": "Как материал монеты связан с ее назначением?",
            "answer": "Серебро указывает на ее участие в повседневных расчетах и региональной торговле."
        }
    ],
    "question_count": 3,
    "source_model": "sonar",
    "source_file": "3031.json",
    "generated_at_unix": 1773388800
}
```

## Field meaning

- `exhibit_id`: stable link to the exhibit in `data/metadata*`.
- `title`, `artist`: copied for convenience when browsing FAQ.
- `questions`: generated list of question-answer pairs (3-5 items).
- `question_count`: cached number of generated question-answer pairs.
- `source_model`: model used for generation.
- `source_file`: source metadata filename used as input.
- `generated_at_unix`: generation timestamp.
