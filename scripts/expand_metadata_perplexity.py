import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI


PAINTING_INFO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "artist": {"type": "string"},
        "year": {"type": "string"},
        "style": {"type": "string"},
        "genre": {"type": "string"},
        "description": {"type": "string"},
        "interesting_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "artist",
        "year",
        "style",
        "genre",
        "description",
        "interesting_facts",
    ],
}


SYSTEM_PROMPT = (
    "Ты — музейный гид. "
    "Твоя задача: по названию картины и автору найти точную информацию и вернуть ее "
    "СТРОГО в указанной структуре. "
    "Вся информация обязательно должна быть НА РУССКОМ ЯЗЫКЕ. "
    "Имя автора и название картины, которые ты получишь на вход, тоже нужно предоставить на русском языке. "
    "Если оригинальное название картины на другом языке, переведи его на русский. "
    "Если есть неопределенность (например, разные датировки), укажи наиболее общепринятую. "
    "Не добавляй никаких лишних полей."
)


def _build_user_prompt(title: str, artist: str) -> str:
    return f"""
Тебе на вход поступит название картины и автор. Тебе нужно найти по ней информацию и выдать ее в строго структурированном виде (json).
Вся информация должна быть на русском языке.

Список информации, которую необходимо предоставить:
* title – название картины
* artist – художник, автор картины
* year – год создания картины (числом;можно диапазон, если принято)
* style – в каком стиле написана картина
* genre – какой жанр у картины
* description – краткое описание картины
* interesting_facts – 3–5 коротких фактов, без воды

Формат ответа (строго):
{{
  "title": "",
  "artist": "",
  "year": "",
  "style": "",
  "genre": "",
  "description": "",
  "interesting_facts": []
}}

Информация:
Название: {title}
Автор: {artist}
""".strip()


def _schema_payload() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "PaintingInfo",
            "schema": PAINTING_INFO_SCHEMA,
            "strict": True,
        },
    }


def _extract_title_artist(
    metadata: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    title = metadata.get("title") or metadata.get("name")
    artist = metadata.get("artist") or metadata.get("author")
    title = title.strip() if isinstance(title, str) else None
    artist = artist.strip() if isinstance(artist, str) else None
    return title, artist


def _safe_json_loads(s: str) -> Any:
    """
    Best-effort JSON parse (some providers occasionally wrap JSON in Markdown).
    """
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).strip()
    return json.loads(s)


def _validate_painting_info(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Response is not a JSON object")

    required = PAINTING_INFO_SCHEMA["required"]
    props = PAINTING_INFO_SCHEMA["properties"]

    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    extra = [k for k in data.keys() if k not in props]
    if extra:
        raise ValueError(f"Unexpected keys present: {extra}")

    for key in ["title", "artist", "year", "style", "genre", "description"]:
        if not isinstance(data.get(key), str):
            raise ValueError(f"Field '{key}' must be a string")

    facts = data.get("interesting_facts")
    if not isinstance(facts, list) or any(not isinstance(x, str) for x in facts):
        raise ValueError("Field 'interesting_facts' must be an array of strings")

    out = {k: data[k] for k in required}
    for key in ["title", "artist", "year", "style", "genre", "description"]:
        out[key] = out[key].strip()
    out["interesting_facts"] = [
        x.strip() for x in out["interesting_facts"] if x.strip()
    ]
    return out


def _call_perplexity(
    client: Any,
    model: str,
    title: str,
    artist: str,
) -> Dict[str, Any]:
    user_prompt = _build_user_prompt(title=title, artist=artist)
    last_err: Optional[Exception] = None

    for attempt in range(1, 6):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_schema_payload(),
            )
            content = (resp.choices[0].message.content or "").strip()
            data = _safe_json_loads(content)
            return _validate_painting_info(data)
        except Exception as e:
            last_err = e

            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = (resp.choices[0].message.content or "").strip()
                data = _safe_json_loads(content)
                return _validate_painting_info(data)
            except Exception as e2:
                last_err = e2

        if attempt < 5:
            delay = 2 ** (attempt - 1)
            delay = min(delay, 20.0)
            delay *= random.uniform(0.85, 1.15)
            time.sleep(delay)

    assert last_err is not None
    raise last_err


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N files (0 = no limit).",
    )
    args = parser.parse_args()

    input_dir = project_root / "data/metadata"
    output_dir = project_root / "data/metadata_expand"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.glob("*.json") if p.is_file()])

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        print(f"No metadata json files found in: {input_dir}")
        return

    model = os.getenv("PERPLEXITY_MODEL", "sonar")
    base_url = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url=base_url)

    processed = 0
    skipped = 0
    failed = 0
    for src_path in files:
        if not src_path.exists():
            print(f"Skip (missing): {src_path}")
            skipped += 1
            continue

        dst_path = output_dir / src_path.name
        if dst_path.exists():
            skipped += 1
            continue

        try:
            metadata = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"FAIL read {src_path.name}: {e}")
            failed += 1
            continue

        title, artist = _extract_title_artist(metadata)
        if not title or not artist:
            print(f"Skip (no title/artist): {src_path.name}")
            skipped += 1
            continue

        try:
            assert client is not None
            info = _call_perplexity(
                client=client,
                model=model,
                title=title,
                artist=artist,
            )
        except Exception as e:
            print(f"FAIL api {src_path.name}: {e}")
            failed += 1
            continue

        dst_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        processed += 1

    print(
        f"Done. processed={processed}, skipped={skipped}, failed={failed}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
