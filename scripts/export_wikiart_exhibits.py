import argparse
import json
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datasets import load_dataset

from PIL import Image


EXCLUDED_GENRES = {
    "religious painting",
    "sketch and study",
    "genre painting",
    "design",
    "nude painting (nu)",
    "illustration",
}


def _parse_year(date_str: Optional[str]) -> str:
    if not date_str:
        return ""
    m = re.search(r"(\d{4})", str(date_str))
    return m.group(1) if m else str(date_str).strip()


def _normalize_genre(genre: str) -> str:
    return (genre or "").strip().lower()


def _image_to_pil(image_obj: Any) -> Optional[Image.Image]:
    if image_obj is None:
        return None
    if isinstance(image_obj, Image.Image):
        return image_obj
    if isinstance(image_obj, dict) and "path" in image_obj:
        try:
            return Image.open(image_obj["path"])
        except Exception:
            return None
    return None


def _should_take(
    artist: str,
    genre: str,
    style: str,
    artist_counts: Counter,
    genre_counts: Counter,
    style_counts: Counter,
    max_per_artist: int,
    max_per_genre: int,
    max_per_style: int,
) -> bool:
    if max_per_artist > 0 and artist and artist_counts[artist] >= max_per_artist:
        return False
    if max_per_genre > 0 and genre and genre_counts[genre] >= max_per_genre:
        return False
    if max_per_style > 0 and style and style_counts[style] >= max_per_style:
        return False
    return True


def _build_metadata(
    exhibit_id: str,
    title: str,
    artist: str,
    year: str,
    style: str,
    genre: str,
    image_rel_path: str,
) -> Dict[str, Any]:
    return {
        "exhibit_id": exhibit_id,
        "title": title,
        "artist": artist,
        "year": year,
        "style": style,
        "genre": genre,
        "description": "",
        "interesting_facts": [],
        "image_path": image_rel_path,
        "additional_info": {},
    }


def export_wikiart_exhibits(
    count: int,
    seed: int,
    max_per_artist: int,
    max_per_genre: int,
    max_per_style: int,
    exhibits_dir: Path,
    metadata_dir: Path,
) -> Tuple[int, Counter, Counter, Counter]:
    exhibits_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Artificio/WikiArt", split="train", streaming=True)
    ds = ds.shuffle(seed=seed)

    artist_counts: Counter = Counter()
    genre_counts: Counter = Counter()
    style_counts: Counter = Counter()

    saved = 0
    scanned = 0

    for row in ds:
        scanned += 1

        title = (row.get("title") or "").strip() or "Untitled"
        artist = (row.get("artist") or "").strip()
        genre = (row.get("genre") or "").strip()
        style = (row.get("style") or "").strip()
        year = _parse_year(row.get("date"))

        if _normalize_genre(genre) in EXCLUDED_GENRES:
            continue

        artist_key = artist or "Unknown"
        genre_key = genre or "Unknown"
        style_key = style or "Unknown"

        if not _should_take(
            artist=artist_key,
            genre=genre_key,
            style=style_key,
            artist_counts=artist_counts,
            genre_counts=genre_counts,
            style_counts=style_counts,
            max_per_artist=max_per_artist,
            max_per_genre=max_per_genre,
            max_per_style=max_per_style,
        ):
            continue

        pil_img = _image_to_pil(row.get("image"))
        if pil_img is None:
            continue

        exhibit_id = str(uuid.uuid4())

        img_filename = f"{exhibit_id}.jpg"
        img_path = exhibits_dir / img_filename
        try:
            pil_img = pil_img.convert("RGB")
            pil_img.save(img_path, format="JPEG", quality=95, optimize=True)
        except Exception:
            try:
                if img_path.exists():
                    img_path.unlink()
            except Exception:
                pass
            continue

        image_rel_path = f"data/exhibits/{img_filename}"
        metadata = _build_metadata(
            exhibit_id=exhibit_id,
            title=title,
            artist=artist,
            year=year,
            style=style,
            genre=genre,
            image_rel_path=image_rel_path,
        )

        meta_path = metadata_dir / f"{exhibit_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        artist_counts[artist_key] += 1
        genre_counts[genre_key] += 1
        style_counts[style_key] += 1

        saved += 1
        if saved % 25 == 0 or saved == count:
            print(
                f"[progress] saved={saved}/{count} scanned={scanned} "
                f"unique_artists={len([k for k, v in artist_counts.items() if k])} "
                f"unique_genres={len([k for k, v in genre_counts.items() if k])} "
                f"unique_styles={len([k for k, v in style_counts.items() if k])}"
            )

        if saved >= count:
            break

    return saved, artist_counts, genre_counts, style_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=500, help="How many exhibits to export."
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--max-per-artist",
        type=int,
        default=10,
        help="Maximum exhibits per artist (0 = no limit).",
    )
    parser.add_argument(
        "--max-per-genre",
        type=int,
        default=30,
        help="Maximum exhibits per genre (0 = no limit).",
    )
    parser.add_argument(
        "--max-per-style",
        type=int,
        default=20,
        help="Maximum exhibits per style (0 = no limit).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import EXHIBITS_DIR, METADATA_DIR

    saved, artist_counts, genre_counts, style_counts = export_wikiart_exhibits(
        count=args.count,
        seed=args.seed,
        max_per_artist=args.max_per_artist,
        max_per_genre=args.max_per_genre,
        max_per_style=args.max_per_style,
        exhibits_dir=EXHIBITS_DIR,
        metadata_dir=METADATA_DIR,
    )

    print("\n[done]")
    print(f"saved: {saved}")
    print(f"unique artists: {len([k for k, v in artist_counts.items() if k])}")
    print(f"unique genres: {len([k for k, v in genre_counts.items() if k])}")
    print(f"unique styles: {len([k for k, v in style_counts.items() if k])}")


if __name__ == "__main__":
    main()
