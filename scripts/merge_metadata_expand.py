import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MERGE_KEYS: Tuple[str, ...] = (
    "title",
    "artist",
    "year",
    "style",
    "genre",
    "description",
    "interesting_facts",
)


def _is_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return (
            len(
                [
                    x
                    for x in value
                    if (isinstance(x, str) and x.strip()) or not isinstance(x, str)
                ]
            )
            > 0
        )
    if isinstance(value, dict):
        return len(value) > 0
    return True


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )


def _merge_metadata(
    *,
    base: Dict[str, Any],
    expanded: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    updated_fields = []
    merged = dict(base)

    for key in MERGE_KEYS:
        if key not in expanded:
            continue
        v = expanded.get(key)
        if _is_non_empty_value(v):
            merged[key] = v
            updated_fields.append(key)

    return merged, updated_fields


def _iter_expand_files(expand_dir: Path) -> Iterable[Path]:
    return sorted([p for p in expand_dir.glob("*.json") if p.is_file()])


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N files (0 = no limit).",
    )
    args = parser.parse_args()

    metadata_dir = project_root / "data/metadata"
    expand_dir = project_root / "data/metadata_expand"

    if not expand_dir.exists():
        raise SystemExit(f"Expand dir not found: {expand_dir}")
    if not metadata_dir.exists():
        raise SystemExit(f"Metadata dir not found: {metadata_dir}")

    files = list(_iter_expand_files(expand_dir))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        print(f"No expanded json files found in: {expand_dir}")
        return

    updated = 0
    created = 0
    skipped = 0
    failed = 0
    for expand_path in files:
        base_path = metadata_dir / expand_path.name

        try:
            expanded = _load_json(expand_path)
        except Exception as e:
            print(f"FAIL read expand {expand_path.name}: {e}")
            failed += 1
            continue

        if base_path.exists():
            try:
                base = _load_json(base_path)
            except Exception as e:
                print(f"FAIL read base {base_path.name}: {e}")
                failed += 1
                continue
        else:
            skipped += 1
            continue

        merged, changed = _merge_metadata(
            base=base,
            expanded=expanded,
        )

        if not changed:
            skipped += 1
            continue

        try:
            _write_json(base_path, merged)
            updated += 1
        except Exception as e:
            print(f"FAIL write {base_path.name}: {e}")
            failed += 1
            continue

    print(
        f"Done. updated={updated}, created={created}, skipped={skipped}, failed={failed} | "
        f"metadata_dir={metadata_dir} expand_dir={expand_dir}"
    )


if __name__ == "__main__":
    main()
