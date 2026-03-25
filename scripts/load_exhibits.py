import json
import sys
from pathlib import Path

from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import EXHIBITS_DIR, METADATA_DIR
from database.schemas import ExhibitMetadata
from database.vector_db import VectorDatabase
from models.text_encoder import TextEncoder
from models.vision_encoder import VisionEncoder


def load_exhibits_from_directory(
    exhibits_dir: Path = EXHIBITS_DIR, metadata_dir: Path = METADATA_DIR
):
    """
    Load exhibits from directory.

    Args:
        exhibits_dir (Path): Directory with exhibit images
        metadata_dir (Path): Directory with exhibit metadata
    """
    db = VectorDatabase()
    vision_encoder = VisionEncoder()
    text_encoder = TextEncoder()

    image_files = (
        list(exhibits_dir.glob("*.jpg"))
        + list(exhibits_dir.glob("*.jpeg"))
        + list(exhibits_dir.glob("*.png"))
    )

    if not image_files:
        print(f"No image files found in {exhibits_dir}")
        return

    print(f"Found {len(image_files)} exhibit images")

    for image_file in image_files:
        exhibit_id = image_file.stem

        metadata_file = metadata_dir / f"{exhibit_id}.json"
        if not metadata_file.exists():
            print(f"No metadata file for {exhibit_id}")
            continue

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        metadata_dict["exhibit_id"] = exhibit_id
        metadata_dict["image_path"] = str(image_file)

        try:
            metadata = ExhibitMetadata(**metadata_dict)
        except Exception as e:
            print(f"Error while creating metadata for {exhibit_id}: {e}")
            continue

        try:
            image = Image.open(image_file)
            embedding = vision_encoder.encode_image(image)
        except Exception as e:
            print(f"Error while encoding image for {exhibit_id}: {e}")
            continue

        title_text = " ".join(
            filter(
                None,
                [
                    metadata.artist or "",
                    metadata.title,
                    metadata.school or "",
                    metadata.year or "",
                ],
            )
        ).strip()

        ai = metadata.additional_info
        desc_parts = [metadata.display_description]
        for key in (
            "anotation",
            "glossary",
            "text",
            "epoque",
            "category",
            "place",
            "techniq",
        ):
            val = ai.get(key)
            if val and isinstance(val, str):
                desc_parts.append(val)
        if metadata.material:
            desc_parts.append(metadata.material)
        desc_text = " ".join(desc_parts).strip()

        try:
            title_emb = text_encoder.encode_text(title_text).tolist()
            desc_emb = (
                text_encoder.encode_text(desc_text).tolist() if desc_text else None
            )
        except Exception as e:
            print(f"Error while encoding text for {exhibit_id}: {e}")
            title_emb = None
            desc_emb = None

        try:
            db.add_exhibit(
                exhibit_id,
                embedding.tolist(),
                metadata,
                title_embedding=title_emb,
                title_text=title_text,
                desc_embedding=desc_emb,
                desc_text=desc_text,
            )
            print(f"Added exhibit: {metadata.title} (ID: {exhibit_id})")
        except Exception as e:
            print(f"Error while adding exhibit {exhibit_id} to database: {e}")

    print("\nExhibit loading completed! Building BM25 indexes...")
    db.build_bm25_indexes()
    print("BM25 indexes ready.")


if __name__ == "__main__":
    load_exhibits_from_directory()
