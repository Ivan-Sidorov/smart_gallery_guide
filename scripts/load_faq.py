import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import FAQ_DIR
from database.schemas import FAQDocument
from database.vector_db import VectorDatabase
from models.text_encoder import TextEncoder


def load_faq_from_directory(faq_dir: Path = FAQ_DIR):
    """
    Load FAQ items from JSON files into the vector database.

    Each file is ``{exhibit_id}.json`` and follows the FAQDocument schema.
    Every question is encoded with TextEncoder and stored as a separate
    entry in the FAQ vector index so it can be retrieved by semantic search.
    """
    db = VectorDatabase()
    text_encoder = TextEncoder()

    faq_files = list(faq_dir.glob("*.json"))
    if not faq_files:
        print(f"No FAQ files found in {faq_dir}")
        return

    print(f"Found {len(faq_files)} FAQ files")

    total_questions = 0
    skipped_files = 0

    for faq_file in faq_files:
        try:
            with open(faq_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {faq_file.name}: {e}")
            skipped_files += 1
            continue

        try:
            doc = FAQDocument(**data)
        except Exception as e:
            print(f"Error parsing {faq_file.name}: {e}")
            skipped_files += 1
            continue

        for qa in doc.questions:
            try:
                embedding = text_encoder.encode_text(qa.question).tolist()
                db.add_faq(
                    question=qa.question,
                    answer=qa.answer,
                    exhibit_id=doc.exhibit_id,
                    embedding=embedding,
                )
                total_questions += 1
            except Exception as e:
                print(
                    f"Error encoding question for exhibit {doc.exhibit_id}: {e}"
                )

    print(
        f"\nFAQ loading completed! "
        f"Loaded {total_questions} questions from {len(faq_files) - skipped_files} files."
    )
    if skipped_files:
        print(f"Skipped {skipped_files} files due to errors.")

    print("Building BM25 indexes...")
    db.build_bm25_indexes()
    print("BM25 indexes ready.")


if __name__ == "__main__":
    load_faq_from_directory()
