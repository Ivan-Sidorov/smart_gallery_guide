import asyncio
from pathlib import Path

from PIL import Image

from models.vlm import VLM

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main():
    async with VLM() as vlm:
        image_path = project_root / "data" / "exhibits" / "example.jpg"

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return

        image = Image.open(image_path)

        question = "Что изображено на этой картине?"
        context = "Это картина из коллекции музея."

        answer = await vlm.answer_question(image, question, context=context)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
