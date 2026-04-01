from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
})

logger = logging.getLogger(__name__)


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logger.info("Discovered %d images in %s", len(images), input_dir)
    return images


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def save_image(
    image: Image.Image,
    output_dir: Path,
    original_name: str,
    suffix: str,
    output_format: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(original_name).stem
    original_ext = Path(original_name).suffix

    if output_format:
        ext = f".{output_format.lower()}"
    else:
        ext = original_ext or ".png"

    filename = f"{stem}_{suffix}{ext}"
    output_path = output_dir / filename

    save_kwargs: dict[str, object] = {}
    if ext.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = 95
    elif ext.lower() == ".webp":
        save_kwargs["quality"] = 95

    image.save(output_path, **save_kwargs)
    return output_path
