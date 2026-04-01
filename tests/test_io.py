from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from image_augmentation.io import discover_images, load_image, save_image


class TestDiscoverImages:
    def test_finds_supported_formats(self, image_dir: Path) -> None:
        images = discover_images(image_dir)
        names = {p.name for p in images}
        assert "test1.png" in names
        assert "test2.jpg" in names
        assert "test3.bmp" in names

    def test_skips_non_images(self, image_dir: Path) -> None:
        images = discover_images(image_dir)
        names = {p.name for p in images}
        assert "readme.txt" not in names

    def test_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_images(tmp_path / "nonexistent")

    def test_empty_dir(self, tmp_path: Path) -> None:
        images = discover_images(tmp_path)
        assert images == []


class TestLoadImage:
    def test_loads_as_rgb(self, image_dir: Path) -> None:
        img = load_image(image_dir / "test1.png")
        assert img.mode == "RGB"

    def test_loads_jpeg(self, image_dir: Path) -> None:
        img = load_image(image_dir / "test2.jpg")
        assert isinstance(img, Image.Image)

    def test_converts_grayscale_to_rgb(self, tmp_path: Path) -> None:
        gray = Image.new("L", (50, 50), 128)
        path = tmp_path / "gray.png"
        gray.save(path)
        img = load_image(path)
        assert img.mode == "RGB"


class TestSaveImage:
    def test_saves_png(self, sample_image: Image.Image, tmp_path: Path) -> None:
        path = save_image(sample_image, tmp_path, "original.png", "rotate_c0")
        assert path.exists()
        assert path.name == "original_rotate_c0.png"

    def test_saves_with_format_override(self, sample_image: Image.Image, tmp_path: Path) -> None:
        path = save_image(sample_image, tmp_path, "original.png", "blur_c0", output_format="jpeg")
        assert path.exists()
        assert path.suffix == ".jpeg"

    def test_creates_output_dir(self, sample_image: Image.Image, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "output"
        path = save_image(sample_image, out_dir, "test.png", "flip_c0")
        assert path.exists()
        assert out_dir.is_dir()
