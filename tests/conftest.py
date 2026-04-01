from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """A 100x80 RGB test image with a gradient pattern."""
    arr = np.zeros((80, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8)  # red gradient
    arr[:, :, 1] = 128
    arr[:, :, 2] = np.linspace(255, 0, 100, dtype=np.uint8)  # blue gradient
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded RNG for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def image_dir(tmp_path: Path, sample_image: Image.Image) -> Path:
    """A temp directory with 3 test images in different formats."""
    for name in ["test1.png", "test2.jpg", "test3.bmp"]:
        sample_image.save(tmp_path / name)
    # Also create a non-image file that should be skipped
    (tmp_path / "readme.txt").write_text("not an image")
    return tmp_path
