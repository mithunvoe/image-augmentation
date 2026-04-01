from __future__ import annotations

import numpy as np
from PIL import Image

from image_augmentation.augmentations.color import adjust_brightness, adjust_contrast


class TestAdjustBrightness:
    def test_returns_valid_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = adjust_brightness(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_deterministic(self, sample_image: Image.Image) -> None:
        r1 = adjust_brightness(sample_image, np.random.default_rng(42))
        r2 = adjust_brightness(sample_image, np.random.default_rng(42))
        assert list(r1.getdata()) == list(r2.getdata())

    def test_factor_one_preserves_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = adjust_brightness(sample_image, rng, factor_range=(1.0, 1.0))
        assert list(result.getdata()) == list(sample_image.getdata())


class TestAdjustContrast:
    def test_returns_valid_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = adjust_contrast(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_deterministic(self, sample_image: Image.Image) -> None:
        r1 = adjust_contrast(sample_image, np.random.default_rng(42))
        r2 = adjust_contrast(sample_image, np.random.default_rng(42))
        assert list(r1.getdata()) == list(r2.getdata())

    def test_factor_one_preserves_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = adjust_contrast(sample_image, rng, factor_range=(1.0, 1.0))
        assert list(result.getdata()) == list(sample_image.getdata())
