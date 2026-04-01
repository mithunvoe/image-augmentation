from __future__ import annotations

import numpy as np
from PIL import Image

from image_augmentation.augmentations.quality import add_noise, heavy_blur, light_blur


class TestHeavyBlur:
    def test_returns_valid_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = heavy_blur(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_deterministic(self, sample_image: Image.Image) -> None:
        r1 = heavy_blur(sample_image, np.random.default_rng(42))
        r2 = heavy_blur(sample_image, np.random.default_rng(42))
        assert list(r1.getdata()) == list(r2.getdata())


class TestLightBlur:
    def test_returns_valid_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = light_blur(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_deterministic(self, sample_image: Image.Image) -> None:
        r1 = light_blur(sample_image, np.random.default_rng(42))
        r2 = light_blur(sample_image, np.random.default_rng(42))
        assert list(r1.getdata()) == list(r2.getdata())


class TestAddNoise:
    def test_returns_valid_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = add_noise(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
        assert result.mode == sample_image.mode

    def test_deterministic(self, sample_image: Image.Image) -> None:
        r1 = add_noise(sample_image, np.random.default_rng(42))
        r2 = add_noise(sample_image, np.random.default_rng(42))
        assert list(r1.getdata()) == list(r2.getdata())

    def test_different_with_different_seed(self, sample_image: Image.Image) -> None:
        r1 = add_noise(sample_image, np.random.default_rng(1))
        r2 = add_noise(sample_image, np.random.default_rng(2))
        assert list(r1.getdata()) != list(r2.getdata())

    def test_pixel_values_in_range(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = add_noise(sample_image, rng)
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255
