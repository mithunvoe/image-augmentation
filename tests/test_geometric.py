from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from image_augmentation.augmentations.geometric import (
    horizontal_flip,
    random_crop,
    rotate,
    scale,
    warp,
)


class TestRotate:
    def test_returns_valid_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = rotate(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_deterministic_with_same_seed(self, sample_image: Image.Image) -> None:
        r1 = rotate(sample_image, np.random.default_rng(99))
        r2 = rotate(sample_image, np.random.default_rng(99))
        assert list(r1.getdata()) == list(r2.getdata())

    def test_different_with_different_seed(self, sample_image: Image.Image) -> None:
        r1 = rotate(sample_image, np.random.default_rng(1))
        r2 = rotate(sample_image, np.random.default_rng(2))
        assert list(r1.getdata()) != list(r2.getdata())

    def test_custom_range(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = rotate(sample_image, rng, angle_range=(0.0, 0.0))
        # With 0 rotation, image should be very similar to original
        assert result.size == sample_image.size


class TestHorizontalFlip:
    def test_returns_valid_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = horizontal_flip(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_double_flip_restores_original(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        flipped_once = horizontal_flip(sample_image, rng)
        flipped_twice = horizontal_flip(flipped_once, rng)
        assert list(flipped_twice.getdata()) == list(sample_image.getdata())


class TestScale:
    def test_returns_original_dimensions(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = scale(sample_image, rng)
        assert result.size == sample_image.size

    def test_scale_up(self, sample_image: Image.Image) -> None:
        result = scale(sample_image, np.random.default_rng(1), scale_range=(1.5, 1.5))
        assert result.size == sample_image.size

    def test_scale_down(self, sample_image: Image.Image) -> None:
        result = scale(sample_image, np.random.default_rng(1), scale_range=(0.5, 0.5))
        assert result.size == sample_image.size


class TestRandomCrop:
    def test_returns_original_dimensions(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = random_crop(sample_image, rng)
        assert result.size == sample_image.size

    def test_deterministic(self, sample_image: Image.Image) -> None:
        r1 = random_crop(sample_image, np.random.default_rng(42))
        r2 = random_crop(sample_image, np.random.default_rng(42))
        assert list(r1.getdata()) == list(r2.getdata())


class TestWarp:
    def test_returns_valid_image(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = warp(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_deterministic(self, sample_image: Image.Image) -> None:
        r1 = warp(sample_image, np.random.default_rng(42))
        r2 = warp(sample_image, np.random.default_rng(42))
        assert list(r1.getdata()) == list(r2.getdata())

    def test_zero_intensity(self, sample_image: Image.Image, rng: np.random.Generator) -> None:
        result = warp(sample_image, rng, intensity=0.0)
        assert result.size == sample_image.size
