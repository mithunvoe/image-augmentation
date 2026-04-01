from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from image_augmentation.augmentations.combos import (
    crop_flip_bright_blur,
    crop_tilt_blur_contrast,
    heavy_distortion,
    noisy_bright_crop,
    scale_contrast_noise,
    warp_blur_noise,
)

ALL_COMBOS = [
    crop_tilt_blur_contrast,
    noisy_bright_crop,
    warp_blur_noise,
    scale_contrast_noise,
    crop_flip_bright_blur,
    heavy_distortion,
]


class TestCombos:
    @pytest.mark.parametrize("combo_fn", ALL_COMBOS, ids=lambda f: f.__name__)
    def test_returns_valid_image(
        self, combo_fn: object, sample_image: Image.Image, rng: np.random.Generator
    ) -> None:
        result = combo_fn(sample_image, rng)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    @pytest.mark.parametrize("combo_fn", ALL_COMBOS, ids=lambda f: f.__name__)
    def test_deterministic(self, combo_fn: object, sample_image: Image.Image) -> None:
        r1 = combo_fn(sample_image, np.random.default_rng(42))
        r2 = combo_fn(sample_image, np.random.default_rng(42))
        assert list(r1.getdata()) == list(r2.getdata())

    @pytest.mark.parametrize("combo_fn", ALL_COMBOS, ids=lambda f: f.__name__)
    def test_different_with_different_seed(self, combo_fn: object, sample_image: Image.Image) -> None:
        r1 = combo_fn(sample_image, np.random.default_rng(1))
        r2 = combo_fn(sample_image, np.random.default_rng(2))
        assert list(r1.getdata()) != list(r2.getdata())
