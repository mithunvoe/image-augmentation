from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance


def adjust_brightness(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    factor_range: tuple[float, float] = (0.7, 1.3),
) -> Image.Image:
    factor = rng.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    factor_range: tuple[float, float] = (0.7, 1.3),
) -> Image.Image:
    factor = rng.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)
