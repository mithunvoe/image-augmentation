from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def gaussian_blur(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    radius_range: tuple[float, float] = (1.0, 3.0),
) -> Image.Image:
    radius = rng.uniform(radius_range[0], radius_range[1])
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def heavy_blur(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    radius_range: tuple[float, float] = (3.0, 8.0),
) -> Image.Image:
    radius = rng.uniform(radius_range[0], radius_range[1])
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def light_blur(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    radius_range: tuple[float, float] = (1.5, 4.0),
) -> Image.Image:
    radius = rng.uniform(radius_range[0], radius_range[1])
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def add_noise(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    std_range: tuple[float, float] = (5.0, 25.0),
) -> Image.Image:
    std = rng.uniform(std_range[0], std_range[1])
    arr = np.array(image, dtype=np.float64)
    noise = rng.normal(0.0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode=image.mode)
