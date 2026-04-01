from __future__ import annotations

import numpy as np
from PIL import Image

from image_augmentation.augmentations.color import adjust_brightness, adjust_contrast
from image_augmentation.augmentations.geometric import random_crop, rotate, scale, warp
from image_augmentation.augmentations.quality import add_noise, heavy_blur, light_blur


def crop_tilt_blur_contrast(
    image: Image.Image,
    rng: np.random.Generator,
) -> Image.Image:
    """Crop + rotation + heavy blur + high contrast."""
    result = random_crop(image, rng, crop_fraction_range=(0.7, 0.9))
    result = rotate(result, rng, angle_range=(-15.0, 15.0))
    result = heavy_blur(result, rng, radius_range=(3.0, 6.0))
    result = adjust_contrast(result, rng, factor_range=(2.0, 4.0))
    return result


def noisy_bright_crop(
    image: Image.Image,
    rng: np.random.Generator,
) -> Image.Image:
    """Noise + brightness boost + crop."""
    result = add_noise(image, rng, std_range=(15.0, 35.0))
    result = adjust_brightness(result, rng, factor_range=(2.0, 4.0))
    result = random_crop(result, rng, crop_fraction_range=(0.75, 0.9))
    return result


def warp_blur_noise(
    image: Image.Image,
    rng: np.random.Generator,
) -> Image.Image:
    """Perspective warp + light blur + noise."""
    result = warp(image, rng, intensity=0.3)
    result = light_blur(result, rng, radius_range=(1.5, 3.5))
    result = add_noise(result, rng, std_range=(10.0, 25.0))
    return result


def scale_contrast_noise(
    image: Image.Image,
    rng: np.random.Generator,
) -> Image.Image:
    """Scale + high contrast + noise."""
    result = scale(image, rng, scale_range=(0.8, 1.2))
    result = adjust_contrast(result, rng, factor_range=(2.0, 5.0))
    result = add_noise(result, rng, std_range=(15.0, 30.0))
    return result


def crop_flip_bright_blur(
    image: Image.Image,
    rng: np.random.Generator,
) -> Image.Image:
    """Crop + flip + brightness boost + light blur."""
    result = random_crop(image, rng, crop_fraction_range=(0.7, 0.85))
    result = result.transpose(Image.FLIP_LEFT_RIGHT)
    result = adjust_brightness(result, rng, factor_range=(2.0, 4.0))
    result = light_blur(result, rng, radius_range=(1.5, 3.0))
    return result


def heavy_distortion(
    image: Image.Image,
    rng: np.random.Generator,
) -> Image.Image:
    """Warp + rotation + heavy blur + noise + contrast -- maximum distortion."""
    result = warp(image, rng, intensity=0.4)
    result = rotate(result, rng, angle_range=(-20.0, 20.0))
    result = heavy_blur(result, rng, radius_range=(4.0, 8.0))
    result = add_noise(result, rng, std_range=(20.0, 40.0))
    result = adjust_contrast(result, rng, factor_range=(2.0, 4.0))
    return result
