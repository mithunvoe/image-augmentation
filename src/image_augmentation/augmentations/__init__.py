from image_augmentation.augmentations.geometric import (
    rotate,
    horizontal_flip,
    scale,
    random_crop,
    warp,
)
from image_augmentation.augmentations.color import (
    adjust_brightness,
    adjust_contrast,
)
from image_augmentation.augmentations.quality import (
    add_noise,
    heavy_blur,
    light_blur,
)
from image_augmentation.augmentations.combos import (
    crop_tilt_blur_contrast,
    noisy_bright_crop,
    warp_blur_noise,
    scale_contrast_noise,
    crop_flip_bright_blur,
    heavy_distortion,
)

REGISTRY: dict[str, object] = {
    "rotate": rotate,
    "flip": horizontal_flip,
    "scale": scale,
    "crop": random_crop,
    "warp": warp,
    "brightness": adjust_brightness,
    "contrast": adjust_contrast,
    "heavy_blur": heavy_blur,
    "light_blur": light_blur,
    "noise": add_noise,
    "crop_tilt_blur_contrast": crop_tilt_blur_contrast,
    "noisy_bright_crop": noisy_bright_crop,
    "warp_blur_noise": warp_blur_noise,
    "scale_contrast_noise": scale_contrast_noise,
    "crop_flip_bright_blur": crop_flip_bright_blur,
    "heavy_distortion": heavy_distortion,
}

__all__ = ["REGISTRY"]
