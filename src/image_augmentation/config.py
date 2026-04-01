from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AugmentationConfig:
    rotation_range: tuple[float, float] = (-30.0, 30.0)
    scale_range: tuple[float, float] = (0.8, 1.2)
    crop_fraction_range: tuple[float, float] = (0.7, 0.9)
    brightness_range: tuple[float, float] = (2, 5)
    contrast_range: tuple[float, float] = (2, 5)
    heavy_blur_radius_range: tuple[float, float] = (3.0, 8.0)
    light_blur_radius_range: tuple[float, float] = (1.5, 4.0)
    noise_std_range: tuple[float, float] = (15.0, 40.0)
    warp_intensity: float = 0.3
    enabled_augmentations: tuple[str, ...] = (
        "rotate",
        "flip",
        "scale",
        "crop",
        "warp",
        "brightness",
        "contrast",
        "heavy_blur",
        "light_blur",
        "noise",
        "crop_tilt_blur_contrast",
        "noisy_bright_crop",
        "warp_blur_noise",
        "scale_contrast_noise",
        "crop_flip_bright_blur",
        "heavy_distortion",
    )


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    max_workers: int = field(
        default_factory=lambda: min(os.cpu_count() or 4, 8)
    )
    seed: int = 42
    output_format: str | None = None
    copies_per_image: int = 1
