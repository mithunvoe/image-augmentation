from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image_augmentation.augmentations import REGISTRY
from image_augmentation.config import AugmentationConfig, PipelineConfig
from image_augmentation.io import discover_images, load_image, save_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    total_images: int
    successful: int
    failed: int
    total_outputs: int


def _make_rng(seed: int, image_path: Path, copy_index: int) -> np.random.Generator:
    """Create a deterministic RNG seeded from global seed + image path + copy index."""
    path_hash = int(hashlib.sha256(str(image_path).encode()).hexdigest(), 16)
    combined_seed = (seed + path_hash + copy_index) % (2**32)
    return np.random.default_rng(combined_seed)


def _get_aug_kwargs(
    aug_name: str, config: AugmentationConfig
) -> dict[str, object]:
    """Map augmentation names to their config keyword arguments."""
    mapping: dict[str, dict[str, object]] = {
        "rotate": {"angle_range": config.rotation_range},
        "flip": {},
        "scale": {"scale_range": config.scale_range},
        "crop": {"crop_fraction_range": config.crop_fraction_range},
        "warp": {"intensity": config.warp_intensity},
        "brightness": {"factor_range": config.brightness_range},
        "contrast": {"factor_range": config.contrast_range},
        "heavy_blur": {"radius_range": config.heavy_blur_radius_range},
        "light_blur": {"radius_range": config.light_blur_radius_range},
        "noise": {"std_range": config.noise_std_range},
    }
    return mapping.get(aug_name, {})


def process_single_image(
    image_path: Path,
    pipeline_config: PipelineConfig,
    aug_config: AugmentationConfig,
) -> list[Path]:
    """Load an image, apply all enabled augmentations, and save results."""
    image = load_image(image_path)
    outputs: list[Path] = []

    for copy_idx in range(pipeline_config.copies_per_image):
        for aug_name in aug_config.enabled_augmentations:
            aug_fn = REGISTRY.get(aug_name)
            if aug_fn is None:
                logger.warning("Unknown augmentation: %s, skipping", aug_name)
                continue

            rng = _make_rng(pipeline_config.seed, image_path, copy_idx)
            kwargs = _get_aug_kwargs(aug_name, aug_config)
            augmented = aug_fn(image, rng, **kwargs)

            suffix = f"{aug_name}_c{copy_idx}"
            output_path = save_image(
                augmented,
                pipeline_config.output_dir,
                image_path.name,
                suffix,
                pipeline_config.output_format,
            )
            outputs.append(output_path)

    return outputs


def run_pipeline(
    pipeline_config: PipelineConfig,
    aug_config: AugmentationConfig,
) -> PipelineResult:
    """Discover images and process them in parallel using ThreadPoolExecutor."""
    images = discover_images(pipeline_config.input_dir)
    total = len(images)

    if total == 0:
        logger.warning("No images found in %s", pipeline_config.input_dir)
        return PipelineResult(
            total_images=0, successful=0, failed=0, total_outputs=0
        )

    logger.info(
        "Processing %d images with %d workers",
        total,
        pipeline_config.max_workers,
    )

    successful = 0
    failed = 0
    total_outputs = 0

    with ThreadPoolExecutor(max_workers=pipeline_config.max_workers) as executor:
        future_to_path = {
            executor.submit(
                process_single_image, img_path, pipeline_config, aug_config
            ): img_path
            for img_path in images
        }

        for future in as_completed(future_to_path):
            img_path = future_to_path[future]
            try:
                outputs = future.result()
                successful += 1
                total_outputs += len(outputs)
                logger.info(
                    "[%d/%d] Processed %s -> %d outputs",
                    successful + failed,
                    total,
                    img_path.name,
                    len(outputs),
                )
            except Exception:
                failed += 1
                logger.exception("Failed to process %s", img_path.name)

    result = PipelineResult(
        total_images=total,
        successful=successful,
        failed=failed,
        total_outputs=total_outputs,
    )
    logger.info(
        "Pipeline complete: %d/%d successful, %d failed, %d total outputs",
        result.successful,
        result.total_images,
        result.failed,
        result.total_outputs,
    )
    return result
