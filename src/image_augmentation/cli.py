from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from image_augmentation.augmentations import REGISTRY
from image_augmentation.config import AugmentationConfig, PipelineConfig
from image_augmentation.pipeline import run_pipeline


def _parse_range(value: str) -> tuple[float, float]:
    """Parse a 'min,max' string into a float tuple."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected 'min,max' format, got: {value}"
        )
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid number in range: {value}") from e
    if lo > hi:
        raise argparse.ArgumentTypeError(
            f"Min must be <= max, got: {lo} > {hi}"
        )
    return (lo, hi)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-augment",
        description="Generate augmented copies of images from a folder",
    )
    parser.add_argument("input_dir", type=Path, help="Input folder containing images")
    parser.add_argument("output_dir", type=Path, help="Output folder for augmented images")

    parser.add_argument(
        "-a", "--augmentations",
        nargs="+",
        choices=list(REGISTRY.keys()),
        default=None,
        help="Augmentations to apply (default: all)",
    )
    parser.add_argument(
        "-n", "--copies",
        type=int,
        default=1,
        help="Number of augmented copies per image (default: 1)",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of threads (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--format",
        choices=["jpeg", "png", "webp"],
        default=None,
        help="Output format (default: same as input)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # Per-augmentation parameters
    aug_group = parser.add_argument_group("augmentation parameters")
    aug_group.add_argument("--rotation-range", type=_parse_range, default=None, help="Rotation angle range, e.g. '-30,30'")
    aug_group.add_argument("--scale-range", type=_parse_range, default=None, help="Scale factor range, e.g. '0.8,1.2'")
    aug_group.add_argument("--crop-range", type=_parse_range, default=None, help="Crop fraction range, e.g. '0.7,0.9'")
    aug_group.add_argument("--brightness-range", type=_parse_range, default=None, help="Brightness factor range, e.g. '0.7,1.3'")
    aug_group.add_argument("--contrast-range", type=_parse_range, default=None, help="Contrast factor range, e.g. '0.7,1.3'")
    aug_group.add_argument("--heavy-blur-range", type=_parse_range, default=None, help="Heavy blur radius range, e.g. '3.0,8.0'")
    aug_group.add_argument("--light-blur-range", type=_parse_range, default=None, help="Light blur radius range, e.g. '1.5,4.0'")
    aug_group.add_argument("--noise-range", type=_parse_range, default=None, help="Noise std range, e.g. '5.0,25.0'")
    aug_group.add_argument("--warp-intensity", type=float, default=None, help="Warp intensity (default: 0.3)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.input_dir.is_dir():
        logging.error("Input directory does not exist: %s", args.input_dir)
        return 1

    # Build AugmentationConfig
    aug_overrides: dict[str, object] = {}
    if args.augmentations is not None:
        aug_overrides["enabled_augmentations"] = tuple(args.augmentations)
    if args.rotation_range is not None:
        aug_overrides["rotation_range"] = args.rotation_range
    if args.scale_range is not None:
        aug_overrides["scale_range"] = args.scale_range
    if args.crop_range is not None:
        aug_overrides["crop_fraction_range"] = args.crop_range
    if args.brightness_range is not None:
        aug_overrides["brightness_range"] = args.brightness_range
    if args.contrast_range is not None:
        aug_overrides["contrast_range"] = args.contrast_range
    if args.heavy_blur_range is not None:
        aug_overrides["heavy_blur_radius_range"] = args.heavy_blur_range
    if args.light_blur_range is not None:
        aug_overrides["light_blur_radius_range"] = args.light_blur_range
    if args.noise_range is not None:
        aug_overrides["noise_std_range"] = args.noise_range
    if args.warp_intensity is not None:
        aug_overrides["warp_intensity"] = args.warp_intensity

    aug_config = AugmentationConfig(**aug_overrides)

    # Build PipelineConfig
    pipeline_kwargs: dict[str, object] = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "copies_per_image": args.copies,
        "output_format": args.format,
    }
    if args.workers is not None:
        pipeline_kwargs["max_workers"] = args.workers

    pipeline_config = PipelineConfig(**pipeline_kwargs)

    result = run_pipeline(pipeline_config, aug_config)

    if result.failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
