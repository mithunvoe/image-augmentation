from __future__ import annotations

from pathlib import Path

from image_augmentation.config import AugmentationConfig, PipelineConfig
from image_augmentation.pipeline import run_pipeline


class TestRunPipeline:
    def test_processes_all_images(self, image_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        config = PipelineConfig(
            input_dir=image_dir,
            output_dir=output_dir,
            max_workers=2,
            seed=42,
        )
        aug_config = AugmentationConfig(
            enabled_augmentations=("rotate", "flip", "heavy_blur"),
        )
        result = run_pipeline(config, aug_config)

        assert result.total_images == 3
        assert result.successful == 3
        assert result.failed == 0
        # 3 images x 3 augmentations x 1 copy = 9 outputs
        assert result.total_outputs == 9

    def test_all_augmentations(self, image_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        config = PipelineConfig(
            input_dir=image_dir,
            output_dir=output_dir,
            max_workers=2,
            seed=42,
        )
        aug_config = AugmentationConfig()  # all augmentations
        result = run_pipeline(config, aug_config)

        assert result.total_images == 3
        assert result.successful == 3
        assert result.failed == 0
        # 3 images x 16 augmentations x 1 copy = 48 outputs
        assert result.total_outputs == 48

    def test_multiple_copies(self, image_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        config = PipelineConfig(
            input_dir=image_dir,
            output_dir=output_dir,
            max_workers=2,
            seed=42,
            copies_per_image=2,
        )
        aug_config = AugmentationConfig(
            enabled_augmentations=("flip",),
        )
        result = run_pipeline(config, aug_config)

        # 3 images x 1 aug x 2 copies = 6 outputs
        assert result.total_outputs == 6

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            input_dir=empty_dir,
            output_dir=output_dir,
        )
        aug_config = AugmentationConfig()
        result = run_pipeline(config, aug_config)

        assert result.total_images == 0
        assert result.successful == 0

    def test_output_files_exist(self, image_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        config = PipelineConfig(
            input_dir=image_dir,
            output_dir=output_dir,
            max_workers=1,
            seed=42,
        )
        aug_config = AugmentationConfig(
            enabled_augmentations=("rotate",),
        )
        run_pipeline(config, aug_config)

        output_files = list(output_dir.iterdir())
        assert len(output_files) == 3
        for f in output_files:
            assert f.stat().st_size > 0
