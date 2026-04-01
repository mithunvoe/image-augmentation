from __future__ import annotations

from pathlib import Path

import pytest

from image_augmentation.cli import build_parser, main


class TestBuildParser:
    def test_parses_required_args(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), str(tmp_path / "out")])
        assert args.input_dir == tmp_path
        assert args.output_dir == tmp_path / "out"

    def test_default_values(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), str(tmp_path / "out")])
        assert args.copies == 1
        assert args.seed == 42
        assert args.format is None
        assert args.augmentations is None

    def test_parses_augmentation_selection(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([
            str(tmp_path), str(tmp_path / "out"),
            "-a", "rotate", "flip", "heavy_blur",
        ])
        assert args.augmentations == ["rotate", "flip", "heavy_blur"]

    def test_parses_range_args(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([
            str(tmp_path), str(tmp_path / "out"),
            "--rotation-range=-15,15",
            "--scale-range", "0.9,1.1",
        ])
        assert args.rotation_range == (-15.0, 15.0)
        assert args.scale_range == (0.9, 1.1)


class TestMain:
    def test_runs_successfully(self, image_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        exit_code = main([
            str(image_dir),
            str(output_dir),
            "-a", "flip", "heavy_blur",
            "-w", "1",
            "--log-level", "WARNING",
        ])
        assert exit_code == 0
        assert output_dir.is_dir()
        output_files = list(output_dir.iterdir())
        assert len(output_files) == 6  # 3 images x 2 augmentations

    def test_returns_error_for_missing_input(self, tmp_path: Path) -> None:
        exit_code = main([
            str(tmp_path / "nonexistent"),
            str(tmp_path / "output"),
            "--log-level", "ERROR",
        ])
        assert exit_code == 1

    def test_multiple_copies(self, image_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        exit_code = main([
            str(image_dir),
            str(output_dir),
            "-a", "flip",
            "-n", "3",
            "-w", "1",
            "--log-level", "WARNING",
        ])
        assert exit_code == 0
        output_files = list(output_dir.iterdir())
        assert len(output_files) == 9  # 3 images x 1 aug x 3 copies
