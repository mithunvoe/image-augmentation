# image-augmentation

A multithreaded CLI tool that generates augmented copies of images from a folder. Built on Pillow and NumPy, with deterministic seeding per image so runs are reproducible.

## Requirements

- Python `>=3.10` (the project pins `3.10` via `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Install

Using `uv` (recommended — it reads `uv.lock` for reproducible installs):

```bash
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

## Run

Once installed, the `image-augment` console script is available.

### Quick start

```bash
# Augment every image in ./input/ once, writing to ./output/
uv run image-augment ./input ./output

# Or, without uv
image-augment ./input ./output
```

### Common examples

```bash
# 5 augmented copies per source image
uv run image-augment ./input ./output -n 5

# Only run rotation + brightness, with custom ranges
uv run image-augment ./input ./output \
  -a rotate brightness \
  --rotation-range "-45,45" \
  --brightness-range "0.6,1.4"

# Force all outputs to JPEG, use 16 worker threads
uv run image-augment ./input ./output --format jpeg -w 16

# Reproducible run with a fixed seed
uv run image-augment ./input ./output -s 1337
```

### Run as a module

```bash
uv run python -m image_augmentation ./input ./output
```

## CLI reference

```
image-augment INPUT_DIR OUTPUT_DIR [options]
```

Positional:

- `INPUT_DIR` — folder containing source images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`)
- `OUTPUT_DIR` — folder where augmented images are written (created if missing)

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-a, --augmentations` | all | Subset of augmentations to apply |
| `-n, --copies` | `1` | Number of augmented copies per source image |
| `-w, --workers` | `min(cpu, 8)` | Thread pool size |
| `-s, --seed` | `42` | Global seed for reproducibility |
| `--format` | input format | Output format: `jpeg`, `png`, or `webp` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Augmentation parameter overrides (all take `"min,max"` ranges unless noted):

`--rotation-range`, `--scale-range`, `--crop-range`, `--brightness-range`, `--contrast-range`, `--heavy-blur-range`, `--light-blur-range`, `--noise-range`, `--warp-intensity` (single float).

### Available augmentations

Single transforms: `rotate`, `flip`, `scale`, `crop`, `warp`, `brightness`, `contrast`, `heavy_blur`, `light_blur`, `noise`.

Composite transforms: `crop_tilt_blur_contrast`, `noisy_bright_crop`, `warp_blur_noise`, `scale_contrast_noise`, `crop_flip_bright_blur`, `heavy_distortion`.

## Output naming

Each augmented file is named `{original_stem}_{augmentation}_c{copy_index}{ext}`, e.g. `cat_rotate_c0.jpg`.

## Tests

```bash
uv run pytest
uv run pytest --cov=src --cov-report=term-missing
```

## Project layout

```
src/image_augmentation/
├── cli.py              # argparse entrypoint
├── config.py           # AugmentationConfig, PipelineConfig (frozen dataclasses)
├── pipeline.py         # ThreadPoolExecutor pipeline + deterministic seeding
├── io.py               # discover_images, load_image, save_image
└── augmentations/
    ├── geometric.py    # rotate, flip, scale, crop, warp
    ├── color.py        # brightness, contrast
    ├── quality.py      # noise, heavy_blur, light_blur
    └── combos.py       # composite augmentations
```
