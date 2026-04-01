from __future__ import annotations

import numpy as np
from PIL import Image


def rotate(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    angle_range: tuple[float, float] = (-30.0, 30.0),
) -> Image.Image:
    angle = rng.uniform(angle_range[0], angle_range[1])
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    return rotated


def horizontal_flip(
    image: Image.Image,
    rng: np.random.Generator,
) -> Image.Image:
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def scale(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> Image.Image:
    factor = rng.uniform(scale_range[0], scale_range[1])
    w, h = image.size
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    scaled = image.resize((new_w, new_h), Image.BICUBIC)

    # Center-crop or pad back to original size
    result = Image.new(image.mode, (w, h), (0, 0, 0))
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2

    if factor >= 1.0:
        # Crop the scaled image to original size
        crop_x = (new_w - w) // 2
        crop_y = (new_h - h) // 2
        result = scaled.crop((crop_x, crop_y, crop_x + w, crop_y + h))
    else:
        # Paste scaled image centered on black background
        result.paste(scaled, (paste_x, paste_y))

    return result


def random_crop(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    crop_fraction_range: tuple[float, float] = (0.7, 0.9),
) -> Image.Image:
    fraction = rng.uniform(crop_fraction_range[0], crop_fraction_range[1])
    w, h = image.size
    crop_w = max(1, int(w * fraction))
    crop_h = max(1, int(h * fraction))

    left = int(rng.integers(0, max(1, w - crop_w)))
    top = int(rng.integers(0, max(1, h - crop_h)))

    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.BICUBIC)


def warp(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    intensity: float = 0.3,
) -> Image.Image:
    w, h = image.size
    # Random perspective transform using 8 coefficients
    # Perturb the four corners slightly
    margin_x = w * intensity * 0.5
    margin_y = h * intensity * 0.5

    # Source corners (original image corners)
    # Destination corners (perturbed)
    dx = [rng.uniform(-margin_x, margin_x) for _ in range(4)]
    dy = [rng.uniform(-margin_y, margin_y) for _ in range(4)]

    coeffs = _find_perspective_coeffs(
        # destination quadrilateral
        [
            (dx[0], dy[0]),
            (w + dx[1], dy[1]),
            (w + dx[2], h + dy[2]),
            (dx[3], h + dy[3]),
        ],
        # source quadrilateral (full image)
        [
            (0, 0),
            (w, 0),
            (w, h),
            (0, h),
        ],
    )
    return image.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def _find_perspective_coeffs(
    dst: list[tuple[float, float]],
    src: list[tuple[float, float]],
) -> tuple[float, ...]:
    """Compute the 8 perspective transform coefficients."""
    matrix: list[list[float]] = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])

    import numpy as np

    a = np.array(matrix, dtype=np.float64)
    b = np.array([p for pair in src for p in pair], dtype=np.float64)
    result = np.linalg.solve(a, b)
    return tuple(result.tolist())
