"""Seven deterministic artistic filters for RGB uint8 images."""

from collections.abc import Callable

import cv2
import numpy as np


MAX_SIDE = 1600


def resize_for_processing(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Return a copy resized to fit within *max_side* without changing aspect ratio."""
    _validate_image(image)
    if max_side < 1:
        raise ValueError("max_side must be positive")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("image must have shape (height, width, 3) and be nonempty")


def _pencil_sketch(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 25, 25)
    sigma = max(1.5, min(image.shape[:2]) / 100)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    sketch = np.minimum(gray.astype(np.float32) / np.maximum(background, 1) * 255, 255)
    sketch[background == 0] = 255
    # Retain a little graphite shading without crushing dark areas into black.
    shading = 0.88 + 0.12 * gray.astype(np.float32) / 255
    sketch = np.clip(sketch * shading, 0, 255).astype(np.uint8)
    sketch[background == 0] = 255
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)


def _cartoon(image: np.ndarray) -> np.ndarray:
    smooth = image.copy()
    for _ in range(2):
        smooth = cv2.bilateralFilter(smooth, 7, 40, 40)
    # Fixed bins keep output deterministic and avoid per-image k-means randomness.
    quantized = np.clip(np.round(smooth.astype(np.float32) / 20) * 20, 0, 255)
    color = 0.7 * quantized + 0.3 * smooth
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160, L2gradient=True)
    edges = cv2.GaussianBlur(edges, (3, 3), 0.6).astype(np.float32) / 255
    return np.clip(color * (1 - 0.72 * edges[:, :, None]), 0, 255).astype(np.uint8)


def _oil_painting(image: np.ndarray) -> np.ndarray:
    # Pick the most frequent intensity band in a small local window, then average
    # the colors from pixels in that band. This is a local oil-painting filter.
    base = cv2.bilateralFilter(image, 5, 30, 25)
    intensity = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY) // 16
    best_count = np.zeros(intensity.shape, dtype=np.float32)
    result = np.zeros_like(base)
    color = base.astype(np.float32)
    radius = int(np.clip(round(min(image.shape[:2]) / 240), 1, 4))
    kernel = (2 * radius + 1, 2 * radius + 1)
    for band in range(16):
        mask = (intensity == band).astype(np.float32)
        count = cv2.boxFilter(mask, -1, kernel, normalize=False)
        selected = count > best_count
        if not np.any(selected):
            continue
        best_count[selected] = count[selected]
        for channel in range(3):
            total = cv2.boxFilter(color[:, :, channel] * mask, -1, kernel, normalize=False)
            mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            result[:, :, channel][selected] = np.clip(mean[selected], 0, 255).astype(np.uint8)
    # Blend back structure near strong edges to preserve faces and fine contours.
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, -1, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, -1, 0, 1, ksize=3)
    detail = (0.15 + 0.55 * np.clip(cv2.magnitude(gx, gy) / 160, 0, 1))[:, :, None]
    return np.clip(result * (1 - detail) + base * detail, 0, 255).astype(np.uint8)


def _emboss(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
    relief = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    embossed = 128 + 105 * np.tanh(relief / 160)
    return cv2.cvtColor(np.clip(embossed, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def _edge_art(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 65, 145)
    return cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)


def _sepia(image: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        dtype=np.float32,
    )
    return np.clip(image.astype(np.float32) @ matrix.T, 0, 255).astype(np.uint8)


def _retro(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    faded = cv2.addWeighted(image, 0.72, _sepia(image), 0.28, 0)
    faded = faded.astype(np.float32) * 0.94 + 10
    yy, xx = np.ogrid[-1:1:complex(height), -1:1:complex(width)]
    vignette = np.clip(1.0 - 0.12 * (xx * xx + yy * yy), 0.78, 1.0)
    rng = np.random.default_rng(12345)
    grain = rng.normal(0, 1.8, (height, width, 1))
    return np.clip(faded.astype(np.float32) * vignette[:, :, None] + grain, 0, 255).astype(np.uint8)


FILTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Pencil sketch": _pencil_sketch,
    "Cartoon effect": _cartoon,
    "Oil painting effect": _oil_painting,
    "Emboss": _emboss,
    "Edge art": _edge_art,
    "Sepia": _sepia,
    "Retro effect": _retro,
}


def apply_filter(image: np.ndarray, name: str, strength: int = 100) -> np.ndarray:
    """Apply *name* at 0–100% intensity to an RGB uint8 image."""
    _validate_image(image)
    if name not in FILTERS:
        raise ValueError(f"Unknown filter: {name}")
    if not isinstance(strength, int) or not 0 <= strength <= 100:
        raise ValueError("strength must be an integer from 0 to 100")
    if strength == 0:
        return image.copy()
    filtered = FILTERS[name](image)
    if strength == 100:
        return filtered
    return cv2.addWeighted(image, 1 - strength / 100, filtered, strength / 100, 0)
