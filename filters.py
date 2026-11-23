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


def _pencil_sketch(image: np.ndarray, line_scale: float = 1.0, shading: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 25, 25)
    sigma = max(1.0, min(image.shape[:2]) / 100 * line_scale)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    sketch = np.minimum(gray.astype(np.float32) / np.maximum(background, 1) * 255, 255)
    sketch[background == 0] = 255
    # Retain a little graphite shading without crushing dark areas into black.
    darkness = 1 - gray.astype(np.float32) / 255
    # Sparse diagonal graphite marks become stronger in shadowed regions.
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    hatch = np.maximum(0, np.cos((xx + yy * 0.8) * 1.8)) ** 10
    cross = np.maximum(0, np.cos((xx - yy * 0.8) * 1.8)) ** 10
    shadow = 0.13 * darkness + 0.1 * darkness * hatch
    shadow += 0.08 * np.maximum(darkness - 0.5, 0) * cross
    sketch = np.clip(sketch * (1 - shadow * shading), 0, 255).astype(np.uint8)
    sketch[background == 0] = 255
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)


def _cartoon(image: np.ndarray, color_step: float = 32, ink_strength: float = 0.82) -> np.ndarray:
    smooth = image.copy()
    for _ in range(2):
        smooth = cv2.bilateralFilter(smooth, 9, 55, 45)
    # Fixed bins keep output deterministic and avoid per-image k-means randomness.
    quantized = np.clip(np.round(smooth.astype(np.float32) / color_step) * color_step, 0, 255)
    color = 0.9 * quantized + 0.1 * smooth
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160, L2gradient=True)
    # Keep a dark ink core, soften only the transition to the color regions.
    core = edges.astype(np.float32) / 255
    softened = cv2.GaussianBlur(core, (3, 3), 0.6)
    ink = np.maximum(core, softened * 0.79) * ink_strength
    return np.clip(color * (1 - ink[:, :, None]), 0, 255).astype(np.uint8)


def _oil_painting(image: np.ndarray, brush_size: float = 1.0, paint_relief: float = 3.0) -> np.ndarray:
    """Paint layered, direction-aware brush strokes with subtle pigment relief."""
    height, width = image.shape[:2]
    base = cv2.bilateralFilter(image, 7, 40, 35)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(gray, -1, 1, 0)
    gy = cv2.Sobel(gray, -1, 0, 1)
    # Brush direction follows contours, perpendicular to the brightness gradient.
    angles = np.degrees(np.arctan2(gy, gx)) + 90
    magnitude = cv2.magnitude(gx, gy)
    canvas = cv2.GaussianBlur(base, (0, 0), 2).copy()
    relief = np.zeros((height, width), dtype=np.float32)
    rng = np.random.default_rng(74)
    unit = max(1, min(height, width) / 150) * brush_size
    for scale in (2.5, 1.2, 0.4):
        radius = max(1, round(unit * scale))
        step = max(2, round(radius * 1.25))
        points = [(y, x) for y in range(0, height, step) for x in range(0, width, step)]
        rng.shuffle(points)
        for y, x in points:
            y = int(np.clip(y + rng.integers(-step // 2, step // 2 + 1), 0, height - 1))
            x = int(np.clip(x + rng.integers(-step // 2, step // 2 + 1), 0, width - 1))
            # Fine strokes are reserved for details; flat areas retain broad paint.
            if scale < 1 and magnitude[y, x] < 20:
                continue
            angle = float(angles[y, x] + rng.normal(0, 12))
            axes = (max(2, round(radius * 2.5)), max(1, round(radius * 0.65)))
            pigment = base[y, x].astype(np.float32) + rng.normal(0, 1.2)
            color = tuple(int(v) for v in np.clip(pigment, 0, 255))
            cv2.ellipse(canvas, (x, y), axes, angle, 0, 360, color, -1, cv2.LINE_AA)
            cv2.ellipse(relief, (x, y), axes, angle, 0, 360, float(rng.uniform(0.3, 1)), -1)
    # Lighting on overlapping paint ridges gives a restrained impasto texture.
    relief = cv2.GaussianBlur(relief, (0, 0), 0.6)
    lighting = cv2.Sobel(relief, -1, 1, 0) - cv2.Sobel(relief, -1, 0, 1)
    painted = canvas.astype(np.float32) + lighting[:, :, None] * paint_relief
    return np.clip(painted, 0, 255).astype(np.uint8)


def _emboss(image: np.ndarray, depth: float = 1.0, direction: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32)
    angle = np.deg2rad(direction)
    kernel = np.cos(angle) * kernel + np.sin(angle) * np.rot90(kernel)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
    relief = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    embossed = 128 + 105 * np.tanh(relief * depth / 160)
    return cv2.cvtColor(np.clip(embossed, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def _edge_art(image: np.ndarray, sensitivity: float = 1.0, line_width: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
    gradients = cv2.magnitude(gx, gy)
    nonzero = gradients[gradients > 2]
    high = float(np.clip(np.percentile(nonzero, 72), 30, 180)) if nonzero.size else 80
    high /= sensitivity
    edges = cv2.Canny(blurred, high * 0.4, high, L2gradient=True)
    if line_width > 1:
        size = int(line_width)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)


def _sepia(image: np.ndarray, warmth: float = 1.0) -> np.ndarray:
    # Monochrome brown toning: luminance controls a single continuous palette.
    luminance = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    stops = np.array([0, 0.25, 0.5, 0.75, 1], dtype=np.float32)
    palette = np.array([[15, 10, 7], [77, 55, 35], [145, 111, 74],
                        [207, 177, 133], [255, 247, 226]], dtype=np.float32)
    toned = np.stack([np.interp(luminance, stops, palette[:, c]) for c in range(3)], axis=-1)
    gray = luminance[:, :, None] * 255
    return np.clip(gray * (1 - warmth) + toned * warmth, 0, 255).astype(np.uint8)


def _retro(image: np.ndarray, grain_amount: float = 2.2, vignette_amount: float = 0.12) -> np.ndarray:
    height, width = image.shape[:2]
    color = image.astype(np.float32)
    luminance = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)[:, :, None]
    faded = color * 0.8 + luminance * 0.2
    faded = faded * 0.87 + 16  # Lift blacks and lower whites like faded prints.
    tone = luminance / 255
    faded[:, :, 0] += 7 * tone[:, :, 0]
    faded[:, :, 2] += 5 - 12 * tone[:, :, 0]
    yy, xx = np.meshgrid(np.linspace(-1, 1, height) if height > 1 else [0],
                         np.linspace(-1, 1, width) if width > 1 else [0], indexing="ij")
    vignette = np.clip(1.0 - vignette_amount * (xx * xx + yy * yy), 0.4, 1.0)
    rng = np.random.default_rng(12345)
    grain = rng.normal(0, grain_amount, (height, width, 1))
    return np.clip(faded.astype(np.float32) * vignette[:, :, None] + grain, 0, 255).astype(np.uint8)


FILTERS: dict[str, Callable[..., np.ndarray]] = {
    "Pencil sketch": _pencil_sketch,
    "Cartoon effect": _cartoon,
    "Oil painting effect": _oil_painting,
    "Emboss": _emboss,
    "Edge art": _edge_art,
    "Sepia": _sepia,
    "Retro effect": _retro,
}

# name -> (Vietnamese label, minimum, maximum, default, step, help)
FILTER_PARAMETERS = {
    "Pencil sketch": {
        "line_scale": ("Cỡ nét chì", 0.5, 2.0, 1.0, 0.1, "Tăng để nét và vùng bóng rộng hơn."),
        "shading": ("Độ đậm bóng chì", 0.0, 2.0, 1.0, 0.1, "Giảm nếu ảnh bị xám hoặc nhiều vân chì."),
    },
    "Cartoon effect": {
        "color_step": ("Mức đơn giản hóa màu", 8.0, 64.0, 32.0, 4.0, "Tăng để màu gom thành mảng rõ hơn."),
        "ink_strength": ("Độ đậm viền mực", 0.0, 1.0, 0.82, 0.02, "Giảm nếu viền đen che chi tiết."),
    },
    "Oil painting effect": {
        "brush_size": ("Cỡ cọ", 0.5, 2.0, 1.0, 0.1, "Giảm cho khuôn mặt; tăng cho nền phong cảnh."),
        "paint_relief": ("Độ nổi lớp sơn", 0.0, 8.0, 3.0, 0.5, "Tăng để ánh sáng trên lớp sơn rõ hơn."),
    },
    "Emboss": {
        "depth": ("Độ sâu chạm nổi", 0.3, 2.5, 1.0, 0.1, "Giảm nếu cạnh quá gắt."),
        "direction": ("Xoay hướng ánh sáng", 0.0, 360.0, 0.0, 15.0, "Xoay hướng tạo cạnh sáng và tối, đơn vị độ."),
    },
    "Edge art": {
        "sensitivity": ("Độ nhạy bắt nét", 0.5, 2.0, 1.0, 0.1, "Tăng để lấy thêm nét; giảm để loại nét vụn."),
        "line_width": ("Độ dày nét", 1.0, 3.0, 1.0, 1.0, "Tăng nếu nét quá mảnh."),
    },
    "Sepia": {
        "warmth": ("Độ ấm nâu", 0.0, 1.0, 1.0, 0.05, "0 là đen trắng; 1 là tông nâu đầy đủ."),
    },
    "Retro effect": {
        "grain_amount": ("Hạt phim", 0.0, 8.0, 2.2, 0.2, "Giảm cho ảnh nhỏ hoặc đã nhiễu."),
        "vignette_amount": ("Tối góc", 0.0, 0.3, 0.12, 0.02, "Giảm nếu góc ảnh bị tối mất chi tiết."),
    },
}


def apply_filter(image: np.ndarray, name: str, strength: int = 100, **parameters: float) -> np.ndarray:
    """Apply *name* at 0–100% intensity to an RGB uint8 image."""
    _validate_image(image)
    if name not in FILTERS:
        raise ValueError(f"Unknown filter: {name}")
    if not isinstance(strength, int) or not 0 <= strength <= 100:
        raise ValueError("strength must be an integer from 0 to 100")
    for key, value in parameters.items():
        if key not in FILTER_PARAMETERS[name]:
            raise ValueError(f"Unknown parameter for {name}: {key}")
        _, minimum, maximum, _, step, _ = FILTER_PARAMETERS[name][key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be a finite number from {minimum} to {maximum}")
        if key == "line_width" and value != int(value):
            raise ValueError("line_width must be an integer")
    if strength == 0:
        return image.copy()
    filtered = FILTERS[name](image, **parameters)
    if strength == 100:
        return filtered
    return cv2.addWeighted(image, 1 - strength / 100, filtered, strength / 100, 0)
