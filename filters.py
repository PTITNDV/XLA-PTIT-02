"""Seven deterministic artistic filters for RGB uint8 images."""

from collections.abc import Callable

import numpy_ops as ops
import numpy as np


MAX_SIDE = 1600


# Giới hạn cạnh dài của ảnh RGB uint8, giữ tỷ lệ hai chiều.
def resize_for_processing(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    _validate_image(image)
    if max_side < 1:
        raise ValueError("max_side must be positive")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return ops.resize_area(image, size)


# Kiểm tra ảnh là mảng NumPy uint8 có ba kênh và không rỗng.
def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("image must have shape (height, width, 3) and be nonempty")


# Tạo tranh chì: ảnh xám -> bilateral -> nền Gaussian -> chia ảnh.
def _pencil_sketch(image: np.ndarray, line_scale: float = 1.0, shading: float = 1.0) -> np.ndarray:
    gray = ops.grayscale(image)
    gray = ops.bilateral_filter(gray, 5, 25, 25)
    sigma = max(1.0, min(image.shape[:2]) / 100 * line_scale)
    background = ops.gaussian_blur(gray, (0, 0), sigma=sigma)
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
    return ops.gray_to_rgb(sketch)


# Tạo hoạt hình bằng làm mượt giữ biên, gom màu và phủ nét mực.
def _cartoon(image: np.ndarray, color_step: float = 32, ink_strength: float = 0.82) -> np.ndarray:
    smooth = image.copy()
    for _ in range(2):
        smooth = ops.bilateral_filter(smooth, 9, 55, 45)
    # Fixed bins keep output deterministic and avoid per-image k-means randomness.
    quantized = np.clip(np.round(smooth.astype(np.float32) / color_step) * color_step, 0, 255)
    color = 0.9 * quantized + 0.1 * smooth
    gray = ops.grayscale(smooth)
    edges = ops.canny(gray, 70, 160)
    # Keep a dark ink core, soften only the transition to the color regions.
    core = edges.astype(np.float32) / 255
    softened = ops.gaussian_blur(core, (3, 3), 0.6)
    ink = np.maximum(core, softened * 0.79) * ink_strength
    return np.clip(color * (1 - ink[:, :, None]), 0, 255).astype(np.uint8)


# Mô phỏng sơn dầu bằng ba lớp cọ elip theo cấu trúc ảnh.
def _oil_painting(image: np.ndarray, brush_size: float = 1.0, paint_relief: float = 3.0) -> np.ndarray:
    height, width = image.shape[:2]
    base = ops.bilateral_filter(image, 7, 40, 35)
    gray = ops.grayscale(base).astype(np.float32)
    gray = ops.gaussian_blur(gray, (0, 0), 1.2)
    gx = ops.sobel(gray, 1, 0)
    gy = ops.sobel(gray, 0, 1)
    # Brush direction follows contours, perpendicular to the brightness gradient.
    angles = np.degrees(np.arctan2(gy, gx)) + 90
    magnitude = np.hypot(gx, gy)
    canvas = ops.gaussian_blur(base, (0, 0), 2).copy()
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
            ops.paint_ellipse(canvas, (x, y), axes, angle, color)
            ops.paint_ellipse(relief, (x, y), axes, angle, float(rng.uniform(0.3, 1)))
    # Lighting on overlapping paint ridges gives a restrained impasto texture.
    relief = ops.gaussian_blur(relief, (0, 0), 0.6)
    lighting = ops.sobel(relief, 1, 0) - ops.sobel(relief, 0, 1)
    painted = canvas.astype(np.float32) + lighting[:, :, None] * paint_relief
    return np.clip(painted, 0, 255).astype(np.uint8)


# Tạo cảm giác chạm nổi bằng kernel định hướng trên ảnh xám.
def _emboss(image: np.ndarray, depth: float = 1.0, direction: float = 0.0) -> np.ndarray:
    gray = ops.grayscale(image)
    kernel = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32)
    angle = np.deg2rad(direction)
    kernel = np.cos(angle) * kernel + np.sin(angle) * np.rot90(kernel)
    gray = ops.gaussian_blur(gray, (3, 3), 0.7)
    relief = ops.correlate(gray, kernel)
    embossed = 128 + 105 * np.tanh(relief * depth / 160)
    return ops.gray_to_rgb(np.clip(embossed, 0, 255).astype(np.uint8))


# Tạo hình nét đen nền trắng từ các biên Canny.
def _edge_art(image: np.ndarray, sensitivity: float = 1.0, line_width: float = 1.0) -> np.ndarray:
    gray = ops.grayscale(image)
    blurred = ops.gaussian_blur(gray, (5, 5), 0)
    gx = ops.sobel(blurred, 1, 0)
    gy = ops.sobel(blurred, 0, 1)
    gradients = np.hypot(gx, gy)
    nonzero = gradients[gradients > 2]
    high = float(np.clip(np.percentile(nonzero, 72), 30, 180)) if nonzero.size else 80
    high /= sensitivity
    edges = ops.canny(blurred, high * 0.4, high)
    if line_width > 1:
        size = int(line_width)
        edges = ops.dilate(edges, size)
    return ops.gray_to_rgb(255 - edges)


# Ánh xạ độ sáng sang bảng màu nâu từ tối tới trắng kem.
def _sepia(image: np.ndarray, warmth: float = 1.0) -> np.ndarray:
    # Monochrome brown toning: luminance controls a single continuous palette.
    luminance = ops.grayscale(image).astype(np.float32) / 255
    stops = np.array([0, 0.25, 0.5, 0.75, 1], dtype=np.float32)
    palette = np.array([[15, 10, 7], [77, 55, 35], [145, 111, 74],
                        [207, 177, 133], [255, 247, 226]], dtype=np.float32)
    toned = np.stack([np.interp(luminance, stops, palette[:, c]) for c in range(3)], axis=-1)
    gray = luminance[:, :, None] * 255
    return np.clip(gray * (1 - warmth) + toned * warmth, 0, 255).astype(np.uint8)


# Mô phỏng phim cũ bằng màu phai, hạt sáng và tối góc.
def _retro(image: np.ndarray, grain_amount: float = 2.2, vignette_amount: float = 0.12) -> np.ndarray:
    height, width = image.shape[:2]
    color = image.astype(np.float32)
    luminance = ops.grayscale(image).astype(np.float32)[:, :, None]
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


# Điểm gọi chung: kiểm tra tham số, chạy hiệu ứng và pha cường độ.
def apply_filter(image: np.ndarray, name: str, strength: int = 100, **parameters: float) -> np.ndarray:
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
    return ops.blend(image, filtered, strength / 100)
