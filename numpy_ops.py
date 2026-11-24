"""Handwritten image algorithms. NumPy supplies array arithmetic only.

Filtering uses reflected borders, floating-point accumulation and explicit
rounding to uint8. Canny receives an already smoothed image from its caller.
These implementations follow the algorithms, not OpenCV's exact rounding.
"""

from collections import deque
from functools import lru_cache

import numpy as np


# Làm tròn mảng số, chặn trong [0, 255] rồi đổi sang uint8.
def to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


# Chuyển ảnh RGB (H, W, 3) thành ảnh xám uint8 (H, W).
def grayscale(image: np.ndarray) -> np.ndarray:
    source = image.astype(np.float32)
    return to_uint8(0.299 * source[..., 0] + 0.587 * source[..., 1] + 0.114 * source[..., 2])


# Lặp ảnh xám (H, W) thành ba kênh giống nhau (H, W, 3).
def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[:, :, None], 3, axis=2)


# Tự lọc tương quan chéo: trượt kernel, nhân từng phần tử rồi cộng.
def correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim != 2 or any(size % 2 != 1 for size in kernel.shape):
        raise ValueError("kernel must be a nonempty odd-sized 2D array")
    source = image.astype(np.float32)
    ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
    padding = [(ry, ry), (rx, rx)] + ([(0, 0)] if source.ndim == 3 else [])
    padded = np.pad(source, padding, mode="reflect")
    h, w = source.shape[:2]
    result = np.zeros_like(source)
    for y, x in np.ndindex(kernel.shape):
        if kernel[y, x] != 0:
            result += kernel[y, x] * padded[y:y + h, x:x + w]
    return result


# Tích chập ảnh với kernel: lật kernel 180 độ rồi gọi correlate.
def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return correlate(image, np.asarray(kernel)[::-1, ::-1])


# Làm mờ ảnh bằng hai lượt Gaussian một chiều: ngang rồi dọc.
def gaussian_blur(image: np.ndarray, size: tuple[int, int], sigma: float = 0) -> np.ndarray:
    if sigma <= 0:
        sigma = 0.3 * ((max(size) - 1) * 0.5 - 1) + 0.8
    result = image.astype(np.float32)
    for axis, width in ((1, size[0]), (0, size[1])):
        radius = width // 2 if width else max(1, int(np.ceil(3 * sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        weights = np.exp(-offsets ** 2 / (2 * sigma ** 2))
        weights /= weights.sum()
        kernel = weights[None, :] if axis == 1 else weights[:, None]
        result = correlate(result, kernel)
    return to_uint8(result) if image.dtype == np.uint8 else result


# Làm mượt vùng đồng màu, hạn chế trộn màu qua các biên rõ.
def bilateral_filter(image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    source = image.astype(np.float32)
    radius = diameter // 2
    padding = [(radius, radius), (radius, radius)] + ([(0, 0)] if source.ndim == 3 else [])
    padded = np.pad(source, padding, mode="reflect")
    h, w = source.shape[:2]
    total = np.zeros_like(source)
    normalizer = np.zeros((h, w), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_distance = dx * dx + dy * dy
            if spatial_distance > radius * radius:
                continue
            neighbor = padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            difference = (neighbor - source) ** 2
            if source.ndim == 3:
                difference = difference.sum(axis=2)
            weights = np.exp(-spatial_distance / (2 * sigma_space ** 2)
                             - difference / (2 * sigma_color ** 2))
            normalizer += weights
            total += neighbor * (weights[:, :, None] if source.ndim == 3 else weights)
    divisor = normalizer[:, :, None] if source.ndim == 3 else normalizer
    return to_uint8(total / divisor)


# Tính đạo hàm bậc một bằng kernel Sobel 3x3 tự khai báo.
def sobel(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if (dx, dy) not in ((1, 0), (0, 1)):
        raise ValueError("choose the x or y first derivative")
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    return correlate(image, kernel if dx else kernel.T)


# Làm mảnh biên bằng cách giữ cực đại theo hướng gradient.
def nonmaximum_suppression(magnitude: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    directions = np.floor((np.degrees(np.arctan2(gy, gx)) % 180 + 22.5) / 45).astype(np.uint8) % 4
    h, w = magnitude.shape
    padded = np.pad(magnitude, 1, mode="constant")
    thin = np.zeros_like(magnitude)
    for direction, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        forward = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        backward = padded[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        # One strict comparison prevents a two-pixel plateau from becoming two edges.
        keep = (directions == direction) & (magnitude > forward) & (magnitude >= backward)
        thin[keep] = magnitude[keep]
    return thin


# Ngưỡng kép và nối biên yếu với biên mạnh bằng hàng đợi BFS.
def hysteresis(thin: np.ndarray, low: float, high: float) -> np.ndarray:
    strong = (thin >= high) & (thin > 0)
    weak = (thin >= low) & (thin > 0)
    queue = deque(zip(*np.nonzero(strong)))
    h, w = thin.shape
    while queue:
        y, x = queue.popleft()
        for ny in range(max(0, y - 1), min(h, y + 2)):
            for nx in range(max(0, x - 1), min(w, x + 2)):
                if weak[ny, nx] and not strong[ny, nx]:
                    strong[ny, nx] = True
                    queue.append((ny, nx))
    return strong.astype(np.uint8) * 255


# Phát hiện biên Canny trên ảnh xám đã được làm mượt.
def canny(image: np.ndarray, low: float, high: float) -> np.ndarray:
    gx, gy = sobel(image, 1, 0), sobel(image, 0, 1)
    magnitude = np.hypot(gx, gy)
    thin = nonmaximum_suppression(magnitude, gx, gy)
    return hysteresis(thin, low, high)


# Tạo phần tử cấu trúc hình đĩa dưới dạng mặt nạ bool (size, size).
def ellipse_footprint(size: int) -> np.ndarray:
    radius = size // 2
    yy, xx = np.indices((size, size))
    return (xx - radius) ** 2 + (yy - radius) ** 2 <= max(1, radius) ** 2


# Giãn ảnh xám/nhị phân bằng phần tử cấu trúc đĩa kích thước size.
def dilate(image: np.ndarray, size: int) -> np.ndarray:
    radius = size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="constant")
    result = np.zeros_like(image)
    h, w = image.shape
    for y, x in zip(*np.nonzero(ellipse_footprint(size))):
        # Reflect the offsets: dilation translates each source foreground point.
        sy, sx = 2 * radius - y, 2 * radius - x
        np.maximum(result, padded[sy:sy + h, sx:sx + w], out=result)
    return result


# Tạo lưới tọa độ tương đối quanh tâm cọ, bán kính radius pixel.
@lru_cache(maxsize=64)
def _brush_grid(radius: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return xx.astype(np.float32), yy.astype(np.float32)


# Vẽ trực tiếp một nét cọ elip xoay lên canvas, không trả ảnh mới.
def paint_ellipse(canvas: np.ndarray, center: tuple[int, int], axes: tuple[int, int],
                  angle: float, color: tuple | float) -> None:
    x, y = center
    a, b = axes
    radius = int(np.ceil(max(a, b))) + 1
    x0, x1 = max(0, x - radius), min(canvas.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(canvas.shape[0], y + radius + 1)
    xx, yy = _brush_grid(radius)
    xx = xx[y0 - y + radius:y1 - y + radius, x0 - x + radius:x1 - x + radius]
    yy = yy[y0 - y + radius:y1 - y + radius, x0 - x + radius:x1 - x + radius]
    radians = np.deg2rad(angle)
    cosine, sine = np.cos(radians), np.sin(radians)
    u, v = xx * cosine + yy * sine, -xx * sine + yy * cosine
    distance = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    alpha = np.clip((1 - distance) * min(a, b) + 0.5, 0, 1)
    if canvas.ndim == 3:
        alpha = alpha[:, :, None]
    patch = canvas[y0:y1, x0:x1]
    mixed = patch * (1 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    patch[:] = to_uint8(mixed) if canvas.dtype == np.uint8 else mixed


# Đổi kích thước theo trung bình diện tích các ô pixel nguồn.
def resize_area(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if min(size) < 1:
        raise ValueError("output dimensions must be positive")
    result = image
    # Shrink width first, and integrate strips rather than copying the full
    # high-resolution image to float64. Peak temporary memory stays bounded.
    for axis, count in ((1, size[0]), (0, size[1])):
        source = np.moveaxis(result, axis, 0)
        length = source.shape[0]
        edges = np.linspace(0, length, count + 1)
        indices = np.minimum(edges.astype(int), length - 1)
        fraction = (edges - indices).reshape((-1,) + (1,) * (source.ndim - 1))
        resized = np.empty((count,) + source.shape[1:], dtype=np.float32)
        for start in range(0, source.shape[1], 64):
            strip = source[:, start:start + 64]
            integral = np.concatenate((np.zeros_like(strip[:1], dtype=np.float64),
                                       np.cumsum(strip, axis=0, dtype=np.float64)), axis=0)
            samples = integral[indices] + fraction * strip[indices]
            resized[:, start:start + 64] = np.diff(samples, axis=0) / (length / count)
        result = np.moveaxis(resized, 0, axis)
    return to_uint8(result)


# Pha ảnh gốc và ảnh hiệu ứng theo O=(1-alpha)*original+alpha*filtered.
def blend(original: np.ndarray, filtered: np.ndarray, alpha: float) -> np.ndarray:
    return to_uint8(original.astype(np.float32) * (1 - alpha) + filtered.astype(np.float32) * alpha)
