"""Image decoding and export shared by the UI and tests."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from filters import resize_for_processing


# Giải mã byte tệp ảnh thành mảng RGB uint8 để đưa vào bộ lọc.
def decode_image(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as source:
        source = ImageOps.exif_transpose(source)
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return resize_for_processing(np.asarray(background.convert("RGB")))


# Mã hóa ảnh RGB uint8 thành dữ liệu byte tải xuống.
def encode_image(image: np.ndarray, output_format: str) -> bytes:
    if output_format not in {"PNG", "JPEG"}:
        raise ValueError("output_format must be PNG or JPEG")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format=output_format, quality=95)
    return buffer.getvalue()
