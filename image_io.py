"""Image decoding and export shared by the UI and tests."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from filters import resize_for_processing


def decode_image(data: bytes) -> np.ndarray:
    """Decode an image as RGB, orient it and place transparency over white."""
    with Image.open(BytesIO(data)) as source:
        source = ImageOps.exif_transpose(source)
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return resize_for_processing(np.asarray(background.convert("RGB")))


def encode_image(image: np.ndarray, output_format: str) -> bytes:
    """Encode an RGB array as PNG or JPEG bytes."""
    if output_format not in {"PNG", "JPEG"}:
        raise ValueError("output_format must be PNG or JPEG")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format=output_format, quality=95)
    return buffer.getvalue()
