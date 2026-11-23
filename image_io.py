"""Image decoding and export shared by the UI and tests."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from filters import MAX_SIDE


def decode_image(data: bytes) -> np.ndarray:
    """Decode an image as RGB, orient it and place transparency over white."""
    with Image.open(BytesIO(data)) as source:
        # JPEG decoders can downsample before allocating the full decoded image.
        source.draft("RGB", (MAX_SIDE, MAX_SIDE))
        source = ImageOps.exif_transpose(source)
        source.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return np.asarray(background.convert("RGB")).copy()


def encode_image(image: np.ndarray, output_format: str) -> bytes:
    """Encode an RGB array as PNG or JPEG bytes."""
    if output_format not in {"PNG", "JPEG"}:
        raise ValueError("output_format must be PNG or JPEG")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format=output_format, quality=95)
    return buffer.getvalue()
