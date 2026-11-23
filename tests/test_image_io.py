"""Tests for input normalization and output encoding."""

from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from image_io import decode_image, encode_image


class ImageIOTests(unittest.TestCase):
    def test_grayscale_input_becomes_rgb(self) -> None:
        data = BytesIO()
        Image.new("L", (2, 3), 80).save(data, format="PNG")
        decoded = decode_image(data.getvalue())
        self.assertEqual(decoded.shape, (3, 2, 3))
        np.testing.assert_array_equal(decoded[0, 0], [80, 80, 80])

    def test_transparency_is_composited_on_white(self) -> None:
        data = BytesIO()
        Image.new("RGBA", (1, 1), (255, 0, 0, 0)).save(data, format="PNG")
        decoded = decode_image(data.getvalue())
        np.testing.assert_array_equal(decoded[0, 0], [255, 255, 255])

    def test_exported_images_can_be_opened(self) -> None:
        image = np.full((4, 5, 3), 100, dtype=np.uint8)
        for output_format in ("PNG", "JPEG"):
            with self.subTest(output_format=output_format):
                with Image.open(BytesIO(encode_image(image, output_format))) as reopened:
                    self.assertEqual(reopened.size, (5, 4))
                    self.assertEqual(reopened.format, output_format)

    def test_invalid_image_data_is_rejected(self) -> None:
        with self.assertRaises(OSError):
            decode_image(b"not an image")

    def test_large_image_is_resized_before_output(self) -> None:
        data = BytesIO()
        Image.new("RGB", (2400, 1200), "red").save(data, format="PNG")
        decoded = decode_image(data.getvalue())
        self.assertEqual(decoded.shape, (800, 1600, 3))

    def test_exif_orientation_is_applied(self) -> None:
        data = BytesIO()
        image = Image.new("RGB", (12, 8), "red")
        exif = image.getexif()
        exif[274] = 6
        image.save(data, format="JPEG", exif=exif)
        self.assertEqual(decode_image(data.getvalue()).shape, (12, 8, 3))


if __name__ == "__main__":
    unittest.main()
