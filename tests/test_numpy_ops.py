"""Analytical fixtures for handwritten processing, independent of OpenCV."""

import subprocess
import sys
import unittest

import numpy as np

import numpy_ops as ops


class NumpyOperationsTests(unittest.TestCase):
    def test_rgb_luminance_and_channel_replication(self):
        colors = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(ops.grayscale(colors), [[76, 150, 29]])
        gray = ops.grayscale(colors)
        np.testing.assert_array_equal(ops.gray_to_rgb(gray)[..., 2], gray)

    def test_correlation_preserves_asymmetric_kernel_orientation(self):
        ramp = np.tile(np.arange(7, dtype=np.float32), (5, 1))
        kernel = np.array([[-1, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(ops.correlate(ramp, kernel)[:, 1:-1], 2)

    def test_gaussian_impulse_matches_normalized_outer_product(self):
        impulse = np.zeros((9, 9), dtype=np.float32)
        impulse[4, 4] = 1
        weights = np.exp(-np.arange(-1, 2, dtype=np.float32) ** 2 / 2)
        weights /= weights.sum()
        result = ops.gaussian_blur(impulse, (3, 3), 1)
        np.testing.assert_allclose(result[3:6, 3:6], weights[:, None] * weights[None, :], atol=1e-7)
        self.assertAlmostEqual(float(result.sum()), 1, places=6)

    def test_filters_preserve_constant_and_single_axis_images(self):
        for shape in ((1, 1), (1, 7), (7, 1), (5, 7), (5, 7, 3)):
            with self.subTest(shape=shape):
                image = np.full(shape, 83, dtype=np.uint8)
                np.testing.assert_array_equal(ops.gaussian_blur(image, (0, 0), 2), image)
                np.testing.assert_array_equal(ops.bilateral_filter(image, 5, 25, 25), image)

    def test_bilateral_keeps_high_contrast_step(self):
        image = np.zeros((9, 9), dtype=np.uint8)
        image[:, 4:] = 255
        np.testing.assert_array_equal(ops.bilateral_filter(image, 5, 10, 10), image)

    def test_sobel_preserves_sign_and_orientation(self):
        ramp = np.tile(np.arange(7, dtype=np.float32), (7, 1))
        np.testing.assert_array_equal(ops.sobel(ramp, 1, 0)[1:-1, 1:-1], 8)
        np.testing.assert_array_equal(ops.sobel(ramp, 0, 1), 0)
        np.testing.assert_array_equal(ops.sobel(-ramp, 1, 0)[1:-1, 1:-1], -8)

    def test_hysteresis_links_weak_chain_but_discards_isolated_weak(self):
        thin = np.zeros((7, 7), dtype=np.float32)
        thin[1, 1] = 200
        thin[2, 2] = thin[3, 3] = 80
        thin[1, 5] = 80
        thin[4, 4] = 10
        result = ops.hysteresis(thin, 50, 100)
        self.assertEqual(np.count_nonzero(result), 3)
        self.assertEqual(result[3, 3], 255)
        self.assertEqual(result[1, 5], 0)

    def test_canny_detects_step_without_flat_region_edges(self):
        image = np.zeros((16, 16), dtype=np.uint8)
        np.testing.assert_array_equal(ops.canny(image, 50, 100), 0)
        image[:, 8:] = 255
        result = ops.canny(image, 50, 100)
        self.assertTrue(np.any(result[:, 7:9]))
        self.assertFalse(np.any(result[:, :6]))
        self.assertFalse(np.any(result[:, 10:]))
        self.assertTrue(np.all(np.count_nonzero(result, axis=1) == 1))

    def test_area_resize_integer_and_fractional_boxes(self):
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)
        np.testing.assert_array_equal(ops.resize_area(image, (2, 2)), [[2, 4], [10, 12]])
        image = np.array([[0, 30, 90]], dtype=np.uint8)
        np.testing.assert_array_equal(ops.resize_area(image, (2, 1)), [[10, 70]])

    def test_painted_ellipse_stays_local_and_respects_rotation(self):
        horizontal = np.zeros((21, 21), dtype=np.float32)
        vertical = horizontal.copy()
        ops.paint_ellipse(horizontal, (10, 10), (5, 2), 0, 1.0)
        ops.paint_ellipse(vertical, (10, 10), (5, 2), 90, 1.0)
        np.testing.assert_allclose(horizontal.T, vertical, atol=1e-6)
        self.assertEqual(horizontal[10, 10], 1)
        self.assertEqual(horizontal[0, 0], 0)
        ops.paint_ellipse(horizontal, (0, 0), (5, 2), 45, 1.0)

    def test_blend_midpoint_without_overflow(self):
        black = np.zeros((2, 2, 3), dtype=np.uint8)
        white = np.full_like(black, 255)
        np.testing.assert_array_equal(ops.blend(black, white, 0.5), 128)
        np.testing.assert_array_equal(ops.blend(black, white, 0), black)
        np.testing.assert_array_equal(ops.blend(black, white, 1), white)

    def test_all_filters_work_with_opencv_import_blocked(self):
        code = """
import sys
sys.modules['cv2'] = None
import numpy as np
from filters import FILTERS, apply_filter
from image_io import encode_image, decode_image
image = np.full((8, 9, 3), 100, dtype=np.uint8)
for name in FILTERS:
    result = apply_filter(image, name)
    assert result.shape == image.shape
assert decode_image(encode_image(image, 'PNG')).shape == image.shape
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
