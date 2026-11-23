"""Behavioral checks for every artistic filter."""

import unittest

import numpy as np

from filters import FILTERS, apply_filter, resize_for_processing


class ArtisticFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        yy, xx = np.indices((72, 96))
        self.image = np.stack(
            ((xx * 3) % 256, (yy * 4) % 256, ((xx + yy) * 2) % 256), axis=-1
        ).astype(np.uint8)

    def test_all_filters_produce_valid_distinct_images(self) -> None:
        for name in FILTERS:
            with self.subTest(name=name):
                output = apply_filter(self.image, name)
                self.assertEqual(output.shape, self.image.shape)
                self.assertEqual(output.dtype, np.uint8)
                self.assertFalse(np.array_equal(output, self.image))

    def test_strength_zero_returns_independent_original(self) -> None:
        for name in FILTERS:
            with self.subTest(name=name):
                result = apply_filter(self.image, name, 0)
                np.testing.assert_array_equal(result, self.image)
                self.assertFalse(np.shares_memory(result, self.image))

    def test_retro_is_reproducible(self) -> None:
        np.testing.assert_array_equal(
            apply_filter(self.image, "Retro effect"),
            apply_filter(self.image, "Retro effect"),
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_filter(self.image[:, :, 0], "Sepia")
        with self.assertRaises(ValueError):
            apply_filter(self.image, "Missing filter")
        with self.assertRaises(ValueError):
            apply_filter(self.image, "Sepia", 101)

    def test_resize_preserves_aspect_ratio(self) -> None:
        enlarged = np.tile(self.image, (10, 10, 1))
        result = resize_for_processing(enlarged, 300)
        self.assertEqual(result.shape, (225, 300, 3))


if __name__ == "__main__":
    unittest.main()
