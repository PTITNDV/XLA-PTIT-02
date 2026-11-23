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

    def test_oil_painting_is_reproducible_and_does_not_mutate_input(self) -> None:
        original = self.image.copy()
        first = apply_filter(self.image, "Oil painting effect")
        second = apply_filter(self.image, "Oil painting effect")
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(self.image, original)

    def test_emboss_flat_surfaces_have_neutral_background(self) -> None:
        for value in (0, 100, 255):
            image = np.full((32, 32, 3), value, dtype=np.uint8)
            np.testing.assert_array_equal(apply_filter(image, "Emboss"), 128)

    def test_sketch_of_flat_black_image_is_white_paper(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        np.testing.assert_array_equal(apply_filter(image, "Pencil sketch"), 255)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_filter(self.image[:, :, 0], "Sepia")
        with self.assertRaises(ValueError):
            apply_filter(self.image, "Missing filter")
        with self.assertRaises(ValueError):
            apply_filter(self.image, "Sepia", 101)

    def test_edge_art_contains_only_black_and_white(self) -> None:
        output = apply_filter(self.image, "Edge art")
        self.assertTrue(set(np.unique(output)).issubset({0, 255}))
        np.testing.assert_array_equal(output[:, :, 0], output[:, :, 1])

    def test_sepia_palette_is_monotonic_and_warm(self) -> None:
        ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
        output = apply_filter(ramp, "Sepia").astype(int)
        self.assertTrue(np.all(np.diff(output, axis=1) >= 0))
        self.assertTrue(np.all(output[:, :, 0] >= output[:, :, 1]))
        self.assertTrue(np.all(output[:, :, 1] >= output[:, :, 2]))

    def test_every_filter_handles_single_pixel(self) -> None:
        image = np.full((1, 1, 3), 100, dtype=np.uint8)
        for name in FILTERS:
            with self.subTest(name=name):
                self.assertEqual(apply_filter(image, name).shape, image.shape)

    def test_resize_preserves_aspect_ratio(self) -> None:
        enlarged = np.tile(self.image, (10, 10, 1))
        result = resize_for_processing(enlarged, 300)
        self.assertEqual(result.shape, (225, 300, 3))


if __name__ == "__main__":
    unittest.main()
