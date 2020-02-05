import numpy as np
from numpy.testing import assert_array_equal

from helpers.image import ImageConverter, ImageIO


class TestImageConverter:

    def test_flat_hashable_from_rgba(self):
        vector = np.array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ])
        given = np.reshape(vector, (2, 3, 4))
        expected_flat = (
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
            (17, 18, 19, 20),
            (21, 22, 23, 24),
        )
        expected_shape = (2, 3, 4)
        result = ImageConverter.flat_hashable_from_rgba(given)
        assert result[0] == expected_flat
        assert result[1] == expected_shape

    def test_grid_from_rgba_creates_expected_array_from_5x5(self):
        """RGBA values from image are converted to -1, 0 or 1."""
        image_array = ImageIO.rgba_png_to_array("tests/data/black_white_and_blank_5x5.png")
        result = ImageConverter.grid_from_rgba(image_array)
        expected = np.array([
            [0, 1, -1, 0, -1],
            [-1, -1, -1, 0, -1],
            [-1, -1, 1, 0, -1],
            [-1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1],
        ])
        assert_array_equal(result, expected)
