import pytest
import numpy as np
from numpy.testing import assert_array_equal

from core.image import ImageConvert, ImageIO


class TestImageConvert:

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
        result = ImageConvert.flat_hashable_from_rgba(given)
        assert result[0] == expected_flat
        assert result[1] == expected_shape

    def test_grid_from_rgba_creates_expected_array_from_5x5(self):
        """RGBA values from image are converted to -1, 0 or 1."""
        image_array = ImageIO.rgba_png_to_array("tests/data/black_white_and_blank_5x5.png")
        result = ImageConvert.grid_from_rgba(image_array)
        expected = np.array([
            [0, 1, -1, 0, -1],
            [-1, -1, -1, 0, -1],
            [-1, -1, 1, 0, -1],
            [-1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1],
        ])
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "given, expected",
        (
            (0.0, (0, 0, 0, 0)),
            (0.249, (0, 0, 0, 0)),
            (0.25, (1, 1, 1, 1)),
            (0.2501, (1, 1, 1, 1)),
            (0.5, (2, 2, 2, 2)),
            (0.999, (3, 3, 3, 3)),
            (0.6, (2, 2, 2, 2)),
        )
    )
    def test_continuous_value_to_discrete_rgba_palette_returns_expected_rgba(self, given, expected):
        palette = ((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3))
        result = ImageConvert.continuous_value_to_discrete_palette(given, palette)
        assert result == expected

    def test_matrix_to_rgb_palette_and_alphas_returns_expected(self):
        palette = ((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3))
        matrix = np.array([
            [0.0, 0.2, 0.9],
            [0.9, 0.49, 0.6],
            [0.1, 0.7, 0.8],
        ])
        expected_alphas = np.array([
            [0, 0, 3],
            [3, 1, 2],
            [0, 2, 3],
        ])
        c0 = [0, 0, 0]
        c1 = [1, 1, 1]
        c2 = [2, 2, 2]
        c3 = [3, 3, 3]
        expected_rgba = np.array([
            [c0, c0, c3],
            [c3, c1, c2],
            [c0, c2, c3],
        ])
        result = ImageConvert.matrix_to_rgb_palette_and_alphas(matrix, palette)
        assert_array_equal(result[0], expected_rgba)
        assert_array_equal(result[1], expected_alphas)
