import numpy as np
import pytest
from numpy.testing import assert_array_equal

from helpers.render import MapGridToScreen


class TestMapGridToScreen:

    @pytest.mark.parametrize(
        "cell, expected",
        [
            (np.array([0, 0]), np.array([100, 120])),
            (np.array([1, 0]), np.array([132, 120])),
            (np.array([2, 0]), np.array([164, 120])),
            (np.array([1, 1]), np.array([132, 140])),
        ]
    )
    def test_bottom_left_of_cell_returns_expected_values(self, cell, expected):
        result = MapGridToScreen.bottom_left_of_cell(
            grid_cell=cell,
            cell_dimenstions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
        )
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "cell, expected",
        [
            (np.array([0, 0]), np.array([100, 100])),
            (np.array([1, 0]), np.array([132, 100])),
            (np.array([2, 0]), np.array([164, 100])),
            (np.array([1, 1]), np.array([132, 120])),
        ]
    )
    def test_top_left_of_cell_returns_expected_values(self, cell, expected):
        result = MapGridToScreen.top_left_of_cell(
            grid_cell=cell,
            cell_dimenstions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
        )
        assert_array_equal(result, expected)
