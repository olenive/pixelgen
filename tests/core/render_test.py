import pytest
from numpy.testing import assert_array_equal

from core.render import MapGridToScreen


class TestMapGridToScreen:

    @pytest.mark.parametrize(
        "cell, expected",
        [
            ((0, 0), (100, 120)),
            ((1, 0), (100, 140)),
            ((2, 0), (100, 160)),
            ((1, 1), (132, 140)),
        ]
    )
    def test_bottom_left_of_cell_returns_expected_values(self, cell, expected):
        result = MapGridToScreen.bottom_left_of_cell(
            grid_cell=cell,
            cell_dimensions=(32, 20),
            top_left_position_of_grid=(100, 100),
        )
        assert result == expected

    @pytest.mark.parametrize(
        "cell, expected",
        [
            ((0, 0), (100, 100)),
            ((1, 0), (100, 120)),
            ((2, 0), (100, 140)),
            ((1, 1), (132, 120)),
        ]
    )
    def test_top_left_of_cell_returns_expected_values(self, cell, expected):
        result = MapGridToScreen.top_left_of_cell(
            grid_cell=cell,
            cell_dimensions=(32, 20),
            top_left_position_of_grid=(100, 100),
        )
        assert_array_equal(result, expected)
