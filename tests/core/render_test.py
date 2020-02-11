import numpy as np
import pytest
from numpy.testing import assert_array_equal

from core.render import MapGridToScreen, Renderable, PrepareForRendering
from helpers.comparisons import AssertSame


class TestMapGridToScreen:

    @pytest.mark.parametrize(
        "cell, expected",
        [
            (np.array([0, 0]), np.array([100, 120])),
            (np.array([1, 0]), np.array([100, 140])),
            (np.array([2, 0]), np.array([100, 160])),
            (np.array([1, 1]), np.array([132, 140])),
        ]
    )
    def test_bottom_left_of_cell_returns_expected_values(self, cell, expected):
        cell_copy = np.copy(cell)
        cell_dimensions = np.array([32, 20])
        top_left_position_of_grid = np.array([100, 100])
        cell_dimensions_copy = np.copy(cell_dimensions)
        top_left_position_of_grid_copy = np.copy(top_left_position_of_grid)
        result = MapGridToScreen.bottom_left_of_cell(
            grid_cell=cell,
            cell_dimensions=cell_dimensions,
            top_left_position_of_grid=top_left_position_of_grid,
        )
        # Check that the function gives the expected result.
        assert_array_equal(result, expected)
        # Check that the function is pure.
        assert_array_equal(cell, cell_copy)
        assert_array_equal(cell_dimensions, cell_dimensions_copy)
        assert_array_equal(top_left_position_of_grid, top_left_position_of_grid_copy)

    @pytest.mark.parametrize(
        "cell, expected",
        [
            (np.array([0, 0]), np.array([100, 100])),
            (np.array([1, 0]), np.array([100, 120])),
            (np.array([2, 0]), np.array([100, 140])),
            (np.array([1, 1]), np.array([132, 120])),
        ]
    )
    def test_top_left_of_cell_returns_expected_values(self, cell, expected):
        cell_copy = np.copy(cell)
        cell_dimensions = np.array([32, 20])
        top_left_position_of_grid = np.array([100, 100])
        cell_dimensions_copy = np.copy(cell_dimensions)
        top_left_position_of_grid_copy = np.copy(top_left_position_of_grid)
        result = MapGridToScreen.top_left_of_cell(
            grid_cell=cell,
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
        )
        # Check that the function gives the expected result.
        assert_array_equal(result, expected)
        # Check that the function is pure.
        assert_array_equal(cell, cell_copy)
        assert_array_equal(cell_dimensions, cell_dimensions_copy)
        assert_array_equal(top_left_position_of_grid, top_left_position_of_grid_copy)


class TestRenderable:

    def test_instantiate_new_renderable_returns_an_immutable_object(self):
        renderable = Renderable(
            get_image=lambda: "dummy",
            position=(1, 2),
            priority=(3, 4),
        )
        assert renderable.get_image() == "dummy"
        assert renderable.priority == (3, 4)
        with pytest.raises(AttributeError):
            renderable.position = (5, 6)
        assert renderable.position == (1, 2)
