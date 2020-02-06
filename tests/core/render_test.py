import numpy as np
import pytest
from numpy.testing import assert_array_equal
from typing import Iterable, Tuple

from core.render import MapGridToScreen, PrepareForRendering


def compare_ids_positions_priorities(
    first: Iterable[Tuple[str, np.array, Tuple[int, int]]],
    second: Iterable[Tuple[str, np.array, Tuple[int, int]]],
) -> None:
    assert len(first) == len(second)
    for i in range(len(first)):
        assert len(first[i]) == 3
        assert len(second[i]) == 3
        assert first[i][0] == second[i][0]
        assert_array_equal(first[i][1], second[i][1])
        assert first[i][2] == second[i][2]


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
            (np.array([1, 0]), np.array([132, 100])),
            (np.array([2, 0]), np.array([164, 100])),
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


class TestPrepareForRendering:

    @pytest.mark.parametrize(
        "top_left_of_tile, expected_position, expected_priority",
        [
            (np.array([0, 0]), np.array([0, 0]), (0, 20)),
            (np.array([10, 20]), np.array([10, 20]), (0, 40)),
        ]
    )
    def test_ids_positions_priorities_for_floor_tile_returns_expected_data(
        self, top_left_of_tile, expected_position, expected_priority
    ):
        result = PrepareForRendering.ids_positions_priorities_for_floor_tile(
            cell_dimensions=np.array([32, 20]),
            top_left_of_tile=top_left_of_tile
        )
        expected = (
            ('data/sprites/dummy_floor_sand.png', expected_position, expected_priority),
        )
        compare_ids_positions_priorities(result, expected)

    def test_order_by_priority_returns_correct_order_based_on_priorites(self):
        unsorted_images_positions_priorities = (
            # e.g. some higher level tiles
            ("img2", np.array([0, 15]), (2, 15)),
            ("img2", np.array([0, 10]), (2, 10)),
            # e.g. some floor tiles
            ("img1", np.array([0, 10]), (0, 10)),
            ("img1", np.array([0, 20]), (0, 20)),
            ("img1", np.array([10, 10]), (0, 10)),
            ("img1", np.array([10, 20]), (0, 20)),
            # e.g. a chacter somewhere on the floor
            ("imgX", np.array([5, 16]), (1, 16)),
            # e.g. a wall and roof
            ("img3", np.array([10, 20]), (1, 20)),
            ("img4", np.array([10, 12]), (1, 12)),
        )
        expected = (
            # e.g. some floor tiles
            ("img1", np.array([0, 10]), (0, 10)),
            ("img1", np.array([10, 10]), (0, 10)),
            ("img1", np.array([0, 20]), (0, 20)),
            ("img1", np.array([10, 20]), (0, 20)),

            ("img4", np.array([10, 12]), (1, 12)),
            ("imgX", np.array([5, 16]), (1, 16)),  # In this example the character is behind the wall (and roof).
            ("img3", np.array([10, 20]), (1, 20)),
            # e.g. some higher level tiles
            ("img2", np.array([0, 10]), (2, 10)),
            ("img2", np.array([0, 15]), (2, 15)),
        )
        result = PrepareForRendering.order_by_priority(unsorted_images_positions_priorities)
        assert result == expected
