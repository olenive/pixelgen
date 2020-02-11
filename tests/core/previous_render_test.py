import numpy as np
import pytest
from numpy.testing import assert_array_equal

from core.render import MapGridToScreen, PrepareForRendering
from helpers.comparisons import AssertSame


PATH_TO_FLOOR_SPRITE = "data/sprites/dummy_floor_sand_32x20.png"
PATH_TO_WALL_SPRITE = "data/sprites/dummy_wall_terracotta_32x12.png"
PATH_TO_ROOF_SPRITE = "data/sprites/dummy_roof_blue_32x20.png"


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


class TestPrepareForRendering:

    sprite_dimensions = {
        "wall sprite": (32, 12),
        "roof sprite": (32, 20),
    }

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
            top_left_of_tile=top_left_of_tile,
        )
        expected = (
            ("floor sprite", expected_position, expected_priority),
        )
        AssertSame.ids_positions_priorities(result, expected)

    def test_ids_positions_priorities_for_wall_tile_returns_expected_data_at_100_100(self):
        result = PrepareForRendering.ids_positions_priorities_for_wall_tile(
            cell_dimensions=np.array([32, 20]),
            top_left_of_tile=np.array([100, 100]),
            sprite_dimensions=TestPrepareForRendering.sprite_dimensions,
        )
        expected = (
            ("wall sprite", np.array([100, 108]), (1, 120)),
            ("roof sprite", np.array([100, 88]), (1, 108)),
        )
        AssertSame.ids_positions_priorities(result, expected)

    def test_ids_positions_priorities_for_wall_tile_returns_expected_data_at_132_120(self):
        result = PrepareForRendering.ids_positions_priorities_for_wall_tile(
            cell_dimensions=np.array([32, 20]),
            top_left_of_tile=np.array([132, 120]),
            sprite_dimensions=TestPrepareForRendering.sprite_dimensions,
        )
        expected = (
            ("wall sprite", np.array([132, 128]), (1, 140)),
            ("roof sprite", np.array([132, 108]), (1, 128)),
        )
        AssertSame.ids_positions_priorities(result, expected)

    def test_ids_positions_priorities_for_tile_in_top_left_corner(self):
        result = PrepareForRendering.ids_positions_priorities_for_tile(
            tile_type=0,
            cell_dimensions=np.array([32, 20]),
            top_left_of_tile=np.array([100, 100]),
            sprite_dimensions=TestPrepareForRendering.sprite_dimensions,
        )
        expected = (
            ("floor sprite", np.array([100, 100]), (0, 120)),
            # Note that priority depends on where the bottom of the floor tile starts, not its top left corner.
        )
        AssertSame.ids_positions_priorities(result, expected)

    def test_ids_positions_priorities_for_tile_in_row_2_column_3(self):
        result = PrepareForRendering.ids_positions_priorities_for_tile(
            tile_type=0,
            cell_dimensions=np.array([32, 20]),
            top_left_of_tile=np.array([164, 120]),
            sprite_dimensions=TestPrepareForRendering.sprite_dimensions,
        )
        expected = (
            ("floor sprite", np.array([164, 120]), (0, 140)),
            # Note that priority depends on where the bottom of the floor tile starts, not its top left corner.
        )
        AssertSame.ids_positions_priorities(result, expected)

    def test_collect_images_for_grid_of_2x3_floor_tiles(self):
        result = PrepareForRendering.collect_images_for_grid(
            grid=np.full((2, 3), 0),
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
            sprite_dimensions=TestPrepareForRendering.sprite_dimensions,
        )
        img = "floor sprite"
        expected = (
            (img, np.array([100, 100]), (0, 120)),
            (img, np.array([132, 100]), (0, 120)),
            (img, np.array([164, 100]), (0, 120)),
            (img, np.array([100, 120]), (0, 140)),
            (img, np.array([132, 120]), (0, 140)),
            (img, np.array([164, 120]), (0, 140)),
        )
        AssertSame.ids_positions_priorities(result, expected)

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
        AssertSame.ids_positions_priorities(result, expected)
