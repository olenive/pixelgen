"""Tests that open a window and draw a map of tiles on the screen."""

import numpy as np

from core.render import ExampleDisplay, MultiTilesetDisplay
from helpers.comparisons import AssertSame
from helpers.image import ImageIO, ImageConverter


PATH_TO_FLOOR_SPRITE = "data/sprites/dummy_floor_sand_32x20.png"
PATH_TO_WALL_SPRITE = "data/sprites/dummy_wall_terracotta_32x12.png"
PATH_TO_ROOF_SPRITE = "data/sprites/dummy_roof_blue_32x20.png"


class TestDrawTiles:

    sprite_dimensions = {
        "wall sprite": (32, 12),
        "roof sprite": (32, 20),
    }

    def test_draw_floor_2x3(self):
        all_floor = np.full((2, 3), 0)
        example_display = ExampleDisplay(
            tile_grid=all_floor,
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
            sprite_dimensions=TestDrawTiles.sprite_dimensions,
        )
        sprite_id = "floor sprite"
        expected = (
            # First row
            (sprite_id, np.array([100, 100]), (0, 120)),
            (sprite_id, np.array([132, 100]), (0, 120)),
            (sprite_id, np.array([164, 100]), (0, 120)),
            # Second row
            (sprite_id, np.array([100, 120]), (0, 140)),
            (sprite_id, np.array([132, 120]), (0, 140)),
            (sprite_id, np.array([164, 120]), (0, 140)),
        )
        example_display.run(maximum_frames=30)  # Display for maximum_frames at 30 frames per second.
        AssertSame.ids_positions_priorities(example_display.ids_positions_priorities, expected)

    def test_draw_2x3_floor_with_wall_at_1_1(self):
        """Draw six floor tiles with a wall tile in the middle of the second row."""
        one_wall = np.array([
            [0, 0, 0],
            [0, 1, 0],
        ])
        example_display = ExampleDisplay(
            tile_grid=one_wall,
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
            sprite_dimensions=TestDrawTiles.sprite_dimensions,
        )
        floor = "floor sprite"
        wall = "wall sprite"
        roof = "roof sprite"
        expected = (
            # First row
            (floor, np.array([100, 100]), (0, 120)),
            (floor, np.array([132, 100]), (0, 120)),
            (floor, np.array([164, 100]), (0, 120)),
            # Second row
            (floor, np.array([100, 120]), (0, 140)),
            (floor, np.array([164, 120]), (0, 140)),
            (roof, np.array([132, 108]), (1, 128)),
            (wall, np.array([132, 128]), (1, 140)),
        )
        example_display.run(maximum_frames=30)  # Display for maximum_frames at 30 frames per second.
        AssertSame.ids_positions_priorities(example_display.ids_positions_priorities, expected)

    def test_draw_3x3_floor_with_wall_at_1_1(self):
        """Draw six floor tiles with a wall tile in the middle of the second row."""
        one_wall = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        example_display = ExampleDisplay(
            tile_grid=one_wall,
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
            sprite_dimensions=TestDrawTiles.sprite_dimensions,
        )
        floor = "floor sprite"
        wall = "wall sprite"
        roof = "roof sprite"
        expected = (
            # First row
            (floor, np.array([100, 100]), (0, 120)),
            (floor, np.array([132, 100]), (0, 120)),
            (floor, np.array([164, 100]), (0, 120)),
            # Second row
            (floor, np.array([100, 120]), (0, 140)),
            (floor, np.array([164, 120]), (0, 140)),
            # Third row
            (floor, np.array([100, 140]), (0, 160)),
            (floor, np.array([132, 140]), (0, 160)),
            (floor, np.array([164, 140]), (0, 160)),
            # Wall
            (roof, np.array([132, 108]), (1, 128)),
            (wall, np.array([132, 128]), (1, 140)),
        )
        example_display.run(maximum_frames=30)  # Display for maximum_frames at 30 frames per second.
        AssertSame.ids_positions_priorities(example_display.ids_positions_priorities, expected)

    def test_draw_evaluation_grid(self):
        """Draw grid loaded from PNG file in data directory."""
        image_array = ImageIO.rgba_png_to_array("data/evaluation_grid_01.png")
        grid = ImageConverter.grid_from_rgba(image_array)
        example_display = ExampleDisplay(
            tile_grid=grid,
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([10, 10]),
            sprite_dimensions=TestDrawTiles.sprite_dimensions,
        )
        example_display.run(maximum_frames=30)

    def test_draw_buttons_containing_examples_of_different_tile_sets(self):
        """Draw multiple examples of tilesets on a single screen.

        In this case they are actually the same tile set...
        """
        one_wall = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        cell_size = (32, 20)
        display = MultiTilesetDisplay(
            tile_grid=one_wall,
            button_grid_size=(5, 3),
            cell_dimensions=np.array(cell_size),
            button_dimensions=(cell_size[0] * 3 + 20, cell_size[1] * 3 + 20),
            top_left_position_of_grid=np.array([100, 100]),
            sprite_dimensions=TestDrawTiles.sprite_dimensions,
            button_inner_boarder=np.array([10, 10])
        )
        display.draw_buttons(maximum_frames=9930)  # Display for maximum_frames at 30 frames per second.
