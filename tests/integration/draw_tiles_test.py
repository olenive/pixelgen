"""Tests that open a window and draw a map of tiles on the screen."""

import numpy as np
from core.render import InteractiveDisplay
from helpers.comparisons import AssertSame


PATH_TO_FLOOR_SPRITE = "data/sprites/dummy_floor_sand_32x20.png"
PATH_TO_WALL_SPRITE = "data/sprites/dummy_wall_terracotta_32x12.png"
PATH_TO_ROOF_SPRITE = "data/sprites/dummy_roof_blue_32x20.png"


class TestDrawTiles:

    sprite_dimensions = {
        PATH_TO_WALL_SPRITE: (32, 12),
        PATH_TO_ROOF_SPRITE: (32, 20),
    }

    def test_draw_floor_2x3(self):
        all_floor = np.full((2, 3), 0)
        interactive = InteractiveDisplay(
            tile_grid=all_floor,
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
            sprite_dimensions=TestDrawTiles.sprite_dimensions,
        )
        sprite_id = PATH_TO_FLOOR_SPRITE
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
        interactive.run(maximum_frames=30)  # Display for maximum_frames at 30 frames per second.
        AssertSame.ids_positions_priorities(interactive.ids_positions_priorities, expected)

    def test_draw_2x3_floor_with_wall_at_1_1(self):
        """Draw six floor tiles with a wall tile in the middle of the second row."""
        one_wall = np.array([
            [0, 0, 0],
            [0, 1, 0],
        ])
        interactive = InteractiveDisplay(
            tile_grid=one_wall,
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100]),
            sprite_dimensions=TestDrawTiles.sprite_dimensions,
        )
        floor = PATH_TO_FLOOR_SPRITE
        wall = PATH_TO_WALL_SPRITE
        roof = PATH_TO_ROOF_SPRITE
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
        interactive.run(maximum_frames=30)  # Display for maximum_frames at 30 frames per second.
        AssertSame.ids_positions_priorities(interactive.ids_positions_priorities, expected)
