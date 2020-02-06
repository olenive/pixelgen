"""Tests that open a window and draw a map of tiles on the screen."""

import numpy as np
from core.render import InteractiveDisplay


PATH_TO_FLOOR_SPRITE = "data/sprites/dummy_floor_sand.png"


class TestDrawTiles:

    def test_draw_floor_2x3(self):
        all_floor = np.full((2, 3), 0)
        interactive = InteractiveDisplay(
            tile_grid=all_floor,
            cell_dimensions=np.array([32, 20]),
            top_left_position_of_grid=np.array([100, 100])
        )
        print(interactive.sprite_info)
        sprite_id = PATH_TO_FLOOR_SPRITE
        expected = (
            # First row
            (sprite_id, np.array([100, 100]), (0, 100)),
            (sprite_id, np.array([132, 100]), (0, 100)),
            (sprite_id, np.array([164, 100]), (0, 100)),
            # Second row
            (sprite_id, np.array([100, 120]), (0, 120)),
            (sprite_id, np.array([132, 120]), (0, 120)),
            (sprite_id, np.array([164, 120]), (0, 120)),
        )
        # import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        interactive.run(maximum_frames=60)  # Display for two seconds.
        assert interactive == expected
