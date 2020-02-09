import os
import neat
import numpy as np

from core.stagnation import InteractiveStagnation
from core.render import MultiTilesetDisplay


PATH_TO_FLOOR_SPRITE = os.path.join("data", "sprites", "dummy_floor_sand_32x20.png")
PATH_TO_WALL_SPRITE = os.path.join("data", "sprites", "dummy_wall_terracotta_32x12.png")
PATH_TO_ROOF_SPRITE = os.path.join("data", "sprites", "dummy_roof_blue_32x20.png")


class InteractiveEvolution:

    sprite_dimensions = {
        PATH_TO_WALL_SPRITE: (32, 12),
        PATH_TO_ROOF_SPRITE: (32, 20),
    }

    def _make_config(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "genome_configurations", "interactive_config")
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            InteractiveStagnation,
            config_path,
        )

    def display_buttons(self):
        """Make a window the user can interact with."""
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
            sprite_dimensions=InteractiveEvolution.sprite_dimensions,
            button_inner_boarder=np.array([10, 10])
        )
        display.draw_buttons(maximum_frames=None)

    def eval_fitness(self, genomes, config):
        pass


if __name__ == "__main__":
    interactive_evolution = InteractiveEvolution()
    interactive_evolution.display_buttons()
