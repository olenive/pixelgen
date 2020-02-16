import os
import numpy as np
import pygame
import neat
from typing import Dict

from core.tiles import TilePrototypeMaker, TilePrototype
from core.render import Render
from ui.buttons import ToggleableIllustratedButtonArray


def main():

    def config_for_this_example(path_to_config_file: str) -> neat.Config:
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            path_to_config_file,
        )

    tile_types_to_configs = {
        "floor": config_for_this_example(os.path.join("genome_configurations", "example_02_configs", "floor")),
        "wall": config_for_this_example(os.path.join("genome_configurations", "example_02_configs", "wall")),
        "roof": config_for_this_example(os.path.join("genome_configurations", "example_02_configs", "roof")),
    }

    tile_types_to_populations_configs = {
        tile: (neat.Population(tile_types_to_configs[tile]), config) for tile, config in tile_types_to_configs.items()
    }

    # Make tile prototypes for each genome in each population.
    tile_prototype_maker = TilePrototypeMaker(
        tiles_types_to_populations_configs=tile_types_to_populations_configs,
    )
    tiles_genomes_prototypes: Dict[str, Dict[int, TilePrototype]] = tile_prototype_maker.prototype_populations()

    # Render buttons and check that they are toggleable.
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    ])

    button_width = np.shape(grid)[1] * 32 + 10
    button_height = np.shape(grid)[0] * 20 + 30

    buttons_array = ToggleableIllustratedButtonArray(
        tile_grid=grid,
        rows_columns=(9, 1),
        cell_dimensions=(32, 20),
        button_dimensions=(button_width, button_height),
        top_left_position_of_grid=(15, 15),
        sprite_dimensions={  # Sprite id -> sprite width and height
            "floor": (32, 20),
            "wall": (32, 12),
            "roof": (32, 20),
        },
        button_inner_boarder=(5, 20),  # Used to create space between the image in the button boarder.
        tiles_genomes_prototypes=tiles_genomes_prototypes,
    )

    screen = pygame.display.set_mode((button_width + 30, button_height * 9 + 50))
    maximum_frames = None
    running = True
    frame_counter = 0
    while running:
        if maximum_frames is not None:
            frame_counter += 1
            if frame_counter >= maximum_frames:
                running = False

        for event in pygame.event.get():
            # Exit the pygame window without causing errors.
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break

            # Toggle buttons in response to click.
            if event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons_array.buttons:
                    if button.rect.collidepoint(pygame.mouse.get_pos()):
                        button.state = not button.state

        if running:  # This if statement prevents a segfault from occuring when closing the pygame window.
            screen.fill((0, 0, 0))

            # Draw button contents
            renderables = buttons_array.collect_renderables()
            Render.on_screen(screen, renderables)

            # Draw button boarders.
            buttons_array.draw_button_boarders(screen)

            pygame.display.flip()


if __name__ == "__main__":
    main()
