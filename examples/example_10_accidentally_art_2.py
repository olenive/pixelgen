"""Try an image generating function that produces per-pixel outputs from the NN."""

import os
import numpy as np
import pygame
import neat
from typing import Dict, Tuple, Iterable, Any, List

from core.tiles import TilePrototypeMaker, TilePrototype
from core.render import Render
from ui.buttons import ToggleableIllustratedButtonArray
from core.neat_interfaces import NeatInterfaces
from core.image import ImageConvert


PATH_TO_CONFIG_FILE_DIRECTORY = os.path.join("genome_configurations", "example_10_configs")

sprite_palettes = {
    "floor": (
        (117, 65, 29, 255),
        (156, 130, 70, 255),
        (145, 131, 97, 255),
        (102, 99, 93, 255),
    ),
    "wall": (
        (252, 186, 3, 255),
        (200, 169, 3, 255),
        (152, 89, 3, 255),
        (171, 77, 14, 255),
        (71, 77, 14, 255),
    ),
    "roof": (
        (10, 20, 50, 255),
        (20, 10, 100, 255),
        (50, 10, 200, 255),
        (0, 0, 0, 255),
    ),
}


def rgb_from_nn(
    neural_network: Any,
    nn_input: Iterable[int],
    sprite_dimensions: Tuple[int, int],
    palette: Iterable[Tuple[int, int, int, int]],
    tile_type=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Directly obtain RGB values from neural network without using the colour palette supplied."""

    def _normalise(value, maximum):
        return (value + 1) / maximum

    alphas = np.full(sprite_dimensions, 255)
    nn_3d_output = np.full((*sprite_dimensions, 3), np.nan)
    for irow in range(sprite_dimensions[0]):
        for icol in range(sprite_dimensions[1]):
            x = _normalise(irow, sprite_dimensions[0])
            y = _normalise(icol, sprite_dimensions[1])
            nn_3d_output[irow, icol, :] = neural_network.activate(tuple(list(nn_input) + [x, y]))
    return np.round(nn_3d_output * 255), alphas


def config_for_this_example(path_to_config_file: str) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        path_to_config_file,
    )


def prototype_tiles_from_genomes(
    tile_types_to_populations_configs: Dict[str, Tuple[neat.Population, neat.Config]]
) -> Dict[str, Dict[int, TilePrototype]]:
    """Make tile prototypes for each genome in each population.

    Returns a dictionary of tile types to dictionaries of genome ids to TilePrototype objects.
    """
    tile_prototype_maker = TilePrototypeMaker(
        tiles_types_to_populations_configs=tile_types_to_populations_configs,
        sprite_palettes=sprite_palettes,
        image_generating_function=rgb_from_nn,
    )
    return tile_prototype_maker.prototype_populations()


def _set_genome_fitnesses(
    dict_with_populations: Dict[str, Tuple[neat.Population, neat.Config]],
    array_of_buttons: ToggleableIllustratedButtonArray,
) -> None:
    """Would be nice to have a clearer more direct way of mapping buttons to genome ids."""
    for button in array_of_buttons.buttons:
        fitness = int(button.state)
        # Add a check to make sure sets of keys between dict in button and dict_with_populations match?
        for tile_type, (neat_population, _) in dict_with_populations.items():
            genome_id = button.tile_types_to_genome_ids[tile_type]
            neat_population.population[genome_id].fitness = fitness


def _advance_populations(dict_with_populations: Dict[str, Tuple[neat.Population, neat.Config]]) -> None:
    """Can this be less ugly?"""
    for _, (population, _) in dict_with_populations.items():
        NeatInterfaces.advance_to_next_generation(population)


def main():

    # Determine NEAT configurations for each tile type.
    tile_types_to_configs = {
        "floor": config_for_this_example(os.path.join(PATH_TO_CONFIG_FILE_DIRECTORY, "floor")),
        "wall": config_for_this_example(os.path.join(PATH_TO_CONFIG_FILE_DIRECTORY, "wall")),
        "roof": config_for_this_example(os.path.join(PATH_TO_CONFIG_FILE_DIRECTORY, "roof")),
    }

    # Initialise populations of genomes for each tile type.
    tile_types_to_populations_configs: Dict[str, Tuple[neat.Population, neat.Config]] = {
        tile: (neat.Population(tile_types_to_configs[tile]), config) for tile, config in tile_types_to_configs.items()
    }

    tiles_genomes_prototypes = prototype_tiles_from_genomes(tile_types_to_populations_configs)

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
    generation_counter = 1
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

            # Advance generations.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    generation_counter += 1
                    print(f"Generation: {generation_counter}")
                    # Update genome fitnesses and generate new genomes within each population (mutate populations).
                    _set_genome_fitnesses(tile_types_to_populations_configs, buttons_array)
                    _advance_populations(tile_types_to_populations_configs)
                    tiles_genomes_prototypes = prototype_tiles_from_genomes(tile_types_to_populations_configs)
                    # Create a new array of buttons using the new populations of genonmes.
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

        if running:  # This if statement prevents a segfault from occuring when closing the pygame window.
            screen.fill((50, 50, 50))

            # Draw button contents
            renderables = buttons_array.collect_renderables()
            Render.on_screen(screen, renderables)

            # Draw button boarders.
            buttons_array.draw_button_boarders(screen)

            pygame.display.flip()


if __name__ == "__main__":
    main()
