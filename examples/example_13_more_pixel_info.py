"""Try an image generating function that produces per-pixel outputs from the NN."""

import os
import numpy as np
from functools import reduce
import pygame
import neat
from typing import Dict, Tuple, Iterable, Any, List
from pathlib import Path

from core.tiles import TilePrototypeMaker, TilePrototype
from core.render import Render
from ui.buttons import ToggleableIllustratedButtonArray, TextButton
from core.neat_interfaces import NeatInterfaces
from helpers.conversions import Convert
from helpers.timestamps import Timestamps
from helpers.io import Pickler
from core.image import MakeSurface


PATH_TO_CONFIG_FILE_DIRECTORY = os.path.join("genome_configurations", "example_13_configs")
EXPORT_DIRECTORY = os.path.join("generated_tile_sets", "pngs_from_example_13", "")


sprite_palettes = {
    "floor": tuple(map(Convert.hex_to_rgb, ("393224", "74695B", "869894", "818B8D"))),
    "wall": tuple(map(Convert.hex_to_rgb, ("D7A847", "B1A77C", "986D54", "372313"))),
    "roof": tuple(map(Convert.hex_to_rgb, ("9199A7", "3760C5", "6C70B2", "232028"))),
}


def rgb_from_nn(
    neural_network: Any,
    nn_input: Iterable[int],
    sprite_dimensions: Tuple[int, int],
    palette: Iterable[Tuple[int, int, int, int]],
    tile_type=None,
    brightness=255,
    alpha_default=255,
) -> Tuple[np.ndarray, np.ndarray]:
    """Directly obtain RGB values from neural network without using the colour palette supplied."""

    def _normalise(value, maximum):
        return (value + 1) / maximum

    def _near_edge(index, maximum_length, tile_type) -> List[int]:
        if tile_type == "floor":
            return [0, 0, 0, 0]

        if index == 0:
            return [1, 1, 0, 0]
        elif index == 1:
            return [1, 0, 0, 0]
        elif index == maximum_length - 1:
            return [0, 0, 1, 1]
        elif index == maximum_length - 2:
            return [0, 0, 0, 1]
        else:
            return [0, 0, 0, 0]

    def _index_of_nearest_on_palette(palette: Iterable[Tuple[int, int, int, int]], nn_3_floats: List[float]) -> int:
        rgb_from_nn = np.array(nn_3_floats) * 255
        rgb_distances = [np.linalg.norm(np.array(rgba[0: 3]) - rgb_from_nn) for rgba in palette]
        return np.argmin(rgb_distances)

    def _select_rgb(palette: Iterable[Tuple[int, int, int, int]], index: int):
        return palette[index][0: 3]

    def _average_rgbs(rgb_1, rgb_2):
        return tuple((np.round([np.mean(x) for x in zip(rgb_1, rgb_2)])).astype(int))

    def _nudge_rgbs(origin: np.ndarray, destination: np.ndarray, nudge_factor=0.2) -> np.ndarray:
        direction_vector = destination - origin
        distance = np.linalg.norm(direction_vector)
        unit_vector = direction_vector / distance
        nudge_magnitude = distance
        return origin + unit_vector * nudge_magnitude * nudge_factor

    alphas = np.full(sprite_dimensions, alpha_default)
    rgb_out = np.full((*sprite_dimensions, 3), np.nan)
    for irow in range(sprite_dimensions[0]):
        for icol in range(sprite_dimensions[1]):
            x = _normalise(irow, sprite_dimensions[0])
            y = _normalise(icol, sprite_dimensions[1])
            mx = 1 - x
            my = 1 - y
            near_x_edge = _near_edge(irow, sprite_dimensions[0], tile_type)
            near_y_edge = _near_edge(icol, sprite_dimensions[1], tile_type)
            nn_out_3d: List = neural_network.activate(
                tuple(list(nn_input) + [x, y, mx, my] + near_x_edge + near_y_edge)
            )
            index = _index_of_nearest_on_palette(palette, nn_out_3d)
            rgb_from_palette = _select_rgb(palette, index)
            resulting_rgb = _nudge_rgbs(np.array(rgb_from_palette), np.array(nn_out_3d) * 255)
            rgb_out[irow, icol, :] = resulting_rgb

    return np.round(rgb_out).astype(int), alphas


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
    for _, (population, _) in dict_with_populations.items():
        NeatInterfaces.advance_to_next_generation(population)


def _export_selection(toggleable_buttons: ToggleableIllustratedButtonArray):
    print("Exporting selected tile images to PNG files and pickling tile prototype objects.")

    def _concatenate_ints(given: Iterable[int]) -> str:
        return reduce((lambda x, y: str(x) + "_" + str(y)), given)

    export_directory: str = os.path.join(EXPORT_DIRECTORY, Timestamps.iso_now_seconds())
    print(export_directory + "\n")
    Path(export_directory).mkdir(parents=True, exist_ok=True)
    for button in toggleable_buttons.buttons:
        if button.state:
            for tile_type in button.prototypes.keys():
                prototype: TilePrototype = button.prototypes[tile_type]
                # NOTE: it may be easier to work with the prototype objects than to parse .png file names.
                Pickler.dump(prototype, os.path.join(export_directory, "prototype.pickle"))
                for inputs_key in prototype.inputs_to_rgbs_and_alphas.keys():
                    # Save PNGs.
                    # The tuple of integers called inputs_key is used as a description of the relevant tiles
                    # surrounding the given tile.
                    file_name: str = tile_type + '_' + _concatenate_ints(inputs_key) + '.png'
                    file_path: str = os.path.join(export_directory, file_name)
                    rgb_array, alphas_array = prototype.inputs_to_rgbs_and_alphas[inputs_key]
                    sprite = MakeSurface.from_rgb_and_alpha_arrays(rgb_array, alphas_array)
                    pygame.image.save(sprite, file_path)

            # TODO: Save genomes.


def main():

    # Initialize pygame fonts so that we can print text on buttons.
    pygame.font.init()

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

    # Create array of buttons containing tiles
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    ])
    button_width = np.shape(grid)[1] * 32 + 10
    button_height = np.shape(grid)[0] * 20 + 30
    toggleable_buttons = ToggleableIllustratedButtonArray(
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

    # Create export PNG button.
    export_pngs_button = TextButton(
        top_left=(button_width + 40, 20),
        dimensions=(185, 30),
        text="EXPORT SELECTED",
        top_left_of_text=(8, 7)
    )

    screen = pygame.display.set_mode((button_width + 30 + 260, button_height * 9 + 50))
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

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Toggle buttons in response to click.
                for button in toggleable_buttons.buttons:
                    if button.rect.collidepoint(pygame.mouse.get_pos()):
                        button.state = not button.state

                # Export PNG files containing selected sprites.
                if export_pngs_button.rect.collidepoint(pygame.mouse.get_pos()):
                    _export_selection(toggleable_buttons)

            # Advance generations.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    generation_counter += 1
                    print(f"Generation: {generation_counter}")
                    # Update genome fitnesses and generate new genomes within each population (mutate populations).
                    _set_genome_fitnesses(tile_types_to_populations_configs, toggleable_buttons)
                    _advance_populations(tile_types_to_populations_configs)
                    tiles_genomes_prototypes = prototype_tiles_from_genomes(tile_types_to_populations_configs)
                    # Create a new array of buttons using the new populations of genonmes.
                    toggleable_buttons = ToggleableIllustratedButtonArray(
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

            # Draw toggleable button contents
            renderables = toggleable_buttons.collect_renderables()
            Render.on_screen(screen, renderables)

            # Draw toggleable button boarders.
            toggleable_buttons.draw_button_boarders(screen)

            # Draw other butotns.
            export_pngs_button.draw_button(screen)

            pygame.display.flip()


if __name__ == "__main__":
    main()
