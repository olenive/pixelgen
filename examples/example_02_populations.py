import os
import numpy as np
import neat
from typing import Any, Iterable, Tuple, List

from core.tiles import TilePrototype
from core.image import ImageConvert


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


"""Instantiate a population of genomes for each tile type."""
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    os.path.join("genome_configurations", "example-sprite-making-config")
)
tile_types_to_populations = {
    tile_type: neat.Population(tile_types_to_configs[tile_type]) for tile_type in tile_types_to_configs.keys()
}


"""Make a tile prototype for a given tile type and genome id."""


def make_rgb_and_alpha(
    neural_network: Any,
    nn_input: Iterable[int],
    sprite_dimensions: Tuple[int, int],
    palette: Iterable[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    nn_output: List = neural_network.activate(nn_input)
    reshaped_output = np.reshape(np.array(nn_output), sprite_dimensions)
    return ImageConvert.matrix_to_rgb_palette_and_alphas(reshaped_output, palette)


tile_type = "wall"
genome_id = 3
sprite_dimensions = (32, 12)
palette = ((200, 0, 0, 100), (100, 0, 100, 100), (0, 0, 200, 200), (0, 200, 0, 100))
neural_network = neat.nn.FeedForwardNetwork.create(
    tile_types_to_populations[tile_type].population[genome_id],
    config,
)
inputs_set = (  # Set of possible inputs used to generate the image for this tile.
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
)

example_floor_prototype = TilePrototype(
    tile_type=tile_type,
    dimensions=sprite_dimensions,
    genome_id=genome_id,
    config=config,
    neural_network=neural_network,
    inputs_to_rgbs_and_alphas={
        nn_input: make_rgb_and_alpha(neural_network, nn_input, sprite_dimensions, palette) for nn_input in inputs_set
    },
)
