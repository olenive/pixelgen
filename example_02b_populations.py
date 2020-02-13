import os
import numpy as np
import neat
from typing import Any, Iterable, Tuple, List, Dict

from core.tiles import TilePrototype, TilePrototypeMaker
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
tile_types_to_populations_configs = {
    tile: (neat.Population(tile_types_to_configs[tile]), config) for tile, config in tile_types_to_configs.items()
}


"""Make tile prototypes for each genome in each population."""
tile_prototype_maker = TilePrototypeMaker(
    tiles_types_to_populations_configs=tile_types_to_populations_configs,
)

tile_prototypes: Dict[str, Dict[int, Dict[Iterable[int], TilePrototype]]] = tile_prototype_maker.prototype_populations()
