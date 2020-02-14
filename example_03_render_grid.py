import os
import time
import numpy as np
from typing import Dict
import neat
import pygame

from core.tiles import TilePrototype, TilePrototypeMaker
from core.render import PrepareForRendering, Render


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

tile_prototypes: Dict[str, Dict[int, TilePrototype]] = tile_prototype_maker.prototype_populations()


"""Given a grid of tiles represented by a numpy array of ones and zeros, draw an image of the level."""
grid = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
])


"""Extract prototype objects used to render this grid"""


def gather_prototypes_by_genome_id(
    tiles_genomes_prototypes: Dict[str, Dict[int, TilePrototype]], genome_id: int,
) -> Dict[str, TilePrototype]:
    return {tile: ids_prototypes[genome_id] for tile, ids_prototypes in tiles_genomes_prototypes.items()}


prototypes: Dict[str, TilePrototype] = gather_prototypes_by_genome_id(tile_prototypes, 1)


"""Initialise a pygame window and draw sprites."""
screen = pygame.display.set_mode((600, 900))
screen.fill((0, 0, 0))
renderables = PrepareForRendering.collect_renderables_for_grid(
    grid=grid,
    tile_prototypes=prototypes,
    top_left_position_of_grid=(15, 30),
    cell_dimensions=(32, 20),
    wall_dimensions=(32, 12),
    roof_dimensions=(32, 20),
)
Render.on_screen(screen, renderables)
pygame.display.flip()
time.sleep(30)
