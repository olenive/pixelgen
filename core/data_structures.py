# from collections import namedtuple
import numpy as np
from typing import Tuple, Any, Iterable, Dict, NamedTuple
import neat


class TilePrototype(NamedTuple):
    """Data cache describing a type of object that occupies a single cell of the grid (map/world).

    A tile may have multiple sprites associated with it. For example, a floor tile may have different sprites depending
    on the contents of adjacent grid cells.

    The tile prototype stores image arrays so that they don't need to be computed for every tile instance (or every
    frame).
    """
    tile_type: str
    dimensions: Tuple[int, int]
    genome_id: neat.genome.DefaultGenome  # Can we avoid typing this as DefaultGenome?
    config: neat.genome.DefaultGenomeConfig  # Not sure if this should be config.Config or not.
    neural_network: Any
    inputs_to_rgb_and_alpha: Dict[Iterable[int], Tuple[np.ndarray, np.ndarray]]


# class TilePrototypeMaker:
#     """Given populations of neural networks, generates sprites and returns Tile objects containing Rendrable objects.

#     The main reson for this being an object rather than a function is so that sprites produced using a given set of
#     inputs can be reused rather than being generated again for every frame.

#     The sprite for a given tile depends on the neural network used to generate it and on the inputs to that network.

#     We want to co-evolve several populations of neural networks so that each population can correspond to a particular
#     tile (e.g. floor, wall or roof). Thus, we use different instances of Tile objects for each type of tile. Further,
#     each tile instance of the same type (e.g. floor) can link to sprites generated using different neural networks from
#     a sepecified population.

#     NN input corresponds to 0 or 1 for passable or impassable respectively in the following orders:
#     Wall: [West, East]
#     Floor: [West, North, East]
#     Roof: [West, South, East]
#     """

#     def __init__(
#         self,
#         *,
#         populations_and_configs: Dict[str, Tuple[neat.population.Population, neat.genome.DefaultGenomeConfig]],
#         tile_dimensions={
#             "floor": (32, 20),
#             "wall": (32, 12),
#             "roof": (32, 20),
#         },
#         nn_inputs={
#             "wall": (
#                 (0, 0),
#                 (1, 0),
#                 (0, 1),
#                 (1, 1),
#             ),
#             "floor": (
#                 (0, 0, 0),
#                 (1, 0, 0),
#                 (0, 1, 0),
#                 (0, 0, 1),
#                 (0, 1, 1),
#                 (1, 0, 1),
#                 (1, 1, 0),
#                 (1, 1, 1),
#             ),
#             "roof": (
#                 (0, 0, 0),
#                 (1, 0, 0),
#                 (0, 1, 0),
#                 (0, 0, 1),
#                 (0, 1, 1),
#                 (1, 0, 1),
#                 (1, 1, 0),
#                 (1, 1, 1),
#             ),
#         },
#         paths_to_default_pngs={
#             "floor": os.path.join("data", "sprites", "dummy_floor_sand_32x20.png"),
#             "wall": os.path.join("data", "sprites", "dummy_wall_terracotta_32x12.png"),
#             "roof": os.path.join("data", "sprites", "dummy_roof_blue_32x20.png"),
#         },
#     ) -> None:
#         self.populations_and_configs = populations_and_configs
#         self.floor_tile_dimensions = tile_dimensions
#         self.paths_to_default_pngs = paths_to_default_pngs
#         self.nn_inputs = nn_inputs

#     def default(tile_type: str, )


#     def default_wall(self) -> Tile:
#         mapping_inputs_to_sprite_getter = {i: lambda: self.default_wall_sprite for i in self.wall_inputs}
#         return Tile(
#             tile_type="wall",
#             inputs_to_sprites=mapping_inputs_to_sprite_getter,
#             dimensions=self.wall_tile_dimensions,
#             genome_id=None,
#             config=None,
#             neural_network=None,
#         )

#     def default_floor(self) -> Tile:
#         mapping_inputs_to_sprite_getter = {i: lambda: self.default_floor_sprite for i in self.floor_inputs}
#         return Tile(
#             tile_type="floor",
#             inputs_to_sprites=mapping_inputs_to_sprite_getter,
#             dimensions=self.floor_tile_dimensions,
#             genome_id=None,
#             config=None,
#             neural_network=None,
#         )

#     def default_roof(self) -> Tile:
#         mapping_inputs_to_sprite_getter = {i: lambda: self.default_roof_sprite for i in self.roof_inputs}
#         return Tile(
#             tile_type="roof",
#             inputs_to_sprites=mapping_inputs_to_sprite_getter,
#             dimensions=self.roof_tile_dimensions,
#             genome_id=None,
#             config=None,
#             neural_network=None,
#         )

#     def wall(self, genome_id: Any) -> Tile:
#         return Tile(
#             tile_type="wall",
#             inputs_to_sprites={i: lambda: self.wall_sprites[genome_id] for i in self.wall_inputs},
#             dimensions=self.wall_tile_dimensions,
#             genome_id=None,
#             config=None,
#             neural_network=None,
#         )

#     def floor(self) -> Tile:
#         mapping_inputs_to_sprite_getter = {i: lambda: self.default_floor_sprite for i in self.floor_inputs}
#         return Tile(
#             tile_type="floor",
#             inputs_to_sprites=mapping_inputs_to_sprite_getter,
#             dimensions=self.floor_tile_dimensions,
#             genome_id=None,
#             config=None,
#             neural_network=None,
#         )

#     def roof(self) -> Tile:
#         mapping_inputs_to_sprite_getter = {i: lambda: self.default_roof_sprite for i in self.roof_inputs}
#         return Tile(
#             tile_type="roof",
#             inputs_to_sprites=mapping_inputs_to_sprite_getter,
#             dimensions=self.roof_tile_dimensions,
#             genome_id=None,
#             config=None,
#             neural_network=None,
#         )

