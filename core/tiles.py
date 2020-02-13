# from collections import namedtuple
import os
import numpy as np
from typing import Tuple, Any, Iterable, Dict, NamedTuple, List
import neat

from core.image import ImageConvert


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
    inputs_to_rgbs_and_alphas: Dict[Iterable[int], Tuple[np.ndarray, np.ndarray]]


class TilePrototypeMaker:
    """Generates TilePrototype objects for every tile type, genome and input combination.

    The image for a given tile depends on the neural network used to generate it and on the inputs to that network.
    The NN's output is converted to array that can be used to generate images.  These arrays are stored attached to an
    instance of a TilePrototype so that they don't need to be generated every time.

    NN input corresponds to 0 or 1 for passable or impassable respectively in the following orders:
    Wall: [West, East]
    Floor: [West, North, East]
    Roof: [West, South, East]
    """

    def __init__(
        self,
        *,
        tiles_types_to_populations_configs: Dict[
            str, Tuple[neat.population.Population, neat.genome.DefaultGenomeConfig]
        ],
        sprite_dimensions={
            "floor": (32, 20),
            "wall": (32, 12),
            "roof": (32, 20),
        },
        sprite_palettes={
            "floor": ((117, 65, 29, 255), (156, 130, 70, 255), (145, 131, 97, 255), (102, 99, 93, 255),),
            "wall": ((252, 186, 3, 255), (252, 169, 3, 255), (252, 119, 3, 255), (171, 77, 14, 255),),
            "roof": ((10, 20, 50, 255), (20, 10, 100, 255), (50, 10, 200, 255),),
        },
        nn_inputs={
            "wall": (
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
            ),
            "floor": (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ),
            "roof": (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ),
        },
        paths_to_default_pngs={
            "floor": os.path.join("data", "sprites", "dummy_floor_sand_32x20.png"),
            "wall": os.path.join("data", "sprites", "dummy_wall_terracotta_32x12.png"),
            "roof": os.path.join("data", "sprites", "dummy_roof_blue_32x20.png"),
        },
    ) -> None:
        self.tiles_types_to_populations_configs = tiles_types_to_populations_configs
        self.sprite_dimensions = sprite_dimensions
        self.sprite_palettes = sprite_palettes
        self.nn_inputs = nn_inputs
        self.paths_to_default_pngs = paths_to_default_pngs
        # TODO: Validate given data.
        # Make sure set of sprite types (keys) matches for all dictionaries.

    def rgb_and_alpha(
        neural_network: Any,
        nn_input: Iterable[int],
        sprite_dimensions: Tuple[int, int],
        palette: Iterable[Tuple[int, int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        nn_output: List = neural_network.activate(nn_input)
        reshaped_output = np.reshape(np.array(nn_output), sprite_dimensions)
        return ImageConvert.matrix_to_rgb_palette_and_alphas(reshaped_output, palette)

    def prototype_populations(self) -> Dict[str, Dict[int, Dict[Iterable[int], TilePrototype]]]:
        """Make a dictionary of tile types to dictionaries of genome ids to TilePrototype instances."""
        def _inputs_to_arrays(neural_network, nn_inputs) -> Dict[Iterable[int], Tuple[np.ndarray, np.ndarray]]:
            return {
                nn_input: TilePrototypeMaker.rgb_and_alpha(
                        neural_network,
                        nn_input,
                        self.sprite_dimensions[tile_type],
                        self.sprite_palettes[tile_type],
                    )
                for nn_input in self.nn_inputs[tile_type]
            }

        tile_types_dict = {}
        for tile_type, (population, config) in self.tiles_types_to_populations_configs.items():
            genomes_dict = {}
            for genome_id, genome in population.population.items():
                neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
                inputs_dict = {}
                for nn_input in self.nn_inputs[tile_type]:
                    inputs_dict[nn_input] = TilePrototype(
                        tile_type=tile_type,
                        dimensions=self.sprite_dimensions[tile_type],
                        genome_id=genome_id,
                        config=config,
                        neural_network=neural_network,
                        inputs_to_rgbs_and_alphas=_inputs_to_arrays(neural_network, nn_input)
                    )
                genomes_dict[genome_id] = inputs_dict
            tile_types_dict[tile_type] = genomes_dict
        return tile_types_dict
