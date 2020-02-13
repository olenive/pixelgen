import os
import numpy as np
from collections import namedtuple
from typing import Iterable, Dict, Tuple, Callable, NamedTuple, Any
import pygame
import neat

from core.image import ImageIO


class MapGridToScreen:
    """Functions for determining where an object should be drawn on screen given its tile grid coordinates."""

    def bottom_left_of_cell(
        *,
        grid_cell: np.ndarray,
        cell_dimensions: np.ndarray,
        top_left_position_of_grid: np.ndarray,
    ) -> np.ndarray:
        """Get the pixel position of the bottom left corner of the given cell in the grid."""
        bottom_left = np.copy(top_left_position_of_grid)
        bottom_left[0] += cell_dimensions[0] * (grid_cell[1])
        bottom_left[1] += cell_dimensions[1] * (1 + grid_cell[0])
        return bottom_left

    def top_left_of_cell(
        *,
        grid_cell: np.ndarray,
        cell_dimensions: np.ndarray,
        top_left_position_of_grid: np.ndarray,
    ) -> np.ndarray:
        """Get the pixel position of the top left corner of the given cell in the grid."""
        top_left = np.copy(top_left_position_of_grid)
        top_left[0] += cell_dimensions[0] * (grid_cell[1])
        top_left[1] += cell_dimensions[1] * (grid_cell[0])
        return top_left


class Renderable(NamedTuple):
    """Data describing an object to be rendered on screen."""
    get_image: Callable
    position: Tuple[int, int]
    priority: Tuple[int, int]


class Render:

    def generate_sprite_from_nn(
        input: Iterable[int],
        neural_network: Any,  # Needs to have an activate method.
    ) -> pygame.surface.Surface:
        nn_out = neural_network.activate(input)
        pass


    def on_screen(
        *,
        screen: pygame.surface.Surface,
        renderables: Iterable[Renderable],
    ) -> None:
        """Determine order that sprites should be drawn in and blit them onto the screen.

        Note: ordering is the last step before drawing so that sprites combined from different
        sources or generated be different processes can be ordered correctly relative to eachother.
        """
        ordered_sprites_info = PrepareForRendering.order_by_priority(sprites_info)
        for (image_id, position, _) in ordered_sprites_info:
            screen.blit(images[image_id], position)


class PrepareForRendering:
    """Functions for determining the order and position in which images should apear on screen.

    There are a couple of complications here.
    1) The images need to be rendered in the right order. Images at the back and below need to be rendered first and
       those in front and on top, last.

    2) PNG images apparently need to be converted before pygame can render them correctly. However, this conversion
       seems to require a pygame.display object to be initialized. Thus image loading needs to be delayed?

    Proposed solution:
    Create an data structure that describes what images should be rendered where and in what order. The type of this
    data structure can be Iterable[Tuple[str, np.ndarray, Tuple[int, int]]] where the string is an image id, the
    numpy array contains the position of the image on the screen and the inner most tuple of two integers represents
    rendering priority in the vertical (z) and horizontal (y) planes respectively.  Could use an iterable of named
    tuples to help keep track of what type of data goes where.

    This data structure will then be passed to a rendering function called subsequently after the pygame.display has
    been initialised and the required images from PNG files loaded and converted.
    """

    def ids_positions_priorities_for_floor_tile(
        *,
        cell_dimensions: np.ndarray,
        top_left_of_tile: np.ndarray,
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Make an iterable of tuples describing how to render a floor tile."""
        cell_dims = np.copy(cell_dimensions)
        top_left = np.copy(top_left_of_tile)
        return (
            (
                "floor sprite",
                top_left,
                (0, top_left[1] + cell_dims[1]),
            ),
        )

    def ids_positions_priorities_for_wall_tile(
        *,
        cell_dimensions: np.ndarray,
        top_left_of_tile: np.ndarray,
        sprite_dimensions: Dict[str, Tuple[int, int]],  # Sprite id -> sprite width and height
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Make an iterable of tuples describing how to render a block with a wall and a roof tile.

        Note: higher values of second dimensions correspond to being drawn lower on the screen.
        """
        # Determine sprite sizes
        cell_dims = np.copy(cell_dimensions)
        wall_dimensions = sprite_dimensions["wall sprite"]
        roof_dimensions = sprite_dimensions["roof sprite"]
        # Determine where the top left corner of the wall sprite is to be drawn on the screen.
        wall_top_left = np.copy(top_left_of_tile)
        wall_top_left[1] += cell_dims[1] - wall_dimensions[1]
        # Determine where the top left corner of the roof sprites is to be drawn on the screen.
        roof_top_left = np.copy(wall_top_left)
        roof_top_left[1] -= roof_dimensions[1]
        # Determine where the bottom left corners of each sprite are so as to know drawing priority.
        wall_bottom_left = np.copy(top_left_of_tile)
        wall_bottom_left[1] += cell_dims[1]
        roof_bottom_left = np.copy(wall_top_left)
        return (
            (
                "wall sprite",
                wall_top_left,
                (1, wall_bottom_left[1]),
            ),
            (
                "roof sprite",
                roof_top_left,
                (1, roof_bottom_left[1]),
            ),
        )

    def ids_positions_priorities_for_tile(
        *,
        tile_type: int,
        cell_dimensions: np.ndarray,
        top_left_of_tile: np.ndarray,
        sprite_dimensions: Dict[str, Tuple[int, int]],  # Sprite id -> sprite width and height
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Determine images, positions and rending priorities for a tile based on its type and location."""
        if tile_type == 0:
            return PrepareForRendering.ids_positions_priorities_for_floor_tile(
                cell_dimensions=cell_dimensions,
                top_left_of_tile=top_left_of_tile,
            )
        elif tile_type == 1:
            return PrepareForRendering.ids_positions_priorities_for_wall_tile(
                cell_dimensions=cell_dimensions,
                top_left_of_tile=top_left_of_tile,
                sprite_dimensions=sprite_dimensions,
            )
        else:
            raise ValueError(f"Unexpected tile type: {tile_type =}")

    def collect_images_for_grid(
        *,
        grid: np.ndarray,
        cell_dimensions: np.ndarray,
        top_left_position_of_grid: np.ndarray,
        sprite_dimensions: Dict[str, Tuple[int, int]],  # Sprite id -> sprite width and height
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Determine what images should be drawn to represent the tiles on a grid.

        Takes a 2D grid containing a single tile type at each position and returns the required image ids, locations
        and draw priorities.
        """
        out = []
        rows, columns = np.shape(grid)
        cell_dimensions_copy = np.copy(cell_dimensions)
        top_left_of_grid_copy = np.copy(top_left_position_of_grid)
        for irow in range(rows):
            for icol in range(columns):
                top_left_of_cell = MapGridToScreen.top_left_of_cell(
                    grid_cell=np.array([irow, icol]),
                    cell_dimensions=cell_dimensions_copy,
                    top_left_position_of_grid=top_left_of_grid_copy,
                )
                tile: Iterable[Tuple[str, np.array, Tuple[int, int]]] = \
                    PrepareForRendering.ids_positions_priorities_for_tile(
                        tile_type=grid[irow, icol],
                        cell_dimensions=cell_dimensions_copy,
                        top_left_of_tile=top_left_of_cell,
                        sprite_dimensions=sprite_dimensions,
                    )
                out += tile
        return tuple(out)

    def order_by_priority(
        image_info: Iterable[Tuple[str, np.array, Tuple[int, int]]]
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Determine what order images should be drawn in.

        Takes an iterable of image ids, positions and priorities.
        Returns the same type of data structure but ordered according to priorities.
        """
        return sorted(image_info, key=lambda x: x[2][0] * 100000000000 + x[2][1])

