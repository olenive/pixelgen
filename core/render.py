import numpy as np
from itertools import chain
from typing import NamedTuple
from typing import Iterable, Dict, Tuple, Callable
import pygame

from core.image import MakeSurface
from core.tiles import TilePrototype


class MapGridToScreen:
    """Functions for determining where an object should be drawn on screen given its tile grid coordinates."""

    def bottom_left_of_cell(
        *,
        grid_cell: Tuple[int, int],
        cell_dimensions: Tuple[int, int],
        top_left_position_of_grid: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Get the pixel position of the bottom left corner of the given cell in the grid."""
        return (
            top_left_position_of_grid[0] + cell_dimensions[0] * (grid_cell[1]),
            top_left_position_of_grid[1] + cell_dimensions[1] * (1 + grid_cell[0]),
        )

    def top_left_of_cell(
        *,
        grid_cell: Tuple[int, int],
        cell_dimensions: Tuple[int, int],
        top_left_position_of_grid: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Get the pixel position of the top left corner of the given cell in the grid."""
        return (
            top_left_position_of_grid[0] + cell_dimensions[0] * (grid_cell[1]),
            top_left_position_of_grid[1] + cell_dimensions[1] * (grid_cell[0]),
        )


class Renderable(NamedTuple):
    """Data describing an object to be rendered on screen."""
    array_getter: Callable
    position: Tuple[int, int]
    priority: Tuple[int, int]


class Render:

    def order_by_priority(
        image_info: Iterable[Tuple[str, np.array, Tuple[int, int]]]
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Determine what order images should be drawn in.

        Takes an iterable of image ids, positions and priorities.
        Returns the same type of data structure but ordered according to priorities.
        """
        return sorted(image_info, key=lambda x: x[2][0] * 100000000000 + x[2][1])

    def on_screen(
        screen: pygame.surface.Surface,
        renderables: Iterable[Renderable],
    ) -> None:
        """Determine order that sprites should be drawn in and blit them onto the screen.

        Note: ordering is the last step before drawing so that sprites combined from different
        sources or generated by different processes can be ordered correctly relative to eachother.
        """
        ordered_sprites_info = Render.order_by_priority(renderables)
        for (array_getter, position, _) in ordered_sprites_info:
            rgb_array, alphas_array = array_getter()
            sprite = MakeSurface.from_rgb_and_alpha_arrays(rgb_array, alphas_array)
            screen.blit(sprite, position)


class PrepareForRendering:
    """Functions for determining the order and position in which images should apear on screen.

    There are a couple of complications here.
    1) The images need to be rendered in the right order. Images at the back and below need to be rendered first and
       those in front and on top, last.

    2) PNG images apparently need to be converted before pygame can render them correctly. However, this conversion
       seems to require a pygame.display object to be initialized. Thus image loading needs to be delayed?
    """

    def floor_tile_renderables(
        *,
        array_getter: Callable,
        top_left_of_tile: Tuple[int, int],
        dimensions: Tuple[int, int]
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Make an iterable of tuples describing how to render a floor tile."""
        return (
            Renderable(array_getter, top_left_of_tile, (0, top_left_of_tile[1] + dimensions[1])),
        )

    def wall_and_roof_tile_renderables(
        *,
        wall_array_getter: Callable,
        roof_array_getter: Callable,
        top_left_of_tile: Tuple[int, int],
        cell_dimensions: Tuple[int, int],
        wall_dimensions: Tuple[int, int],
        roof_dimensions: Tuple[int, int],
    ) -> Iterable[Renderable]:
        """Make an iterable of tuples describing how to render a block with a wall and a roof tile.

        Note: higher values of second dimensions correspond to being drawn lower on the screen.
        """
        # Determine where the top left corner of the wall sprite is to be drawn on the screen.
        wall_top_left = list(top_left_of_tile)
        wall_top_left[1] += cell_dimensions[1] - wall_dimensions[1]
        # Determine where the top left corner of the roof sprites is to be drawn on the screen.
        roof_top_left = list(wall_top_left)
        roof_top_left[1] -= roof_dimensions[1]
        # Determine where the bottom left corners of each sprite are so as to know drawing priority.
        wall_bottom_left = list(top_left_of_tile)
        wall_bottom_left[1] += cell_dimensions[1]
        roof_bottom_left = wall_top_left
        return (
            Renderable(wall_array_getter, tuple(wall_top_left), (1, wall_bottom_left[1])),
            Renderable(roof_array_getter, tuple(roof_top_left), (1, roof_bottom_left[1])),
        )

    def renderables_for_cell_tiles(
        *,
        tile_prototypes: Dict[str, TilePrototype],
        grid_context_3x3: np.ndarray,  # 3x3 matrix of cells centered on current cell
        top_left_of_tile: Tuple[int, int],
        cell_dimensions: Tuple[int, int],
        wall_dimensions: Tuple[int, int],
        roof_dimensions: Tuple[int, int],
    ) -> Iterable[Renderable]:
        """Make Rendrable for a cell based on its contents and context."""
        this_cell_in_grid = grid_context_3x3[1, 1]
        if this_cell_in_grid == 0:
            floor_prototype = tile_prototypes["floor"]
            floor_inputs = (grid_context_3x3[1, 0], grid_context_3x3[0, 1], grid_context_3x3[1, 2],)
            return PrepareForRendering.floor_tile_renderables(
                array_getter=(lambda: floor_prototype.inputs_to_rgbs_and_alphas[floor_inputs]),
                top_left_of_tile=top_left_of_tile,
                dimensions=cell_dimensions,
            )
        elif this_cell_in_grid == 1:
            wall_prototype = tile_prototypes["wall"]
            roof_prototype = tile_prototypes["roof"]
            wall_inputs = (grid_context_3x3[1, 0], grid_context_3x3[1, 2],)
            roof_inputs = (grid_context_3x3[1, 0], grid_context_3x3[2, 1], grid_context_3x3[1, 2],)
            return PrepareForRendering.wall_and_roof_tile_renderables(
                wall_array_getter=(lambda: wall_prototype.inputs_to_rgbs_and_alphas[wall_inputs]),
                roof_array_getter=(lambda: roof_prototype.inputs_to_rgbs_and_alphas[roof_inputs]),
                top_left_of_tile=top_left_of_tile,
                cell_dimensions=cell_dimensions,
                wall_dimensions=wall_dimensions,
                roof_dimensions=roof_dimensions,
            )
        else:
            raise ValueError(f"Unexpected value in middle cell: {this_cell_in_grid =}")

    def collect_renderables_for_grid(
        *,
        grid: np.ndarray,
        tile_prototypes: Dict[str, TilePrototype],
        top_left_position_of_grid: Tuple[int, int],
        cell_dimensions: Tuple[int, int],
        wall_dimensions: Tuple[int, int],
        roof_dimensions: Tuple[int, int],
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Determine what images should be drawn to represent the tiles on a grid.

        Takes a 2D grid of integer contianing cells and use it to generate Renderable objects.
        """
        def _create_grid_context_3x3(grid, i, j) -> np.ndarray:
            """This may need optimising."""

            def _at(i, j) -> int:
                """Get contentes of grid at (i, j) or return 0 if (i, j) is outside the grid."""
                try:
                    return grid[i, j]
                except(IndexError):
                    return 0

            return np.array([
                [_at(i - 1, j - 1), _at(i - 1, j), _at(i - 1, j + 1)],
                [_at(i,     j - 1), _at(i,     j), _at(i,     j + 1)],
                [_at(i + 1, j - 1), _at(i + 1, j), _at(i + 1, j + 1)],
            ])

        out = []
        rows, columns = np.shape(grid)
        for irow in range(rows):
            for icol in range(columns):
                top_left_of_cell = MapGridToScreen.top_left_of_cell(
                    grid_cell=(irow, icol),
                    cell_dimensions=cell_dimensions,
                    top_left_position_of_grid=top_left_position_of_grid,
                )
                cell_renderables: Iterable[Renderable] = PrepareForRendering.renderables_for_cell_tiles(
                    tile_prototypes=tile_prototypes,
                    grid_context_3x3=_create_grid_context_3x3(grid, irow, icol),
                    top_left_of_tile=top_left_of_cell,
                    cell_dimensions=cell_dimensions,
                    wall_dimensions=wall_dimensions,
                    roof_dimensions=roof_dimensions,
                )
                out += cell_renderables
        return tuple(out)
