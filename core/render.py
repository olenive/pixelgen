import numpy as np
from typing import Iterable, Dict, Tuple

import pygame


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
                "data/sprites/dummy_floor_sand.png",
                top_left,
                (0, top_left[1] + cell_dims[1]),
            ),
        )

    def ids_positions_priorities_for_tile(
        *,
        tile_type: int,
        cell_dimensions: np.ndarray,
        top_left_of_tile: np.ndarray,
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Determine images, positions and rending priorities for a tile based on its type and location."""
        if tile_type == 0:
            return PrepareForRendering.ids_positions_priorities_for_floor_tile(
                cell_dimensions=cell_dimensions,
                top_left_of_tile=top_left_of_tile,
            )
        elif tile_type == 1:
            raise NotImplementedError(f"TODO: implement wall tiles")  # TODO:
        else:
            raise ValueError(f"Unexpected tile type: {tile_type =}")

    def collect_images_for_grid(
        *,
        grid: np.ndarray,
        cell_dimensions: np.ndarray,
        top_left_position_of_grid: np.ndarray,
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


class InteractiveDisplay:
    """Create a window and display images.

    NOTE: Unfortunately the PNG images need to be loaded after the window is initialised otherwise a
    "cannot convert without pygame.display initialized." error is raised by pygame.
    """

    def load_png(self, path: str) -> pygame.Surface:
        """Load PNG data from given file path and carry out conversions required by pygame."""
        surface = pygame.image.load(path)
        converted = surface.convert()
        with_alphas = converted.convert_alpha()
        return with_alphas

    def images_from_paths(self, paths: Iterable[str]) -> Dict[str, pygame.Surface]:
        """For now we are assuming all images come from PNG files."""
        return {path: self.load_png(path) for path in paths}

    def __init__(
        self,
        *,
        tile_grid: np.ndarray,
        cell_dimensions: np.ndarray,
        top_left_position_of_grid: np.ndarray,
    ) -> None:
        self.tile_grid = np.copy(tile_grid)
        self.cell_dimensions = np.copy(cell_dimensions)
        self.top_left_position_of_grid = np.copy(top_left_position_of_grid)
        self.sprite_info = PrepareForRendering.collect_images_for_grid(
            grid=self.tile_grid,
            cell_dimensions=self.cell_dimensions,
            top_left_position_of_grid=self.top_left_position_of_grid,
        )
        self.window_scale = 20
        self.window_width = 32 * self.window_scale
        self.window_height = 20 * self.window_scale

        self.image_paths = (
            "data/sprites/dummy_floor_blue.png",
            "data/sprites/dummy_floor_sand.png",
            "data/sprites/dummy_roof_blue.png",
            "data/sprites/dummy_wall_terracotta.png"
        )

        # Initialise pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Interactive NEAT-python pixel image generation.")

        # Load images (this needs to happen after a pygame display is initialised)
        self.images: Dict[str, pygame.Surface] = self.images_from_paths(self.image_paths)

    def _draw_sprites(
        self,
        sprites_info: Iterable[Tuple[str, np.array, Tuple[int, int]]],
    ) -> None:
        for (image_id, position, _) in sprites_info:
            self.screen.blit(self.images[image_id], position)

    def run(self, maximum_frames=None):
        running = True
        frame_counter = 0
        while running:
            if maximum_frames is not None:
                frame_counter += 1
                if frame_counter >= maximum_frames:
                    running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    running = False
                    break

            if running:  # This if statement prevents a segfault from occuring when closing the pygame window.
                self.screen.fill((0, 0, 0))
                self.ids_positions_priorities = PrepareForRendering.collect_images_for_grid(
                    grid=self.tile_grid,
                    cell_dimensions=self.cell_dimensions,
                    top_left_position_of_grid=self.top_left_position_of_grid,
                )
                self._draw_sprites(
                    PrepareForRendering.order_by_priority(
                        self.ids_positions_priorities
                    )
                )
                pygame.display.flip()
