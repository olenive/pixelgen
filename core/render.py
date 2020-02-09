import os
import numpy as np
from typing import Iterable, Dict, Tuple, Callable
import pygame

from helpers.image import ImageIO


PATH_TO_FLOOR_SPRITE = os.path.join("data", "sprites", "dummy_floor_sand_32x20.png")
PATH_TO_WALL_SPRITE = os.path.join("data", "sprites", "dummy_wall_terracotta_32x12.png")
PATH_TO_ROOF_SPRITE = os.path.join("data", "sprites", "dummy_roof_blue_32x20.png")


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


class Render:

    def sprites(
        *,
        screen: pygame.surface.Surface,
        images: Dict[str, pygame.surface.Surface],
        sprites_info: Iterable[Tuple[str, np.array, Tuple[int, int]]],
    ) -> None:
        """Determine order that sprites should be drawn in and blit them onto the screen.

        Note: ordering is the last step before drawing so that sprites combined from different
        sources or generated be different processes can be ordered correctly relative to eachother.
        """
        ordered_sprites_info = PrepareForRendering.order_by_priority(sprites_info)
        for (image_id, position, _) in ordered_sprites_info:
            screen.blit(images[image_id], position)


class SpriteMaker:
    """Used to obtain default images or generate images using a neural network from a NEAT genome & config."""

    def __init__(self):
        self.default_sprite_image_paths = {
            "floor sprite": PATH_TO_FLOOR_SPRITE,
            "wall sprite": PATH_TO_WALL_SPRITE,
            "roof sprite": PATH_TO_ROOF_SPRITE,
        }
        # Load images (this needs to happen after a pygame display is initialised)
        self.default_images = {key: ImageIO.load_png(path) for key, path in self.default_sprite_image_paths.items()}

    def get_image(self, *, sprite_type: str, generating_function: Callable = None) -> pygame.surface.Surface:
        if generating_function is None:
            return self.default_images[sprite_type]


class ExampleDisplay:
    """Create a window and display images.

    NOTE: Unfortunately the PNG images need to be loaded after the window is initialised otherwise a
    "cannot convert without pygame.display initialized." error is raised by pygame.
    """

    def _collect_images_for_this_grid(self):
        """Wrapper to avoid repetition between call in init and subsequent calls

        Note: the call in init may not be needed anyway?
        """
        return PrepareForRendering.order_by_priority(
            PrepareForRendering.collect_images_for_grid(
                grid=self.tile_grid,
                cell_dimensions=self.cell_dimensions,
                top_left_position_of_grid=self.top_left_position_of_grid,
                sprite_dimensions=self.sprite_dimensions,
            )
        )

    def __init__(
        self,
        *,
        tile_grid: np.ndarray,
        cell_dimensions: np.ndarray,
        top_left_position_of_grid: np.ndarray,
        sprite_dimensions: Dict[str, Tuple[int, int]],  # Sprite id -> sprite width and height
    ) -> None:
        self.tile_grid = np.copy(tile_grid)
        self.cell_dimensions = np.copy(cell_dimensions)
        self.top_left_position_of_grid = np.copy(top_left_position_of_grid)
        self.sprite_dimensions = sprite_dimensions

        self.window_scale = 40
        self.window_width = 32 * self.window_scale
        self.window_height = 20 * self.window_scale

        self.image_paths = (
            PATH_TO_FLOOR_SPRITE,
            PATH_TO_WALL_SPRITE,
            PATH_TO_ROOF_SPRITE,
        )

        # Initialise pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Interactive NEAT-python pixel image generation.")

        # Load images (this needs to happen after a pygame display is initialised)
        sprite_maker = SpriteMaker()
        sprite_types = ("floor sprite", "wall sprite", "roof sprite")
        self.images: Dict[str, pygame.Surface] = \
            {sprite_type: sprite_maker.get_image(sprite_type=sprite_type) for sprite_type in sprite_types}

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
                self.ids_positions_priorities = self._collect_images_for_this_grid()
                Render.sprites(
                    screen=self.screen,
                    images=self.images,
                    sprites_info=self.ids_positions_priorities,
                )
                pygame.display.flip()


class ToggleableButton:
    """Rectangular button containing an image that can be toggled on or off.

    Rather than holding the image surface directly this object hold instructions for which image to draw in the form of
    an iterable of tuples of image ids, image positions and draw priorites.
    """

    def __init__(
        self,
        *,
        top_left: Tuple[int, int],
        dimensions: Tuple[int, int],
        image_info: Iterable[Tuple[str, np.array, Tuple[int, int]]],
        initial_state: bool = False
    ):
        self.top_left = top_left
        self.dimensions = dimensions
        self.state = initial_state
        self.image_info = image_info
        self.rect = pygame.Rect(top_left, dimensions)


class MultiTilesetDisplay:
    """Show more than one tile set in a single window."""

    def __init__(
        self,
        *,
        tile_grid: np.ndarray,
        button_grid_size: np.ndarray,
        cell_dimensions: np.ndarray,
        button_dimensions: np.ndarray,
        top_left_position_of_grid: np.ndarray,
        sprite_dimensions: Dict[str, Tuple[int, int]],  # Sprite id -> sprite width and height
        button_inner_boarder: np.ndarray,  # Used to create space between button boarder and the image in the button.
    ) -> None:
        self.tile_grid = np.copy(tile_grid)
        self.button_grid_size = np.copy(button_grid_size)
        self.cell_dimensions = np.copy(cell_dimensions)
        self.button_dimensions = np.copy(button_dimensions)
        self.top_left_position_of_grid = np.copy(top_left_position_of_grid)
        self.sprite_dimensions = sprite_dimensions
        self.button_inner_boarder = np.copy(button_inner_boarder)

        self.window_scale = 40
        self.window_width = 32 * self.window_scale
        self.window_height = 20 * self.window_scale

        self.image_paths = (
            PATH_TO_FLOOR_SPRITE,
            PATH_TO_WALL_SPRITE,
            PATH_TO_ROOF_SPRITE,
        )

        # Initialise pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Interactive NEAT-python pixel image generation.")

        # Load images (this needs to happen after a pygame display is initialised)
        sprite_maker = SpriteMaker()
        sprite_types = ("floor sprite", "wall sprite", "roof sprite")
        self.images: Dict[str, pygame.Surface] = \
            {sprite_type: sprite_maker.get_image(sprite_type=sprite_type) for sprite_type in sprite_types}

    def _make_buttons(
        self, button_grid_size: Tuple[int, int], button_dimensions: Tuple[int, int]
    ) -> Iterable[ToggleableButton]:
        """Create button objects that can the be used to draw buttons on the screen and detect clicks."""
        buttons = []
        for irow in range(button_grid_size[0]):
            for icol in range(button_grid_size[1]):
                top_left_position_of_button = MapGridToScreen.top_left_of_cell(
                    grid_cell=(irow, icol),
                    cell_dimensions=self.button_dimensions,
                    top_left_position_of_grid=self.top_left_position_of_grid,
                )
                button_images_info: Iterable[str, np.ndarray, Tuple[int, int]] = \
                    PrepareForRendering.collect_images_for_grid(
                        grid=self.tile_grid,
                        cell_dimensions=self.cell_dimensions,
                        top_left_position_of_grid=top_left_position_of_button + self.button_inner_boarder, 
                        sprite_dimensions=self.sprite_dimensions,
                    )
                buttons.append(
                    ToggleableButton(
                        top_left=top_left_position_of_button,
                        dimensions=self.button_dimensions,
                        image_info=button_images_info,
                    )
                )
        return buttons

    def _collect_button_image_info(
        self, buttons: Iterable[ToggleableButton]
    ) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Gather image info from multiple buttons into a single tuple of image ids, positions and priorities."""
        images_info = []
        for button in buttons:
            images_info += button.image_info
        return tuple(images_info)

    def _draw_button_boarders(self, screen, buttons):
        for button in buttons:
            if button.state:
                pygame.draw.rect(screen, [190, 190, 190, 200], button.rect, width=2)

    def draw_buttons(self, maximum_frames=None):
        """Draw a grid of buttons containing multiple tile sets."""
        buttons = self._make_buttons(
            button_grid_size=self.button_grid_size,
            button_dimensions=self.button_dimensions,
        )

        running = True
        frame_counter = 0
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
                    for button in buttons:
                        if button.rect.collidepoint(pygame.mouse.get_pos()):
                            button.state = not button.state

            if running:  # This if statement prevents a segfault from occuring when closing the pygame window.
                self.screen.fill((0, 0, 0))
                button_images_info = self._collect_button_image_info(buttons)
                Render.sprites(
                    screen=self.screen,
                    images=self.images,
                    sprites_info=button_images_info,
                )
                self._draw_button_boarders(self.screen, buttons)
                pygame.display.flip()
