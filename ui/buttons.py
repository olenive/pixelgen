from typing import Tuple, Iterable
import pygame

from core.render import Renderable


class ToggleableIllustratedButton:
    """A button that can be toggled and also displays an image that can be generated from a tile grid.

    Rather than holding the image surface directly this object hold instructions for which image to draw in the form of
    an iterable of tuples of Renderable objects.

    Created to display sprites to the user while they pick which ones they like.
    """

    def __init__(
        self,
        *,
        top_left: Tuple[int, int],
        dimensions: Tuple[int, int],
        renderables: Iterable[Renderable],
        initial_state: bool = False
    ):
        self.top_left = top_left
        self.dimensions = dimensions
        self.state = initial_state
        self.renderables = renderables
        self.rect = pygame.Rect(top_left, dimensions)

    def _draw_button_boarders(self, screen, buttons):
        for button in buttons:
            if button.state:
                pygame.draw.rect(screen, [190, 190, 190, 200], button.rect, width=2)


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
