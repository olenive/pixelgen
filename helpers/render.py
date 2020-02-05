import numpy as np
from typing import Iterable, Dict

import pygame


class MapGridToScreen:

    def bottom_left_of_cell(
        *,
        grid_cell: np.ndarray,
        cell_dimenstions: np.ndarray,
        top_left_position_of_grid=np.array([0, 0])
    ) -> np.ndarray:
        """Get the pixel position of the bottom left corner of the given cell in the grid."""
        bottom_left = top_left_position_of_grid
        bottom_left[0] += cell_dimenstions[0] * (grid_cell[0])
        bottom_left[1] += cell_dimenstions[1] * (1 + grid_cell[1])
        return bottom_left

    def top_left_of_cell(
        *,
        grid_cell: np.ndarray,
        cell_dimenstions: np.ndarray,
        top_left_position_of_grid=np.array([0, 0])
    ) -> np.ndarray:
        """Get the pixel position of the top left corner of the given cell in the grid."""
        top_left = top_left_position_of_grid
        top_left[0] += cell_dimenstions[0] * (grid_cell[0])
        top_left[1] += cell_dimenstions[1] * (grid_cell[1])
        return top_left


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

    def __init__(self, grid_source_path: str):
        self.grid_source_path = grid_source_path

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

        # Load images (need to happen after a pygame display is initialised)
        self.images: Dict[str, pygame.Surface] = self.images_from_paths(self.image_paths)

    def _draw_full_floor(self):
        """Method for figuring out how things work, delete once things work."""
        grid_rows = 5
        grid_columns = 7
        # grid = np.full((grid_rows, grid_columns), 0)
        for irow in range(grid_rows):
            for icol in range(grid_columns):
                self.screen.blit(
                    self.images["data/sprites/dummy_floor_sand.png"],
                    MapGridToScreen.top_left_of_cell(
                        grid_cell=np.array([irow, icol]),
                        cell_dimenstions=np.array([32, 20]),
                        top_left_position_of_grid=np.array([50, 50])
                    ),
                )

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    running = False
                    break

                # if event.type == pygame.MOUSEBUTTONDOWN:
                #     clicked_button = -1
                #     for n, button in enumerate(buttons):
                #         if rects[n].collidepoint(pygame.mouse.get_pos()):
                #             clicked_button = n
                #             break

                #     if event.button == 1:
                #         selected[clicked_button] = not selected[clicked_button]
                #     else:
                #         self.make_high_resolution(genomes[clicked_button], config)

            if running:
                self.screen.fill((0, 0, 0))
                self._draw_full_floor()
                # for n, button in enumerate(buttons):
                #     screen.blit(button, rects[n])
                #     if selected[n]:
                #         pygame.draw.rect(screen, (255, 0, 0), rects[n], 3)
                pygame.display.flip()
