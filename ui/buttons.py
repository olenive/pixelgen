from typing import Tuple, Iterable, Dict
import pygame
import numpy as np
from dataclasses import dataclass

from core.render import Renderable, MapGridToScreen, PrepareForRendering
from core.tiles import TilePrototype


class ToggleableIllustratedButton:
    """A button that can be toggled and also displays an image that can be generated from a tile grid.

    Rather than holding the image surface directly this object hold instructions for which image to draw in the form of
    an iterable of tuples of Renderable objects.

    Created to display sprites to the user while they pick which ones they like.
    """
    def __init__(
        self,
        *,
        button_id: int,
        top_left: Tuple[int, int],
        dimensions: Tuple[int, int],
        prototypes: Dict[str, TilePrototype],
        renderables: Iterable[Renderable],
        tile_types_to_genome_ids: Dict[str, int],
        initial_state: bool = False,
    ):
        self.button_id = button_id
        self.top_left = top_left
        self.dimensions = dimensions
        self.state = initial_state
        self.prototypes = prototypes
        self.renderables = renderables
        self.rect = pygame.Rect(top_left, dimensions)
        self.tile_types_to_genome_ids = tile_types_to_genome_ids


class ToggleableIllustratedButtonArray:
    """Show more than one tile set in a single window."""

    def __init__(
        self,
        *,
        tile_grid: np.ndarray,
        rows_columns: Tuple[int, int],
        cell_dimensions: Tuple[int, int],
        button_dimensions: Tuple[int, int],
        top_left_position_of_grid: Tuple[int, int],
        sprite_dimensions: Dict[str, Tuple[int, int]],  # Sprite id -> sprite width and height
        button_inner_boarder: Tuple[int, int],  # Used to create space between the image in the button boarder.
        tiles_genomes_prototypes: Dict[str, Dict[int, TilePrototype]],
    ) -> None:
        self.tile_grid = tile_grid
        self.rows_columns = rows_columns
        self.cell_dimensions = cell_dimensions
        self.button_dimensions = button_dimensions
        self.top_left_position_of_grid = top_left_position_of_grid
        self.sprite_dimensions = sprite_dimensions
        self.button_inner_boarder = button_inner_boarder
        self.tiles_genomes_prototypes = tiles_genomes_prototypes
        self.buttons = self._make_buttons()

    def _gather_prototypes_by_genome_id(
        tiles_genomes_prototypes: Dict[str, Dict[int, TilePrototype]], index: int,
    ) -> Dict[str, TilePrototype]:
        out = {}
        for tile, ids_prototypes in tiles_genomes_prototypes.items():
            # Aribitrarily associate genomes with tile types based on their ordered incices.
            genome_id = sorted(ids_prototypes.keys())[index]
            out[tile] = ids_prototypes[genome_id]
        return out

    def _make_buttons(self) -> Iterable[ToggleableIllustratedButton]:
        """Create button objects that can then be used to draw buttons on the screen and detect clicks."""
        buttons = []
        button_index = 0
        for irow in range(self.rows_columns[0]):
            for icol in range(self.rows_columns[1]):
                top_left_position_of_button = MapGridToScreen.top_left_of_cell(
                    grid_cell=(irow, icol),
                    cell_dimensions=self.button_dimensions,
                    top_left_position_of_grid=self.top_left_position_of_grid,
                )
                # TODO: Here we are assuming that there is an incidental one to one mapping between buttons and genomes.
                # This may not be the case in the future and an explicit mapping may be needed.
                prototypes: Dict[str, TilePrototype] = ToggleableIllustratedButtonArray._gather_prototypes_by_genome_id(
                    self.tiles_genomes_prototypes, button_index
                )
                button_renderables: Iterable[Renderable] = PrepareForRendering.collect_renderables_for_grid(
                    grid=self.tile_grid,
                    tile_prototypes=prototypes,
                    top_left_position_of_grid=(
                        tuple(np.array(top_left_position_of_button) + np.array(self.button_inner_boarder))
                    ),
                    cell_dimensions=self.cell_dimensions,
                    wall_dimensions=self.sprite_dimensions["wall"],
                    roof_dimensions=self.sprite_dimensions["roof"],
                )
                buttons.append(
                    ToggleableIllustratedButton(
                        button_id=button_index,
                        top_left=top_left_position_of_button,
                        dimensions=self.button_dimensions,
                        prototypes=prototypes,
                        renderables=button_renderables,
                        tile_types_to_genome_ids={tile: prototype.genome_id for tile, prototype in prototypes.items()}
                    )
                )
                button_index += 1
        return tuple(buttons)

    def collect_renderables(self) -> Iterable[Tuple[str, np.array, Tuple[int, int]]]:
        """Gather Renderable objects from many buttons into a single tuple."""
        renderables = []
        for button in self.buttons:
            renderables += button.renderables
        return tuple(renderables)

    def draw_button_boarders(self, screen):
        for button in self.buttons:
            if button.state:
                pygame.draw.rect(screen, [240, 240, 240, 200], button.rect, width=4)
            else:
                pygame.draw.rect(screen, [90, 90, 90, 200], button.rect, width=1)


class TextButton:
    """A button that can have some text displayed over it."""

    def __init__(
        self,
        top_left: Tuple[int, int],
        dimensions: Tuple[int, int],
        text: str,
        top_left_of_text=(5, 5),
        text_colour=(10, 10, 10),
    ):
        self.top_left = top_left
        self.dimensions = dimensions
        self.text = text
        self.top_left_of_text = top_left_of_text
        self.text_colour = text_colour
        self.rect = pygame.Rect(top_left, dimensions)

    # def delay_font_use(self, font_name: str, size: int):
    #     """This will give an error if called before pygame.init() or pygame.font.init()"""
    #     return pygame.font.SysFont(font_name, size)

    @staticmethod
    def _relative_offset(outside: Tuple[int, int], inside: Tuple[int, int]) -> Tuple[int, int]:
        return (outside[0] + inside[0], outside[1] + inside[1])

    def draw_button(self, screen):
        pygame.draw.rect(screen, [190, 190, 190, 200], self.rect, width=0)
        myfont = pygame.font.SysFont('Comic Sans MS', 25)
        textsurface = myfont.render(self.text, False, self.text_colour)
        screen.blit(textsurface, self._relative_offset(self.top_left, self.top_left_of_text))
