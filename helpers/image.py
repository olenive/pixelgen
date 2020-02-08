import numpy as np
import pygame
from typing import Iterable, Tuple, Dict
from PIL import Image


class ImageIO:

    def rgba_png_to_array(path_to_png: str) -> np.ndarray:
        rgba_image = Image.open(path_to_png)
        return np.asarray(rgba_image, dtype="int32")

    def load_png(path: str) -> pygame.Surface:
        """Load PNG data from given file path and carry out conversions required by pygame.

        Note: the conversions may only be possible after a pygame.display object is initialised.
        """
        surface = pygame.image.load(path)
        converted = surface.convert()
        with_alphas = converted.convert_alpha()
        return with_alphas

    def images_from_paths(paths: Iterable[str]) -> Dict[str, pygame.Surface]:
        """For now we are assuming all images come from PNG files."""
        return {path: ImageIO.load_png(path) for path in paths}


class ImageConverter:

    def flat_hashable_from_rgba(
        rgba_array: np.ndarray
    ) -> Tuple[Iterable[Tuple[int, int, int, int]], Tuple[int, int, int]]:
        """Flatten an array produced by reading an RGBA image into a vector of tuples of RGBA values.

        Also return the shape of the original array.
        """
        rows, columns, colours = np.shape(rgba_array)
        rgba_column = np.reshape(rgba_array, (rows * columns, colours))
        rgba_tuples = tuple(map(lambda x: tuple(x), rgba_column))
        return rgba_tuples, np.shape(rgba_array)

    def grid_from_rgba(rgba_array: np.ndarray) -> np.ndarray:
        """Map an array produced by reading an RGBA image to a 2D array of cell types."""
        mapping = {
            tuple([0, 0, 0, 0]): -1,  # nothing -> -1
            tuple([0, 0, 0, 255]): 0,  # black -> 0
            tuple([255, 255, 255, 255]): 1,  # white -> 1
        }
        flat, (rows, columns, colours) = ImageConverter.flat_hashable_from_rgba(rgba_array)
        out = np.full(rows * columns, np.nan)
        for i, v in enumerate(flat):
            out[i] = mapping[v]
        return np.reshape(out, (rows, columns))
