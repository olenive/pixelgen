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


class ImageConvert:

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
        """Map an array produced by reading an RGBA image to a 2D array of cell types.

        This is used to encode convert level maps stored as PNG files to a matrix of integers.  These can later be used
        to generate terrain.
        """
        mapping = {
            tuple([0, 0, 0, 0]): -1,  # nothing -> -1
            tuple([0, 0, 0, 255]): 0,  # black -> 0
            tuple([255, 255, 255, 255]): 1,  # white -> 1
        }
        flat, (rows, columns, colours) = ImageConvert.flat_hashable_from_rgba(rgba_array)
        out = np.full(rows * columns, np.nan)
        for i, v in enumerate(flat):
            out[i] = mapping[v]
        return np.reshape(out, (rows, columns))

    def continuous_value_to_discrete_palette(
        value: float,
        palette: Iterable[Tuple[int, int, int, int]],
        value_range=(0, 1),
    ) -> Tuple[int, int, int, int]:
        """Use a float to select an set of values from a palette of colours."""
        bin_size = (value_range[1] - value_range[0]) / len(palette)
        bins = tuple([i * bin_size for i in range(len(palette))])
        index = np.digitize(value, bins) - 1
        return palette[index]

    def matrix_to_rgb_palette_and_alphas(
        matrix: np.ndarray,
        palette: Iterable[Tuple[int, int, int, int]],
        value_range=(0, 1),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Produces a 3D array of RGB values selected from the palette using a 2D numpy array of values.

        Also returns a 2D array of alpha values.

        The alpha values are separate because pygame.surfarray works with them separately.
        """
        rgb_palette = tuple([v[0: 3] for v in palette])
        alpha_palette = tuple([v[3] for v in palette])
        rows, columns = np.shape(matrix)
        rgb_out = np.empty((rows, columns, 3))
        alpha_out = np.empty((rows, columns))
        for irow in range(rows):
            for icol in range(columns):
                rgb_out[irow, icol] = ImageConvert.continuous_value_to_discrete_palette(
                    matrix[irow, icol], rgb_palette)
                alpha_out[irow, icol] = ImageConvert.continuous_value_to_discrete_palette(
                    matrix[irow, icol], alpha_palette)
        return rgb_out, alpha_out


class MutateSurface:
    """Collection of impure functions operating on Surface objects."""

    def set_alphas(surface: pygame.surface.Surface, alphas: np.ndarray) -> None:
        """Access the alpha values of a Surface and set them to the values in the supplied array.

        Keeping the reference to the surface contained inside the functions scope results in the surface being unlocked
        so that it an be later used with the blit method.  Otherwise the surface is locked while the reference
        array exists.
        """
        surface_alphas = pygame.surfarray.pixels_alpha(surface)
        surface_alphas[:] = alphas[:]


class MakeSurface:
    """Wraper for functions that make a Surfaces from arrays.

    The surface needs to have the same dimensions as the array and applying alpha requires mutating the Surface object.
    """

    def from_rgb_and_alpha_arrays(rgb_array: np.ndarray, alpha_array) -> pygame.surface.Surface:
        """Make a surface with an image from a 3D array of RGB values and a 2D array of alpha values."""
        draw_surface = pygame.surface.Surface(np.shape(alpha_array))
        pygame.surfarray.blit_array(draw_surface, rgb_array)
        MutateSurface.set_alphas(draw_surface, alpha_array)
        return draw_surface
