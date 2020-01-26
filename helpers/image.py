import numpy as np
from typing import Iterable, Tuple
from PIL import Image


class ImageIO:

    def rgba_png_to_array(path_to_png: str) -> np.ndarray:
        rgba_image = Image.open(path_to_png)
        return np.asarray(rgba_image, dtype="int32")


class ImageConverter:

    def flat_hashable_from_rgba(
        rgba_array: np.ndarray
    ) -> Tuple[Iterable[Tuple[int, int, int, int]], Tuple[int, int, int]]:
        rows, columns, colours = np.shape(rgba_array)
        rgba_column = np.reshape(rgba_array, (rows * columns, colours))
        rgba_tuples = tuple(map(lambda x: tuple(x), rgba_column))
        return rgba_tuples, np.shape(rgba_array)

    def grid_from_rgba(rgba_array: np.ndarray) -> np.ndarray:
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
