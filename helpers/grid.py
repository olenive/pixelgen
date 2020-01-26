import numpy as np


class GridMap:

    def top_down_to_oblique_2_over_3(grid: np.ndarray) -> np.ndarray:
        """Generate a grid of sprites from a map of passable or impassable terrain.

        The aim is to generate sprites showing a top-down oblique projection of the terrain.
        A flat square surface of dimension 1 by 1 in x and y respectively is represented by a rectangle of dimensions
        1 by 2/3.
        Vertical walls are represented by a rectangele of dimensions 1 x 1/3.
        Thus, an n by m input top-down map grid will be represented by a 2n + 1 by m grid indicating sprite types.
        See grid_test.py for examples.
        """
        map_rows, map_columns = np.shape(grid)
        out = np.full([2 * map_rows + 1, map_columns], np.nan)
        for map_row in range(map_rows):
            out_row = (map_row + 1) * 2
            for map_column in range(map_columns):
                if grid[map_row, map_column] == 1:
                    out[out_row, map_column] = 1
                    out[out_row - 1, map_column] = 2
                    out[out_row - 2, map_column] = 2
                elif grid[map_row, map_column] == 0:
                    out[out_row, map_column] = 0
                    out[out_row - 1, map_column] = 0
                else:
                    raise ValueError(
                        f"Unexpected grid value {grid[map_row, map_column] = } for {map_row =} and {map_column = }."
                    )
        return out
