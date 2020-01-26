import numpy as np
from numpy.testing import assert_array_equal

from helpers.grid import GridMap


class TestGridMap:

    def test_top_down_to_oblique_2_over_3_no_walls(self):
        given = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        expected = np.array([
            [np.nan, np.nan, np.nan],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = GridMap.top_down_to_oblique_2_over_3(given)
        assert_array_equal(result, expected)

    def test_top_down_to_oblique_2_over_3_one_wall_in_middle(self):
        given = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        expected = np.array([
            [np.nan, np.nan, np.nan],
            [0, 0, 0],
            [0, 2, 0],
            [0, 2, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = GridMap.top_down_to_oblique_2_over_3(given)
        assert_array_equal(result, expected)

    def test_top_down_to_oblique_2_over_3_walls_in_middle_column_top_two(self):
        given = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        expected = np.array([
            [np.nan, 2, np.nan],
            [0, 2, 0],
            [0, 2, 0],
            [0, 2, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = GridMap.top_down_to_oblique_2_over_3(given)
        assert_array_equal(result, expected)    

    def test_top_down_to_oblique_2_over_3_walls_in_middle_column(self):
        given = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ])
        expected = np.array([
            [np.nan, 2, np.nan],
            [0, 2, 0],
            [0, 2, 0],
            [0, 2, 0],
            [0, 2, 0],
            [0, 2, 0],
            [0, 1, 0],
        ])
        result = GridMap.top_down_to_oblique_2_over_3(given)
        assert_array_equal(result, expected)

    def test_top_down_to_oblique_2_over_3_walls_in_middle_row(self):
        given = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ])
        expected = np.array([
            [np.nan, np.nan, np.nan],
            [0, 0, 0],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = GridMap.top_down_to_oblique_2_over_3(given)
        assert_array_equal(result, expected)
