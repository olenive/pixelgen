import numpy as np
from numpy.testing import assert_array_equal
from typing import Iterable, Tuple


class AssertSame:
    """Functions commonly used to assert that two data structures are the same."""

    def ids_positions_priorities(
        first: Iterable[Tuple[str, np.array, Tuple[int, int]]],
        second: Iterable[Tuple[str, np.array, Tuple[int, int]]],
    ) -> None:
        assert len(first) == len(second)
        for i in range(len(first)):
            assert len(first[i]) == 3
            assert len(second[i]) == 3
            assert first[i][0] == second[i][0]
            assert_array_equal(first[i][1], second[i][1])
            assert first[i][2] == second[i][2]
