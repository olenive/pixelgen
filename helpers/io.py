import pickle
from typing import Any


class Pickler:

    def dump(data: Any, file_path: str) -> None:
        with open(file_path, "wb") as output_file:
            pickle.dump(data, output_file)
