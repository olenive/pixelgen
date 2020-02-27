from typing import Tuple


class Convert:

    def hex_to_rgb(hex: str) -> Tuple[int, int, int]:
        sans_hash = hex.lstrip('#')
        return tuple(int(sans_hash[i: i + 2], 16) for i in (0, 2, 4))
