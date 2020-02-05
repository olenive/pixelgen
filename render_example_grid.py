import numpy as np

from helpers.render import InteractiveDisplay


def main():
    interactive = InteractiveDisplay(grid_source_path="tests/data/black_white_and_blank_5x5.png")
    interactive.run()


if __name__ == "__main__":
    main()
