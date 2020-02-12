"""Example code used to make a sprite from a neural network defined by a NEAT genome and config."""

import os
from typing import Any
import numpy as np
import neat
import pygame
from pygame import surfarray
from typing import List

from helpers.image import ImageIO, ImageConvert


SPRITE_DIMENSIONS = (32, 12)


"""Create a neural network from a genome & config."""
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    os.path.join("genome_configurations", "example-sprite-making-config")
)


def example_genome(key: Any, genome_config):
    genome = config.genome_type(key)
    genome.configure_new(genome_config)
    return genome


genome = example_genome(0, config.genome_config)
net = neat.nn.FeedForwardNetwork.create(genome, config)


"""Feed in input and obtain output from neural network."""
net_output: List = net.activate((1, 1))


"""Make a sprite from the NN's output."""
reshaped_net_output = np.reshape(np.array(net_output), SPRITE_DIMENSIONS)
palette = ((200, 0, 0, 100), (100, 0, 100, 100), (0, 0, 200, 200), (0, 200, 0, 100))
rgb_array, alphas = ImageConvert.matrix_to_rgb_palette_and_alphas(reshaped_net_output, palette)


"""Initialise a pygame window and draw the sprite."""
screen = pygame.display.set_mode((500, 800))
screen.fill((0, 0, 0))

# TODO: Figure out alphas and surface locking.
draw_surface_1 = pygame.surface.Surface(SPRITE_DIMENSIONS)
pygame.surfarray.blit_array(draw_surface_1, rgb_array)
screen.blit(draw_surface_1, (10, 30))

draw_surface_2 = pygame.surface.Surface(SPRITE_DIMENSIONS)
pygame.surfarray.blit_array(draw_surface_2, rgb_array)


class MutateSurface:
    """Collection of impure functions operating on Surface objects."""

    def set_alphas(surface: pygame.surface.Surface, alphas: np.ndarray) -> None:
        """Access the alpha values of a Surface and set them to the values in the supplied array.

        Keeping the reference to the surface contained inside the functions scope results in the surface being unlocked
        so that it an be subseuently used with the blit method.  Otherwise the surface is locked while the reference
        array exists.
        """
        surface_alphas = pygame.surfarray.pixels_alpha(surface)
        surface_alphas[:] = alphas[:]


MutateSurface.set_alphas(draw_surface_2, alphas)


screen.blit(draw_surface_2, (50, 30))


pygame.display.flip()
