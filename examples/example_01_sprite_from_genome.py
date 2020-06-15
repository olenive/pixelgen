"""Example code used to make a sprite from a neural network defined by a NEAT genome and config."""

import os
import time
from typing import Any
import numpy as np
import neat
import pygame
from typing import List

from core.image import ImageConvert, MakeSurface


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
sprite = MakeSurface.from_rgb_and_alpha_arrays(rgb_array, alphas)
screen.blit(sprite, (50, 30))
pygame.display.flip()

time.sleep(30)
