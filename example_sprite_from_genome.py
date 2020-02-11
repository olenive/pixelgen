"""Example code used to make a sprite from a neural network defined by a NEAT genome and config."""

import os
from typing import Any
import numpy as np
import neat
import pygame
from pygame import surfarray


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
net_output = net.activate((1, 1))


"""Make a sprite from the NN's output."""


"""Initialise a pygame window and draw the sprite."""


def draw_example(array: np.ndarray, maximum_frames=None) -> None:
    screen = pygame.display.set_mode((200, 120))
    running = True
    frame_counter = 0
    while running:
        if maximum_frames is not None:
            frame_counter += 1
            if frame_counter >= maximum_frames:
                running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break

        if running:  # This if statement prevents a segfault from occuring when closing the pygame window.
            screen.fill((0, 0, 0))
            surfarray.blit_array(screen, array)
            pygame.display.flip()