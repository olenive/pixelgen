import pygame
import time
import numpy as np


pygame.init()
pygame.display.set_caption("Sandbox")
canvas = pygame.display.set_mode((800, 600))

path = "data/sprites/example.png"

image = pygame.image.load(path)
size = image.get_size()


print(f"{image.get_size() = }")

tStart = time.time()

for i in range(30):
    for j in range(10):
        canvas.blit(image, (10, 10))
    pygame.display.flip()

tTotal = time.time() - tStart
print("tTotal image    ", tTotal)


array3d = pygame.surfarray.array3d(image)

tStart = time.time()

black = np.ones((800, 600, 3)) * 100

for i in range(30):
    for j in range(10):
        # canvas.blit(image, (10, 10))
        # result = black + array3d
        pygame.surfarray.blit_array(canvas, black)
    pygame.display.flip()

tTotal = time.time() - tStart
print("tTotal surfarray", tTotal)


tStart = time.time()

black = np.ones((800, 600, 3)) * 200
black_image = pygame.surfarray.make_surface(black)

for i in range(30):
    for j in range(10):
        # canvas.blit(image, (10, 10))
        # result = black + array3d
        canvas.blit(black_image, (0, 0))
    pygame.display.flip()

tTotal = time.time() - tStart
print("tTotal surfarray", tTotal)


tStart = time.time()

black = np.ones((800, 600, 3)) * 200
black_image = pygame.surfarray.make_surface(black)

for i in range(30):
    for j in range(10):
        # canvas.blit(image, (10, 10))
        # result = black + array3d
        black_image = pygame.surfarray.make_surface(black)
        canvas.blit(black_image, (0, 0))
    pygame.display.flip()

tTotal = time.time() - tStart
print("tTotal surfarray", tTotal)