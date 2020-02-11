import pygame
import time


imgPath = "data/evaluation_grid_01.png"

img = pygame.image.load(imgPath)
sz = img.get_size()
ret = pygame.Surface(sz)

tStart = time.time()
for x in range(sz[0]):
    for y in range(sz[1]):
        clr = img.get_at((x, y))
        clr2 = (clr[0]*.5, clr[1]*.5, clr[2]*.5)
        ret.set_at((x, y), clr2)

tTotal = time.time() - tStart
print("tTotal image    ", tTotal)


imgSA = pygame.surfarray.array3d(img)
ret = pygame.Surface(sz)
retSA = pygame.surfarray.array3d(ret)

tStart = time.time()
for x in range(sz[0]):
    for y in range(sz[1]):
        clr = imgSA[x][y]
        clr2 = (clr[0]*.5, clr[1]*.5, clr[2]*.5)
        retSA[x][y] = clr2

tTotal = time.time() - tStart
print("tTotal surfarray", tTotal)
