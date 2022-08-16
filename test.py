import cream_beta as cream
import SNN as cream
import tool.Csys as Csys
import tool.progress_bar as pb


import numpy, random, time
from threading import Thread

Csys.clear()
 



import pygame, sys

window = pygame.display.set_mode((1920,1080),pygame.FULLSCREEN)
pygame.mouse.set_visible(False)

while True:
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            pygame.quit()
            sys.exit()