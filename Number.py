import pygame, sys, numpy

import SNN as cream

network = cream.snn([784, 20, 10], cream.sigmoid)

pygame.init()
window = pygame.display.set_mode((720,480),pygame.RESIZABLE)
pygame.display.set_caption("Hand Writing Number Detection")
clock = pygame.time.Clock()

width = 16
grid = pygame.Surface((width*28+1,width*28+1),pygame.SRCALPHA).convert_alpha()
canvas = pygame.Surface((width*28+1,width*28+1),pygame.SRCALPHA).convert_alpha()

for i in range(29):
    pygame.draw.line(grid, (0,0,0),(i*width,0), (i*width,width*28+1),1)
    pygame.draw.line(grid, (0,0,0),(0,i*width), (width*28+1,i*width),1)

gr = numpy.zeros((28,28))

class system:
    def event(events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if (pygame.mouse.get_pressed()[0] == 1):
            x, y = pygame.mouse.get_pos()
            try:
                px = int((x - int((480-(width*28+1))/2))/width)
                py = int((y - int((480-(width*28+1))/2))/width)

                gr[max(0,py-1):min(27,py+1)][max(0,px-1):min(27,px+1)] = 1
                pygame.draw.rect(canvas,(0,255,0),[(px-1)*width,(py-1)*width,width*3,width*3])
            except:
                pass
        if (pygame.mouse.get_pressed()[2] == 1):
            x, y = pygame.mouse.get_pos()
            try:
                px = int((x - int((480-(width*28+1))/2))/width)
                py = int((y - int((480-(width*28+1))/2))/width)

                gr[max(0,py-1):min(27,py+1)][max(0,px-1):min(27,px+1)] = 0
                pygame.draw.rect(canvas,(255,255,255),[(px-1)*width,(py-1)*width,width*3,width*3])
            except:
                pass

    def display():
        window.fill((255,255,255))
        window.blit(canvas,(int((480-(width*28+1))/2),int((480-(width*28+1))/2)))
        window.blit(grid,(int((480-(width*28+1))/2),int((480-(width*28+1))/2)))
        pygame.draw.line(window,(0,0,0),(int((480-(width*28+1))/2)*2+width*28+1, 0),(int((480-(width*28+1))/2)*2+width*28+1, 480))

        pygame.display.update()
        clock.tick(60)

while True:
    events = pygame.event.get()

    system.event(events)
    system.display()

