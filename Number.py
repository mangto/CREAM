import pygame, sys, numpy, pickle
from pygame import gfxdraw

import SNN as cream
import tool.Csys as Csys

network = cream.snn([784, 20, 10], cream.sigmoid)

pygame.init()
window = pygame.display.set_mode((720,480),pygame.RESIZABLE)
pygame.display.set_caption("Hand Writing Number Detection")
clock = pygame.time.Clock()

global gr
width = 16
grid = pygame.Surface((width*28+1,width*28+1),pygame.SRCALPHA).convert_alpha()
canvas = pygame.Surface((width*28+1,width*28+1),pygame.SRCALPHA).convert_alpha()

for i in range(29):
    pygame.draw.line(grid, (0,0,0),(i*width,0), (i*width,width*28+1),1)
    pygame.draw.line(grid, (0,0,0),(0,i*width), (width*28+1,i*width),1)

gr = numpy.zeros((28,28))
array = numpy.array

network.load_weight(pickle.load(open(".\\Number_data\\weights.dat","rb")))
network.load_bias(pickle.load(open(".\\Number_data\\biases.dat","rb")))

def font(fontname, size):
    return pygame.font.Font(f"C:\\Windows\\Fonts\\{fontname}.TTF",size)

class system:
    class draw:
        def aacircle(surface, x, y, radius, color):
            gfxdraw.aacircle(surface, x, y, radius, color)
            gfxdraw.filled_circle(surface, x, y, radius, color)
        def rrect(surface,rect,color,radius=0.4):
            rect         = pygame.Rect(rect)
            color        = pygame.Color(*color)
            alpha        = color.a
            color.a      = 0
            pos          = rect.topleft
            rect.topleft = 0,0
            rectangle    = pygame.Surface(rect.size,pygame.SRCALPHA)
            circle       = pygame.Surface([min(rect.size)*3]*2,pygame.SRCALPHA)
            pygame.draw.ellipse(circle,(0,0,0),circle.get_rect(),0)
            circle       = pygame.transform.smoothscale(circle,[int(min(rect.size)*radius)]*2)
            radius              = rectangle.blit(circle,(0,0))
            radius.bottomright  = rect.bottomright
            rectangle.blit(circle,radius)
            radius.topright     = rect.topright
            rectangle.blit(circle,radius)
            radius.bottomleft   = rect.bottomleft
            rectangle.blit(circle,radius)

            rectangle.fill((0,0,0),rect.inflate(-radius.w,0))
            rectangle.fill((0,0,0),rect.inflate(0,-radius.h))

            rectangle.fill(color,special_flags=pygame.BLEND_RGBA_MAX)
            rectangle.fill((255,255,255,alpha),special_flags=pygame.BLEND_RGBA_MIN)
            return surface.blit(rectangle,pos)
        def trirect(surface,x,y,sx,sy,tri,color,edge=(1,1,1,1)):
            if sx < tri*2:
                sx = tri*2
            if sy < tri*2:
                sy = tri*2

            pygame.draw.rect(surface,color,[x+tri,y,sx-tri*2,sy])
            pygame.draw.rect(surface,color,[x,y+tri,sx,sy-tri*2])
            if edge[0] == 1:
                pygame.draw.polygon(surface,color,[[x,y+tri],[x+tri,y],[x+tri,y+tri]])
            else:
                pygame.draw.rect(surface,color,[x,y,tri,tri])
            if edge[1] == 1:
                pygame.draw.polygon(surface,color,[[x+sx-tri,y+1],[x+sx-1,y+tri],[x+sx-tri,y+tri]])
            else:
                pygame.draw.rect(surface,color,[x+sx-tri,y,tri,tri])
            if edge[2] == 1:
                pygame.draw.polygon(surface,color,[[x,y+sy-tri],[x+tri,y+sy-1],[x+tri,y+sy-tri]])
            else:
                pygame.draw.rect(surface,color,[x,y+sy-tri,tri,tri])
            if edge[3] == 1:
                pygame.draw.polygon(surface,color,[[x+sx-1,y+sy-tri],[x+sx-tri,y+sy-1],[x+sx-tri,y+sy-tri]])
            else:
                pygame.draw.rect(surface,color,[x+sx-tri,y+sy-tri,tri,tri])
        def textsize(text, font):
            text_obj = font.render(text, True, (0,0,0))
            text_rect=text_obj.get_rect()
            return text_rect.size
        def text(text, font, window, x, y, cenleft="center", color=(0,0,0)):
            text_obj = font.render(text, True, color)
            text_rect=text_obj.get_rect()
            if(cenleft == "center"):
                text_rect.centerx = x
                text_rect.centery = y
            elif(cenleft == "left"):
                text_rect.left=x
                text_rect.top=y
            elif(cenleft == "right"):
                text_rect.right=x
                text_rect.top=y
            elif(cenleft == "cenleft"):
                text_rect.left=x
                text_rect.centery=y
            elif(cenleft == "cenright"):
                text_rect.right=x
                text_rect.centery=y
            window.blit(text_obj, text_rect)
        def gettsize(text,font):
            return font.render(text,True,(0,0,0)).get_rect().size


    def event(events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if (pygame.mouse.get_pressed()[0] == 1):
            x, y = pygame.mouse.get_pos()
            px = int((x - int((480-(width*28+1))/2))/width)
            py = int((y - int((480-(width*28+1))/2))/width)

            gr[max(0,py-1)][max(0,px-1):min(27,px+1)] = [1]*(min(27,px+1)-max(0,px-1))
            gr[py][max(0,px-1):min(27,px+1)] = [1]*(min(27,px+1)-max(0,px-1))
            gr[min(27,py+1)][max(0,px-1):min(27,px+1)] = [1]*(min(27,px+1)-max(0,px-1))
            pygame.draw.rect(canvas,(0,255,0),[(px-1)*width,(py-1)*width,width*3,width*3])
        if (pygame.mouse.get_pressed()[2] == 1):
            x, y = pygame.mouse.get_pos()
            px = int((x - int((480-(width*28+1))/2))/width)
            py = int((y - int((480-(width*28+1))/2))/width)

            gr[max(0,py-1)][max(0,px-1):min(27,px+1)] = [0]*(min(27,px+1)-max(0,px-1))
            gr[py][max(0,px-1):min(27,px+1)] = [0]*(min(27,px+1)-max(0,px-1))
            gr[min(27,py+1)][max(0,px-1):min(27,px+1)] = [0]*(min(27,px+1)-max(0,px-1))
            pygame.draw.rect(canvas,(255,255,255),[(px-1)*width,(py-1)*width,width*3,width*3])

    def display():
        window.fill((255,255,255))
        window.blit(canvas,(int((480-(width*28+1))/2),int((480-(width*28+1))/2)))
        window.blit(grid,(int((480-(width*28+1))/2),int((480-(width*28+1))/2)))
        pygame.draw.line(window,(0,0,0),(int((480-(width*28+1))/2)*2+width*28+1, 0),(int((480-(width*28+1))/2)*2+width*28+1, 480))

        pygame.display.update()
        clock.tick(60)
n = 0
while True:

    events = pygame.event.get()
    n += 1
    if (n >= 30):
        n = 0
        Csys.clear()
        network.forward(numpy.reshape(gr, (1, 784))[0])
        system.draw.text([round(i,2) for i in network.activations[-1]],)

    system.event(events)
    system.display()

