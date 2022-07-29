import pygame, sys
import win32api
from pygame import gfxdraw

pygame.init()

window = pygame.display.set_mode((600,400), pygame.RESIZABLE)
pygame.display.set_caption("CREAM Network Visualizer")
pygame.display.set_icon(pygame.image.load(".\\visualizer\\icon.png"))
clock = pygame.time.Clock()

lastleft1 = 0
lastleft2 = 0
lastright2 = 0
lastright1 = 0
lastmiddle1 = 0
ui_list = []

def font(fontname, size):
    return pygame.font.Font(f"C:\\Windows\\Fonts\\{fontname}.TTF",size)

class mouse:
    def middlebtdown():
        global lastmiddle1
        middle = win32api.GetKeyState(0x04)
        if int(lastmiddle1) >=0 and middle <0:
            lastmiddle1 = middle
            return True
        else:
            lastmiddle1 = middle
            return False
    def rightbtdown():
        global lastright1
        right = win32api.GetKeyState(0x02)
        if int(lastright1) >= 0 and right <0:
            lastright1 = right
            return True
        else:
            lastright1=right
            return False
    def rightbtup():
        global lastright2
        right = win32api.GetKeyState(0x02)
        if int(lastright2) < 0 and right >=0:
            lastright2 = right
            return True
        else:
            lastright2=right
            return False
    def leftbtdown():
        global lastleft1
        left = win32api.GetKeyState(0x01)
        if int(lastleft1) >=0 and left <0:
            lastleft1 = left
            return True
        else:
            lastleft1 = left
            return False
    def leftbtup():
        global lastleft2
        left = win32api.GetKeyState(0x01)
        if int(lastleft2) < 0 and left >= 0:
            lastleft2 = left
            return True
        
        else:
            lastleft2 = left
            return False
class color:
    light = (160, 161, 163)
    key = (255,59,82)
    dark = (50, 51, 60)
class System:
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
        def trans_rect(window,x,y,sx,sy,color):
            s = pygame.Surface((sx,sy), pygame.SRCALPHA)
            s.fill(color)
            window.blit(s, (x,y))
    class ui:
        ui_tag = eval(open('.\\visualizer\\data\\ui.json','r',encoding='utf-8').read())
        class button:
            def __init__(self,surface:pygame.Surface, x:int, y:int, sx:int, sy:int, icon:pygame.Surface=False,
                            color=(255,255,255),edge_color=(0,0,0), edge_thick=1,opacity:int=255,round:bool=False, roundness=1.0,
                            text:str="",text_color=(0,0,0),font:font=font("ARIAL",15),
                            addshadow=True, clickable=True,CustomCorrectionX=0,CustomCorrectionY=0,
                            showline=True,
                            tag=""
                        ):
                ui_list.append(self)

                self.surface=surface
                self.x=x
                self.y=y
                self.sx=sx
                self.sy=sy
                self.icon=icon
                self.color=color
                self.edge_color=edge_color
                self.edge_thick=edge_thick
                self.shape = round
                self.opacity = opacity
                self.text=text
                self.text_color = text_color
                self.font=font
                self.round=round
                self.roundness=roundness
                self.addshadow=addshadow
                self.clickable=clickable
                self.CustomCorrectionX=CustomCorrectionX
                self.CustomCorrectionY=CustomCorrectionY
                self.showline=showline
                self.tag=tag

                self.onmouse = False
                self.onmousecolor = (int(color[0]*0.9),int(color[1]*0.9),int(color[2]*0.9))

                self.image = pygame.Surface((sx,sy),pygame.SRCALPHA,32).convert_alpha()
                self.onmouseS = pygame.Surface((sx,sy),pygame.SRCALPHA,32).convert_alpha()
                self.omtexted = pygame.Surface((sx,sy),pygame.SRCALPHA,32).convert_alpha()
                self.texted = pygame.Surface((sx,sy),pygame.SRCALPHA,32).convert_alpha()
                self.hitbox = pygame.Surface(surface.get_size())

                if (round == False):
                    pygame.draw.rect(self.image,color,[0,0,sx,sy])
                    pygame.draw.rect(self.image,edge_color,[0,0,sx,sy],edge_thick)
                    pygame.draw.rect(self.hitbox,(255,255,255),[x,y,sx,sy])

                    pygame.draw.rect(self.onmouseS,self.onmousecolor,[0,0,sx,sy])
                    pygame.draw.rect(self.onmouseS,edge_color,[0,0,sx,sy],edge_thick)
                else:
                    System.draw.rrect(self.image,[0,0,sx,sy],edge_color,1)
                    System.draw.rrect(self.image,[edge_thick,edge_thick,sx-2*edge_thick,sy-2*edge_thick],color,roundness)
                    System.draw.rrect(self.hitbox,[x,y,sx,sy],(255,255,255),1)

                    System.draw.rrect(self.onmouseS,[0,0,sx,sy],edge_color,1)
                    System.draw.rrect(self.onmouseS,[edge_thick,edge_thick,sx-2*edge_thick,sy-2*edge_thick],self.onmousecolor,roundness)


                if (icon != False):
                    icon_size = icon.get_size()
                    self.image.blit(icon,(int((sx-icon_size[0])/2),int((sy-icon_size[1])/2)))
                    self.onmouseS.blit(icon,(int((sx-icon_size[0])/2),int((sy-icon_size[1])/2)))

                self.texted.blit(self.image,(0,0))
                self.omtexted.blit(self.onmouseS,(0,0))
                System.draw.text(text,font,self.texted,int(sx/2),int(sy/2),"center",self.text_color)
                System.draw.text(text,font,self.omtexted,int(sx/2),int(sy/2),"center",self.text_color)

                self.opacitied = self.texted
                self.omopacited = self.omtexted
                self.opacitied.set_alpha(opacity)
                self.omopacited.set_alpha(opacity)

                self.shadow, self.correctionx,self.correctiony=System.ui.shadow(self.opacitied,5,10)
            def draw(self, mx, my):
                
                if (self.addshadow == True):
                    if (self.CustomCorrectionX==0 and self.CustomCorrectionY ==0):
                        self.surface.blit(self.shadow,(self.x-self.correctionx,self.y-self.correctiony))
                    else:
                        self.surface.blit(self.shadow,(self.x+self.CustomCorrectionX,self.y+self.CustomCorrectionY))
                self.surface.blit(self.opacitied,(self.x,self.y))

                if (self.hitbox.get_at((mx,my)) == (255,255,255) and self.clickable==True):
                    self.onmouse = True
                    if (self.showline == True):
                        pygame.draw.rect(window,(56,190,128),[self.x,self.y,self.sx,self.sy],1)
                    else: self.surface.blit(self.omopacited,(self.x,self.y))

                    if (mouse.leftbtup() == True):
                        if (self.tag in System.ui.ui_tag['button']):
                            exec(f"{System.ui.ui_tag['button'][self.tag]}")
                else:
                    self.onmouse = False

                System.draw.text(self.text,self.font,window,self.x+int(self.sx/2),self.y+int(self.sy/2),"center",color=self.text_color)
            def set_text(self,text:str):
                self.text = text
                self.texted = pygame.Surface((self.sx,self.sy),pygame.SRCALPHA,32).convert_alpha()
                self.omtexted = pygame.Surface((self.sx,self.sy),pygame.SRCALPHA,32).convert_alpha()
                self.texted.blit(self.image,(0,0))
                self.omtexted.blit(self.onmouseS, (0,0))
                self.opacitied = self.texted
                self.opacitied.set_alpha(self.opacity)
                self.omopacited = self.omtexted
                self.omopacited.set_alpha(self.opacity)
            def set_opacity(self,opacity:int):
                self.opacity = opacity
                self.opacitied = self.texted
                self.opacitied.set_alpha(opacity)
    
    def display(events):
        if (events != []):
            window.fill((255,255,255))

            pygame.display.update()
            clock.tick(30)

    def event(events):
        for event in events:
            if (event.type == pygame.QUIT):
                pygame.quit()
                sys.exit()
while True:
    events = pygame.event.get()

    System.display(events)
    System.event(events)