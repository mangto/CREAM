import pygame, sys, os, math, win32api, json, keyboard

pygame.init()

window = pygame.display.set_mode((980,540))
icon = pygame.Surface((1,1))
icon.fill((255,255,255))
pygame.display.set_caption("Cream Image Dataset Maker")
pygame.display.set_icon(icon)
clock = pygame.time.Clock()
path = os.path.split(os.path.abspath(__file__))[0]


images = os.listdir(path + "\\images")
selected_index = 0
selected_image = pygame.image.load(path + "\\images\\" + images[selected_index]) if len(images) != 0 else pygame.Surface((720,405))
save_enabled = True
return_enabled = True

def font(fontname, size):
    return pygame.font.Font(f"C:\\Windows\\Fonts\\{fontname}.TTF",size)

lastleft1 = 0
lastleft2 = 0
lastright2 = 0
lastright1 = 0
lastmiddle1 = 0
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

class system:
    class draw:
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
        def makeTriangle(scale, internalAngle, rotation):
            #define the points in a uint space
            ia = (math.radians(internalAngle) * 2) - 1
            p1 = (0, -1)
            p2 = (math.cos(ia), math.sin(ia))
            p3 = (math.cos(ia) * -1, math.sin(ia))

            #rotate the points
            ra = math.radians(rotation) 
            rp1x = p1[0] * math.cos(ra) - p1[1] * math.sin(ra)
            rp1y = p1[0] * math.sin(ra) + p1[1] * math.cos(ra)                 
            rp2x = p2[0] * math.cos(ra) - p2[1] * math.sin(ra)
            rp2y = p2[0] * math.sin(ra) + p2[1] * math.cos(ra)                        
            rp3x = p3[0] * math.cos(ra) - p3[1] * math.sin(ra)                         
            rp3y = p3[0] * math.sin(ra) + p3[1] * math.cos(ra)
            rp1 = ( rp1x, rp1y )
            rp2 = ( rp2x, rp2y )
            rp3 = ( rp3x, rp3y )

            #scale the points 
            sp1 = [rp1[0] * scale, rp1[1] * scale]
            sp2 = [rp2[0] * scale, rp2[1] * scale]
            sp3 = [rp3[0] * scale, rp3[1] * scale]
                            
            return sp1, sp2, sp3
        def offsetTriangle(triangle, offsetx, offsety):
            triangle[0][0] += offsetx;  triangle[0][1] += offsety;
            triangle[1][0] += offsetx;  triangle[1][1] += offsety;
            triangle[2][0] += offsetx;  triangle[2][1] += offsety;

            return triangle
        def draw_rect_alpha(surface, color, rect):
            shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            surface.blit(shape_surf, rect)
    class math:
        def sign(number:int):
            if number < 0: return -1
            else: return 1

    scroll = 0
    class ui:
        ui_list = []
        class text_list_viewer:
            def __init__(self, x, y, sx, sy, button_color:tuple=(24, 132, 236)):
                system.ui.ui_list.append(self)
                self.x = x
                self.y = y
                self.sx = sx
                self.sy = sy
                self.button_color = button_color
                self.classes = eval(open(path + "\\data\\classes.json", 'r', encoding='utf8').read())
                self.class_name = list(self.classes.keys())
                self.class_color = list(self.classes.values())
                self.selected_index = 0

                self.button_surface = pygame.Surface((self.sx, self.sy), pygame.SRCALPHA).convert_alpha()
                pygame.draw.rect(self.button_surface, button_color, [0, 0, self.sx, self.sy])
                self.shower = pygame.Surface((self.sx, self.sy), pygame.SRCALPHA).convert_alpha()

                self.hitbox = pygame.Surface((980, 540))
                pygame.draw.rect(self.hitbox, (255, 255, 255), [self.x, self.y, self.sx, self.sy])

                self.opened = False
                self.row = (540-self.y-self.sy-10)//self.sy
                self.canvas = pygame.Surface((self.sx, min(self.row, len(self.class_name))*self.sy))
                self.canvas_y = 0
                self.canvas_hitbox = pygame.Surface((980, 540))
                pygame.draw.rect(self.canvas_hitbox, (255, 255, 255), [self.x, self.y+self.sy+10, self.sx, min(self.row, len(self.class_name))*self.sy])
                self.min_canvas_y = -30 * len(self.class_name) + min(self.row, len(self.class_name))*self.sy
                self.focused = False

            def draw(self, mx, my):
                self.shower = pygame.Surface((self.sx, self.sy), pygame.SRCALPHA).convert_alpha()
                self.shower.blit(self.button_surface, (0, 0))
                system.draw.text(self.class_name[self.selected_index], font("Arial", 16), self.shower, 8, 15, "cenleft", self.class_color[self.selected_index])
                pygame.draw.polygon(self.shower, self.class_color[self.selected_index], system.draw.offsetTriangle(system.draw.makeTriangle(5, 45 ,180), self.sx-10, self.sy/2))

                window.blit(self.shower, (self.x, self.y))

                if (self.hitbox.get_at((mx, my)) == (255, 255, 255)):
                    if (pygame.mouse.get_pressed()[0] == True):
                        
                        if (self.opened): self.opened = False
                        else: self.opened = True; self.focused = False
                if (self.opened):
                    self.canvas = pygame.Surface((self.sx, min(self.row, len(self.class_name))*self.sy))
                    self.canvas.fill(self.button_color)

                    if (self.canvas_hitbox.get_at((mx, my)) == (255, 255, 255)):
                        self.focused = True
                        self.canvas_y += system.scroll
                        self.canvas_y = min(self.canvas_y, 0)
                        if (self.canvas_y < 0): self.canvas_y = max(self.canvas_y, self.min_canvas_y)
                        index = ((my - self.y - self.sy -10) - self.canvas_y) // 30
                        pygame.draw.rect(self.canvas, (242, 121, 60), [0, 30*index+self.canvas_y, self.sx, self.sy], 2)

                        if (pygame.mouse.get_pressed()[0] == True): self.selected_index = index; self.opened = False;
                    else:
                        if (pygame.mouse.get_pressed()[0] == True and self.focused): self.opened = False
                    pygame.draw.rect(self.canvas, (24, 132, 236), [0, 30*self.selected_index+self.canvas_y, self.sx, self.sy], 2)
                    for i,(name, color) in enumerate(zip(self.class_name, self.class_color)):
                        system.draw.text(name, font("Arial", 16), self.canvas, 8, self.canvas_y+15+30*i,"cenleft", color)

                    window.blit(self.canvas, (self.x, self.y+self.sy+10))
        class image_list_viewer:
            def __init__(self, x, y, sx, sy, image_list):
                system.ui.ui_list.append(self)
                self.canvas_y = 5

                self.x = x
                self.y = y
                self.sx = sx
                self.sy = sy
                self.original = [pygame.image.load(path + "\\images\\" + image) for image in image_list]
                self.images = [pygame.transform.smoothscale(image, (200,130)) for image in self.original]
                self.min_canvas_y = len(self.images)*135*(-1)+self.sy

                self.hitbox = pygame.Surface((980, 540))
                pygame.draw.rect(self.hitbox, (255, 255, 255), [self.x, self.y, self.sx, self.sy])

            
            def draw(self, mx, my):
                global selected_index, selected_image

                if (self.hitbox.get_at((mx, my)) == (255, 255, 255)):
                    self.canvas_y += system.scroll
                    self.canvas_y = min(5, self.canvas_y)
                    if (self.canvas_y < 0): self.canvas_y = max(self.canvas_y, self.min_canvas_y)

                    if (pygame.mouse.get_pressed()[0] == 1):
                        abs_pos = my - self.canvas_y
                        if (abs_pos//135 < len(self.images)):
                            selected_index = abs_pos//135
                            selected_image = self.original[selected_index]



                for i, image in enumerate(self.images):
                    window.blit(image, (10, self.canvas_y + 135*i))

                pygame.draw.rect(window, (24, 132, 236), (10, 135*selected_index+self.canvas_y, 200, 130), 2)
        class border_maker:
            def __init__(self, x, y, sx, sy):
                system.ui.ui_list.append(self)
                self.x = x
                self.y = y
                self.sx = sx
                self.sy = sy

                self.hitbox = pygame.Surface((980, 540))
                pygame.draw.rect(self.hitbox, (255, 255, 255), [self.x, self.y, self.sx, self.sy])
                self.new_box = [(-1, -1), (-1, -1)]
                self.enabled = True
                self.borders = [{name:[] for name in class_selector.class_name} for i in range(len(image_list_view.images))]
                self.history = [[] for i in range(len(image_list_view.images))]

            def draw(self, mx, my):
                if (self.enabled):
                    if (self.hitbox.get_at((mx, my)) == (255, 255, 255)):
                        if (mouse.leftbtdown()):
                            self.new_box[0] = (mx, my)

                        if (pygame.mouse.get_pressed()[0]):
                            self.new_box[1] = (mx, my)
                        
                        if (mouse.leftbtup() and (mx, my) != self.new_box[0]):
                            self.borders[selected_index][class_selector.class_name[class_selector.selected_index]].append(self.new_box)
                            self.history[selected_index].append([class_selector.class_name[class_selector.selected_index]])
                            self.new_box = [(-1, -1), (-1, -1)]
                    if (class_selector.opened): self.enabled = False
                
                else:
                    if (class_selector.opened == False and pygame.mouse.get_pressed()[0] == 0): self.enabled = True

                for name in self.borders[selected_index]:
                    for border in self.borders[selected_index][name]:
                        pygame.draw.rect(window, class_selector.classes[name], (border[0][0], border[0][1], border[1][0] - border[0][0], border[1][1] - border[0][1]), 1)
                pygame.draw.aaline(window, (0,0,0), self.new_box[0], self.new_box[1])
                pygame.draw.rect(window, class_selector.class_color[class_selector.selected_index], (self.new_box[0][0], self.new_box[0][1], self.new_box[1][0] - self.new_box[0][0], self.new_box[1][1] - self.new_box[0][1]), 1)

    def display(events):
        if (len(events) > 0):
            window.fill((44, 49, 60))
            pygame.draw.rect(window,(33, 37, 43),(0,0,220,540)) # image pannel
            window.blit(selected_image, (240,68))

            mx, my = pygame.mouse.get_pos()
            for ui in system.ui.ui_list:
                ui.draw(mx, my)

            pygame.display.update()
        clock.tick(144)

    def event(events):
        global save_enabled, return_enabled
        for event in events:
            system.scroll = 0

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEWHEEL:
                system.scroll = system.math.sign(event.y) * 70
            
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_BACKSPACE:
                    history = border.history[selected_index]

                    if (history != []):
                        name = history[-1][0]
                        border.history[selected_index] = border.history[selected_index][:-1]
                        border.borders[selected_index][name] = border.borders[selected_index][name][:-1]
                
        if keyboard.is_pressed("ctrl+s") and save_enabled:
            json.dump(border.borders, open(path+"\\data\\dataset.json", 'w', encoding='utf8'), indent=4)
            json.dump(border.history, open(path+"\\data\\history.json", 'w', encoding='utf8'), indent=4)
            save_enabled = False
        if keyboard.is_pressed("s") != True:
            save_enabled = True
        if keyboard.is_pressed("ctrl+z") and return_enabled:
            history = border.history[selected_index]

            if (history != []):
                name = history[-1][0]
                border.history[selected_index] = border.history[selected_index][:-1]
                border.borders[selected_index][name] = border.borders[selected_index][name][:-1]
            return_enabled = False
        if keyboard.is_pressed("z") != True:
            return_enabled = True




image_list_view = system.ui.image_list_viewer(0, 0, 220, 540, images)
class_selector = system.ui.text_list_viewer(230, 10, 140, 30, (33, 37, 43))
border = system.ui.border_maker(240,68,720,405)
border.borders = eval(open(path + "\\data\\dataset.json", 'r', encoding='utf8').read())
border.history = eval(open(path + "\\data\\history.json", 'r', encoding='utf8').read())


reversed(system.ui.ui_list)

while __name__ == "__main__":
    events = pygame.event.get()
    system.event(events)
    system.display(events)