import pygame

class Button:
    def __init__(self, text, width, height, pos, win, font):
        self.win = win
        self.font = font
        self.pressed = False

        # top rectangle
        self.top_rect = pygame.Rect(pos,(width, height))
        self.top_color = "#475F77"

        # text
        self.text_surface = self.font.render(text,True,"#FFFFFF")
        self.text_rect = self.text_surface.get_rect(center = self.top_rect.center)
    
    def draw(self):
        pygame.draw.rect(self.win, self.top_color, self.top_rect, border_radius = 12)
        self.win.blit(self.text_surface, self.text_rect)
        self.is_clicked()
    
    def is_clicked(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            self.top_color = "#D74B4B"
            if pygame.mouse.get_pressed()[0]:
                self.pressed = True
            else:
                if self.pressed == True:
                    self.pressed = False
        else:
            self.top_color = "#475F77"
    
    def check_pressed(self):
        return self.pressed == True

class Checkbox():
    def __init__(self,text, width, height, pos, win, font, offset, checked):
        self.win = win
        self.font = font
        self.text = text
        self.width = width
        self.height = height
        self.pos = pos
        self.pressed = False
        self.checked = checked
        self.offset = offset
        self.text_len = len(self.text * int(self.width/2))
        self.cb_x = pos[0]+self.text_len
        self.cb_y = pos[1]+(offset/2)

        # top rectangle
        self.top_rect = pygame.Rect((self.cb_x, self.cb_y),(width, height))
        self.top_color = "#475F77"

        # text
        self.text_checkbox = self.font.render(self.text,True, (0,0,0))
        self.cross_rect = pygame.Rect((self.cb_x+(offset/2), self.cb_y+(offset/2)), (width-offset, height-offset))
    
    def draw(self):
        self.win.fill((255,255,255), ((self.cb_x, self.pos[1]), (self.width, self.height))) #clear the text
        pygame.draw.rect(self.win, self.top_color, self.top_rect)
        if self.is_checked():
            pygame.draw.rect(self.win, (150, 150, 150), self.cross_rect)
        self.win.blit(self.text_checkbox, (self.pos))
        self.is_clicked()

    def change_state(self):
        if self.pressed:
            if self.checked == False:
                self.checked = True
            else:
                self.checked = False
    
    def is_checked(self):
        return self.checked
    
    def is_clicked(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            if pygame.mouse.get_pressed()[0]:
                self.pressed = True
            else:
                if self.pressed:
                    self.change_state()
                    self.pressed = False

class DropDown():

    def __init__(self, color_menu, color_option, x, y, w, h, font, main, options):
        self.color_menu = color_menu
        self.color_option = color_option
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.main = main
        self.options = options
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1

    def draw(self, surf):
        
        pygame.draw.rect(surf, self.color_menu[self.menu_active], self.rect, 0)
        msg = self.font.render(self.main, 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center = self.rect.center))

        if self.draw_menu:
            for i, text in enumerate(self.options):
                rect = self.rect.copy()
                rect.y += (i+1) * self.rect.height
                pygame.draw.rect(surf, self.color_option[1 if i == self.active_option else 0], rect, 0)
                msg = self.font.render(text, 1, (0, 0, 0))
                surf.blit(msg, msg.get_rect(center = rect.center))

    def update(self, event_list):
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)
        
        self.active_option = -1
        for i in range(len(self.options)):
            rect = self.rect.copy()
            rect.y += (i+1) * self.rect.height
            if rect.collidepoint(mpos):
                self.active_option = i
                break

        if not self.menu_active and self.active_option == -1:
            self.draw_menu = False

        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    self.draw_menu = not self.draw_menu
                elif self.draw_menu and self.active_option >= 0:
                    self.draw_menu = False
                    return self.active_option
        return -1