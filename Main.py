# GUI.py
from queue import PriorityQueue
import pygame
import time
pygame.init()


class Grid:
    WHITE = (255, 255, 255)

    def __init__(self, rows, cols, width, height, win):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.win = win
        self.start = None
        self.end = None
        self.nodes = self.make_grid()
    
    def make_grid(self):
        grid = []
        gap = self.width // self.rows
        for i in range(self.rows):
            grid.append([]) 
            for j in range(self.rows):
                node = Node(i, j, gap, self.rows, self.win)
                grid[i].append(node)
        return grid

    def draw_grid(self):
        BLACK = (0,0,0)
        gap = self.width // self.rows

        for row in self.nodes:
            for node in row:
                node.draw()

        for i in range(self.rows+1):
            pygame.draw.line(self.win, BLACK, (0, i * gap), (self.width, i * gap))
            for j in range(self.cols+1):
                pygame.draw.line(self.win, BLACK, (j * gap, 0), (j * gap, self.width))

        pygame.display.update()

    def click(self, pos):
        """
        :param: pos
        :return: (row, col)
        """
        if pos[0] < self.width and pos[1] < self.height:
            gap = self.width / self.rows
            x = pos[1] // gap
            y = pos[0] // gap
            return (int(y),int(x))
        else:
            return None

    def clear(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.nodes[i][j].reset()
        self.start = None
        self.end = None

    def clear_algorithm(self):
        for row in self.nodes:
            for node in row:
                if node.is_empty():
                    node.reset()
                    
    def select(self, row, col, addNode):
        current_node = self.nodes[row][col]

        if addNode:
            if current_node.is_empty():
                if not self.start:
                    current_node.make_start()
                    self.start = current_node
                elif not self.end:
                    current_node.make_end()
                    self.end = current_node
                else:
                    current_node.make_wall()
        else:
            if not current_node.is_empty():
                if current_node.is_start():
                    self.start = None
                elif current_node.is_end():
                    self.end = None
                current_node.reset()

    def h(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)
    
    def reconstruct_path(self, parent, current):
        while current in parent:
            current = parent[current]
            current.make_path()
            # Draw the Path by steps
            self.draw_grid()

    def run_algorithm(self):
        #A* Search Algorithm
        start_node = self.start
        end_node = self.end

        # Update Neighbors for all nodes
        for row in self.nodes:
            for node in row:
                node.update_neighbors(self.nodes)

        #clear board
        self.clear_algorithm()

        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, start_node))
        parent = {}
        g_score = {node: float("inf") for row in self.nodes for node in row}
        g_score[start_node] = 0
        f_score = {node: float("inf") for row in self.nodes for node in row}
        f_score[start_node] = self.h(start_node.get_pos(), end_node.get_pos())

        open_set_hash = {start_node}

        while not open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.QUIT

            current = open_set.get()[2]
            open_set_hash.remove(current)
            
            if current == end_node:
                end_node.make_end()
                self.reconstruct_path(parent, end_node)
                start_node.make_start()
                return True # Make path
            
            for neighbor in current.neighbors:
                temp_g_score = g_score[current] + 1

                if temp_g_score < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.h(neighbor.get_pos(), end_node.get_pos())
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        neighbor.make_open()
            
            # Update the grid with open & closed nodes
            self.draw_grid()
            
            if current != start_node:
                current.make_closed()
        
        return False

class Node:
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    PURPLE = (128, 0, 128)
    ORANGE= (255, 165, 0)
    GREY = (128, 128, 128)
    TURQUOISE = (64, 224, 208)

    def __init__(self, row, col, width, total_rows, win):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = self.WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.win = win
        self.start = False
        self.end = False

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == self.RED

    def is_open(self):
        return self.color == self.GREEN
    
    def is_wall(self):
        return self.color == self.BLACK
    
    def is_start(self):
        return self.color == self.BLUE
    
    def is_end(self):
        return self.color == self.ORANGE
    
    def is_empty(self):
        if self.color == self.WHITE or self.color == self.GREEN or self.color == self.RED or self.color == self.PURPLE:
            return True
    
    def reset(self):
        self.start = False
        self.end = False
        self.wall = False
        self.color = self.WHITE

    def make_closed(self):
        self.color = self.RED
    
    def make_open(self):
        self.color = self.GREEN
    
    def make_wall(self):
        self.color = self.BLACK
    
    def make_start(self):
        self.start = True
        self.color = self.BLUE
    
    def make_end(self):
        self.end = True
        self.color = self.ORANGE

    def make_path(self):
        self.color = self.PURPLE

    def draw(self):
        pygame.draw.rect(self.win, self.color, (self.x, self.y, self.width, self.width))
    
    def update_neighbors(self, nodes):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not nodes[self.row + 1][self.col].is_wall(): # Down
            self.neighbors.append(nodes[self.row + 1][self.col])

        if self.row > 0 and not nodes[self.row - 1][self.col].is_wall(): # Up
                self.neighbors.append(nodes[self.row - 1][self.col])        
        
        if self.col < self.total_rows - 1 and not nodes[self.row][self.col + 1].is_wall(): # Left
                self.neighbors.append(nodes[self.row][self.col + 1])
        
        if self.col > 0 and not nodes[self.row][self.col - 1].is_wall(): # Right
                self.neighbors.append(nodes[self.row][self.col - 1])

    def __lt__(self,other):
        return False

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


def redraw_window(win, board, time, run_button):
    # Draw time
    fnt = pygame.font.SysFont("cambria", 40)
    if time != None:
        win.fill((255,255,255), (600, 800, 600, 800)) #clear the text
        text = fnt.render("Time: " + str(time), 1, (0,0,0))
        win.blit(text, (600, 800))

    # Draw grid and board
    board.draw_grid()

    # Draw Button
    run_button.draw()

def main():
    width = 800
    height = 920
    rows = 25
    cols = 25
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("A* Search")
    board = Grid(rows, cols, width, width, win)
    run = True
    start = None
    play_time = None
    font = pygame.font.SysFont("cambria", 35)
    run_button = Button("Run Algorithm", 290, 45, (250,805),win,font)
    WHITE = (255, 255, 255)
    win.fill(WHITE)
    run_button_pressed = False

    while run:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
                    pygame.QUIT
                
                if event.key == pygame.K_DELETE:
                    board.clear()

                if event.key == pygame.K_SPACE:
                    start = time.perf_counter()
                    board.run_algorithm()
                    play_time = round(time.perf_counter() - start, 2)

                if event.key == pygame.K_RETURN:
                    board.clear_algorithm()

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                boardClicked = board.click(pos)
                if boardClicked:
                    board.select(boardClicked[0], boardClicked[1], True)
            if pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                boardClicked = board.click(pos)
                if boardClicked:
                    board.select(boardClicked[0], boardClicked[1], False)
        
        if run_button_pressed == False and run_button.check_pressed() == True:
            run_button_pressed = True
            start = time.perf_counter()
            board.run_algorithm()
            play_time = round(time.perf_counter() - start, 2)
        else:
            if run_button_pressed == True and run_button.check_pressed() == False:
                run_button_pressed = False
        
        # Draw Board + Time
        redraw_window(win, board, play_time, run_button)
        pygame.display.update()
 
main()
pygame.quit()
