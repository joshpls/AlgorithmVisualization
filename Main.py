from threading import Timer
import pygame
import math
import time
from queue import PriorityQueue
from pygame import font

DISPLAYWIDTH = (800)
WIN = pygame.display.set_mode((DISPLAYWIDTH, DISPLAYWIDTH))
pygame.display.set_caption("A* Search Algorithm Visualization")
pygame.init()

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
ROWS = (25)

myfont = pygame.font.Font('freesansbold.ttf', 30)

class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN
    
    def is_wall(self):
        return self.color == BLACK
    
    def is_start(self):
        return self.color == BLUE
    
    def is_end(self):
        return self.color == ORANGE
    
    def reset(self):
        self.color = WHITE

    def make_closed(self):
        self.color = RED
    
    def make_open(self):
        self.color = GREEN
    
    def make_wall(self):
        self.color = BLACK
    
    def make_start(self):
        self.color = BLUE
    
    def make_end(self):
        self.color = ORANGE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
    
    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_wall(): # Down
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_wall(): # Up
                self.neighbors.append(grid[self.row - 1][self.col])        
        
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_wall(): # Left
                self.neighbors.append(grid[self.row][self.col + 1])
        
        if self.col > 0 and not grid[self.row][self.col - 1].is_wall(): # Right
                self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self,other):
        return False

class button():
    def __init__(self, color, x,y,width,height, text=''):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def draw(self,win,outline=None):
        #Call this method to draw the button on the screen
        if outline:
            pygame.draw.rect(win, outline, (self.x-2,self.y-2,self.width+4,self.height+4),0)
            
        pygame.draw.rect(win, self.color, (self.x,self.y,self.width,self.height),0)
        
        if self.text != '':
            font = pygame.font.SysFont('comicsans', 60)
            text = font.render(self.text, 1, (0,0,0))
            win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))

    def isOver(self, pos):
        #Pos is the mouse position or a tuple of (x,y) coordinates
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                return True
            
        return False

# Test Button
blueButton = button(BLUE,350,750, 100, 50, "Click Me!")

# Methods

def getStarted():
    global started
    return started

def setStarted(this):
    global started
    started = this

def paused(draw):
    global isPaused
    global stepThrough
    isPaused = True
    if getStarted():
        print("Game Paused")
        while isPaused:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.QUIT
                    #setStarted(False)
                    isPaused = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        #setStarted(False)
                        isPaused = False
                        stepThrough = False
                    if event.key == pygame.K_RIGHT:
                        print("Step Through")
                        stepThrough = True
                        isPaused = False
            draw()
            pygame.display.update()
        print("Game Unpaused")

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(parent, current, draw):
    while current in parent:
        current = parent[current]
        current.make_path()
        draw()

def showTextBox(x, y):
    textBox = myfont.render(textboxValue, True, BLACK)
    WIN.blit(textBox, (x, y))

def clearTextBox():
    global textboxValue
    textboxValue = ""

def setup_algorithm(draw, grid, start, end):
    pass

def run_algorithm(draw, grid, start, end):
    global stepThrough
    startTime = time.perf_counter()
    waitTime = 0.00
    waitTimeElapsed = 0
    endTime = 0
    setStarted(True)
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    parent = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        if stepThrough:
            paused(draw)

        time.sleep(waitTime)
        waitTimeElapsed = waitTimeElapsed + waitTime

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    stepThrough = False
                    print("pause key pressed")
                    paused(draw)
        
        current = open_set.get()[2]
        open_set_hash.remove(current)
        
        if current == end:
            endTime = time.perf_counter()
            endTime = endTime - waitTimeElapsed
            reconstruct_path(parent, end, draw)
            start.make_start()
            end.make_end()
            setStarted(False)
            global textboxValue
            textboxValue = f"Found path in: {endTime - startTime:0.2f} seconds"
            return True # Make path
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                parent[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        draw()
        
        if current != start:
            current.make_closed()
    
    return False

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([]) 
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)
    
    draw_grid(win, rows, width)
    showTextBox(200, 650)
    #blueButton.draw(win, BLACK)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

# Main Method
def main(win, width):

    grid = make_grid(ROWS, width)
    global textboxValue
    textboxValue = ""
    global started 
    started = False
    global isPaused
    isPaused = True
    global stepThrough
    stepThrough = False

    start = None
    end = None
    run = True

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            pos = pygame.mouse.get_pos()
            row, col = get_clicked_pos(pos, ROWS, width)
            node = grid[row][col]
            if event.type == pygame.QUIT:
                run = False
            if pygame.mouse.get_pressed()[0]: # left
                clearTextBox()
                
                if blueButton.isOver(pos):
                    blueButton.text = "Clicked!"
                if not start and node != end:
                    start = node
                    start.make_start()
                elif not end and node != start:
                    end = node
                    end.make_end()
                elif node != end and node != start:
                    node.make_wall()
            elif pygame.mouse.get_pressed()[2]: # Right
                clearTextBox()
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None
            if event.type == pygame.KEYDOWN:
                clearTextBox()
                if event.key == pygame.K_SPACE:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    run_algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
                if event.key == pygame.K_ESCAPE:
                    if start == None:
                        run = False
                        pygame.QUIT
                    
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
    pygame.QUIT

# Run Main
main(WIN, DISPLAYWIDTH)
