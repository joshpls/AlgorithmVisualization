# Path Finding Visualizor
from queue import PriorityQueue
import pygame
import time
import Gui
import random
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
    
    def generate_maze(self):
        # Update Neighbors for all nodes
        for row in self.nodes:
            for node in row:
                if not node.is_start() and not node.is_end():
                    node.reset()
                node.update_neighbors(self.nodes, False)

        x = random.randint(0, (len(self.nodes)-1))
        y = random.randint(0, (len(self.nodes)-1))
        random_start = self.nodes[y][x]

        stack = []
        backlog = []
        stack.append(random_start)

        while len(stack) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.clear_algorithm()
                        return -1
            # Set current node to the last added to the stack - LIFO
            current = stack.pop()
            current.make_visited()
            #current.make_path()

            # Loop through all neighbors of current node (up,down,left,right)
            count = 0
            neighbors = []
            for neighbor in current.neighbors:
                if not neighbor.is_visited():
                    if not neighbor.is_end() and not neighbor.is_start():
                        neighbors.append(neighbor)

            if len(neighbors) > 0:
                ran = random.randint(0, len(neighbors)-1)

                for neighbor in neighbors:
                    if count != ran:
                        backlog.append(neighbor)
                        if not neighbor.is_start() and not neighbor.is_end():
                            neighbor.make_wall()
                            neighbor.make_visited()
                    else:   
                        stack.append(neighbors[ran])
                    count+=1
            else:
                if len(backlog) > 0:
                    stack.append(backlog.pop())
                else:
                    return True        

            # Update the grid with open & closed nodes
            if show_steps:
                self.draw_grid()
        
        return False

    # BFS Search Algorithm - Unweighted and gaurentee's the shortest path
    def bfs(self, start_node, show_steps):
        parent = {}
        queue = []
        queue.append(start_node)

        # Loop through while queue is not empty
        while len(queue) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.clear_algorithm()
                        return -1
            
            # Set current node to first time inserted into the queue - FIFO
            current = queue.pop(0)
            
            for neighbor in current.neighbors:
                if not neighbor.is_visited():
                    parent[neighbor] = current
                    if not neighbor.is_end() and not neighbor.is_start():
                        # Mark neighbor node open and visited
                        neighbor.make_open()
                        neighbor.make_visited()
                    queue.append(neighbor)

                    if neighbor.is_end():
                        # Found the end - Reconstruct the Path
                        return self.reconstruct_path(parent, neighbor, show_steps)

                # Update the grid with open & closed nodes
                if show_steps:
                    self.draw_grid()

                # Close current node
                if current != start_node:
                    current.make_closed()
        
        return -1

    # DFS Search Algorithm - Weighted and does not gaurentee shortest path
    def dfs(self, start_node, show_steps):
        stack = []
        parent = {}
        stack.append(start_node)

        while len(stack) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.clear_algorithm()
                        return -1
            # Set current node to the last added to the stack - LIFO
            current = stack.pop()
            
            if current.is_end():
                # Found the end - Reconstruct the Path
                return self.reconstruct_path(parent, current, show_steps)

            if not current.is_start():
                # Close and mark node as visited
                current.make_closed()
                current.make_visited()

            # Loop through all neighbors of current node (up,down,left,right)
            for neighbor in current.neighbors:
                if not neighbor.is_visited():
                    parent[neighbor] = current
                    stack.append(neighbor)

                    if not neighbor.is_start() and not neighbor.is_end():
                        neighbor.make_open()
                
                # Update the grid with open & closed nodes
                if show_steps:
                    self.draw_grid()
        
        return -1
    
    # A* Search Algorithm - Weigthed and gaurentee's the shortest path
    def astar(self, start_node, end_node, show_steps):
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
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.clear_algorithm()
                        return -1

            current = open_set.get()[2]
            open_set_hash.remove(current)
            
            if current.is_end():
                return self.reconstruct_path(parent, end_node, show_steps)

            if current != start_node:
                current.make_closed()
            
            for neighbor in current.neighbors:
                temp_g_score = g_score[current] + 2

                if temp_g_score < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.h(neighbor.get_pos(), end_node.get_pos())
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        if not neighbor.is_end():
                            neighbor.make_open()
            
            # Update the grid with open & closed nodes
            if show_steps:
                self.draw_grid()

        return -1

    # Greedy - Best First Search Algorithm - Weighted and does not gaurentee shortest path
    def greedy(self, start_node, end_node, show_steps):
        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, start_node))
        parent = {}
        f_score = {node: float("inf") for row in self.nodes for node in row}
        f_score[start_node] = self.h(start_node.get_pos(), end_node.get_pos())

        while not open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.clear_algorithm()
                        return -1

            current = open_set.get()[2]
            
            if current.is_end():
                return self.reconstruct_path(parent, end_node, show_steps)

            if current != start_node:
                current.make_closed()

            for neighbor in current.neighbors:
                if not neighbor.is_closed() and not neighbor.is_open():
                    f_score[neighbor] = self.h(neighbor.get_pos(), end_node.get_pos())
                    parent[neighbor] = current
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    if not neighbor.is_end() and not neighbor.is_start():
                        neighbor.make_open()
        
            # Update the grid with open & closed nodes
            if show_steps:
                self.draw_grid()

        return -1

    def h(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, parent, current, show_steps):
        count = 0
        while current in parent:
            current = parent[current]
            if current == self.start:
                return count
            current.make_path()
            count+=1
            # Draw the Path by steps
            if show_steps:
                self.draw_grid()

    def run_algorithm(self, value, show_steps, diagonal):
        # Check if Algorithm is selected
        if value < 0:
            return -1

        start_node = self.start
        end_node = self.end

        # Check if Start and End nodes are added
        if start_node == None or end_node == None:
            return -1

        # Update Neighbors for all nodes
        for row in self.nodes:
            for node in row:
                node.update_neighbors(self.nodes, diagonal)

        #clear board
        self.clear_algorithm()

        if value == 0:
            return self.astar(start_node, end_node, show_steps)
        elif value == 1:
            return self.bfs(start_node, show_steps)
        elif value == 2:
            return self.dfs(start_node, show_steps)
        else:
            return self.greedy(start_node, end_node, show_steps) #Greedy Best First Search
        

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
        self.visited = False

    def get_pos(self):
        return self.row, self.col

    def is_visited(self):
        return self.visited

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
        self.visited = False
        self.color = self.WHITE
    
    def make_visited(self):
        self.visited = True

    def make_closed(self):
        self.color = self.RED
    
    def make_open(self):
        self.color = self.GREEN
    
    def make_wall(self):
        self.color = self.BLACK
    
    def make_start(self):
        self.start = True
        self.visited = False
        self.color = self.BLUE
    
    def make_end(self):
        self.end = True
        self.visited = False
        self.color = self.ORANGE

    def make_path(self):
        self.color = self.PURPLE

    def draw(self):
        pygame.draw.rect(self.win, self.color, (self.x, self.y, self.width, self.width))
    
    def update_neighbors(self, nodes, diagonal):
        self.neighbors = []
        if self.row > 0 and not nodes[self.row - 1][self.col].is_wall(): # Up
            self.neighbors.append(nodes[self.row - 1][self.col])  

        if self.row < self.total_rows - 1 and not nodes[self.row + 1][self.col].is_wall(): # Down
            self.neighbors.append(nodes[self.row + 1][self.col])
        
        if self.col < self.total_rows - 1 and not nodes[self.row][self.col + 1].is_wall(): # Right
                self.neighbors.append(nodes[self.row][self.col + 1])
        
        if self.col > 0 and not nodes[self.row][self.col - 1].is_wall(): # Left
                self.neighbors.append(nodes[self.row][self.col - 1])

        if diagonal:
            if self.row > 0 and self.col > 0 and not nodes[self.row - 1][self.col - 1].is_wall(): # Up-left
                self.neighbors.append(nodes[self.row - 1][self.col - 1])

            if self.row > 0 and self.col < self.total_rows - 1 and not nodes[self.row - 1][self.col + 1].is_wall(): # Up-Right
                self.neighbors.append(nodes[self.row - 1][self.col + 1])
            
            if self.row < self.total_rows - 1 and self.col < self.total_rows - 1 and not nodes[self.row + 1][self.col + 1].is_wall(): # Down-Right
                self.neighbors.append(nodes[self.row + 1][self.col + 1])

            if self.row < self.total_rows - 1 and self.col > 0 and not nodes[self.row + 1][self.col - 1].is_wall(): # Down-Left
                self.neighbors.append(nodes[self.row + 1][self.col - 1])

    def __lt__(self,other):
        return False

def redraw_window(win, board, event_list, time, display_count, run_button, maze_button, clear_button, steps_cb, diagonal_cb):
    # Draw time
    if time != None:
        fnt = pygame.font.SysFont("cambria", 35)
        win.fill((255,255,255), (600, 800, 600, 800)) #clear the text
        time_text = fnt.render("Time: " + str(time), 1, (0,0,0))
        win.blit(time_text, (600, 800))

    if display_count >= 0:
        fnt = pygame.font.SysFont("cambria", 20)
        time_text = fnt.render("Path to Goal: " + str(display_count), 1, (71,95,119))
        win.blit(time_text, (615, 850))

    # Draw grid and board
    board.draw_grid()

    # Draw Buttons
    run_button.draw(event_list)
    maze_button.draw(event_list)
    clear_button.draw(event_list)

    # Draw Checkboxes
    steps_cb.draw(event_list)
    diagonal_cb.draw(event_list)

if __name__ == "__main__":
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
    small_font = pygame.font.SysFont("cambria", 20)
    tiny_font = pygame.font.SysFont("cambria", 17)
    run_button = Gui.Button("Run Algorithm", 290, 45, (250,805),win,font)
    maze_button = Gui.Button("Generate Maze", 180, 25, (295,855),win,small_font)
    clear_button = Gui.Button("Clear All", 120, 25, (325,885),win,small_font)
    steps_cb = Gui.Checkbox("Steps:", 18, 18, (172,808),win,tiny_font, 6, True)
    diagonal_cb = Gui.Checkbox("Diagonal:", 18, 18, (145,828),win,tiny_font, 6, False)
    WHITE = (255, 255, 255)
    list1 = Gui.DropDown(
        [(WHITE), (0,50,255)],
        [(WHITE), (0,50,255)],
        5, 810, 130, 17, 
        tiny_font,
        "Select Algorithm", ["A* Search", "Breadth First", "Depth First", "Greedy-BFS"])

    win.fill("WHITE")
    show_steps = False
    algorithm = -1
    display_count = -1

    while run:
        
        event_list = pygame.event.get()
        for event in event_list:
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
                    display_count = board.run_algorithm(algorithm, show_steps, diagonal_movement)
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

        # Get algorithm from drop down menu
        selected_option = list1.update(event_list)
        if selected_option >= 0:
            list1.main = list1.options[selected_option]
            algorithm = selected_option

        # Check in Run Button is Pressed, if so run algorithm
        if run_button.check_pressed():
            start = time.perf_counter()
            display_count = board.run_algorithm(algorithm, show_steps, diagonal_movement)
            play_time = round(time.perf_counter() - start, 2)
        
        # Generate Maze
        if maze_button.check_pressed():
            board.generate_maze()
        
        # Clear Board
        if clear_button.check_pressed():
            board.clear()
        
        # Show Steps
        if steps_cb.is_checked():
            show_steps = True
        else:
            show_steps = False

        if diagonal_cb.is_checked():
            diagonal_movement = True
        else:
            diagonal_movement = False

        win.fill(WHITE, ((5, 810), (130, 300)))
        list1.draw(win)
        
        # Draw Board + Time
        redraw_window(win, board, event_list, play_time, display_count, run_button, maze_button, clear_button, steps_cb, diagonal_cb)
        pygame.display.update()

pygame.quit()
