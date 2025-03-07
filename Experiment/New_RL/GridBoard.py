# gridboard.py

import numpy as np
import random
import sys
import matplotlib.pyplot as plt

def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)

class BoardPiece:

    def __init__(self, name, code, pos):
        self.name = name #name of the piece
        self.code = code #an ASCII character to display on the board
        self.pos = pos #2-tuple e.g. (1,4)

class BoardMask:

    def __init__(self, name, mask, code):
        self.name = name
        self.mask = mask
        self.code = code

    def get_positions(self): #returns tuple of arrays
        return np.nonzero(self.mask)

def zip_positions2d(positions): #positions is tuple of two arrays
    x,y = positions
    return list(zip(x,y))

class GridBoard:

    def __init__(self, size=40):
        self.size = size #Board dimensions, e.g. 40 x 40
        self.components = {} #name : board piece
        self.masks = {}
        self.obstacle_positions = []  # Store obstacle positions for collision detection
        self.redspots = []  # Store obstacle positions as redspots (compatible with env.py)
        self.vision_distance = 2  # Default vision distance (compatible with env.py)

    def generate_obstacles(self, num_obstacles=20):
        """Generate random obstacles on the board similar to env.py"""
        # Clear existing obstacles
        self.obstacle_positions = []
        self.redspots = []
        
        # Generate random obstacle positions
        while len(self.redspots) < num_obstacles:
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            if (x, y) not in self.redspots:
                self.redspots.append((x, y))
                self.obstacle_positions.append((x, y))
        
        # Create mask for obstacles
        obstacle_mask = np.zeros((self.size, self.size), dtype=np.int8)
        for x, y in self.redspots:
            obstacle_mask[x, y] = 1
            
        # Add the mask to the board
        self.addMask('obstacles', obstacle_mask, '-')
        
        return self.redspots

    def place_goal_random(self, obstacles=None):
        """Place goal randomly on the board (not on obstacles)"""
        if obstacles is None:
            obstacles = self.redspots
            
        while True:
            goal_x = random.randint(0, self.size-1)
            goal_y = random.randint(0, self.size-1)
            if (goal_x, goal_y) not in obstacles:
                if 'Goal' in self.components:
                    self.components['Goal'].pos = (goal_x, goal_y)
                else:
                    self.addPiece('Goal', '+', (goal_x, goal_y))
                return (goal_x, goal_y)

    def place_agent_random(self, obstacles=None, goal=None):
        """Place agent randomly on the board (not on obstacles or goal)"""
        if obstacles is None:
            obstacles = self.redspots
            
        if goal is None and 'Goal' in self.components:
            goal = self.components['Goal'].pos
            
        while True:
            agent_x = random.randint(0, self.size-1)
            agent_y = random.randint(0, self.size-1)
            if (agent_x, agent_y) not in obstacles and (agent_x, agent_y) != goal:
                if 'Player' in self.components:
                    self.components['Player'].pos = (agent_x, agent_y)
                else:
                    self.addPiece('Player', 'P', (agent_x, agent_y))
                return (agent_x, agent_y)

    def get_vision(self, pos, distance=None):
        """Get the cells visible from the current position within a given distance"""
        if distance is None:
            distance = self.vision_distance
            
        x, y = pos
        visible_positions = []
        
        # Generate 5x5 vision grid (like in env.py)
        for dy in range(-distance, distance+1):
            for dx in range(-distance, distance+1):
                vision_x = max(0, min(self.size-1, x + dx))
                vision_y = max(0, min(self.size-1, y + dy))
                visible_positions.append((vision_x, vision_y))
                
        return visible_positions

    def get_observation(self, agent_pos=None, goal_pos=None):
        """Get observation vector similar to env.py"""
        if agent_pos is None and 'Player' in self.components:
            agent_pos = self.components['Player'].pos
            
        if goal_pos is None and 'Goal' in self.components:
            goal_pos = self.components['Goal'].pos
            
        if agent_pos is None or goal_pos is None:
            return None
            
        agent_x, agent_y = agent_pos
        
        # Get 5x5 vision cells (25 total)
        visible_cells = []
        for dy in range(-2, 3):  # -2, -1, 0, 1, 2
            for dx in range(-2, 3):  # -2, -1, 0, 1, 2
                vision_x = max(0, min(self.size-1, agent_x + dx))
                vision_y = max(0, min(self.size-1, agent_y + dy))
                
                # Determine what's in this cell
                if (vision_x, vision_y) in self.redspots:
                    color = 1.0  # Red (obstacle)
                elif (vision_x, vision_y) == goal_pos:
                    color = 2.0  # Green (goal)
                else:
                    color = 0.0  # White (empty)
                    
                visible_cells.append(color)
        
        # Add agent and goal coordinates to the state
        goal_x, goal_y = goal_pos
        coordinates_info = [float(agent_x), float(agent_y), float(goal_x), float(goal_y)]
        
        # Combine visible cells and coordinate information
        return visible_cells + coordinates_info

    def addPiece(self, name, code, pos=(0,0)):
        newPiece = BoardPiece(name, code, pos)
        self.components[name] = newPiece

    #basically a set of boundary elements
    def addMask(self, name, mask, code):
        #mask is a 2D-numpy array with 1s where the boundary elements are
        newMask = BoardMask(name, mask, code)
        self.masks[name] = newMask

    def movePiece(self, name, pos):
        move = True
        # Check if the move would place the piece on an obstacle
        if pos in self.obstacle_positions:
            move = False
        # Check other masks
        for _, mask in self.masks.items():
            if pos in zip_positions2d(mask.get_positions()):
                move = False
        
        # Make sure the position is within bounds
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            move = False
            
        if move:
            self.components[name].pos = pos

    def delPiece(self, name):
        if name in self.components:
            del self.components[name]

    def render(self):
        dtype = '<U2'
        displ_board = np.zeros((self.size, self.size), dtype=dtype)
        displ_board[:] = ' '

        for name, piece in self.components.items():
            displ_board[piece.pos] = piece.code

        for name, mask in self.masks.items():
            positions = mask.get_positions()
            displ_board[positions] = mask.code

        return displ_board

    def render_plt(self, show=True):
        """Render the board using matplotlib (similar to env.py render)"""
        grid = np.zeros((self.size, self.size))
        
        # Set redspots (obstacles)
        for x, y in self.redspots:
            grid[y, x] = 1
        
        # Set goal
        if 'Goal' in self.components:
            goal_pos = self.components['Goal'].pos
            grid[goal_pos[1], goal_pos[0]] = 2
        
        # Set agent position
        if 'Player' in self.components:
            agent_pos = self.components['Player'].pos
            grid[agent_pos[1], agent_pos[0]] = 3
        
        # Create custom colormap: white -> red -> green -> blue
        colors = ['white', 'red', 'lime', 'blue']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap=cmap)
        plt.title('GridWorld Environment')
        plt.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.2)
        
        if show:
            plt.show()
        else:
            plt.close()

    def render_np(self):
        num_pieces = len(self.components) + len(self.masks)
        displ_board = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)
        layer = 0
        for name, piece in self.components.items():
            pos = (layer,) + piece.pos
            displ_board[pos] = 1
            layer += 1

        for name, mask in self.masks.items():
            x,y = mask.get_positions()
            z = np.repeat(layer,len(x))
            a = (z,x,y)
            displ_board[a] = 1
            layer += 1
        return displ_board

def addTuple(a,b):
    return tuple([sum(x) for x in zip(a,b)])
