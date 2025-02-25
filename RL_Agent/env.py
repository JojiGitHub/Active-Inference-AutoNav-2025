import numpy as np
from gym import spaces
import torch
import matplotlib.pyplot as plt

# Define constants for actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

import numpy as np
from gym import spaces

class GridWorldEnv():
    def __init__(self, initial_position=(0, 0), 
                 red_zone_positions=[(1, 2), (3, 2), (4, 4), (6, 1)],  # Original red zone positions
                 goal_position=(6, 4),  # Original goal position
                 grid_dimensions=[40, 40]):
        self.grid_dimensions = grid_dimensions
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=2, shape=(17,), dtype=np.float32)
        
        self.agent_x, self.agent_y = initial_position
        self.initial_position = initial_position
        self.goal_position = goal_position
        self.red_zone_positions = red_zone_positions
        self.total_steps = 0
        self.max_steps = 12  # Keep the tight constraint since goal is reachable in ~10 steps
        
    def get_visible_cells(self, current_position, grid_dimensions):
        agent_x, agent_y = current_position
        visible_colors = []
        
        # Update the vision pattern to be more logical - a clear view of surroundings
        # Vision radius 2 in a more natural pattern
        vision_pattern = [
            # Immediate surroundings (radius 1)
            (0, 0),   # Current position
            (-1, 0),  # Left
            (1, 0),   # Right
            (0, -1),  # Up
            (0, 1),   # Down
            (-1, -1), # Upper left
            (-1, 1),  # Lower left
            (1, -1),  # Upper right
            (1, 1),   # Lower right
            
            # Extended vision (radius 2)
            (-2, 0),  # Far left
            (2, 0),   # Far right
            (0, -2),  # Far up
            (0, 2),   # Far down
            (-2, -2), # Far upper left
            (-2, 2),  # Far lower left
            (2, -2),  # Far upper right
            (2, 2),   # Far lower right
        ]
        
        # Get colors for each position in vision pattern
        for dx, dy in vision_pattern:
            x_pos = max(0, min(grid_dimensions[0] - 1, agent_x + dx))
            y_pos = max(0, min(grid_dimensions[1] - 1, agent_y + dy))
            
            if (x_pos, y_pos) in self.red_zone_positions:
                color = 1.0  # Red
            elif (x_pos, y_pos) == self.goal_position:
                color = 2.0  # Green (goal)
            else:
                color = 0.0  # White
            visible_colors.append(color)
        
        return visible_colors

    def step(self, action):
        old_position = (self.agent_x, self.agent_y)
        
        # Execute action
        if action == UP:
            self.agent_y = max(0, self.agent_y - 1)
        elif action == DOWN:
            self.agent_y = min(self.grid_dimensions[1] - 1, self.agent_y + 1)
        elif action == LEFT:
            self.agent_x = max(0, self.agent_x - 1)
        elif action == RIGHT:
            self.agent_x = min(self.grid_dimensions[0] - 1, self.agent_x + 1)
        elif action == STAY:
            pass
        
        new_position = (self.agent_x, self.agent_y)
        self.total_steps += 1
        
        # Calculate manhattan distance (more suitable for grid movement)
        curr_manhattan = abs(new_position[0] - self.goal_position[0]) + abs(new_position[1] - self.goal_position[1])
        prev_manhattan = abs(old_position[0] - self.goal_position[0]) + abs(old_position[1] - self.goal_position[1])
        
        # Start with small negative reward to encourage speed
        reward = -0.5
        
        # Terminal states
        if new_position in self.red_zone_positions:
            reward = -20.0
            done = True
        elif new_position == self.goal_position:
            # Massive bonus for reaching goal quickly
            steps_bonus = max(0, 10 - self.total_steps) * 20  # 20 points per step under 10
            reward = 50.0 + steps_bonus
            done = True
        else:
            # Progress rewards/penalties
            if curr_manhattan < prev_manhattan:
                reward += 2.0  # Reward for getting closer
            elif curr_manhattan > prev_manhattan:
                reward -= 4.0  # Bigger penalty for moving away
                
            # Action penalties
            if old_position == new_position and action != STAY:
                reward -= 2.0  # Penalty for hitting walls
            elif action == STAY:
                reward -= 5.0  # Bigger penalty for staying still
                
            done = False
        
        # Harsh timeout penalty
        if self.total_steps >= 12:  # Timeout at 12 steps
            reward = -30.0
            done = True
        
        # Get observation
        state = self.get_visible_cells(new_position, self.grid_dimensions)
        
        return np.array(state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.agent_x, self.agent_y = self.initial_position
        state = self.get_visible_cells(self.initial_position, self.grid_dimensions)
        self.total_steps = 0
        return np.array(state, dtype=np.float32)

    def render(self, show=False):
        grid = np.zeros((self.grid_dimensions[1], self.grid_dimensions[0]))  # Note the order: (rows, cols)
        
        # Set red zones
        for x, y in self.red_zone_positions:
            grid[y, x] = 1  # Swapped to [y,x] for correct visualization
        
        # Set goal (green)
        goal_x, goal_y = self.goal_position
        grid[goal_y, goal_x] = 2  # Swapped to [y,x] for correct visualization
        
        # Set agent position
        grid[self.agent_y, self.agent_x] = 3  # Swapped to [y,x] for correct visualization
        
        # Create custom colormap: white -> red -> green -> blue
        colors = ['white', 'red', 'lime', 'blue']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        plt.imshow(grid, cmap=cmap)
        plt.title('Grid World Environment')
        if show:
            plt.show()
        else:
            plt.close()