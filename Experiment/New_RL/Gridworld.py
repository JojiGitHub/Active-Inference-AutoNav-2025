# gridworld.py

from GridBoard import *
import sys
import os
import matplotlib.pyplot as plt
from gym import spaces
import numpy as np

# Define constants for actions (to match env.py)
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

# Action name mapping for consistent interface
ACTION_NAMES = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT", 
    RIGHT: "RIGHT",
    STAY: "STAY"
}

class Gridworld:

    def __init__(self, size=40, mode='random', num_obstacles=20, random_seed=42, max_steps=100, use_coppeliasim=False):
        # Validate input parameters
        if size < 5:  # Minimum size requirement
            print(f"Warning: Grid size {size} is too small. Setting to minimum size of 5")
            size = 5
            
        # Adjust num_obstacles based on grid size to prevent overcrowding
        max_possible_obstacles = (size - 2) * (size - 2) // 4  # Leave room for movement
        if num_obstacles > max_possible_obstacles:
            print(f"Warning: Too many obstacles ({num_obstacles}) for grid size {size}. Reducing to {max_possible_obstacles}")
            num_obstacles = max_possible_obstacles

        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Store configuration
        self.size = size
        self.mode = mode
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.total_steps = 0
        self.random_seed = random_seed
        self.use_coppeliasim = False  # Always set to False to disable CoppeliaSim

        # Create the board
        self.board = GridBoard(size=size)
        
        # Initialize board with empty pieces (they'll be placed in reset())
        self.board.addPiece('Player', 'P', (0,0))
        self.board.addPiece('Goal', '+', (0,0))
        
        # Store current position
        self.x = 0
        self.y = 0
        
        # Setup observation and action spaces for compatibility with Gym interface
        self.action_space = spaces.Discrete(5)  # 5 actions: UP, DOWN, LEFT, RIGHT, STAY
        
        # Observation: 25 vision cells + agent x,y + goal x,y (29 total values)
        self.observation_space = spaces.Box(
            low=0, 
            high=self.size, 
            shape=(29,), 
            dtype=np.float32
        )
        
        # Reset to initialize the environment
        self.reset()
    
    def add_border_walls(self):
        """Add border walls around the grid using obstacles/redspots"""
        # Clear existing border walls first
        self.board.border_positions = []
        
        # Create walls on all four sides of the grid
        for x in range(self.size):
            # Top wall
            self.board.redspots.append((x, 0))
            self.board.obstacle_positions.append((x, 0))
            self.board.border_positions.append((x, 0))
            
            # Bottom wall
            self.board.redspots.append((x, self.size-1))
            self.board.obstacle_positions.append((x, self.size-1))
            self.board.border_positions.append((x, self.size-1))
        
        # Left and right walls (skip corners to avoid double-counting)
        for y in range(1, self.size-1):
            # Left wall
            self.board.redspots.append((0, y))
            self.board.obstacle_positions.append((0, y))
            self.board.border_positions.append((0, y))
            
            # Right wall
            self.board.redspots.append((self.size-1, y))
            self.board.obstacle_positions.append((self.size-1, y))
            self.board.border_positions.append((self.size-1, y))
        
        # Create mask for obstacles/walls
        obstacle_mask = np.zeros((self.size, self.size), dtype=np.int8)
        for x, y in self.board.obstacle_positions:
            obstacle_mask[x, y] = 1
            
        # Add the mask to the board
        self.board.addMask('obstacles', obstacle_mask, '-')
        
    def reset(self):
        """Reset the environment to an initial state and return the initial observation"""
        self.total_steps = 0
        
        # Add border walls first - this is now done for ALL environment types
        self.add_border_walls()
        
        # Generate additional obstacles based on mode
        if self.mode == 'random':
            # Add random obstacles using the specified method
            self.generate_reproducible_obstacles(self.random_seed)
        elif self.mode == 'static':
            # Use predefined obstacle pattern for static mode
            self.initGridStatic()
        elif self.mode == 'player':
            # Static obstacles but random player position
            self.initGridPlayer()
        elif self.mode == 'clusters':
            # Generate clustered obstacles
            self.generate_clustered_obstacles()
        elif self.mode == 'wall' or self.mode == 'walls':  # Support both 'wall' and 'walls' for compatibility
            # Generate maze-like walls
            self.generate_maze_walls()
        else:
            # Default to random if mode is not recognized
            print(f"Warning: Environment mode '{self.mode}' not recognized. Using 'random'.")
            self.generate_reproducible_obstacles(self.random_seed)
        
        # Place goal in random position (not on obstacles)
        self.goal_position = self.board.place_goal_random()
        
        # Place player in random position (not on obstacles or goal)
        self.player_position = self.board.place_agent_random()
        
        # Store current position for easier access
        self.x, self.y = self.player_position
        
        # Get observation
        observation = self.get_observation()
        
        return np.array(observation, dtype=np.float32)
    
    def create_bounding_locations(self, position, dimensions):
        """
        Creates bounding locations for an object.
        
        Parameters:
        -----------
        position : tuple (x, y)
            Center position of the object
        dimensions : tuple (a, b)
            Dimensions of the object (width, depth)
            
        Returns:
        --------
        tuple
            Eight points: (top_right, bottom_left, top_left, bottom_right, mid_top, mid_bottom, mid_left, mid_right)
        """
        (x, y) = position
        (a, b) = dimensions

        # Bounding locations
        top_right = (x + a/2, y + b/2)
        bottom_left = (x - a/2, y - b/2)
        top_left = (x - a/2, y + b/2)
        bottom_right = (x + a/2, y - b/2)

        # Midpoints
        mid_top = ((top_right[0] + top_left[0]) / 2, (top_right[1] + top_left[1]) / 2)
        mid_bottom = ((bottom_right[0] + bottom_left[0]) / 2, (bottom_right[1] + bottom_left[1]) / 2)
        mid_left = ((top_left[0] + bottom_left[0]) / 2, (top_left[1] + bottom_left[1]) / 2)
        mid_right = ((top_right[0] + bottom_right[0]) / 2, (top_right[1] + bottom_right[1]) / 2)
        
        return top_right, bottom_left, top_left, bottom_right, mid_top, mid_bottom, mid_left, mid_right
    
    def generate_reproducible_obstacles(self, seed):
        """Generate obstacles in reproducible positions based on a seed"""
        random.seed(seed)
        
        # Clear existing obstacles (except borders)
        self.board.obstacle_positions = []
        self.board.redspots = []
        
        # Add border walls first
        self.add_border_walls()
        
        # Calculate valid ranges for obstacle placement
        min_grid_x = 2  # Keep minimum distance from left border
        max_grid_x = self.size - 3  # Keep minimum distance from right border
        min_grid_y = 2  # Keep minimum distance from top border
        max_grid_y = self.size - 3  # Keep minimum distance from bottom border
        
        # Check if we have valid ranges
        if max_grid_x <= min_grid_x or max_grid_y <= min_grid_y:
            print(f"Warning: Grid size {self.size} is too small for additional obstacles")
            return
            
        # Calculate available positions
        valid_positions = []
        for x in range(min_grid_x, max_grid_x + 1):
            for y in range(min_grid_y, max_grid_y + 1):
                valid_positions.append((x, y))
                
        # Shuffle positions and select obstacles
        random.shuffle(valid_positions)
        num_obstacles_to_place = min(self.num_obstacles, len(valid_positions))
        
        # Place obstacles
        for i in range(num_obstacles_to_place):
            pos = valid_positions[i]
            if (pos not in self.board.obstacle_positions and 
                ('Player' not in self.board.components or pos != self.board.components['Player'].pos) and
                ('Goal' not in self.board.components or pos != self.board.components['Goal'].pos)):
                self.board.obstacle_positions.append(pos)
                self.board.redspots.append(pos)
        
        # Update the obstacle mask
        self.update_obstacle_mask()

    def get_observation(self):
        """Get a state representation compatible with env.py"""
        return self.board.get_observation()

    def generate_clustered_obstacles(self):
        """Generate clustered obstacle patterns"""
        # Number of clusters (ensure at least 1 cluster)
        num_clusters = max(1, min(4, self.num_obstacles // 5))
        obstacles_per_cluster = self.num_obstacles // num_clusters
        
        # Generate clusters
        for _ in range(num_clusters):
            # Pick a random center that's not on the border
            center_x = random.randint(2, self.size - 3)
            center_y = random.randint(2, self.size - 3)
            
            # Add obstacles around the center
            for _ in range(obstacles_per_cluster):
                # Get random offset from center (closer to center is more likely)
                dx = int(random.gauss(0, 1.5))
                dy = int(random.gauss(0, 1.5))
                
                x = max(1, min(self.size - 2, center_x + dx))
                y = max(1, min(self.size - 2, center_y + dy))
                
                # Add the obstacle if not already present
                pos = (x, y)
                if pos not in self.board.obstacle_positions:
                    self.board.redspots.append(pos)
                    self.board.obstacle_positions.append(pos)
        
        # Update the obstacle mask
        self.update_obstacle_mask()

    def generate_maze_walls(self):
        """Generate maze-like walls in the environment"""
        # Number of wall segments
        num_walls = min(8, self.num_obstacles // 4)
        
        for _ in range(num_walls):
            # Decide if horizontal or vertical wall
            is_horizontal = random.choice([True, False])
            
            # Wall start position (avoid edges)
            start_x = random.randint(2, self.size - 3)
            start_y = random.randint(2, self.size - 3)
            
            # Wall length
            length = random.randint(3, self.size // 4)
            
            # Create wall
            if is_horizontal:
                # Horizontal wall
                for dx in range(length):
                    x = min(self.size - 2, start_x + dx)
                    y = start_y
                    pos = (x, y)
                    if pos not in self.board.obstacle_positions:
                        self.board.redspots.append(pos)
                        self.board.obstacle_positions.append(pos)
            else:
                # Vertical wall
                for dy in range(length):
                    x = start_x
                    y = min(self.size - 2, start_y + dy)
                    pos = (x, y)
                    if pos not in self.board.obstacle_positions:
                        self.board.redspots.append(pos)
                        self.board.obstacle_positions.append(pos)
        
        # Update the obstacle mask
        self.update_obstacle_mask()

    def update_obstacle_mask(self):
        """Update the obstacle mask based on the current obstacle positions"""
        obstacle_mask = np.zeros((self.size, self.size), dtype=np.int8)
        for x, y in self.board.obstacle_positions:
            obstacle_mask[x, y] = 1
        
        # Add the mask to the board
        self.board.addMask('obstacles', obstacle_mask, '-')

    def initGridStatic(self):
        """Initialize the board with a static configuration of obstacles"""
        # Define additional static obstacle positions in grid coordinates
        static_obstacles = [
            (1, 1), (1, 2), (1, 3), (2, 1), (3, 1),  # Corner shape
            (self.size//2, self.size//2), (self.size//2+1, self.size//2), (self.size//2, self.size//2+1),  # Middle obstacles
            (self.size-2, self.size-2), (self.size-3, self.size-2), (self.size-2, self.size-3)  # Bottom-right corner
        ]
        
        # Add the static obstacles to the board
        for pos in static_obstacles:
            if 0 <= pos[0] < self.size and 0 <= pos[1] < self.size:
                if pos not in self.board.redspots:  # Avoid duplicates
                    self.board.redspots.append(pos)
                    self.board.obstacle_positions.append(pos)
        
        # Update the obstacle mask
        self.update_obstacle_mask()

    def initGridPlayer(self):
        """Initialize with static obstacles but random player position"""
        # Set up static obstacles
        self.initGridStatic()
        
        # Player will be placed randomly in reset()

    def validateMove(self, direction, pos=None):
        """
        Validate if a move is possible from the given position
        
        Args:
            direction: Direction to move (UP, DOWN, LEFT, RIGHT, STAY)
            pos: Current position (if None, use player position)
            
        Returns:
            int: 0 = valid move, 1 = invalid (wall/obstacle), 2 = pit (no pits in this version)
        """
        if pos is None:
            pos = self.player_position
            
        x, y = pos
        
        # Calculate new position based on direction
        if direction == UP:  # UP
            new_pos = (x, max(0, y-1))
        elif direction == DOWN:  # DOWN
            new_pos = (x, min(self.size-1, y+1))
        elif direction == LEFT:  # LEFT
            new_pos = (max(0, x-1), y)
        elif direction == RIGHT:  # RIGHT
            new_pos = (min(self.size-1, x+1), y)
        else:  # STAY
            new_pos = pos
            
        # Check if new position is on an obstacle
        if new_pos in self.board.obstacle_positions:
            return 1  # Invalid move
        
        # Check if new position is outside bounds
        nx, ny = new_pos
        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
            return 1  # Invalid move
            
        return 0  # Valid move

    def step(self, action):
        """
        Execute one environment step
        
        Args:
            action: Integer action index
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.total_steps += 1
        old_position = (self.x, self.y)
        old_manhattan = abs(old_position[0] - self.goal_position[0]) + abs(old_position[1] - self.goal_position[1])
        
        # Get action name for easier processing
        action_name = ACTION_NAMES.get(action, "STAY")
        
        # Execute the action
        self.makeMove(action_name)
        
        # Get updated position
        new_position = (self.x, self.y)
        new_manhattan = abs(new_position[0] - self.goal_position[0]) + abs(new_position[1] - self.goal_position[1])
        
        # Calculate reward based on new position, obstacles, goal, and manhattan distance change
        manhattan_improvement = old_manhattan - new_manhattan
        reward = self.reward(manhattan_improvement)
        
        # Check if done
        done = (new_position == self.goal_position)  # Goal reached
        done = done or (new_position in self.board.obstacle_positions)  # Hit obstacle
        done = done or (self.total_steps >= self.max_steps)  # Max steps reached
        
        # Get observation
        observation = self.get_observation()
        
        # Additional info for compatibility with env.py
        info = {
            'manhattan_distance': new_manhattan,
            'action_name': action_name,
            'manhattan_improvement': manhattan_improvement
        }
        
        return np.array(observation, dtype=np.float32), reward, done, info

    def makeMove(self, action):
        """
        Make a move in the environment
        
        Args:
            action: String action name ('u', 'd', 'l', 'r', or 's')
        """
        # Store the old position
        old_x, old_y = self.x, self.y
        
        # Convert abbreviated action to full action name
        if action == 'u' or action == 'UP':  # up
            new_pos = (old_x, max(0, old_y-1))
        elif action == 'd' or action == 'DOWN':  # down
            new_pos = (old_x, min(self.size-1, old_y+1))
        elif action == 'l' or action == 'LEFT':  # left
            new_pos = (max(0, old_x-1), old_y)
        elif action == 'r' or action == 'RIGHT':  # right
            new_pos = (min(self.size-1, old_x+1), old_y)
        else:  # stay
            new_pos = (old_x, old_y)
            
        # Move piece and update position
        self.board.movePiece('Player', new_pos)
        self.player_position = self.board.components['Player'].pos
        self.x, self.y = self.player_position

    def reward(self, manhattan_improvement=0):
        """
        Calculate reward based on current state and Manhattan distance improvement
        
        Args:
            manhattan_improvement: Change in Manhattan distance to goal (positive if closer)
            
        Returns:
            float: Reward value
        """
        # Check for special cases first
        if (self.player_position in self.board.obstacle_positions):
            return -10  # Hit obstacle
        elif (self.player_position == self.goal_position):
            return 10  # Reached goal
        else:
            # Base reward is a small step penalty
            base_reward = -0.1
            
            # Add reward based on Manhattan distance improvement
            distance_reward = 0.0
            if manhattan_improvement > 0:
                # Reward for moving closer to the goal
                distance_reward = manhattan_improvement * 0.5
            elif manhattan_improvement < 0:
                # Penalty for moving away from the goal
                distance_reward = manhattan_improvement * 0.5  # Will be negative
            
            # For STAY action, add a small penalty to encourage movement
            if manhattan_improvement == 0:
                distance_reward = -0.1
                
            return base_reward + distance_reward

    def display(self):
        """Display the board in text format"""
        return self.board.render()
        
    def render(self, show=True):
        """Render the board using matplotlib (similar to env.py render)"""
        return self.board.render_plt(show)
        
    def close(self):
        """Close the environment and clean up resources"""
        plt.close('all')  # Close any open matplotlib figures
                
    def visualize_steps(self, num_steps=10, render_every=1, action_selector=None):
        """
        Run the environment for a number of steps and visualize the agent's movement
        
        Args:
            num_steps: Number of steps to run
            render_every: How often to render the environment (every N steps)
            action_selector: Function that takes the current state and returns an action
                            If None, random actions will be used
        
        Returns:
            list: Rewards collected at each step
        """
        # Reset the environment to get a fresh start
        state = self.reset()
        
        # Render the initial state
        print("Initial state:")
        self.render()
        
        rewards = []
        done = False
        steps_taken = 0
        
        while not done and steps_taken < num_steps:
            # Select action
            if action_selector is not None:
                action = action_selector(state)
            else:
                action = self.action_space.sample()  # Random action
            
            # Take a step
            state, reward, done, info = self.step(action)
            rewards.append(reward)
            steps_taken += 1
            
            # Render if it's time to
            if steps_taken % render_every == 0 or done:
                print(f"\nStep {steps_taken} - Action: {ACTION_NAMES[action]}, Reward: {reward}")
                print(f"Position: {self.player_position}, Goal: {self.goal_position}")
                print(f"Manhattan distance: {info['manhattan_distance']}")
                self.render()
                
            # Break if done
            if done:
                print(f"\nEnvironment finished after {steps_taken} steps")
                print(f"Total reward: {sum(rewards)}")
                if self.player_position == self.goal_position:
                    print("Goal reached!")
                elif self.player_position in self.board.obstacle_positions:
                    print("Hit an obstacle!")
                else:
                    print("Max steps reached!")
                    
        return rewards
