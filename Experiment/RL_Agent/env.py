import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
from scipy.special import comb
from gym import spaces
import sys
import os

# Add CoppeliaSim directory to path to import the module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CoppeliaSim'))
try:
    from CoppeliaSim.coppeliasim import sim
except ImportError:
    print("Warning: CoppeliaSim module not found. Simulator functionality may be limited.")

# Define constants for actions
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

class GridWorldEnv:
    """
    Grid World Environment for reinforcement learning with optional CoppeliaSim integration.
    """
    
    def __init__(self, random_seed=42, grid_dimensions=[40, 40], max_steps=100, use_coppeliasim=False):
        """
        Initialize the environment
        
        Args:
            random_seed (int): Random seed for reproducibility
            grid_dimensions (list): Grid world dimensions [width, height]
            max_steps (int): Maximum steps before episode termination
            use_coppeliasim (bool): Whether to use CoppeliaSim for visualization
        """
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Store configuration
        self.grid_dimensions = grid_dimensions
        self.max_steps = max_steps
        self.total_steps = 0
        self.vision_distance = 2  # How far the agent can see (radius)
        self.use_coppeliasim = use_coppeliasim
        
        # Setup Gym spaces for RL compatibility
        self.action_space = spaces.Discrete(5)  # 5 actions: UP, DOWN, LEFT, RIGHT, STAY
        
        # Observation space - vector of 29 values (25 vision cells + agent x,y + goal x,y)
        self.observation_space = spaces.Box(
            low=0, 
            high=max(grid_dimensions), 
            shape=(29,), 
            dtype=np.float32
        )
        
        # Initialize environment (CoppeliaSim or Grid-only)
        try:
            if self.use_coppeliasim:
                self.initialize_coppelia_environment(random_seed)
            else:
                self.initialize_grid_environment(random_seed)
        except Exception as e:
            print(f"Error during initialization: {e}")
            print("Falling back to grid-only mode")
            self.use_coppeliasim = False
            self.initialize_grid_environment(random_seed)
    
    def initialize_grid_environment(self, random_seed):
        """Initialize grid environment with obstacles, goal, and agent"""
        # Create random obstacles (red zones)
        num_obstacles = random.randint(20, 50)
        self.redspots = []
        
        # Generate obstacle positions until we have enough
        while len(self.redspots) < num_obstacles:
            x = random.randint(0, self.grid_dimensions[0]-1)
            y = random.randint(0, self.grid_dimensions[1]-1)
            if (x, y) not in self.redspots:
                self.redspots.append((x, y))
        
        # Set goal position
        while True:
            goal_x = random.randint(0, self.grid_dimensions[0]-1)
            goal_y = random.randint(0, self.grid_dimensions[1]-1)
            if (goal_x, goal_y) not in self.redspots:
                self.goal = (goal_x, goal_y)
                break
        
        # Set agent position
        while True:
            agent_x = random.randint(0, self.grid_dimensions[0]-1)
            agent_y = random.randint(0, self.grid_dimensions[1]-1)
            if (agent_x, agent_y) not in self.redspots and (agent_x, agent_y) != self.goal:
                self.x = agent_x
                self.y = agent_y
                self.init_loc = (agent_x, agent_y)
                self.current_location = (agent_x, agent_y)
                break
    
    def initialize_coppelia_environment(self, random_seed):
        """Initialize environment with CoppeliaSim integration"""
        # Get environment setup from CoppeliaSim
        obstacle_positions, goal_position, obstacle_handles, goal_handle, bubbleRob_position, redspots = self.initialize_environment(random_seed)
        
        # Set agent and environment properties
        self.obstacle_handles = obstacle_handles
        self.goal_handle = goal_handle
        
        # Convert positions to grid coordinates
        agent_pos = self.move_to_grid(bubbleRob_position[0], bubbleRob_position[1])
        self.goal = self.move_to_grid(goal_position[0], goal_position[1])
        self.redspots = redspots
        
        # Initialize agent state
        self.x, self.y = agent_pos
        self.init_loc = agent_pos
        self.current_location = agent_pos
        
        # Initialize path following components
        self.initialize_robot()
        self.initialize_path_following()
    
    def initialize_robot(self):
        """Initialize robot handle and properties"""
        try:
            self.bubbleRobHandle = sim.getObject('/bubbleRob')
            if self.bubbleRobHandle == -1:
                raise Exception("Could not get handle to bubbleRob")
        except Exception as e:
            print(f"Error getting robot handle: {e}")
            raise
    
    def initialize_path_following(self):
        """Initialize path following variables"""
        self.initial_z, self.initial_orientation = self.get_initial_object_properties()
        pos = sim.getObjectPosition(self.bubbleRobHandle, -1)
        self.ctrlPts = [[pos[0], pos[1], 0.05, 0.0, 0.0, 0.0, 1.0]]
        self.ctrlPts_flattened = [coord for point in self.ctrlPts for coord in point]
        self.posAlongPath = 0
        self.velocity = 0.08
        self.previousSimulationTime = sim.getSimulationTime()
    
    def get_initial_object_properties(self):
        """Get initial robot properties"""
        initial_pos = sim.getObjectPosition(self.bubbleRobHandle, -1)
        initial_orient = sim.getObjectQuaternion(self.bubbleRobHandle, -1)
        return initial_pos[2], initial_orient
    
    def move_to_grid(self, x, y):
        """Convert CoppeliaSim coordinates to grid coordinates"""
        x = x + 2.5
        y = y + 2.5
        
        if x > 5 or x < 0 or y > 5 or y < 0:
            return None
        
        x_grid = round(x / 0.125)
        y_grid = round(y / 0.125)
        
        if x_grid > 40 or x_grid < 0 or y_grid > 40 or y_grid < 0:
            return None
        
        return (x_grid, y_grid)
    
    def grid_to_coordinates(self, x_grid, y_grid):
        """Convert grid coordinates to CoppeliaSim coordinates"""
        if x_grid > 40 or x_grid < 0 or y_grid > 40 or y_grid < 0:
            return None
        
        x = x_grid * 0.125
        y = y_grid * 0.125
        
        x = x - 2.5
        y = y - 2.5
        
        return (x, y, 0.05)
    
    def bezier_recursive(self, ctrlPts, t):
        """Calculate point on Bezier curve"""
        n = (len(ctrlPts) // 7) - 1
        point = np.zeros(3)
        total_weight = 0
        
        for i in range(n + 1):
            binomial_coeff = comb(n, i)
            weight = binomial_coeff * ((1 - t) ** (n - i)) * (t ** i)
            point_coords = np.array(ctrlPts[i * 7:i * 7 + 3])
            point += weight * point_coords
            total_weight += weight
        
        if total_weight > 0:
            point = point / total_weight
        
        point[2] = self.initial_z
        return point
    
    def calculate_total_length(self, ctrl_points, subdivisions=1000):
        """Calculate total length of Bezier path"""
        total_length = 0.0
        prev_point = self.bezier_recursive(ctrl_points, 0)
        
        for i in range(1, subdivisions + 1):
            t = i / subdivisions
            curr_point = self.bezier_recursive(ctrl_points, t)
            total_length += np.linalg.norm(curr_point - prev_point)
            prev_point = curr_point
        return total_length
    
    def get_point_and_tangent(self, t, ctrl_points):
        """Get point and tangent on Bezier curve"""
        point = self.bezier_recursive(ctrl_points, t)
        
        delta = 0.001
        t_next = min(1.0, t + delta)
        next_point = self.bezier_recursive(ctrl_points, t_next)
        
        tangent = next_point - point
        if np.linalg.norm(tangent) > 0:
            tangent = tangent / np.linalg.norm(tangent)
        
        return point, tangent
    
    def update_orientation(self, position, tangent):
        """Update robot orientation based on path tangent"""
        if np.linalg.norm(tangent[:2]) > 0:
            yaw = np.arctan2(tangent[1], tangent[0])
            orientation_quaternion = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
            sim.setObjectQuaternion(self.bubbleRobHandle, -1, orientation_quaternion)
    
    def follow_path(self):
        """Follow the Bezier path"""
        self.posAlongPath = 0
        total_length = self.calculate_total_length(self.ctrlPts_flattened)
        self.previousSimulationTime = sim.getSimulationTime()
        
        while self.posAlongPath < total_length:
            t = sim.getSimulationTime()
            deltaT = t - self.previousSimulationTime
            
            if deltaT <= 0.0:
                self.previousSimulationTime = t
                continue
            
            self.posAlongPath += self.velocity * deltaT
            
            if self.posAlongPath >= total_length - 0.001:
                self.posAlongPath = total_length
                break
            
            t_norm = np.clip(self.posAlongPath / total_length, 0, 1)
            current_pos, tangent = self.get_point_and_tangent(t_norm, self.ctrlPts_flattened)
            current_pos[2] = self.initial_z
            
            sim.setObjectPosition(self.bubbleRobHandle, -1, current_pos.tolist())
            self.update_orientation(current_pos, tangent)
            
            self.previousSimulationTime = t
            sim.step()
            time.sleep(0.05)
    
    def get_observation(self):
        """
        Get vector representation of the environment state for RL agent
        
        Returns:
            list: Observation vector with cell colors and position information
        """
        agent_x, agent_y = self.current_location
        visible_colors = []
        
        # Complete 5x5 vision grid (25 cells including center)
        # Build a proper 5x5 grid centered at the agent's position
        for dy in range(-2, 3):  # -2, -1, 0, 1, 2
            for dx in range(-2, 3):  # -2, -1, 0, 1, 2
                x_pos = max(0, min(self.grid_dimensions[0] - 1, agent_x + dx))
                y_pos = max(0, min(self.grid_dimensions[1] - 1, agent_y + dy))
                
                if (x_pos, y_pos) in self.redspots:
                    color = 1.0  # Red (obstacle)
                elif (x_pos, y_pos) == self.goal:
                    color = 2.0  # Green (goal)
                else:
                    color = 0.0  # White (empty)
                    
                visible_colors.append(color)
        
        # Add agent and goal coordinates to the state
        goal_x, goal_y = self.goal
        coordinates_info = [
            float(agent_x),
            float(agent_y), 
            float(goal_x), 
            float(goal_y)
        ]
        
        # Combine visible colors and coordinate information
        return visible_colors + coordinates_info
    
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Integer action index
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        try:
            # Track steps and previous position
            self.total_steps += 1
            old_position = self.current_location
            old_manhattan = abs(old_position[0] - self.goal[0]) + abs(old_position[1] - self.goal[1])
            
            # Get action name
            action_name = ACTION_NAMES.get(action, "STAY")
            
            # Update position based on action
            if action == UP:
                self.y = max(0, self.y - 1)
            elif action == DOWN:
                self.y = min(self.grid_dimensions[1] - 1, self.y + 1)
            elif action == LEFT:
                self.x = max(0, self.x - 1)
            elif action == RIGHT:
                self.x = min(self.grid_dimensions[0] - 1, self.x + 1)
            
            # Update current location and calculate distances
            self.current_location = (self.x, self.y)
            new_position = self.current_location
            new_manhattan = abs(new_position[0] - self.goal[0]) + abs(new_position[1] - self.goal[1])
            
            # Update CoppeliaSim if enabled
            if self.use_coppeliasim:
                # ... existing CoppeliaSim code ...
                pass
            
            # ULTRA SIMPLE REWARD FUNCTION
            reward = 0.0
            done = False
            
            # Goal reached - large fixed reward
            if new_position == self.goal:
                reward = 1.0  # Binary reward
                done = True
            
            # Obstacle hit - episode ends
            elif new_position in self.redspots:
                reward = -1.0  # Fixed negative reward
                done = True
            
            # Simple direction-based reward
            else:
                # Positive reward only when moving closer to the goal
                if new_manhattan < old_manhattan:
                    reward = 0.1  # Small positive reward
                # Negative reward for moving away or hitting walls
                elif new_manhattan > old_manhattan:
                    reward = -0.1  # Small negative reward
                else:
                    reward = -0.05  # Smaller negative reward for no progress
            
            # Time limit reached
            if self.total_steps >= self.max_steps:
                done = True
            
            # Get observation
            state = self.get_observation()
            info = {
                'manhattan_distance': new_manhattan,
                'action_name': ACTION_NAMES.get(action, "UNKNOWN"),
                'manhattan_improvement': old_manhattan - new_manhattan
            }
            
            return np.array(state, dtype=np.float32), reward, done, info
            
        except Exception as e:
            print(f"Error during environment step: {e}")
            # Return safe defaults
            return (
                np.zeros(29, dtype=np.float32),
                -0.1,
                True,
                {'manhattan_distance': 999, 'action_name': 'ERROR', 'manhattan_improvement': 0}
            )
    
    def reset(self):
        """
        Reset environment to initial state
        
        Returns:
            numpy array: Observation vector
        """
        try:
            self.x, self.y = self.init_loc
            self.current_location = (self.x, self.y)
            self.total_steps = 0
            
            # Reset CoppeliaSim position if enabled
            if self.use_coppeliasim:
                try:
                    # Reset BubbleRob position
                    coords = self.grid_to_coordinates(self.x, self.y)
                    if coords:
                        sim.setObjectPosition(self.bubbleRobHandle, -1, [coords[0], coords[1], 0.12])
                        
                    # Reset path following variables
                    self.initialize_path_following()
                except Exception as e:
                    print(f"Error resetting CoppeliaSim: {e}")
            
            # Ensure goal and agent positions are valid
            if not hasattr(self, 'goal') or not isinstance(self.goal, tuple) or len(self.goal) != 2:
                # Set a default goal if invalid
                self.goal = (self.grid_dimensions[0] - 1, self.grid_dimensions[1] - 1)
            
            # Return RL-compatible observation
            obs = self.get_observation()
            return np.array(obs, dtype=np.float32)
            
        except Exception as e:
            print(f"Error during environment reset: {e}")
            # Return a safe default observation as fallback
            return np.zeros(29, dtype=np.float32)  # Updated to match new observation size
    
    def render(self, show=True):
        """Render the environment as a grid"""
        grid = np.zeros((self.grid_dimensions[1], self.grid_dimensions[0]))
        
        # Set red zones
        for x, y in self.redspots:
            if 0 <= x < self.grid_dimensions[0] and 0 <= y < self.grid_dimensions[1]:
                grid[y, x] = 1
        
        # Set goal (green)
        goal_x, goal_y = self.goal
        if 0 <= goal_x < self.grid_dimensions[0] and 0 <= goal_y < self.grid_dimensions[1]:
            grid[goal_y, goal_x] = 2
        
        # Set agent position (blue)
        grid[self.y, self.x] = 3
        
        # Create custom colormap: white -> red -> green -> blue
        colors = ['white', 'red', 'lime', 'blue']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap=cmap)
        plt.title(f'Environment - Step {self.total_steps}')
        plt.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.2)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def close(self):
        """Close environment and clean up resources"""
        if self.use_coppeliasim:
            try:
                # Stop simulation
                sim.stopSimulation()
            except Exception as e:
                print(f"Error closing CoppeliaSim: {e}")
    
    # CoppeliaSim environment setup methods
    def initialize_environment(self, seed):
        """Initialize the environment with obstacles and objects"""
        random.seed(seed)
        num_obstacles = random.randint(20, 50)
        print(f"Initializing environment with seed {seed} and {num_obstacles} obstacles")
        
        obstacle_positions = []
        obstacle_handles = []
        obstacle_dimensions = [0.3, 0.3, 0.8]
        redspots = []
        
        # Create obstacles
        for i in range(num_obstacles):
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < 100:
                x = round(random.uniform(-2.5 + (obstacle_dimensions[0]/2 + 0.1), 
                                      2.5 - (obstacle_dimensions[0]/2 + 0.1)), 2)
                y = round(random.uniform(-2.5 + (obstacle_dimensions[1]/2 + 0.1), 
                                      2.5 - (obstacle_dimensions[1]/2 + 0.1)), 2)
                valid_position = self.is_position_valid(x, y, obstacle_dimensions, obstacle_positions, obstacle_dimensions)
                attempts += 1
                
            if valid_position:
                obstacle_positions.append((x, y))
                obstacle = self.create_cuboid(
                    dimensions=obstacle_dimensions,
                    position=[x, y, 0.4],
                    color=[1, 0, 0],
                    mass=1,
                    respondable=True,
                    name=f"Obstacle{i}"
                )
                obstacle_handles.append(obstacle)
                
                # Get grid points for redspots
                grid_points = self.get_all_grid_points_in_obstacle((x, y), obstacle_dimensions)
                for point in grid_points:
                    grid_pos = self.move_to_grid(point[0], point[1])
                    if isinstance(grid_pos, tuple) and grid_pos not in redspots:
                        redspots.append(grid_pos)
        
        # Create goal location
        goal_dimensions = [0.125, 0.125, 0.01]
        valid_position = False
        attempts = 0
        goal_position = None
        goal_handle = None
        
        while not valid_position and attempts < 100:
            x = round(random.uniform(-2.5 + (goal_dimensions[0]/2 + 0.1), 
                                   2.5 - (goal_dimensions[0]/2 + 0.1)), 2)
            y = round(random.uniform(-2.5 + (goal_dimensions[1]/2 + 0.1), 
                                   2.5 - (goal_dimensions[1]/2 + 0.1)), 2)
            valid_position = self.is_position_valid(x, y, goal_dimensions, obstacle_positions, obstacle_dimensions)
            attempts += 1
            
        if valid_position:
            goal_handle = self.create_cuboid(
                dimensions=goal_dimensions,
                position=[x, y, 0.005],
                color=[0, 1, 0],
                mass=0.1,
                respondable=True,
                name="Goal_Loc"
            )
            goal_position = (x, y, 0.005)
        
        # Place bubbleRob
        bubbleRob_dimensions = [0.2, 0.2, 0.2]
        valid_position = False
        attempts = 0
        bubbleRob_position = None
        
        while not valid_position and attempts < 100:
            x = round(random.uniform(-2.0 + (bubbleRob_dimensions[0]/2), 
                                   2.0 - (bubbleRob_dimensions[0]/2)), 2)
            y = round(random.uniform(-2.0 + (bubbleRob_dimensions[1]/2), 
                                   2.0 - (bubbleRob_dimensions[1]/2)), 2)
            
            valid_position = self.is_position_valid(x, y, bubbleRob_dimensions, obstacle_positions, obstacle_dimensions)
            attempts += 1
        
        if valid_position:
            bubbleRob_handle = sim.getObject('/bubbleRob')
            if bubbleRob_handle != -1:
                sim.setObjectPosition(bubbleRob_handle, -1, [x, y, 0.12])
                bubbleRob_position = (x, y, 0.12)
        
        return obstacle_positions, goal_position, obstacle_handles, goal_handle, bubbleRob_position, redspots
    
    def is_position_valid(self, new_x, new_y, object_dimensions, obstacle_positions, obstacle_dimensions):
        """Check if a position is valid (no overlap with obstacles or boundaries)"""
        new_position = (new_x, new_y)
        new_bounds = self.create_bounding_locations(new_position, object_dimensions)
        new_top_right, new_bottom_left, _, _, _, _, _, _ = new_bounds
        
        if (new_x + object_dimensions[0]/2 > 2.5 or 
            new_x - object_dimensions[0]/2 < -2.5 or
            new_y + object_dimensions[1]/2 > 2.5 or
            new_y - object_dimensions[1]/2 < -2.5):
            return False
            
        for pos in obstacle_positions:
            existing_x, existing_y = pos[0], pos[1]
            existing_bounds = self.create_bounding_locations((existing_x, existing_y), obstacle_dimensions)
            existing_top_right, existing_bottom_left, _, _, _, _, _, _ = existing_bounds
            
            if not (new_top_right[0] < existing_bottom_left[0] or 
                    existing_top_right[0] < new_bottom_left[0] or
                    new_top_right[1] < existing_bottom_left[1] or 
                    existing_top_right[1] < new_bottom_left[1]):
                return False
                
        return True
    
    def create_bounding_locations(self, position, dimensions):
        """Calculate bounding box locations for an object"""
        x, y = position
        a, b, c = dimensions
        
        top_right = (x + a/2, y + b/2)
        bottom_left = (x - a/2, y - b/2)
        top_left = (x - a/2, y + b/2)
        bottom_right = (x + a/2, y - b/2)
        
        mid_top = ((top_right[0] + top_left[0]) / 2, (top_right[1] + top_left[1]) / 2)
        mid_bottom = ((bottom_right[0] + bottom_left[0]) / 2, (bottom_right[1] + bottom_left[1]) / 2)
        mid_left = ((top_left[0] + bottom_left[0]) / 2, (top_left[1] + bottom_left[1]) / 2)
        mid_right = ((top_right[0] + bottom_right[0]) / 2, (top_right[1] + bottom_right[1]) / 2)
        
        return top_right, bottom_left, top_left, bottom_right, mid_top, mid_bottom, mid_left, mid_right
    
    def get_all_grid_points_in_obstacle(self, position, dimensions):
        """Get all grid points within an obstacle"""
        x, y = position
        a, b, c = dimensions
        
        x_min = x - a/2
        x_max = x + a/2
        y_min = y - b/2
        y_max = y + b/2
        
        epsilon = 0.0001
        x_start = math.ceil(x_min / 0.125) * 0.125
        x_end = math.floor(x_max / 0.125) * 0.125
        y_start = math.ceil(y_min / 0.125) * 0.125
        y_end = math.floor(y_max / 0.125) * 0.125
        
        all_grid_points = []
        current_x = x_start
        while current_x <= x_end + epsilon:
            current_y = y_start
            while current_y <= y_end + epsilon:
                all_grid_points.append((round(current_x, 3), round(current_y, 3)))
                current_y += 0.125
            current_x += 0.125
        
        return all_grid_points
    
    def create_cuboid(self, dimensions, position, orientation=None, color=None, mass=0, respondable=False, name="cuboid"):
        """Create a cuboid object in CoppeliaSim"""
        if orientation is None:
            orientation = [0, 0, 0]
        
        options = 8 if respondable else 0
        
        cuboid_handle = sim.createPrimitiveShape(
            sim.primitiveshape_cuboid,
            dimensions,
            options
        )
        
        sim.setObjectAlias(cuboid_handle, name)
        sim.setObjectPosition(cuboid_handle, -1, position)
        sim.setObjectOrientation(cuboid_handle, -1, orientation)
        
        if mass > 0:
            sim.setShapeMass(cuboid_handle, mass)
        
        if color is not None:
            sim.setShapeColor(cuboid_handle, None, 0, color)
        
        return cuboid_handle

