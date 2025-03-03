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
        # Updated observation space to include agent and goal coordinates (4 additional values)
        self.observation_space = spaces.Box(low=0, high=max(grid_dimensions), shape=(21,), dtype=np.float32)
        
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
        
        # Adding agent and goal coordinates to the state
        goal_x, goal_y = self.goal_position
        # Add agent's coordinates and goal's coordinates to the state
        coordinates_info = [
            float(agent_x), 
            float(agent_y), 
            float(goal_x), 
            float(goal_y)
        ]
        
        # Combine visible colors and coordinate information
        return visible_colors + coordinates_info

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

import time
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import CoppeliaSim libraries
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    COPPELIA_AVAILABLE = True
except ImportError:
    print("CoppeliaSim RemoteAPI not found. Running without simulation visualization.")
    COPPELIA_AVAILABLE = False

class CoppeliaSim:
    def __init__(self, random_seed=None, num_obstacles=None, grid_dimensions=[40, 40]):
        """
        Initialize CoppeliaSim environment
        
        Args:
            random_seed (int, optional): Seed for random number generation. If None, a random seed will be used.
            num_obstacles (int, optional): Default number of obstacles to create. If None, will be randomized based on seed.
            grid_dimensions (list, optional): Dimensions of the grid. Defaults to [40, 40].
        """
        # Set random seed
        self.random_seed = random_seed if random_seed is not None else int(time.time())
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        print(f"Using random seed: {self.random_seed}")
        
        # Generate a deterministic number of obstacles based on random_seed if not provided
        if num_obstacles is None:
            # Using a seeded random number to get a repeatable obstacle count between 20 and 50
            r = random.Random(self.random_seed)
            self.num_obstacles = r.randint(20, 50)
            print(f"Randomized number of obstacles: {self.num_obstacles} (based on seed: {self.random_seed})")
        else:
            self.num_obstacles = num_obstacles
            print(f"Using specified number of obstacles: {self.num_obstacles}")
        
        # Define observation space and action space for RL compatibility
        self.grid_dimensions = grid_dimensions
        self.action_space = spaces.Discrete(5)  # UP, DOWN, LEFT, RIGHT, STAY
        self.observation_space = spaces.Box(low=0, high=max(grid_dimensions), shape=(21,), dtype=np.float32)
        
        # Initialize agent state tracking
        self.agent_grid_position = (0, 0)  # Default initial position in grid coordinates
        self.initial_grid_position = (0, 0)
        self.total_steps = 0
        # Increase max_steps to allow more time to reach the goal in large environments with many obstacles
        self.max_steps = 50  # Increased from 12 to 50 to accommodate larger environment
        
        # Connect to CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        
        # Try to get the bubbleRob handle
        try:
            self.bubbleRobHandle = self.sim.getObject('/bubbleRob')
            print("Connected to bubbleRob")
        except Exception as e:
            print(f"Error connecting to bubbleRob: {e}")
            
        # Get initial position properties
        try:
            self.initial_position = self.sim.getObjectPosition(self.bubbleRobHandle, -1)
            self.initial_z = self.initial_position[2]
            self.initial_orientation = self.sim.getObjectQuaternion(self.bubbleRobHandle, -1)
        except:
            self.initial_position = [0, 0, 0.05]
            self.initial_z = 0.05
            self.initial_orientation = [0, 0, 0, 1]
            
        # Create path following properties
        self.ctrlPts = [[self.initial_position[0], self.initial_position[1], self.initial_z, 0.0, 0.0, 0.0, 1.0]]
        self.ctrlPts_flattened = [coord for point in self.ctrlPts for coord in point]
        self.posAlongPath = 0
        self.velocity = 0.08
        self.previousSimulationTime = 0
        
        # Initialize the environment (obstacles will be created later)
        self.red_zone_positions = []
        self.redspots_coppelia = []  # Store CoppeliaSim coordinates of red zones
        self.goal_position = None
        
    def create_bounding_locations(self, position, dimensions):
        """Create bounding locations for collision detection"""
        (x, y) = position
        (a, b, c) = dimensions
        
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
    
    def is_position_valid(self, new_x, new_y, object_dimensions):
        """
        Check if a new obstacle position would overlap with existing obstacles
        
        Args:
            new_x (float): X coordinate of the new obstacle
            new_y (float): Y coordinate of the new obstacle
            object_dimensions (list/tuple): Dimensions [x, y, z] of the object
            
        Returns:
            bool: True if position is valid (no overlaps), False otherwise
        """
        # Convert position and dimensions to format needed for bounding locations
        new_position = (new_x, new_y)
        
        # Get bounding points for the new obstacle
        new_bounds = self.create_bounding_locations(new_position, object_dimensions)
        new_top_right, new_bottom_left, _, _, _, _, _, _ = new_bounds
        
        # Check if the new obstacle would be inside the room bounds
        if (new_x + object_dimensions[0]/2 > 2.5 or 
            new_x - object_dimensions[0]/2 < -2.5 or
            new_y + object_dimensions[1]/2 > 2.5 or
            new_y - object_dimensions[1]/2 < -2.5):
            return False
        
        # Check against all existing obstacles
        for pos in self.redspots_coppelia:
            existing_x, existing_y = pos[0], pos[1]
            existing_position = (existing_x, existing_y)
            
            # Get bounding points for existing obstacle
            existing_bounds = self.create_bounding_locations(existing_position, object_dimensions)
            existing_top_right, existing_bottom_left, _, _, _, _, _, _ = existing_bounds
            
            # Check for overlap using AABB collision detection
            # If one rectangle is to the left of the other
            if (new_top_right[0] < existing_bottom_left[0] or 
                existing_top_right[0] < new_bottom_left[0]):
                continue
                
            # If one rectangle is above the other
            if (new_top_right[1] < existing_bottom_left[1] or 
                existing_top_right[1] < new_bottom_left[1]):
                continue
                
            # If we get here, the rectangles overlap
            return False
            
        # If we've checked all obstacles and found no overlaps
        return True
        
    def is_bubbleRob_position_valid(self, new_x, new_y):
        """
        Check if a position for bubbleRob is valid (no overlaps with obstacles or goal)
        
        Args:
            new_x (float): X coordinate for bubbleRob
            new_y (float): Y coordinate for bubbleRob
            
        Returns:
            bool: True if position is valid (no overlaps), False otherwise
        """
        # BubbleRob dimensions (approximate)
        bubbleRob_dimensions = [0.2, 0.2, 0.2]
        
        # Check if within bounds (-2 to 2 as specified)
        if new_x > 2 or new_x < -2 or new_y > 2 or new_y < -2:
            return False
        
        # Create bubbleRob position and bounds
        new_position = (new_x, new_y)
        new_bounds = self.create_bounding_locations(new_position, bubbleRob_dimensions)
        new_top_right, new_bottom_left, _, _, _, _, _, _ = new_bounds
        
        # Check against obstacles
        obstacle_dimensions = [0.3, 0.3, 0.8]
        for pos in self.redspots_coppelia:
            existing_x, existing_y = pos[0], pos[1]
            existing_position = (existing_x, existing_y)
            
            # Get bounding points for existing obstacle
            existing_bounds = self.create_bounding_locations(existing_position, obstacle_dimensions)
            existing_top_right, existing_bottom_left, _, _, _, _, _, _ = existing_bounds
            
            # Check for overlap using AABB collision detection
            if not (new_top_right[0] < existing_bottom_left[0] or 
                    existing_top_right[0] < new_bottom_left[0] or
                    new_top_right[1] < existing_bottom_left[1] or 
                    existing_top_right[1] < new_bottom_left[1]):
                # Overlap detected
                return False
                
        # Check against goal if it exists
        if hasattr(self, 'goal_coppelia_position') and self.goal_coppelia_position:
            goal_x, goal_y = self.goal_coppelia_position[0], self.goal_coppelia_position[1]
            goal_pos = (goal_x, goal_y)
            flat_object_dimensions = [0.3, 0.3, 0.01]
            
            # Get bounding points for goal
            goal_bounds = self.create_bounding_locations(goal_pos, flat_object_dimensions)
            goal_top_right, goal_bottom_left, _, _, _, _, _, _ = goal_bounds
            
            # Check for overlap with goal
            if not (new_top_right[0] < goal_bottom_left[0] or 
                    goal_top_right[0] < new_bottom_left[0] or
                    new_top_right[1] < goal_bottom_left[1] or 
                    goal_top_right[1] < new_bottom_left[1]):
                # Overlap detected
                return False
        
        # If we get here, position is valid
        return True
    
    def create_cuboid(self, dimensions, position, orientation=None, color=None, mass=0, respondable=False, name="cuboid"):
        """Create a cuboid in CoppeliaSim with customizable properties"""
        # Default orientation if none specified
        if orientation is None:
            orientation = [0, 0, 0]
        
        # Set the options flag based on parameters
        options = 0
        if respondable:
            options = options | 8  # bit 3 (8) = respondable
        
        # Create the cuboid primitive
        cuboid_handle = self.sim.createPrimitiveShape(
            self.sim.primitiveshape_cuboid,  # shape type
            dimensions,  # size parameters [x, y, z]
            options  # options
        )
        
        # Set object name
        self.sim.setObjectAlias(cuboid_handle, name)
        
        # Set position
        self.sim.setObjectPosition(cuboid_handle, -1, position)
        
        # Set orientation
        self.sim.setObjectOrientation(cuboid_handle, -1, orientation)
        
        # Set mass if it's a dynamic object
        if mass > 0:
            self.sim.setShapeMass(cuboid_handle, mass)
        
        # Set color if specified
        if color is not None:
            # In the new API, we can set the color directly on the shape
            self.sim.setShapeColor(cuboid_handle, None, 0, color)  # 0 = ambient/diffuse color component
        
        return cuboid_handle
    
    def initialize_environment(self):
        """Create random obstacles in CoppeliaSim
        
        Returns:
            tuple: (red_zone_positions, goal_position)
        """
        print(f"Initializing environment with {self.num_obstacles} obstacles...")
        
        # Re-seed random number generator for consistent obstacle placement based on seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
            
        # Clear previous obstacles if any
        try:
            # Try to remove any existing obstacles
            for i in range(max(50, self.num_obstacles)):  # Ensure we clean up enough obstacles (up to 50)
                try:
                    obstacle_handle = self.sim.getObject(f'/Obstacle{i}')
                    self.sim.removeObject(obstacle_handle)
                except:
                    pass
            
            # Try to remove goal location
            try:
                goal_handle = self.sim.getObject('/Goal_Loc')
                self.sim.removeObject(goal_handle)
            except:
                pass
        except:
            pass
            
        positions = []
        self.redspots_coppelia = []
        self.red_zone_positions = []
        obstacle_dimensions = [0.3, 0.3, 0.8]  # Same dimensions for all obstacles
        
        # Create random obstacles as specified by num_obstacles
        for i in range(self.num_obstacles):
            # Try to find a valid position (up to 100 attempts)
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < 100:
                # Generate random position with limits (within -2.5 to 2.5 range, accounting for obstacle size)
                x = round(random.uniform(-2.5+(0.3+0.01), 2.5-(0.3+0.01)), 2)
                y = round(random.uniform(-2.5+(0.3+0.01), 2.5-(0.3+0.01)), 2)
                
                # Check if this position is valid (no overlaps)
                valid_position = self.is_position_valid(x, y, obstacle_dimensions)
                attempts += 1
            
            if valid_position:
                # Store CoppeliaSim coordinates
                self.redspots_coppelia.append((x, y))
                
                # Convert to grid coordinates
                grid_pos = self.move_to_grid(x, y)
                if isinstance(grid_pos, tuple):  # Check if conversion was successful
                    self.red_zone_positions.append(grid_pos)
                
                # Create the obstacle in CoppeliaSim
                self.create_cuboid(
                    dimensions=obstacle_dimensions,
                    position=[x, y, 0.4],
                    orientation=[0, 0, 0],
                    color=[1, 0, 0],  # Red color
                    mass=1,
                    respondable=True,
                    name=f"Obstacle{i}"
                )
                print(f"Created obstacle {i} at CoppeliaSim:({x}, {y}), Grid:({grid_pos})")
            else:
                print(f"Could not find valid position for Obstacle{i} after {attempts} attempts")
                
        # Report obstacle creation statistics
        print(f"Successfully created {len(self.redspots_coppelia)} obstacles out of {self.num_obstacles} requested")
        
        # Create a random goal position
        flat_object_dimensions = [0.3, 0.3, 0.01]
        valid_position = False
        attempts = 0
        goal_x, goal_y = 0, 0
        
        while not valid_position and attempts < 100:
            goal_x = round(random.uniform(-2.5+(0.3+0.01), 2.5-(0.3+0.01)), 2)
            goal_y = round(random.uniform(-2.5+(0.3+0.01), 2.5-(0.3+0.01)), 2)
            
            # Check if goal is valid (far enough from obstacles)
            valid_position = self.is_position_valid(goal_x, goal_y, flat_object_dimensions)
            attempts += 1
        
        if valid_position:
            # Store the goal's CoppeliaSim position for future collision checks
            self.goal_coppelia_position = [goal_x, goal_y]
            
            # Create goal
            self.create_cuboid(
                dimensions=flat_object_dimensions,
                position=[goal_x, goal_y, 0.005],  # Lower Z height for the goal
                orientation=[0, 0, 0],
                color=[0, 1, 0],  # Green color
                mass=1,
                respondable=True,
                name="Goal_Loc"
            )
            
            # Convert goal to grid coordinates
            goal_grid = self.move_to_grid(goal_x, goal_y)
            if isinstance(goal_grid, tuple):
                self.goal_position = goal_grid
                print(f"Created goal at CoppeliaSim:({goal_x}, {goal_y}), Grid:({goal_grid})")
        else:
            print(f"Could not find valid position for goal after {attempts} attempts")
            # Use a fallback position if needed
            goal_x, goal_y = 2.0, 2.0
            self.goal_coppelia_position = [goal_x, goal_y]
            self.goal_position = self.move_to_grid(goal_x, goal_y)
        
        return self.red_zone_positions, self.goal_position
    
    def visualize_environments_mapping(self):
        """Create a visualization showing both CoppeliaSim and GridWorld environments"""
        # Commented out visualization code - can be re-enabled if needed
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # CoppeliaSim visualization
        ax1.set_title("CoppeliaSim Environment")
        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylim(-2.5, 2.5)
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # Draw obstacles
        for x, y in self.redspots_coppelia:
            ax1.add_patch(Rectangle((x-0.15, y-0.15), 0.3, 0.3, color='red', alpha=0.7))
        
        # Draw goal if it exists
        if hasattr(self, 'goal_position'):
            goal_x, goal_y, _ = self.grid_to_coordinates(self.goal_position[0], self.goal_position[1])
            ax1.add_patch(Rectangle((goal_x-0.15, goal_y-0.15), 0.3, 0.3, color='green', alpha=0.7))
        
        # Draw agent
        ax1.plot(self.initial_position[0], self.initial_position[1], 'bo', markersize=10)
        
        # Grid visualization
        ax2.set_title("GridWorld Environment")
        ax2.set_xlim(0, 40)
        ax2.set_ylim(0, 40)
        ax2.grid(True)
        ax2.set_aspect('equal')
        
        # Draw obstacles
        for x, y in self.red_zone_positions:
            ax2.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, color='red', alpha=0.7))
        
        # Draw goal if it exists
        if hasattr(self, 'goal_position'):
            ax2.add_patch(Rectangle((self.goal_position[0]-0.5, self.goal_position[1]-0.5), 1, 1, color='green', alpha=0.7))
        
        # Draw agent
        if hasattr(self, 'bubbleRobHandle'):
            agent_pos = self.move_to_grid(self.initial_position[0], self.initial_position[1])
            if isinstance(agent_pos, tuple):
                ax2.plot(agent_pos[0], agent_pos[1], 'bo', markersize=10)
        
        plt.tight_layout()
        plt.savefig('environment_mapping.png')
        print("Environment mapping visualization saved to 'environment_mapping.png'")
        plt.show()
        """
        print("Visualization disabled - using CoppeliaSim's visualization instead")
    
    def move_to_grid(self, x, y):
        """Moves CoppeliaSim coordinates (x,y) to a 40x40 grid, z coordinate remains constant"""
        # Translate x,y coordinate 2.5 up and 2.5 right
        x = x + 2.5
        y = y + 2.5
        
        # Ensure coordinates (x,y) are within (0,0) and (5,5)
        if x > 5 or x < 0:
            return "Invalid x coordinate!"
        elif y > 5 or y < 0:
            return "Invalid y coordinate!"
        
        # Convert x, y to grid indices by dividing by 0.125 (since each grid cell is 0.125 wide)
        x_grid = round(x / 0.125)
        y_grid = round(y / 0.125)
        
        # Ensure that the coordinates are within valid grid range (0 to 40)
        if x_grid > 40 or x_grid < 0:
            return "Invalid x grid point!"
        if y_grid > 40 or y_grid < 0:
            return "Invalid y grid point!"
        
        # Return the grid indices
        return (x_grid, y_grid)
    
    def grid_to_coordinates(self, x_grid, y_grid):
        """Converts a valid 40x40 grid point back into CoppeliaSim (x,y,z) coordinates"""
        # Ensure the grid points are within valid range (0 to 40)
        if x_grid > 40 or x_grid < 0:
            return "Invalid x grid point!"
        if y_grid > 40 or y_grid < 0:
            return "Invalid y grid point!"
        
        # Convert grid indices to world coordinates
        x = x_grid * 0.125
        y = y_grid * 0.125
        
        # Translate back to CoppeliaSim coordinates
        x = x - 2.5
        y = y - 2.5
        
        # Return the original (x, y, z) coordinates
        return (x, y, 0.05)
    
    def bezier_recursive(self, ctrlPts, t):
        """Calculate point on Bezier curve at parameter t"""
        from math import comb
        n = (len(ctrlPts) // 7) - 1
        point = np.zeros(3)
        total_weight = 0
        
        for i in range(n + 1):
            binomial_coeff = comb(n, i)
            weight = binomial_coeff * ((1 - t) ** (n - i)) * (t ** i)
            point_coords = np.array(ctrlPts[i * 7:i * 7 + 3])
            point += weight * point_coords
            total_weight += weight
        
        # Normalize point to ensure precise interpolation
        if total_weight > 0:
            point = point / total_weight
        
        point[2] = self.initial_z  # Set exact Z coordinate
        return point
    
    def calculate_total_length(self, ctrl_points, subdivisions=1000):
        """Calculate total path length with more precise sampling"""
        total_length = 0.0
        prev_point = self.bezier_recursive(ctrl_points, 0)
        for i in range(1, subdivisions + 1):
            t = i / subdivisions
            curr_point = self.bezier_recursive(ctrl_points, t)
            total_length += np.linalg.norm(curr_point - prev_point)
            prev_point = curr_point
        return total_length
    
    def get_point_and_tangent(self, t, ctrl_points):
        """Calculate point and tangent at given parameter"""
        # Get current point
        point = self.bezier_recursive(ctrl_points, t)
        
        # Calculate tangent using small delta
        delta = 0.001
        t_next = min(1.0, t + delta)
        next_point = self.bezier_recursive(ctrl_points, t_next)
        
        tangent = next_point - point
        if np.linalg.norm(tangent) > 0:
            tangent = tangent / np.linalg.norm(tangent)
        
        return point, tangent
    
    def update_orientation(self, position, tangent):
        """Update orientation based on path tangent"""
        if np.linalg.norm(tangent[:2]) > 0:  # Only use X and Y components
            yaw = np.arctan2(tangent[1], tangent[0])
            orientation_quaternion = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
            self.sim.setObjectQuaternion(self.bubbleRobHandle, -1, orientation_quaternion)
    
    def follow_path(self, ctrl_points):
        """Follow path with precise path following"""
        # Calculate total length for current control points
        total_length = self.calculate_total_length(ctrl_points)
        # Reset position along path for new path
        self.posAlongPath = 0
        
        self.previousSimulationTime = self.sim.getSimulationTime()
        
        while self.posAlongPath < total_length:
            t = self.sim.getSimulationTime()
            deltaT = t - self.previousSimulationTime
            
            if deltaT <= 0.0:
                self.previousSimulationTime = t
                continue
            
            self.posAlongPath += self.velocity * deltaT
            
            if self.posAlongPath >= total_length - 0.001:
                self.posAlongPath = total_length
                print("Reached the end of the path!")
                break
            
            # Calculate normalized parameter
            t_norm = np.clip(self.posAlongPath / total_length, 0, 1)
            # Get current position and tangent
            current_pos, tangent = self.get_point_and_tangent(t_norm, ctrl_points)
            
            # Ensure Z coordinate
            current_pos[2] = self.initial_z
            
            # Update position and orientation
            self.sim.setObjectPosition(self.bubbleRobHandle, -1, current_pos.tolist())
            self.update_orientation(current_pos, tangent)
            
            self.previousSimulationTime = t
            self.sim.step()
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def move_agent(self, action):
        """Move agent in CoppeliaSim based on RL agent action"""
        # Start simulation if not already running
        try:
            self.sim.startSimulation()
        except:
            pass
            
        # Get current position
        current_pos = self.sim.getObjectPosition(self.bubbleRobHandle, -1)
        
        # Calculate target position based on action
        # Action mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        target_x, target_y = current_pos[0], current_pos[1]
        step_size = 0.125  # Grid cell size
        
        # Calculate offsets for curved paths
        offset_x, offset_y = 0, 0
        
        if action == 0:  # UP
            target_y -= step_size
            offset_x = 0.02  # Curve slightly
        elif action == 1:  # DOWN
            target_y += step_size
            offset_x = -0.02  # Curve slightly
        elif action == 2:  # LEFT
            target_x -= step_size
            offset_y = 0.02  # Curve slightly
        elif action == 3:  # RIGHT
            target_x += step_size
            offset_y = -0.02  # Curve slightly
        # STAY action doesn't change position
        
        if action != 4:  # Not STAY
            # Create control points for the path
            # Start with current position
            ctrl_points = [[current_pos[0], current_pos[1], self.initial_z, 0.0, 0.0, 0.0, 1.0]]
            
            # Add midpoint with offset for curve
            # midpoint_x = (current_pos[0] + target_x) / 2 + offset_x
            # midpoint_y = (current_pos[1] + target_y) / 2 + offset_y
            # ctrl_points.append([midpoint_x, midpoint_y, self.initial_z, 0.0, 0.0, 0.0, 1.0])
            
            # Add target position
            ctrl_points.append([target_x, target_y, self.initial_z, 0.0, 0.0, 0.0, 1.0])
            
            # Flatten control points for path creation
            ctrl_points_flattened = [coord for point in ctrl_points for coord in point]
            
            # Create path and follow it
            try:
                # Create the path
                pathHandle = self.sim.createPath(
                    ctrl_points_flattened,
                    0,  # Options: open path
                    100,  # Subdivision for smoothness
                    1,  # No smoothness
                    0,  # Orientation mode
                    [0.0, 0.0, 1.0]  # Up vector
                )
                
                # Follow the path
                self.follow_path(ctrl_points_flattened)
                
                # Remove the path
                self.sim.removeObject(pathHandle)
            except Exception as e:
                print(f"Error during path following: {e}")
    
    def reset_agent_position(self, grid_pos=(0, 0)):
        """Reset agent position to the specified grid position"""
        world_x, world_y, world_z = self.grid_to_coordinates(grid_pos[0], grid_pos[1])
        if isinstance(world_x, str):  # Check if conversion error occurred
            print(f"Error converting grid position: {grid_pos}")
            return
            
        try:
            self.sim.setObjectPosition(self.bubbleRobHandle, -1, [world_x, world_y, world_z])
            print(f"Reset agent position to grid {grid_pos}, world coordinates: {[world_x, world_y, world_z]}")
        except Exception as e:
            print(f"Error resetting agent position: {e}")
    
    def reset(self):
        """
        Reset the environment to initial state. Compatible with Gym interface.
        
        Returns:
            numpy.ndarray: Initial state observation
        """
        # If environment not initialized yet, do that first
        if not self.red_zone_positions or not self.goal_position:
            self.initialize_environment(self.num_obstacles)
        
        # Re-seed random generator for agent position
        r = random.Random(self.random_seed + self.total_steps)  # Use steps to vary position on resets
            
        # Generate a valid initial position for the agent
        valid_position = False
        attempts = 0
        initial_x, initial_y = 0, 0
        
        while not valid_position and attempts < 100:
            # Generate random position within safe range
            initial_x = round(r.uniform(-2.0, 2.0), 2)
            initial_y = round(r.uniform(-2.0, 2.0), 2)
            
            # Check if position is valid
            valid_position = self.is_bubbleRob_position_valid(initial_x, initial_y)
            attempts += 1
            
        # If we couldn't find a valid position, use a safe default
        if not valid_position:
            initial_x, initial_y = 0, 0  # Center position, might need adjustment
            
        # Reset agent position in CoppeliaSim
        self.sim.setObjectPosition(self.bubbleRobHandle, -1, [initial_x, initial_y, self.initial_z])
        
        # Store the initial grid position
        grid_pos = self.move_to_grid(initial_x, initial_y)
        if isinstance(grid_pos, tuple):
            self.agent_grid_position = grid_pos
            self.initial_grid_position = grid_pos
        else:
            # Fallback to a default position if conversion failed
            self.agent_grid_position = (0, 0)
            self.initial_grid_position = (0, 0)
            
        # Reset step counter
        self.total_steps = 0
        
        # Return initial state observation
        return np.array(self.get_state(), dtype=np.float32)
        
    def step(self, action):
        """
        Take an action in the environment. Compatible with Gym interface.
        
        Args:
            action (int): Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY)
            
        Returns:
            tuple: (new_state, reward, done, info)
        """
        # Store current position before moving
        old_pos = self.agent_grid_position
        old_world_pos = self.sim.getObjectPosition(self.bubbleRobHandle, -1)
        
        # Execute action in the simulation
        self.move_agent(action)
        
        # Get new position from CoppeliaSim
        new_world_pos = self.sim.getObjectPosition(self.bubbleRobHandle, -1)
        
        # Convert to grid position
        new_grid_pos = self.move_to_grid(new_world_pos[0], new_world_pos[1])
        if isinstance(new_grid_pos, tuple):
            self.agent_grid_position = new_grid_pos
        else:
            # If conversion failed, don't update grid position
            new_grid_pos = old_pos
            
        # Increment step counter
        self.total_steps += 1
        
        # Calculate reward and check if episode is done
        reward, done = self.calculate_reward(old_pos, new_grid_pos, action)
        
        # Get new state
        new_state = self.get_state()
        
        # Additional info
        info = {
            'steps': self.total_steps,
            'old_position': old_pos,
            'new_position': new_grid_pos,
            'goal_position': self.goal_position,
            'manhattan_distance': abs(new_grid_pos[0] - self.goal_position[0]) + 
                                 abs(new_grid_pos[1] - self.goal_position[1])
        }
        
        return np.array(new_state, dtype=np.float32), reward, done, info
    
    def get_state(self):
        """
        Get the current state observation.
        
        Returns:
            list: State observation including visible cells and coordinates
        """
        # Current agent position
        agent_x, agent_y = self.agent_grid_position
        
        # Initialize visible colors for the 17 cells in vision pattern
        visible_colors = []
        
        # Vision pattern similar to GridWorld
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
            x_pos = max(0, min(self.grid_dimensions[0] - 1, agent_x + dx))
            y_pos = max(0, min(self.grid_dimensions[1] - 1, agent_y + dy))
            
            if (x_pos, y_pos) in self.red_zone_positions:
                color = 1.0  # Red
            elif (x_pos, y_pos) == self.goal_position:
                color = 2.0  # Green (goal)
            else:
                color = 0.0  # White
            visible_colors.append(color)
            
        # Add agent and goal coordinates
        goal_x, goal_y = self.goal_position
        coordinates_info = [
            float(agent_x), 
            float(agent_y), 
            float(goal_x), 
            float(goal_y)
        ]
        
        # Full state observation
        return visible_colors + coordinates_info
    
    def calculate_reward(self, old_position, new_position, action):
        """
        Calculate reward and determine if the episode is done.
        
        Args:
            old_position (tuple): Previous grid position (x, y)
            new_position (tuple): Current grid position (x, y)
            action (int): Action taken
            
        Returns:
            tuple: (reward, done)
        """
        # Calculate manhattan distance to goal
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
            steps_bonus = max(0, 30 - self.total_steps) * 5  # Adjusted bonus calculation for longer episodes
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
        
        # Timeout penalty - adjusted for increased max steps
        if self.total_steps >= self.max_steps:
            reward = -30.0
            done = True
            
        return reward, done
    
    def render(self, show=True):
        """
        Render the environment. CoppeliaSim already provides visualization,
        but this method creates a simplified top-down view.
        
        Args:
            show (bool): Whether to show the plot immediately
        """
        # Commented out visualization code - can be re-enabled if needed
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw room boundaries
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title("CoppeliaSim Environment")
        ax.grid(True)
        
        # Draw red zones (obstacles)
        for x, y in self.redspots_coppelia:
            ax.add_patch(Rectangle((x-0.15, y-0.15), 0.3, 0.3, color='red', alpha=0.7))
        
        # Draw goal
        if hasattr(self, 'goal_coppelia_position'):
            goal_x, goal_y = self.goal_coppelia_position
            ax.add_patch(Rectangle((goal_x-0.15, goal_y-0.15), 0.3, 0.3, color='green', alpha=0.7))
        
        # Get current agent position
        try:
            agent_pos = self.sim.getObjectPosition(self.bubbleRobHandle, -1)
            ax.plot(agent_pos[0], agent_pos[1], 'bo', markersize=10)
        except:
            pass
        
        # Display information
        plt.figtext(0.02, 0.02, f'Steps: {self.total_steps}/{self.max_steps}', 
                   color='black', backgroundcolor='white', fontsize=10)
        plt.figtext(0.02, 0.05, f'Grid Position: {self.agent_grid_position}', 
                   color='black', backgroundcolor='white', fontsize=10)
        plt.figtext(0.02, 0.08, f'Goal Position: {self.goal_position}', 
                   color='black', backgroundcolor='white', fontsize=10)
        
        # Save or show
        if show:
            plt.show()
        else:
            plt.close()
        """
        print(f"Position: {self.agent_grid_position}, Goal: {self.goal_position}, Steps: {self.total_steps}/{self.max_steps}")
            
    def close(self):
        """Close the environment and stop the simulation"""
        try:
            self.sim.stopSimulation()
            print("CoppeliaSim simulation stopped")
        except:
            pass

def custom_render(env, coppelia_env=None):
    """Create a custom visualization showing both GridWorld and CoppeliaSim environments"""
    # Commented out visualization code - can be re-enabled if needed
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    grid = np.zeros((env.grid_dimensions[1], env.grid_dimensions[0]))  # Note the order: (rows, cols)
    
    # Set red zones
    for x, y in env.red_zone_positions:
        grid[y, x] = 1  # Swapped to [y,x] for correct visualization
    
    # Set goal (green)
    goal_x, goal_y = env.goal_position
    grid[goal_y, goal_x] = 2  # Swapped to [y,x] for correct visualization
    
    # Set agent position
    grid[env.agent_y, env.agent_x] = 3  # Swapped to [y,x] for correct visualization
    
    # Create custom colormap: white -> red -> green -> blue
    colors = ['white', 'red', 'lime', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    plt.imshow(grid, cmap=cmap)
    plt.title('Grid World Environment')
    
    # Add additional information
    plt.figtext(0.02, 0.02, f'Agent position: ({env.agent_x}, {env.agent_y})', 
                color='black', backgroundcolor='white', fontsize=10)
    plt.figtext(0.02, 0.05, f'Goal position: {env.goal_position}', 
                color='black', backgroundcolor='white', fontsize=10)
    plt.figtext(0.02, 0.08, f'Red zones: {env.red_zone_positions}', 
                color='black', backgroundcolor='white', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    """
    print(f"Custom rendering disabled - using CoppeliaSim's visualization instead")