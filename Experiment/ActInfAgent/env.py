import numpy as np
import random
import time
import math
from scipy.special import comb
import sys
import os

# Flag to control simulation mode (no CoppeliaSim required)
SIMULATION_MODE = False

# Import CoppeliaSim interface properly using the RemoteAPIClient
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    # Create a client and get the sim object
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    print("Successfully connected to CoppeliaSim Remote API")
except ImportError:
    print("Failed to import coppeliasim_zmqremoteapi_client. Make sure the CoppeliaSim Remote API client is properly installed.")
    print("You may need to install it with: pip install coppeliasim-zmqremoteapi-client")
    # Provide a path to try to import from CoppeliaSim directory
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'CoppeliaSim'))
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient()
        sim = client.getObject('sim')
        print("Successfully connected to CoppeliaSim Remote API from CoppeliaSim directory")
    except (ImportError, Exception) as e:
        print(f"Error initializing CoppeliaSim interface: {e}")
        print("Make sure the CoppeliaSim is running and the Remote API server is started.")
        # Create a dummy sim object for testing without CoppeliaSim
        print("Creating dummy sim object for testing without CoppeliaSim")
        SIMULATION_MODE = True
        class DummySim:
            def __getattr__(self, name):
                def dummy_method(*args, **kwargs):
                    print(f"Dummy sim.{name} called with args: {args}, kwargs: {kwargs}")
                    return -1
                return dummy_method
        sim = DummySim()

# Define grid dimensions
grid_dims = [40, 40]

num_obstacles = 0
def move_to_grid(x, y):
    '''Moves coppelia coordinates (x,y) to a 40x40 grid, z coordinate remains constant, outputs coordinate in terms of grid'''
    
    # Translate x,y coordinate 2.5 up and 2.5 right
    x = x + 2.5
    y = y + 2.5
    
    # Ensure coordinates (x,y) are within (0,0) and (5,5)
    if x > 5 or x < 0:
        return "Invalid x coordinate!"
    elif y > 5 or y < 0:
        return "Invalid y coordinate!"
    
    # Convert x, y to grid indices by dividing by 0.05 (since each grid cell is 0.05 wide)
    x_grid = round(x / 0.125)
    y_grid = round(y / 0.125)
    
    # Ensure that the coordinates are within valid grid range (0 to 200)
    if x_grid > 40 or x_grid < 0:
        return "Invalid x grid point!"
    if y_grid > 40 or y_grid < 0:
        return "Invalid y grid point!"
    
    # Return the grid indices
    return (x_grid, y_grid)

    
def grid_to_coordinates(x_grid, y_grid):
    '''Converts a valid 200x200 grid point back into coppelia (x,y,z) coordinates in the range (x,y) = (0,0)-(5,5), z remains constant'''
    # Ensure the grid points are within valid range (0 to 200)
    if x_grid > 40 or x_grid < 0:
        return "Invalid x grid point!"
    if y_grid > 40 or y_grid < 0:
        return "Invalid y grid point!"
    
    # Reverse the grid index conversion by multiplying by 0.05
    x = x_grid * 0.125
    y = y_grid * 0.125
    
    x = x-2.5
    y = y-2.5
    # Return the original (x, y, z) coordinates
    return (x, y, 0.05)   

def get_object_position(object_name):
    # Step 2: Get the object handle by name
    objectHandle = sim.getObject(f'/{object_name}')
    
    if objectHandle == -1:
        raise Exception(f"Object '{object_name}' not found.")
    
    # Step 3: Get the position of the obstacle relative to the world (-1 means world reference)
    objectPosition = sim.getObjectPosition(objectHandle, -1)
    
    # Round each element in the position to the nearest thousandth
    roundedPosition = [round(coord, 3) for coord in objectPosition]
    
    print(f"Coppelia position of {object_name}: {roundedPosition}")
    return roundedPosition

def create_bounding_locations(position, dimensions):
    """
    Creates bounding locations for an object.
    
    Parameters:
    -----------
    position : tuple (x, y)
        Center position of the object
    dimensions : tuple (a, b, c)
        Dimensions of the object (width, depth, height)
        
    Returns:
    --------
    tuple
        Eight points: (top_right, bottom_left, top_left, bottom_right, mid_top, mid_bottom, mid_left, mid_right)
    """
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


        
def check_bounds(loc):
    '''Checks if a location is within the bounds of the grid'''
    
    x, y, z = loc
    if x > 5 or x < 0:
        return None
    elif y > 5 or y < 0:
        return None
    else:
        return loc


def create_cuboid(dimensions, position, orientation=None, color=None, mass=0, respondable=False, name="cuboid"):
    """
    Create a cuboid in CoppeliaSim with customizable properties using the ZMQ Remote API.
    
    Parameters:
    -----------
    dimensions : list/tuple
        [x, y, z] dimensions of the cuboid in meters
    position : list/tuple
        [x, y, z] position coordinates of the cuboid
    orientation : list/tuple, optional
        [alpha, beta, gamma] orientation in Euler angles (radians)
    color : list/tuple, optional
        [r, g, b] color components (0-1 range)
    mass : float, optional
        Mass of the cuboid in kg (0 = static object)
    respondable : bool, optional
        Whether the cuboid should be respondable (participate in dynamics)
    name : str, optional
        Name of the cuboid object
        
    Returns:
    --------
    int
        Handle to the created cuboid object
    """
    # Default orientation if none specified
    if orientation is None:
        orientation = [0, 0, 0]
    
    # Set the options flag based on parameters
    options = 0
    if respondable:
        options = options | 8  # bit 3 (8) = respondable
    
    # Create the cuboid primitive
    cuboid_handle = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid,  # shape type
        dimensions,  # size parameters [x, y, z]
        options  # options
    )
    
    # Set object name
    sim.setObjectAlias(cuboid_handle, name)
    
    # Set position
    sim.setObjectPosition(cuboid_handle, -1, position)
    
    # Set orientation
    sim.setObjectOrientation(cuboid_handle, -1, orientation)
    
    # Set mass if it's a dynamic object
    if mass > 0:
        sim.setShapeMass(cuboid_handle, mass)
    
    # Set color if specified
    if color is not None:
        # In the new API, we can set the color directly on the shape
        sim.setShapeColor(cuboid_handle, None, 0, color)  # 0 = ambient/diffuse color component
    
    return cuboid_handle



redspots = []
def initialize_environment(seed):
    global num_obstacles  # Declare we're using the global variable
    global redspots  # Ensure we're using the global redspots
    global SIMULATION_MODE
    
    # Clear redspots to avoid accumulation from previous runs
    redspots = []
    
    random.seed(seed)
    num_obstacles = random.randint(20, 50)
    print(f"Initializing environment with seed {seed} and {num_obstacles} obstacles")
    
    obstacle_dimensions = [0.25, 0.25, 0.8]  # Same dimensions for all obstacles
    
    def round_to_grid_precision(x):
        """Round a coordinate to the nearest 0.125 interval."""
        return round(x / 0.125) * 0.125
    
    def is_position_valid(new_x, new_y, existing_positions, dimensions):
        """
        Check if a new position is valid and not overlapping with existing positions.
        
        Parameters:
        -----------
        new_x : float
            X coordinate of the new position
        new_y : float
            Y coordinate of the new position
        existing_positions : list
            List of existing obstacle positions
        dimensions : list
            Dimensions of the obstacle
        
        Returns:
        --------
        bool
            True if position is valid, False otherwise
        """
        # First, ensure the center is at a 0.125 interval
        new_x = round_to_grid_precision(new_x)
        new_y = round_to_grid_precision(new_y)
        
        # Check environment bounds with obstacle dimensions
        half_width = dimensions[0] / 2
        half_depth = dimensions[1] / 2
        
        if (new_x - half_width < -2.5 or 
            new_x + half_width > 2.5 or 
            new_y - half_depth < -2.5 or 
            new_y + half_depth > 2.5):
            return False
        
        # Check overlap with existing obstacles
        for (ex, ey) in existing_positions:
            # Check for overlap using Axis-Aligned Bounding Box (AABB) collision detection
            if (abs(new_x - ex) < (dimensions[0] + obstacle_dimensions[0]) and 
                abs(new_y - ey) < (dimensions[1] + obstacle_dimensions[1])):
                return False
        
        return True
    
    def get_grid_points_in_obstacle(center_x, center_y, dimensions):
        """
        Get grid points within an obstacle and a single-cell buffer around its boundary.
        
        Parameters:
        -----------
        center_x : float
            X coordinate of obstacle center
        center_y : float
            Y coordinate of obstacle center
        dimensions : list
            Dimensions of the obstacle
        
        Returns:
        --------
        list
            List of grid coordinates within and immediately around the obstacle boundary
        """
        half_width = dimensions[0] / 2
        half_depth = dimensions[1] / 2
        
        # Define the basic bounds of the obstacle
        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_depth
        y_max = center_y + half_depth
        
        obstacle_points = []
        
        # First, collect all the grid points that fall within the obstacle itself
        # Use intervals that align with the grid
        x = x_min
        while x <= x_max + 0.0001:  # Small epsilon to include boundary
            y = y_min
            while y <= y_max + 0.0001:
                grid_coord = move_to_grid(x, y)
                if isinstance(grid_coord, tuple) and grid_coord not in obstacle_points:
                    obstacle_points.append(grid_coord)
                y += 0.125  # Align with grid intervals
            x += 0.125
        
        # Now add just a single-cell buffer around the existing points
        buffer_points = []
        for point in obstacle_points:
            x_grid, y_grid = point
            
            # Check all 8 adjacent cells (including diagonals)
            adjacents = [
                (x_grid-1, y_grid), (x_grid+1, y_grid),  # Left, Right
                (x_grid, y_grid-1), (x_grid, y_grid+1),  # Down, Up
                (x_grid-1, y_grid-1), (x_grid+1, y_grid+1),  # Diagonal: Bottom-left, Top-right
                (x_grid-1, y_grid+1), (x_grid+1, y_grid-1)   # Diagonal: Top-left, Bottom-right
            ]
            
            for adj in adjacents:
                # Ensure point is within grid bounds and not already added
                if (0 <= adj[0] < grid_dims[0] and 
                    0 <= adj[1] < grid_dims[1] and
                    adj not in obstacle_points and
                    adj not in buffer_points):
                    buffer_points.append(adj)
        
        # Combine the original obstacle points and the buffer points
        all_points = obstacle_points + buffer_points
        
        return all_points
    
    # Stores obstacle center positions
    obstacle_positions = []
    obstacle_handles = []
    
    # Create obstacles
    for i in range(num_obstacles):
        attempts = 0
        while attempts < 100:
            # Generate random position ensuring it aligns with 0.125 grid
            x = round_to_grid_precision(random.uniform(-2.25, 2.25))
            y = round_to_grid_precision(random.uniform(-2.25, 2.25))
            
            # Check if position is valid
            if is_position_valid(x, y, obstacle_positions, obstacle_dimensions):
                obstacle_positions.append((x, y))
                
                # Create obstacle in CoppeliaSim if not in simulation mode
                if not SIMULATION_MODE:
                    try:
                        obstacle_handle = create_cuboid(
                            dimensions=obstacle_dimensions,
                            position=[x, y, 0.4],  # z=0.4 is half the height
                            orientation=[0, 0, 0],
                            color=[1, 0, 0],  # Red color
                            mass=1,
                            respondable=True,
                            name=f"Obstacle{i}"
                        )
                        obstacle_handles.append(obstacle_handle)
                    except Exception as e:
                        print(f"Error creating obstacle: {e}")
                        obstacle_handles.append(i)  # Dummy handle
                else:
                    obstacle_handles.append(i)  # Dummy handle
                
                break
            
            attempts += 1
    
    # Add obstacle grid points to redspots
    for (x, y) in obstacle_positions:
        # Add obstacle center to redspots
        center_grid = move_to_grid(x, y)
        if isinstance(center_grid, tuple) and center_grid not in redspots:
            redspots.append(center_grid)
        
        # Add all grid points within the obstacle
        obstacle_grid_points = get_grid_points_in_obstacle(x, y, obstacle_dimensions)
        for point in obstacle_grid_points:
            if point not in redspots:
                redspots.append(point)
    
    # Find goal location
    goal_dimensions = [0.125, 0.125, 0.01]
    goal_position = None
    goal_handle = None
    
    while attempts < 100:
        # Generate goal position with 0.125 precision
        x = round_to_grid_precision(random.uniform(-2.25, 2.25))
        y = round_to_grid_precision(random.uniform(-2.25, 2.25))
        
        # Check if goal is far enough from obstacles
        is_valid_goal = True
        for (ox, oy) in obstacle_positions:
            if ((x - ox)**2 + (y - oy)**2)**0.5 < 0.375:  # At least 0.125 away
                is_valid_goal = False
                break
        
        if is_valid_goal:
            goal_position = (x, y)
            
            # Create goal in CoppeliaSim if not in simulation mode
            if not SIMULATION_MODE:
                try:
                    goal_handle = create_cuboid(
                        dimensions=goal_dimensions,
                        position=[x, y, 0.005],  # z=0.005 is half the height
                        orientation=[0, 0, 0],
                        color=[0, 1, 0],  # Green color
                        mass=0.1,
                        respondable=True,
                        name="Goal_Loc"
                    )
                except Exception as e:
                    print(f"Error creating goal: {e}")
                    goal_handle = num_obstacles  # Dummy handle
            else:
                goal_handle = num_obstacles  # Dummy handle
            
            break
        
        attempts += 1
    
    if goal_position is None:
        raise ValueError("Could not place goal within environment constraints")
    
    # Place bubbleRob
    bubbleRob_position = None
    attempts = 0
    while attempts < 100:
        # Generate bubbleRob position with 0.125 precision
        x = round_to_grid_precision(random.uniform(-2.25, 2.25))
        y = round_to_grid_precision(random.uniform(-2.25, 2.25))
        
        # Check if bubbleRob is far enough from obstacles and goal
        is_valid_pos = True
        for (ox, oy) in obstacle_positions:
            if ((x - ox)**2 + (y - oy)**2)**0.5 < 0.375:  # At least 0.125 away
                is_valid_pos = False
                break
        
        # Check distance from goal
        if is_valid_pos and goal_position:
            if ((x - goal_position[0])**2 + (y - goal_position[1])**2)**0.5 < 0.375:
                is_valid_pos = False
        
        if is_valid_pos:
            # Place bubbleRob in CoppeliaSim if not in simulation mode
            if not SIMULATION_MODE:
                try:
                    bubbleRob_handle = sim.getObject('/bubbleRob')
                    if bubbleRob_handle != -1:
                        sim.setObjectPosition(bubbleRob_handle, -1, [x, y, 0.12])
                    bubbleRob_position = (x, y, 0.12)
                except Exception as e:
                    print(f"Error placing bubbleRob: {e}")
                    bubbleRob_position = (x, y, 0.12)
            else:
                bubbleRob_position = (x, y, 0.12)
            
            break
        
        attempts += 1
    
    if bubbleRob_position is None:
        raise ValueError("Could not place bubbleRob within environment constraints")
    
    print(f"Total redspots collected: {len(redspots)}")
    print(redspots)
    
    # Flatten obstacle positions for the first return value
    flat_positions = [coord for pos in obstacle_positions for coord in pos]
    
    return flat_positions, goal_position, obstacle_handles, goal_handle, bubbleRob_position, redspots              



def clear_environment(obstacle_handles=None, goal_handle=None):
    """
    Clear all cuboids created by the initialize_environment function
    
    Parameters:
    -----------
    obstacle_handles : list, optional
        List of handles to obstacle objects
    goal_handle : int, optional
        Handle to the goal location object
    
    Returns:
    --------
    bool
        True if all objects were successfully removed
    """
    global num_obstacles  # Declare we're using the global variable
    success = True
    
    # Remove all obstacles if handles are provided
    if obstacle_handles:
        for i, handle in enumerate(obstacle_handles):
            try:
                if sim.isHandle(handle):  # Check if handle is valid
                    sim.removeObject(handle)
                    print(f"Removed Obstacle{i}")
            except Exception as e:
                print(f"Error removing Obstacle{i}: {e}")
                success = False
    
    # Remove goal location if handle is provided
    if goal_handle:
        try:
            if sim.isHandle(goal_handle):  # Check if handle is valid
                sim.removeObject(goal_handle)
                print("Removed Goal_Loc")
        except Exception as e:
            print(f"Error removing Goal_Loc: {e}")
            success = False
    
    # Alternative method: try to remove all objects by name
    # This is useful if handles are not available
    if not obstacle_handles and not goal_handle:
        # Try to remove obstacles by name
        for i in range(num_obstacles):  # Fixed the range to use the global variable
            try:
                object_handle = sim.getObject(f'/Obstacle{i}')
                if object_handle != -1:
                    sim.removeObject(object_handle)
                    print(f"Removed Obstacle{i} by name")
            except Exception as e:
                print(f"Error removing Obstacle{i} by name: {e}")
                success = False
        
        # Try to remove goal by name
        try:
            goal_object = sim.getObject('/Goal_Loc')
            if goal_object != -1:
                sim.removeObject(goal_object)
                print("Removed Goal_Loc by name")
        except Exception as e:
            print(f"Error removing Goal_Loc by name: {e}")
            success = False
    
    return success

def update_vision(current_location, grid_dims, distance):
    """
    Update the agent's field of vision based on the current location and distance
    Returns a list of all grid positions within the vision range
    
    Args:
        current_location (tuple): Current (x,y) position of the agent
        grid_dims (list): Dimensions of the grid [width, height]
        distance (int): Vision range/distance
        
    Returns:
        list: List of (x,y) tuples representing visible grid positions
    """
    x, y = current_location
    x_min = max(0, x - distance)
    x_max = min(grid_dims[0], x + distance + 1)
    y_min = max(0, y - distance)
    y_max = min(grid_dims[1], y + distance + 1)
    
    visible_locations = []
    for y_pos in range(y_min, y_max):
        for x_pos in range(x_min, x_max):
            visible_locations.append((x_pos, y_pos))
            
    return visible_locations

def monitor_position(bubbleRobHandle):
    pos1 = sim.getObjectPosition(bubbleRobHandle, -1)
    time.sleep(0.1)
    pos2 = sim.getObjectPosition(bubbleRobHandle, -1)
    print(f"Position change: {np.array(pos2) - np.array(pos1)}")
    
class CoppeliaEnv():
    def get_initial_object_properties(self):
        """Get initial position and orientation of bubbleRob"""
        try:
            initial_pos = sim.getObjectPosition(self.bubbleRobHandle, -1)
            initial_orient = sim.getObjectQuaternion(self.bubbleRobHandle, -1)
            return initial_pos[2], initial_orient
        except Exception as e:
            print(f"Warning: Could not get object properties: {e}")
            return 0.12, [0, 0, 0, 1]  # Default values

    def __init__(self, redspots, starting_loc, goal):
        # Get robot handle first and ensure it's valid
        try:
            self.bubbleRobHandle = sim.getObject('/bubbleRob')
            if self.bubbleRobHandle == -1:
                print("Warning: Could not get handle to bubbleRob, using simulation mode")
                self.simulation_mode = True
            else:
                self.simulation_mode = False
                
                try:
                    # Get bubbleRob position
                    bubbleRob_position = get_object_position('bubbleRob')
                    
                    # Create initial control points for bubbleRob path
                    self.ctrlPts = [[bubbleRob_position[0], bubbleRob_position[1], 0.05, 0.0, 0.0, 0.0, 1.0]]
                    self.ctrlPts_flattened = [coord for point in self.ctrlPts for coord in point]
                    
                    # Initialize path-related variables
                    self.initial_z, self.initial_orientation = self.get_initial_object_properties()
                except Exception as e:
                    print(f"Error initializing robot position: {e}")
                    self.simulation_mode = True
        except Exception as e:
            print(f"Error getting robot handle, using simulation mode: {e}")
            self.simulation_mode = True
        
        # If in simulation mode, create dummy values
        if hasattr(self, 'simulation_mode') and self.simulation_mode:
            # Create dummy values for simulation mode
            self.ctrlPts = [[0, 0, 0.05, 0.0, 0.0, 0.0, 1.0]]
            self.ctrlPts_flattened = [0, 0, 0.05, 0.0, 0.0, 0.0, 1.0]
            self.initial_z, self.initial_orientation = 0.12, [0, 0, 0, 1]
            self.bubbleRobHandle = -1

        # Initialize coordinates 
        self.x, self.y = starting_loc
        self.init_loc = starting_loc
        self.current_location = (self.x, self.y)
        self.goal = goal
        self.redspots = redspots
        
        # Initialize simulation variables
        self.posAlongPath = 0
        self.velocity = 0.08
        self.previousSimulationTime = sim.getSimulationTime() if not self.simulation_mode else 0
        
        # Initialize agent reward
        self.agent_reward = 0

    def bezier_recursive(self, ctrlPts, t):
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
        total_length = 0.0
        prev_point = self.bezier_recursive(ctrl_points, 0)
        
        for i in range(1, subdivisions + 1):
            t = i / subdivisions
            curr_point = self.bezier_recursive(ctrl_points, t)
            total_length += np.linalg.norm(curr_point - prev_point)
            prev_point = curr_point
        return total_length

    def get_point_and_tangent(self, t, ctrl_points):
        point = self.bezier_recursive(ctrl_points, t)
        
        delta = 0.001
        t_next = min(1.0, t + delta)
        next_point = self.bezier_recursive(ctrl_points, t_next)
        
        tangent = next_point - point
        if np.linalg.norm(tangent) > 0:
            tangent = tangent / np.linalg.norm(tangent)
        
        return point, tangent

    def update_orientation(self, position, tangent):
        if np.linalg.norm(tangent[:2]) > 0:  # Only use X and Y components
            yaw = np.arctan2(tangent[1], tangent[0])
            orientation_quaternion = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
            sim.setObjectQuaternion(self.bubbleRobHandle, -1, orientation_quaternion)

    def follow_path(self):
        # Skip if in simulation mode
        if self.simulation_mode:
            return
            
        # Reset position along path for new path
        self.posAlongPath = 0
        
        # Calculate total length for current control points
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
                print("Reached the end of the path!")
                break
            
            # Calculate normalized parameter
            t_norm = np.clip(self.posAlongPath / total_length, 0, 1)
            # Get current position and tangent
            current_pos, tangent = self.get_point_and_tangent(t_norm, self.ctrlPts_flattened)
            
            # Ensure Z coordinate
            current_pos[2] = self.initial_z
            
            # Update position and orientation
            sim.setObjectPosition(self.bubbleRobHandle, -1, current_pos.tolist())
            self.update_orientation(current_pos, tangent)
            
            self.previousSimulationTime = t
            sim.step()
            time.sleep(0.05)

    def step(self, action_label):
        # Try to start simulation if not in simulation mode
        if not self.simulation_mode:
            try:
                sim.startSimulation()
            except Exception as e:
                print(f"Warning: Could not start simulation: {e}")
                self.simulation_mode = True
        
        # Store previous position before movement
        prev_x, prev_y = self.x, self.y
        offset_x = 0
        offset_y = 0
        
        # Basic movement logic
        if action_label == "UP":
            self.y = max(0, self.y - 1)
            # For UP/DOWN movement, offset the x coordinate
            offset_x = offset_x + 0.02  # Curve to the right
        elif action_label == "DOWN":
            self.y = min(grid_dims[1] - 1, self.y + 1)
            offset_x = offset_x - 0.02  # Curve to the left
        elif action_label == "LEFT":
            self.x = max(0, self.x - 1)
            offset_y = offset_y + 0.02  # Curve upward
        elif action_label == "RIGHT":
            self.x = min(grid_dims[0] - 1, self.x + 1)
            offset_y = offset_y - 0.02  # Curve downward
            
        # Only perform physical simulation if not in simulation mode
        if not self.simulation_mode:
            try:
                # Add new control point
                new_coords = grid_to_coordinates(self.x, self.y)
                if isinstance(new_coords, tuple):  # Make sure we got valid coordinates
                    self.ctrlPts.append([
                        new_coords[0], 
                        new_coords[1],
                        0.05, 0.0, 0.0, 0.0, 1.0
                    ])
                    
                    # Update flattened control points
                    self.ctrlPts_flattened = [coord for point in self.ctrlPts for coord in point]
                    
                    # Create path
                    pathHandle = sim.createPath(
                        self.ctrlPts_flattened,
                        0,  # Options: open path
                        100,  # Subdivision for smoothness
                        0.5,  # No smoothness
                        0,  # Orientation mode
                        [0.0, 0.0, 1.0]  # Up vector
                    )
                    
                    # Follow the path
                    self.follow_path()
                    
                    # Clear control points for next step
                    self.ctrlPts.clear()
                    # Reset with current position
                    self.ctrlPts.append([
                        new_coords[0],
                        new_coords[1],
                        0.05, 0.0, 0.0, 0.0, 1.0
                    ])
            except Exception as e:
                print(f"Warning: Simulation step failed: {e}")
                self.simulation_mode = True
        
        # Update current location
        self.current_location = (self.x, self.y)
        
        print(f"Current location: {self.current_location}")
        
        # Update vision with current coordinates
        self.vision = update_vision(self.current_location, grid_dims, 6)

        self.loc_obs = self.current_location

        # Reset observations at each step
        self.red_obs = ['Null']
        self.white_obs = ['Null']
        self.green_obs = 'Null'

        # Update observations based on vision
        for spot in self.vision:
            if spot in self.redspots:
                if 'Null' in self.red_obs:
                    self.red_obs = [spot]
                else:
                    self.red_obs.append(spot)
            elif spot == self.goal:
                self.green_obs = spot
                
            else:
                if 'Null' in self.white_obs:
                    self.white_obs = [spot]
                else:
                    self.white_obs.append(spot)

        # Update rewards and observations based on current location
        if self.current_location in self.redspots:
            self.agent_reward -= 5
            if 'Null' in self.red_obs:
                self.red_obs = [self.current_location]
            else:
                self.red_obs.append(self.current_location)
        elif self.current_location == self.goal:
            self.agent_reward += 20
            self.green_obs = self.current_location
        else:
            if 'Null' in self.white_obs:
                self.white_obs = [self.current_location]
            else:
                self.white_obs.append(self.current_location)
        
        return self.loc_obs, self.green_obs, self.white_obs, self.red_obs, self.agent_reward
    
    def reset(self):
        self.x, self.y = self.init_loc
        self.current_location = (self.x, self.y)
        print(f'Re-initialized location to {self.current_location}')
        self.loc_obs = self.current_location
        self.green_obs, self.white_obs, self.red_obs, self.agent_reward = 'Null', ['Null'], ['Null'], 0
        
        # Initialize vision with current coordinates
        self.vision = update_vision(self.current_location, grid_dims, 6)

        # Update observations based on vision and redspots
        for spot in self.vision:
            if spot in self.redspots:
                if 'Null' in self.red_obs:
                    self.red_obs = [spot]
                else:
                    self.red_obs.append(spot)
            elif spot == self.goal:
                self.green_obs = spot
            else:
                if 'Null' in self.white_obs:
                    self.white_obs = [spot]
                else:
                    self.white_obs.append(spot)
        
        # Reset robot position in CoppeliaSim if not in simulation mode
        if not self.simulation_mode:
            try:
                coords = grid_to_coordinates(self.x, self.y)
                if isinstance(coords, tuple):  # Make sure we got valid coordinates
                    sim.setObjectPosition(self.bubbleRobHandle, -1, [coords[0], coords[1], self.initial_z])
                    sim.setObjectQuaternion(self.bubbleRobHandle, -1, self.initial_orientation)
                    
                    # Reset control points
                    self.ctrlPts = [[coords[0], coords[1], 0.05, 0.0, 0.0, 0.0, 1.0]]
                    self.ctrlPts_flattened = [coord for point in self.ctrlPts for coord in point]
            except Exception as e:
                print(f"Warning: Could not reset robot position: {e}")
                self.simulation_mode = True

        return self.loc_obs, self.green_obs, self.white_obs, self.red_obs, self.agent_reward



initialize_environment(4)