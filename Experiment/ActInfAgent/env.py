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

def get_all_grid_points_in_obstacle(position, dimensions):
    """
    Get ALL grid points within and on the boundaries of an obstacle.
    
    Parameters:
    -----------
    position : tuple (x, y)
        Center position of the obstacle
    dimensions : tuple (a, b, c)
        Dimensions of the obstacle (width, depth, height)
        
    Returns:
    --------
    list
        All points to be converted to grid coordinates within and on the obstacle
    """
    (x, y) = position
    (a, b, c) = dimensions
    
    # Calculate the exact boundaries of the obstacle
    x_min = x - a/2
    x_max = x + a/2
    y_min = y - b/2
    y_max = y + b/2
    
    # Precision for floating-point comparisons
    epsilon = 1e-6
    
    all_points = []
    
    # Comprehensive point generation
    current_x = x_min
    while current_x <= x_max + epsilon:
        current_y = y_min
        while current_y <= y_max + epsilon:
            all_points.append((current_x, current_y))
            current_y += 0.125
        current_x += 0.125
    
    return all_points
        
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
# Random positional value between 0+dim[index of coordinate(x = 0,y=1)] and 5-dim[index of coordinate]
def initialize_environment(seed):
    global num_obstacles  # Declare we're using the global variable
    global redspots  # Ensure we're using the global redspots
    global SIMULATION_MODE
    
    # Clear redspots to avoid accumulation from previous runs
    redspots = []
    
    random.seed(seed)
    num_obstacles = random.randint(20, 50)
    print(f"Initializing environment with seed {seed} and {num_obstacles} obstacles")
    obstacle_positions = []
    obstacle_handles = []
    obstacle_dimensions = [0.25, 0.25, 0.8]  # Same dimensions for all obstacles
    
    # Function to check if a new position conflicts with existing obstacles
    def is_position_valid(new_x, new_y, object_dimensions):
        # Convert position and dimensions to format needed for bounding locations
        new_position = (new_x, new_y)
        
        # Get bounding points for the new obstacle
        new_bounds = create_bounding_locations(new_position, object_dimensions)
        new_top_right, new_bottom_left, _, _, _, _, _, _ = new_bounds
        
        # Check if the new obstacle would be inside the room bounds
        if (new_x + object_dimensions[0]/2 > 2.5 or 
            new_x - object_dimensions[0]/2 < -2.5 or
            new_y + object_dimensions[1]/2 > 2.5 or
            new_y - object_dimensions[1]/2 < -2.5):
            return False
        
        # Check against all existing obstacles
        for pos in obstacle_positions:
            existing_x, existing_y = pos[0], pos[1]
            existing_position = (existing_x, existing_y)
            
            # Get bounding points for existing obstacle
            existing_bounds = create_bounding_locations(existing_position, obstacle_dimensions)
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
    
    # Create obstacles
    for i in range(num_obstacles):
        # Try to find a valid position (up to 100 attempts)
        valid_position = False
        attempts = 0
        
        while not valid_position and attempts < 100:
            # Generate random position
            x = round(random.uniform(-2.5 + (obstacle_dimensions[0]/2 + 0.1), 
                                     2.5 - (obstacle_dimensions[0]/2 + 0.1)), 2)
            y = round(random.uniform(-2.5 + (obstacle_dimensions[1]/2 + 0.1), 
                                     2.5 - (obstacle_dimensions[1]/2 + 0.1)), 2)
            
            # Check if this position is valid
            valid_position = is_position_valid(x, y, obstacle_dimensions)
            attempts += 1
            
        if valid_position:
            # Add position to our list
            obstacle_positions.append((x, y))
            
            # Create the obstacle in CoppeliaSim if not in simulation mode
            if not SIMULATION_MODE:
                try:
                    obstacle = create_cuboid(
                        dimensions=obstacle_dimensions,
                        position=[x, y, 0.4],  # z=0.4 is half the height
                        orientation=[0, 0, 0],
                        color=[1, 0, 0],
                        mass=1,
                        respondable=True,
                        name=f"Obstacle{i}"
                    )
                    obstacle_handles.append(obstacle)
                    print(f"Created Obstacle{i} at position [{x}, {y}, 0.4]")
                except Exception as e:
                    print(f"Error creating obstacle: {e}")
                    # Use a dummy handle in simulation mode
                    obstacle_handles.append(i)  # Just use the index as a dummy handle
            else:
                # Use a dummy handle in simulation mode
                obstacle_handles.append(i)  # Just use the index as a dummy handle
                print(f"Simulated Obstacle{i} at position [{x}, {y}, 0.4]")
        else:
            print(f"Could not find valid position for Obstacle{i} after {attempts} attempts")
    
    # Now create the single flat object with dimensions [0.125, 0.125, 0.01]
    flat_object_dimensions = [0.125, 0.125, 0.01]
    valid_position = False
    attempts = 0
    goal_position = None
    goal_handle = None
    
    while not valid_position and attempts < 100:
        # Generate random position
        x = round(random.uniform(-2.5 + (flat_object_dimensions[0]/2 + 0.1), 
                                 2.5 - (flat_object_dimensions[0]/2 + 0.1)), 2)
        y = round(random.uniform(-2.5 + (flat_object_dimensions[1]/2 + 0.1), 
                                 2.5 - (flat_object_dimensions[1]/2 + 0.1)), 2)
        
        # Check if this position is valid
        valid_position = is_position_valid(x, y, flat_object_dimensions)
        attempts += 1
        
    if valid_position:
        # Create the goal object in CoppeliaSim if not in simulation mode
        if not SIMULATION_MODE:
            try:
                # Create the flat object (z=0.005 is half the height of 0.01)
                goal_handle = create_cuboid(
                    dimensions=flat_object_dimensions,
                    position=[x, y, 0.005],  
                    orientation=[0, 0, 0],
                    color=[0, 1, 0],  # Green to distinguish from obstacles
                    mass=0.1,
                    respondable=True,
                    name="Goal_Loc"  # Renamed as requested
                )
                print(f"Created Goal_Loc at position [{x}, {y}, 0.005]")
            except Exception as e:
                print(f"Error creating goal object: {e}")
                # Use a dummy handle in simulation mode
                goal_handle = -999  # Dummy handle
        else:
            # Use a dummy handle in simulation mode
            goal_handle = -999  # Dummy handle
            print(f"Simulated Goal_Loc at position [{x}, {y}, 0.005]")
        
        goal_position = (x, y, 0.005)
    else:
        print(f"Could not find valid position for Goal_Loc after {attempts} attempts")
    
    # Now place bubbleRob at a random position with no overlap
    # First, collect all positions to avoid (obstacles and goal)
    all_positions = obstacle_positions.copy()
    if goal_position:
        all_positions.append((goal_position[0], goal_position[1]))  # Only need x,y from goal
    
    # Estimate bubbleRob dimensions (assuming it's roughly 0.2 x 0.2)
    bubbleRob_dimensions = [0.2, 0.2, 0.2]  # Approximate dimensions
    
    valid_position = False
    attempts = 0
    bubbleRob_position = None
    
    # Function to check if bubbleRob position is valid
    def is_bubbleRob_position_valid(new_x, new_y):
        # Check if within bounds (-2 to 2 as specified)
        if new_x > 2 or new_x < -2 or new_y > 2 or new_y < -2:
            return False
        
        # Create bubbleRob position and bounds
        new_position = (new_x, new_y)
        new_bounds = create_bounding_locations(new_position, bubbleRob_dimensions)
        new_top_right, new_bottom_left, _, _, _, _, _, _ = new_bounds
        
        # Check against obstacles
        for i, pos in enumerate(obstacle_positions):
            existing_x, existing_y = pos[0], pos[1]
            existing_position = (existing_x, existing_y)
            
            # Get bounding points for existing obstacle
            existing_bounds = create_bounding_locations(existing_position, obstacle_dimensions)
            existing_top_right, existing_bottom_left, _, _, _, _, _, _ = existing_bounds
            
            # Check for overlap using AABB collision detection
            if not (new_top_right[0] < existing_bottom_left[0] or 
                    existing_top_right[0] < new_bottom_left[0] or
                    new_top_right[1] < existing_bottom_left[1] or 
                    existing_top_right[1] < new_bottom_left[1]):
                # Overlap detected
                return False
                
        # Check against goal if it exists
        if goal_position:
            goal_x, goal_y = goal_position[0], goal_position[1]
            goal_pos = (goal_x, goal_y)
            
            # Get bounding points for goal
            goal_bounds = create_bounding_locations(goal_pos, flat_object_dimensions)
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
    
    while not valid_position and attempts < 100:
        # Generate random position between -2 and 2 as specified
        x = round(random.uniform(-2.0 + (bubbleRob_dimensions[0]/2), 
                                 2.0 - (bubbleRob_dimensions[0]/2)), 2)
        y = round(random.uniform(-2.0 + (bubbleRob_dimensions[1]/2), 
                                 2.0 - (bubbleRob_dimensions[1]/2)), 2)
        
        # Check if this position is valid
        valid_position = is_bubbleRob_position_valid(x, y)
        attempts += 1
    
    if valid_position:
        # Place bubbleRob in CoppeliaSim if not in simulation mode
        if not SIMULATION_MODE:
            try:
                # Get the handle for bubbleRob
                bubbleRob_handle = sim.getObject('/bubbleRob')
                if bubbleRob_handle != -1:  # -1 means object not found
                    # Set bubbleRob position (z=0.05 assuming that's appropriate for bubbleRob's height)
                    sim.setObjectPosition(bubbleRob_handle, -1, [x, y, 0.12])
                    print(f"Placed bubbleRob at position [{x, y, 0.12}]")
                else:
                    print("bubbleRob object not found in the scene!")
            except Exception as e:
                print(f"Error placing bubbleRob: {e}")
                
        else:
            # Just log in simulation mode
            print(f"Simulated bubbleRob placement at [{x}, {y}, 0.12]")
            
        # Store the position regardless of simulation mode
        bubbleRob_position = (x, y, 0.12)
    else:
        print(f"Could not find valid position for bubbleRob after {attempts} attempts")
        # Use default position in case of failure
        bubbleRob_position = (0, 0, 0.12)
    
    # Return all obstacle positions as a flat list: [x1, y1, x2, y2, ...] and other positions
    flat_positions = []
    for pos in obstacle_positions:
        flat_positions.extend(pos)
    
    print("Generating redspots...")
    for obstacle_num in range(len(obstacle_positions)):
        obstacle_name = f'Obstacle{obstacle_num}'
        obstacle_position = obstacle_positions[obstacle_num]
   
        try:
            # In simulation mode, use the obstacle positions directly
            if SIMULATION_MODE:
                obstacle_position_list = [obstacle_position[0], obstacle_position[1], 0.4]
                print(f"Using simulated position for {obstacle_name}: {obstacle_position_list}")
            else:
                # Try to get position from CoppeliaSim
                try:
                    obstacle_position_list = get_object_position(obstacle_name)
                except Exception as e:
                    print(f"Error getting position for {obstacle_name}, using stored position: {e}")
                    obstacle_position_list = [obstacle_position[0], obstacle_position[1], 0.4]
            
            print(f"Processing {obstacle_name}...")
            
            # Get all points within this obstacle
            points_in_obstacle = get_all_grid_points_in_obstacle(
                (obstacle_position_list[0], obstacle_position_list[1]), 
                obstacle_dimensions
            )
            
            # Convert points to grid coordinates and add to redspots
            for point in points_in_obstacle:
                grid_position = move_to_grid(point[0], point[1])
                
                # Only append valid grid positions
                if isinstance(grid_position, tuple) and grid_position not in redspots:
                    redspots.append(grid_position)
                
        except Exception as e:
            print(f"Error processing {obstacle_name}: {e}")

    print(f"Total redspots collected: {len(redspots)}")
    print("Redspots:", redspots)
    
    # Convert goal position to grid coordinates
    if goal_position:
        goal_grid = move_to_grid(goal_position[0], goal_position[1])
        if isinstance(goal_grid, tuple):
            print(f"Goal in grid coordinates: {goal_grid}")
        else:
            print(f"Warning: Invalid goal grid position: {goal_grid}")
    
    # Convert bubbleRob position to grid coordinates
    if bubbleRob_position:
        bubbleRob_grid = move_to_grid(bubbleRob_position[0], bubbleRob_position[1])
        if isinstance(bubbleRob_grid, tuple):
            print(f"BubbleRob in grid coordinates: {bubbleRob_grid}")
        else:
            print(f"Warning: Invalid bubbleRob grid position: {bubbleRob_grid}")
    
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

