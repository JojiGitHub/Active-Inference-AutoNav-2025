import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

import seaborn as sns

import pymdp
from pymdp import utils
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
from math import comb
import random

# Initialize client and sim (keep these at module level)
client = RemoteAPIClient()
sim = client.getObject('sim')

# Move all other initialization code into functions
def get_bubbleRob_handle():
    """Get the handle for bubbleRob safely"""
    try:
        return sim.getObject('/bubbleRob')
    except Exception as e:
        print(f"Error getting bubbleRob handle: {e}")
        return -1

def get_object_position(object_name):
    """Get the position of an object by name"""
    # Get the object handle by name
    objectHandle = sim.getObject(f'/{object_name}')
    
    if objectHandle == -1:
        raise Exception(f"Object '{object_name}' not found.")
    
    # Get the position of the obstacle relative to the world (-1 means world reference)
    objectPosition = sim.getObjectPosition(objectHandle, -1)
    
    # Round each element in the position to the nearest thousandth
    roundedPosition = [round(coord, 3) for coord in objectPosition]
    
    print(f"Coppelia position of {object_name}: {roundedPosition}")
    return roundedPosition

def move_to_grid(x, y):
    '''Moves coppelia coordinates (x,y) to a 40x40 grid, z coordinate remains constant'''
    
    # First check if inputs are valid numbers
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    
    # Translate x,y coordinate 2.5 up and 2.5 right to make (0,0) the top-left corner
    x = x + 2.5
    y = y + 2.5
    
    # Check if coordinates are within valid range (0,5)
    if x > 5.0 or x < 0.0 or y > 5.0 or y < 0.0:
        return None
    
    # Convert to grid indices - each grid cell is 0.125 units
    # Add 0.0625 (half cell width) to ensure we're targeting cell centers
    x_grid = round((x + 0.0625) / 0.125)
    y_grid = round((y + 0.0625) / 0.125)
    
    # Ensure results are within valid grid range (0-39)
    x_grid = max(0, min(39, x_grid))
    y_grid = max(0, min(39, y_grid))
    
    return (x_grid, y_grid)
    
def grid_to_coordinates(x_grid, y_grid, z_height=0.05):
    '''Converts a valid 40x40 grid point back into coppelia (x,y,z) coordinates'''
    # Ensure the grid points are within valid range (0-39 inclusive)
    if not isinstance(x_grid, (int, float)) or not isinstance(y_grid, (int, float)):
        return None
        
    if x_grid > 39 or x_grid < 0 or y_grid > 39 or y_grid < 0:
        return None
    
    # Convert grid indices to coordinates
    # Each grid cell is 0.125 units wide, and we want to center within the cell
    x = (x_grid * 0.125) - 2.5 + 0.0625  # Add half cell width (0.0625) to center
    y = (y_grid * 0.125) - 2.5 + 0.0625
    
    # Ensure coordinates are within CoppeliaSim bounds (-2.5 to 2.5)
    x = max(-2.5, min(2.5, x))
    y = max(-2.5, min(2.5, y))
    
    # Return the coordinates with specified z-height
    return (x, y, z_height)

def create_bounding_locations(position, dimensions):
    (x, y) = position
    (a, b, c) = dimensions
    
    # Center point
    center = (x, y)
    
    # Corner points
    top_right = (x + a/2, y + b/2)
    bottom_left = (x - a/2, y - b/2)
    top_left = (x - a/2, y + b/2)
    bottom_right = (x + a/2, y - b/2)
    
    # Edge midpoints
    mid_top = (x, y + b/2)
    mid_bottom = (x, y - b/2)
    mid_left = (x - a/2, y)
    mid_right = (x + a/2, y)
    
    # Quarter points on edges for better coverage
    quarter_top_left = (x - a/4, y + b/2)
    quarter_top_right = (x + a/4, y + b/2)
    quarter_bottom_left = (x - a/4, y - b/2)
    quarter_bottom_right = (x + a/4, y - b/2)
    quarter_left_top = (x - a/2, y + b/4)
    quarter_left_bottom = (x - a/2, y - b/4)
    quarter_right_top = (x + a/2, y + b/4)
    quarter_right_bottom = (x + a/2, y - b/4)
    
    # Interior points for better coverage
    interior_top_left = (x - a/4, y + b/4)
    interior_top_right = (x + a/4, y + b/4)
    interior_bottom_left = (x - a/4, y - b/4)
    interior_bottom_right = (x + a/4, y - b/4)
    
    return [
        center,
        top_right, bottom_left, top_left, bottom_right,  # Corners
        mid_top, mid_bottom, mid_left, mid_right,  # Edge midpoints
        quarter_top_left, quarter_top_right,  # Top edge quarters
        quarter_bottom_left, quarter_bottom_right,  # Bottom edge quarters
        quarter_left_top, quarter_left_bottom,  # Left edge quarters
        quarter_right_top, quarter_right_bottom,  # Right edge quarters
        interior_top_left, interior_top_right,  # Interior points
        interior_bottom_left, interior_bottom_right
    ]

        
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
#Random positional value between 0+dim[index of coordinate(x = 0,y=1)] and 5-dim[index of coordinate]
def initialize_environment(seed):
    random.seed(seed)
    num_obstacles = random.randint(20, 50)
    print(f"Initializing environment with seed {seed} and {num_obstacles} obstacles")
    obstacle_positions = []
    obstacle_handles = []
    obstacle_dimensions = [0.3, 0.3, 0.8]  # Same dimensions for all obstacles
    redspots = []  # Initialize redspots list
    
    # Function to check if a new position conflicts with existing obstacles
    def is_position_valid(new_x, new_y, object_dimensions):
        # Convert position and dimensions to format needed for bounding locations
        new_position = (new_x, new_y)
        
        # Add a small buffer to dimensions for spacing between obstacles
        buffered_dimensions = [d + 0.1 for d in object_dimensions]  # Add 10cm buffer
        
        # Get bounding points for the new obstacle
        new_bounds = create_bounding_locations(new_position, buffered_dimensions)
        new_top_right = new_bounds[1]  # First corner
        new_bottom_left = new_bounds[2]  # Second corner
        
        # Check if the new obstacle would be inside the room bounds
        if (new_x + object_dimensions[0]/2 > 2.4 or  # Reduced from 2.5 to ensure better edge spacing
            new_x - object_dimensions[0]/2 < -2.4 or
            new_y + object_dimensions[1]/2 > 2.4 or
            new_y - object_dimensions[1]/2 < -2.4):
            return False
        
        # Check against all existing obstacles with buffered collision detection
        for pos in obstacle_positions:
            existing_x, existing_y = pos[0], pos[1]
            existing_position = (existing_x, existing_y)
            
            # Get bounding points for existing obstacle (with buffer)
            existing_bounds = create_bounding_locations(existing_position, buffered_dimensions)
            existing_top_right = existing_bounds[1]
            existing_bottom_left = existing_bounds[2]
            
            # Check for overlap using AABB collision detection with buffer
            if not (new_top_right[0] < existing_bottom_left[0] or 
                    existing_top_right[0] < new_bottom_left[0] or
                    new_top_right[1] < existing_bottom_left[1] or 
                    existing_top_right[1] < new_bottom_left[1]):
                # Overlap detected
                return False
                
            # Additional check for minimum path width
            min_path_width = 0.3  # Minimum width for agent to pass through
            x_dist = abs(new_x - existing_x)
            y_dist = abs(new_y - existing_y)
            if x_dist < object_dimensions[0] + min_path_width and y_dist < object_dimensions[1] + min_path_width:
                return False
                
        return True
    
    # Create obstacles
    for i in range(num_obstacles):
        valid_position = False
        attempts = 0
        
        while not valid_position and attempts < 100:
            x = round(random.uniform(-2.5 + (obstacle_dimensions[0]/2 + 0.1), 
                                     2.5 - (obstacle_dimensions[0]/2 + 0.1)), 2)
            y = round(random.uniform(-2.5 + (obstacle_dimensions[1]/2 + 0.1), 
                                     2.5 - (obstacle_dimensions[1]/2 + 0.1)), 2)
            
            valid_position = is_position_valid(x, y, obstacle_dimensions)
            attempts += 1
            
        if valid_position:
            # Add position to our list
            obstacle_positions.append((x, y))
            
            # Create the obstacle
            obstacle = create_cuboid(
                dimensions=obstacle_dimensions,
                position=[x, y, 0.4],
                orientation=[0, 0, 0],
                color=[1, 0, 0],
                mass=1,
                respondable=True,
                name=f"Obstacle{i}"
            )
            
            obstacle_handles.append(obstacle)
            print(f"Created Obstacle{i} at position [{x}, {y}, 0.4]")
            
            # Get bounding locations for this obstacle and convert to grid positions
            bounding_locs = create_bounding_locations((x, y), obstacle_dimensions)
            for location in bounding_locs:
                bx, by = location
                grid_position = move_to_grid(bx, by)
                # Only add valid grid positions
                if isinstance(grid_position, tuple):
                    if grid_position not in redspots:  # Avoid duplicates
                        redspots.append(grid_position)
                        print(f"Added redspot at grid position {grid_position}")
                else:
                    print(f"Warning: Invalid grid position at {location}: {grid_position}")
                        
        else:
            print(f"Could not find valid position for Obstacle{i} after {attempts} attempts")
    
    # Create goal location with flat dimensions
    flat_object_dimensions = [0.3, 0.3, 0.01]
    valid_position = False
    attempts = 0
    goal_position = None
    goal_handle = None
    
    def is_goal_position_valid(x, y, flat_dimensions):
        # First check basic position validity
        if not is_position_valid(x, y, flat_dimensions):
            return False
            
        # Convert position to grid coordinates
        grid_pos = move_to_grid(x, y)
        if not isinstance(grid_pos, tuple):
            return False
            
        # Check that the goal position isn't in a redspot
        if grid_pos in redspots:
            return False
            
        # Check if there's at least one adjacent position that isn't a redspot
        x_grid, y_grid = grid_pos
        adjacent_positions = [
            (x_grid - 1, y_grid), (x_grid + 1, y_grid),
            (x_grid, y_grid - 1), (x_grid, y_grid + 1)
        ]
        
        valid_adjacent = sum(1 for pos in adjacent_positions if pos not in redspots)
        return valid_adjacent >= 2  # Need at least two valid adjacent positions
    
    while not valid_position and attempts < 100:
        x = round(random.uniform(-2.4 + (flat_object_dimensions[0]/2), 
                               2.4 - (flat_object_dimensions[0]/2)), 2)
        y = round(random.uniform(-2.4 + (flat_object_dimensions[1]/2), 
                               2.4 - (flat_object_dimensions[1]/2)), 2)
        
        valid_position = is_goal_position_valid(x, y, flat_object_dimensions)
        attempts += 1
        
    if valid_position:
        goal_handle = create_cuboid(
            dimensions=flat_object_dimensions,
            position=[x, y, 0.005],
            orientation=[0, 0, 0],
            color=[0, 1, 0],
            mass=0.1,
            respondable=True,
            name="Goal_Loc"
        )
        
        goal_position = (x, y, 0.005)
        print(f"Created Goal_Loc at position [{x}, {y}, 0.005]")
    else:
        print("Could not find valid position for Goal_Loc, using center position")
        # Use center position as fallback
        x, y = 0, 0
        goal_handle = create_cuboid(
            dimensions=flat_object_dimensions,
            position=[x, y, 0.005],
            orientation=[0, 0, 0],
            color=[0, 1, 0],
            mass=0.1,
            respondable=True,
            name="Goal_Loc"
        )
        goal_position = (x, y, 0.005)
        
        # Clear obstacles near the goal to ensure it's reachable
        clear_radius = 0.5
        for i, pos in enumerate(obstacle_positions[:]):
            if abs(pos[0] - x) < clear_radius and abs(pos[1] - y) < clear_radius:
                obstacle_positions.remove(pos)
                if i < len(obstacle_handles):
                    try:
                        sim.removeObject(obstacle_handles[i])
                        obstacle_handles.pop(i)
                    except Exception as e:
                        print(f"Error removing obstacle: {e}")
        print(f"Created Goal_Loc at center position [{x}, {y}, 0.005] and cleared nearby obstacles")
    
    # Place bubbleRob at a random position with no overlap
    bubbleRob_dimensions = [0.2, 0.2, 0.2]
    valid_position = False
    attempts = 0
    bubbleRob_position = None
    
    # Function to check if bubbleRob position is valid
    def is_bubbleRob_position_valid(new_x, new_y):
        if new_x > 2 or new_x < -2 or new_y > 2 or new_y < -2:
            return False
        
        new_position = (new_x, new_y)
        new_bounds = create_bounding_locations(new_position, bubbleRob_dimensions)
        
        # Get only the corner points for collision detection
        # Index 1-4 are the corner points in the new format
        new_corners = new_bounds[1:5]  # [top_right, bottom_left, top_left, bottom_right]
        new_top_right = new_corners[0]
        new_bottom_left = new_corners[1]
        
        # Check against obstacles
        for pos in obstacle_positions:
            existing_x, existing_y = pos[0], pos[1]
            existing_position = (existing_x, existing_y)
            existing_bounds = create_bounding_locations(existing_position, obstacle_dimensions)
            existing_corners = existing_bounds[1:5]
            existing_top_right = existing_corners[0]
            existing_bottom_left = existing_corners[1]
            
            if not (new_top_right[0] < existing_bottom_left[0] or 
                    existing_top_right[0] < new_bottom_left[0] or
                    new_top_right[1] < existing_bottom_left[1] or 
                    existing_top_right[1] < new_bottom_left[1]):
                return False
                
        # Check against goal if it exists
        if goal_position:
            goal_x, goal_y = goal_position[0], goal_position[1]
            goal_pos = (goal_x, goal_y)
            goal_bounds = create_bounding_locations(goal_pos, flat_object_dimensions)
            goal_corners = goal_bounds[1:5]
            goal_top_right = goal_corners[0]
            goal_bottom_left = goal_corners[1]
            
            if not (new_top_right[0] < goal_bottom_left[0] or 
                    goal_top_right[0] < new_bottom_left[0] or
                    new_top_right[1] < goal_bottom_left[1] or 
                    goal_top_right[1] < new_bottom_left[1]):
                return False
        
        return True
    
    while not valid_position and attempts < 100:
        x = round(random.uniform(-2.0 + (bubbleRob_dimensions[0]/2), 
                                 2.0 - (bubbleRob_dimensions[0]/2)), 2)
        y = round(random.uniform(-2.0 + (bubbleRob_dimensions[1]/2), 
                                 2.0 - (bubbleRob_dimensions[1]/2)), 2)
        
        valid_position = is_bubbleRob_position_valid(x, y)
        attempts += 1
    
    if valid_position:
        try:
            bubbleRob_handle = sim.getObject('/bubbleRob')
            if bubbleRob_handle != -1:
                sim.setObjectPosition(bubbleRob_handle, -1, [x, y, 0.12])
                bubbleRob_position = (x, y, 0.12)
                print(f"Placed bubbleRob at position [{x}, {y}, 0.12]")
            else:
                print("bubbleRob object not found in the scene!")
        except Exception as e:
            print(f"Error placing bubbleRob: {e}")
    else:
        print(f"Could not find valid position for bubbleRob after {attempts} attempts")
    
    # Return all positions and handles - now only including obstacle positions in flat_positions
    flat_positions = []
    for pos in obstacle_positions:
        flat_positions.extend(pos)  # This only contains obstacle positions now
    
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
        for i in range(50):  # Increased from 5 to match potential num_obstacles
            try:
                object_handle = sim.getObject(f'/Obstacle{i}')
                if object_handle != -1:
                    sim.removeObject(object_handle)
                    print(f"Removed Obstacle{i} by name")
            except Exception:
                # Silently ignore obstacles that don't exist
                pass
        
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

def monitor_position(bubbleRobHandle):
    """Monitor position change of an object"""
    pos1 = sim.getObjectPosition(bubbleRobHandle, -1)
    time.sleep(0.1)
    pos2 = sim.getObjectPosition(bubbleRobHandle, -1)
    print(f"Position change: {np.array(pos2) - np.array(pos1)}")

class CoppeliaEnv():
    def get_initial_object_properties(self):
        initial_pos = sim.getObjectPosition(self.bubbleRobHandle, -1)
        initial_orient = sim.getObjectQuaternion(self.bubbleRobHandle, -1)
        return initial_pos[2], initial_orient
    
    def __init__(self, redspots, starting_loc, goal, grid_dims=(40, 40)):
        # Get robot handle first and ensure it's valid
        try:
            self.bubbleRobHandle = sim.getObject('/bubbleRob')
            if self.bubbleRobHandle == -1:
                raise Exception("Could not get handle to bubbleRob")
        except Exception as e:
            print(f"Error getting robot handle: {e}")
            raise
            
        # Initialize coordinates 
        self.x, self.y = starting_loc
        self.init_loc = starting_loc
        self.current_location = (self.x, self.y)
        self.goal = goal
        self.redspots = redspots
        self.grid_dims = grid_dims
        
        # Initialize control points
        coords = grid_to_coordinates(self.x, self.y)
        if isinstance(coords, tuple):
            self.ctrlPts = [[coords[0], coords[1], 0.05, 0.0, 0.0, 0.0, 1.0]]
            self.ctrlPts_flattened = [coord for point in self.ctrlPts for coord in point]
        else:
            # Handle error case where grid_to_coordinates returns a string
            print(f"Warning: Invalid grid coordinates: {coords}")
            # Use current robot position as fallback
            robot_pos = sim.getObjectPosition(self.bubbleRobHandle, -1)
            self.ctrlPts = [[robot_pos[0], robot_pos[1], 0.05, 0.0, 0.0, 0.0, 1.0]]
            self.ctrlPts_flattened = [coord for point in self.ctrlPts for coord in point]
            
        # Initialize path-related variables
        self.initial_z, self.initial_orientation = self.get_initial_object_properties()
        self.posAlongPath = 0
        self.velocity = 0.08
        self.previousSimulationTime = sim.getSimulationTime()
        
        # Initialize distance tracking
        self.total_distance = 0.0
        self.last_position = sim.getObjectPosition(self.bubbleRobHandle, -1)
        
        # Initialize reward
        self.agent_reward = 0
        
        # Initialize vision
        self.vision = update_vision(self.current_location, self.grid_dims, 6)

    # ... rest of the CoppeliaEnv class remains the same ...
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
        # Reset position along path for new path
        self.posAlongPath = 0
        last_pos = sim.getObjectPosition(self.bubbleRobHandle, -1)
        
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
            
            # Update total distance
            new_pos = current_pos.tolist()
            segment_distance = np.linalg.norm(np.array(new_pos) - np.array(last_pos))
            self.total_distance += segment_distance
            last_pos = new_pos
            
            self.update_orientation(current_pos, tangent)
            
            self.previousSimulationTime = t
            sim.step()
            time.sleep(0.05)

    def step(self, action_label):
        sim.startSimulation()
        
        # Store previous position before movement
        prev_x, prev_y = self.x, self.y
        next_x, next_y = self.x, self.y
        
        # Calculate next position based on action
        if action_label == "UP":
            next_y = max(0, self.y - 1)
        elif action_label == "DOWN":
            next_y = min(self.grid_dims[1] - 1, self.y + 1)
        elif action_label == "LEFT":
            next_x = max(0, self.x - 1)
        elif action_label == "RIGHT":
            next_x = min(self.grid_dims[0] - 1, self.x + 1)
        
        # Check if next position would be in a redspot
        next_position = (next_x, next_y)
        if next_position in self.redspots:
            # Don't move to the redspot - stay in current position
            next_x, next_y = self.x, self.y
            self.agent_reward -= 5  # Penalize attempt to move into redspot
            print(f"Movement blocked - redspot at {next_position}")
            return (self.current_location, self.green_obs, self.white_obs, 
                   self.red_obs, self.agent_reward, self.total_distance)
        
        # Update position if move is valid
        self.x, self.y = next_x, next_y
        
        # Update control points for path
        next_coords = grid_to_coordinates(self.x, self.y)
        if isinstance(next_coords, tuple):
            # Calculate path points for smoother movement
            current_coords = grid_to_coordinates(prev_x, prev_y)
            
            # Add start point
            self.ctrlPts = [[current_coords[0], current_coords[1], 0.05, 0.0, 0.0, 0.0, 1.0]]
            
            # Add end point
            self.ctrlPts.append([next_coords[0], next_coords[1], 0.05, 0.0, 0.0, 0.0, 1.0])
            
            self.ctrlPts_flattened = [coord for point in self.ctrlPts for coord in point]
            
            # Create and follow path
            pathHandle = sim.createPath(
                self.ctrlPts_flattened,
                0,  # Options: open path
                100,  # Subdivision for smoothness
                0.5,  # No smoothness
                0,  # Orientation mode
                [0.0, 0.0, 1.0]  # Up vector
            )
            self.follow_path()
        else:
            print(f"Warning: Invalid grid coordinates for next position: {next_coords}")
        
        # Update current location
        self.current_location = (self.x, self.y)
        self.loc_obs = self.current_location
        
        # Clear and reset control points for next step
        self.ctrlPts.clear()
        current_coords = grid_to_coordinates(self.x, self.y)
        if isinstance(current_coords, tuple):
            self.ctrlPts.append([current_coords[0], current_coords[1], 0.05, 0.0, 0.0, 0.0, 1.0])
        
        # Update vision and observations
        self.vision = update_vision(self.current_location, self.grid_dims, 6)
        
        # Reset observations
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
        
        # Update rewards with stronger goal reward and redspot penalty
        if self.current_location == self.goal:
            self.agent_reward += 20  # Strong positive reward for reaching goal
            print("Goal reached!")
        
        # Update distance tracking
        current_pos = sim.getObjectPosition(self.bubbleRobHandle, -1)
        segment_distance = np.linalg.norm(np.array(current_pos) - np.array(self.last_position))
        self.total_distance += segment_distance
        self.last_position = current_pos
        
        return (self.loc_obs, self.green_obs, self.white_obs, self.red_obs, 
                self.agent_reward, self.total_distance)

    def reset(self):
        self.x, self.y = self.init_loc
        self.current_location = (self.x, self.y)
        print(f'Re-initialized location to {self.current_location}')
        self.loc_obs = self.current_location
        self.green_obs, self.white_obs, self.red_obs, self.agent_reward = 'Null', ['Null'], ['Null'], 0
        
        self.total_distance = 0.0
        self.last_position = sim.getObjectPosition(self.bubbleRobHandle, -1)

        return self.loc_obs, self.green_obs, self.white_obs, self.red_obs, self.agent_reward, self.total_distance

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

