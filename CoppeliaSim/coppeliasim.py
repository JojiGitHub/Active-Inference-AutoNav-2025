from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
from math import comb
import random

# Step 1: Create a client and get handles
client = RemoteAPIClient()
sim = client.getObject('sim')

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
    
    # Convert x, y to grid indices by dividing by 0.125 (40x40 grid)
    x_grid = round(x / 0.125)
    y_grid = round(y / 0.125)
    
    # Ensure that the coordinates are within valid grid range (0 to 39)
    if x_grid > 39 or x_grid < 0:
        return "Invalid x grid point!"
    if y_grid > 39 or y_grid < 0:
        return "Invalid y grid point!"
    
    # Return the grid indices
    return (x_grid, y_grid)
    
def grid_to_coordinates(x_grid, y_grid):
    '''Converts a valid 40x40 grid point back into coppelia (x,y,z) coordinates'''
    # Ensure the grid points are within valid range (0 to 39)
    if x_grid > 39 or x_grid < 0:
        return "Invalid x grid point!"
    if y_grid > 39 or y_grid < 0:
        return "Invalid y grid point!"
    
    # Convert grid indices to world coordinates (multiply by 0.125)
    x = x_grid * 0.125 
    y = y_grid * 0.125
    
    # Translate back to original coordinate system
    x = x - 2.5
    y = y - 2.5
    
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
#Random positional value between 0+dim[index of coordinate(x = 0,y=1)] and 5-dim[index of coordinate]
def initialize_environment():
    obstacle_positions = []
    obstacle_handles = []
    obstacle_dimensions = [0.3, 0.3, 0.8]  # Same dimensions for all obstacles
    
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
    
    # Create 5 obstacles
    for i in range(5):
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
            
            # Create the obstacle
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
        else:
            print(f"Could not find valid position for Obstacle{i} after {attempts} attempts")
    
    # Now create the single flat object with dimensions [0.3, 0.3, 0.01]
    flat_object_dimensions = [0.3, 0.3, 0.01]
    valid_position = False
    attempts = 0
    flat_object_position = None
    
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
        # Create the flat object (z=0.005 is half the height of 0.01)
        flat_object = create_cuboid(
            dimensions=flat_object_dimensions,
            position=[x, y, 0.005],  
            orientation=[0, 0, 0],
            color=[0, 1, 0],  # Green to distinguish from obstacles
            mass=0.1,
            respondable=True,
            name="Goal_Loc"
        )
        
        flat_object_position = (x, y, 0.005)
        print(f"Created FlatObject at position [{x}, {y}, 0.005]")
    else:
        print(f"Could not find valid position for FlatObject after {attempts} attempts")
    
    # Return all obstacle positions as a flat list: [x1, y1, x2, y2, ...] and the flat object position
    flat_positions = []
    for pos in obstacle_positions:
        flat_positions.extend(pos)
    
    return flat_positions, flat_object_position, obstacle_handles, flat_object

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
        for i in range(5):
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

clear_environment()

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
        
        # Add tracking for total distance and last position
        self.total_distance = 0.0
        self.last_position = None

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
        
        # Calculate distance traveled in this step
        if self.last_position is not None:
            distance = np.sqrt((new_world_pos[0] - self.last_position[0])**2 + 
                             (new_world_pos[1] - self.last_position[1])**2)
            self.total_distance += distance
        
        # Update last position
        self.last_position = new_world_pos
        
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
        
        # Additional info including distance metrics
        info = {
            'steps': self.total_steps,
            'old_position': old_pos,
            'new_position': new_grid_pos,
            'goal_position': self.goal_position,
            'manhattan_distance': abs(new_grid_pos[0] - self.goal_position[0]) + 
                                 abs(new_grid_pos[1] - self.goal_position[1]),
            'total_distance': self.total_distance,
            'step_distance': distance if self.last_position is not None else 0.0
        }
        
        return np.array(new_state, dtype=np.float32), reward, done, info

    def reset(self):
        """
        Reset the environment to initial state. Compatible with Gym interface.
        
        Returns:
            numpy.ndarray: Initial state observation
        """
        # Reset distance tracking
        self.total_distance = 0.0
        self.last_position = None
        
        # If environment not initialized yet, do that first
        if not self.red_zone_positions or not self.goal_position:
            self.initialize_environment(self.num_obstacles)
        
        # ...existing code...

# Step 2: Get the object handle for the BubbleRob
# bubbleRobHandle = sim.getObject('/bubbleRob')

# # Step 3: Define the control points (with the full formatting)
# ctrlPts = [[0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0], [1, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]]

# ctrlPts_flattened = [coord for point in ctrlPts for coord in point]
# print(ctrlPts_flattened)
# # Get initial object properties
# def get_initial_object_properties(objectHandle):
#     initial_pos = sim.getObjectPosition(objectHandle, -1)
#     initial_orient = sim.getObjectQuaternion(objectHandle, -1)
#     return initial_pos[2], initial_orient

# # Store initial properties
# initial_z, initial_orientation = get_initial_object_properties(bubbleRobHandle)

# # Path creation
# pathHandle = sim.createPath(
#     ctrlPts_flattened,
#     0,  # Options: open path
#     100,  # Subdivision for smoothness
#     1,  # No smoothness
#     0,  # Orientation mode
#     [0.0, 0.0, 1.0]  # Up vector
# )

# # Improved Bezier calculation with precise interpolation
# def bezier_recursive(ctrlPts, t):
#     n = (len(ctrlPts) // 7) - 1
#     point = np.zeros(3)
#     print(f"Operator n = {n}")
#     total_weight = 0
    
#     for i in range(n + 1):
#         binomial_coeff = comb(n, i)
#         weight = binomial_coeff * ((1 - t) ** (n - i)) * (t ** i)
#         point_coords = np.array(ctrlPts[i * 7:i * 7 + 3])
#         print(f"Coordinates = {point_coords}")
#         point += weight * point_coords
#         total_weight += weight
    
#     # Normalize point to ensure precise interpolation
#     if total_weight > 0:
#         point = point / total_weight
    
#     point[2] = initial_z  # Set exact Z coordinate
#     print(f"Final Point{point}") 
#     return point

# # Calculate total path length with more precise sampling
# def calculate_total_length(ctrl_points, subdivisions=1000):
#     total_length = 0.0
#     prev_point = bezier_recursive(ctrl_points, 0)
#     for i in range(1, subdivisions + 1):
#         t = i / subdivisions
#         curr_point = bezier_recursive(ctrl_points, t)
#         total_length += np.linalg.norm(curr_point - prev_point)
#         prev_point = curr_point
#     return total_length

# # Initial calculation
# totalLength = calculate_total_length(ctrlPts_flattened)
# posAlongPath = 0
# velocity = 0.08

# # Calculate point and tangent at given parameter
# def get_point_and_tangent(t, ctrl_points):
#     # Get current point
#     point = bezier_recursive(ctrl_points, t)
    
#     # Calculate tangent using small delta
#     delta = 0.001
#     t_next = min(1.0, t + delta)
#     next_point = bezier_recursive(ctrl_points, t_next)
    
#     tangent = next_point - point
#     if np.linalg.norm(tangent) > 0:
#         tangent = tangent / np.linalg.norm(tangent)
    
#     return point, tangent

# # Update orientation based on path tangent
# def update_orientation(position, tangent):
#     if np.linalg.norm(tangent[:2]) > 0:  # Only use X and Y components
#         yaw = np.arctan2(tangent[1], tangent[0])
#         orientation_quaternion = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
#         sim.setObjectQuaternion(bubbleRobHandle, -1, orientation_quaternion)

# # Modified simulation loop with precise path following
# def follow_path(ctrl_points):
#     global posAlongPath, velocity
    
#     # Calculate total length for current control points
#     total_length = calculate_total_length(ctrl_points)
#     # Reset position along path for new path
#     posAlongPath = 0
    
#     previousSimulationTime = sim.getSimulationTime()
    
#     while posAlongPath < total_length:
#         t = sim.getSimulationTime()
#         deltaT = t - previousSimulationTime
        
#         if deltaT <= 0.0:
#             previousSimulationTime = t
#             continue
        
#         posAlongPath += velocity * deltaT
        
#         if posAlongPath >= total_length - 0.001:
#             posAlongPath = total_length
#             print("Reached the end of the path!")
#             break
        
#         # Calculate normalized parameter
#         t_norm = np.clip(posAlongPath / total_length, 0, 1)
#         # Get current position and tangent
#         current_pos, tangent = get_point_and_tangent(t_norm, ctrl_points)
        
#         # Ensure Z coordinate
#         current_pos[2] = initial_z
        
#         # Update position and orientation
#         sim.setObjectPosition(bubbleRobHandle, -1, current_pos.tolist())
#         update_orientation(current_pos, tangent)
        
#         previousSimulationTime = t
#         sim.step()
#         time.sleep(0.05)

# # Start simulation
# sim.startSimulation()
# follow_path(ctrlPts_flattened)
# time.sleep(10)
# sim.step()
# ctrlPts.append([2,0,0.05,0,0,0,1])
# ctrlPts_flattened = [coord for point in ctrlPts for coord in point]
# pathHandle = sim.createPath(
#     ctrlPts_flattened,
#     0,  # Options: open path
#     100,  # Subdivision for smoothness
#     1,  # No smoothness
#     0,  # Orientation mode
#     [0.0, 0.0, 1.0]  # Up vector
# )
# sim.step()

# # Call follow_path with the new control points
# follow_path(ctrlPts_flattened)
