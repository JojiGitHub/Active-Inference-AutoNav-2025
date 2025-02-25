from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
from math import comb

# Step 1: Create a client and get handles
client = RemoteAPIClient()
sim = client.getObject('sim')

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

my_cuboid = create_cuboid(
    dimensions=[0.3, 0.3, 0.8],
    position=[1.0, 1.0, 0.15],
    orientation=[0, 0, 0],
    color=[1, 0,0],
    mass=1,
    respondable=True,
    name="Obstacle0"
)
def get_cuboid_dimensions(object_name):
    """
    Get the dimensions of a primitive cuboid in CoppeliaSim.
    
    Parameters:
    -----------
    object_name : str
        Name of the cuboid object
        
    Returns:
    --------
    list
        [x, y, z] dimensions of the cuboid in meters
    """
    # Get the object handle - try both with and without the leading slash
    try:
        cuboid_handle = sim.getObject(f'/{object_name}')
    except:
        try:
            cuboid_handle = sim.getObject(object_name)
        except:
            print(f"Error: Could not find object named '{object_name}'")
            return None
    
    # Get the shape data - newer versions of CoppeliaSim return data differently
    try:
        # Try the new API format first
        result = sim.getShapeGeomInfo(cuboid_handle)
        
        # Check the format of the result
        if isinstance(result, tuple) and len(result) == 2:
            # New API format: (shape_type, [dim1, dim2, dim3, ...])
            shape_type, dimensions = result
            
            # For cuboids (type 0), return the first 3 values of dimensions
            if shape_type == 0:  # Cuboid
                return dimensions[:3]
            else:
                print(f"Warning: Object '{object_name}' is not a cuboid (type: {shape_type})")
                return None
        else:
            # Old format or unknown format
            print(f"Unexpected return format from getShapeGeomInfo: {result}")
            
            # Try to handle it anyway
            if isinstance(result, list) and len(result) >= 4:
                return result[1:4]
            else:
                return None
    except Exception as e:
        print(f"Error getting shape geometry: {str(e)}")
        
        # Fall back to bounding box method
        try:
            min_x = sim.getObjectFloatParam(cuboid_handle, sim.objfloatparam_objbbox_min_x)
            min_y = sim.getObjectFloatParam(cuboid_handle, sim.objfloatparam_objbbox_min_y)
            min_z = sim.getObjectFloatParam(cuboid_handle, sim.objfloatparam_objbbox_min_z)
            max_x = sim.getObjectFloatParam(cuboid_handle, sim.objfloatparam_objbbox_max_x)
            max_y = sim.getObjectFloatParam(cuboid_handle, sim.objfloatparam_objbbox_max_y)
            max_z = sim.getObjectFloatParam(cuboid_handle, sim.objfloatparam_objbbox_max_z)
            
            dimensions = [
                abs(max_x - min_x),
                abs(max_y - min_y),
                abs(max_z - min_z)
            ]
            
            return dimensions
        except Exception as e2:
            print(f"Error getting bounding box: {str(e2)}")
            return None

print(get_cuboid_dimensions("Obstacle0"))


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
