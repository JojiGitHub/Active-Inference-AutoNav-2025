from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
from math import comb

# Step 1: Create a client and get handles
client = RemoteAPIClient()
sim = client.getObject('sim')

# Step 2: Get the object handle for the BubbleRob
bubbleRobHandle = sim.getObject('/bubbleRob')

# Step 3: Define the control points (with the full formatting)
ctrlPts = [[0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0], [1, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]]

ctrlPts_flattened = [coord for point in ctrlPts for coord in point]
print(ctrlPts_flattened)
# Get initial object properties
def get_initial_object_properties(objectHandle):
    initial_pos = sim.getObjectPosition(objectHandle, -1)
    initial_orient = sim.getObjectQuaternion(objectHandle, -1)
    return initial_pos[2], initial_orient

# Store initial properties
initial_z, initial_orientation = get_initial_object_properties(bubbleRobHandle)

# Path creation
pathHandle = sim.createPath(
    ctrlPts_flattened,
    0,  # Options: open path
    100,  # Subdivision for smoothness
    1,  # No smoothness
    0,  # Orientation mode
    [0.0, 0.0, 1.0]  # Up vector
)

# Improved Bezier calculation with precise interpolation
def bezier_recursive(ctrlPts, t):
    n = (len(ctrlPts) // 7) - 1
    point = np.zeros(3)
    print(f"Operator n = {n}")
    total_weight = 0
    
    for i in range(n + 1):
        binomial_coeff = comb(n, i)
        weight = binomial_coeff * ((1 - t) ** (n - i)) * (t ** i)
        point_coords = np.array(ctrlPts[i * 7:i * 7 + 3])
        print(f"Coordinates = {point_coords}")
        point += weight * point_coords
        total_weight += weight
    
    # Normalize point to ensure precise interpolation
    if total_weight > 0:
        point = point / total_weight
    
    point[2] = initial_z  # Set exact Z coordinate
    print(f"Final Point{point}") 
    return point

# Calculate total path length with more precise sampling
def calculate_total_length(ctrl_points, subdivisions=1000):
    total_length = 0.0
    prev_point = bezier_recursive(ctrl_points, 0)
    for i in range(1, subdivisions + 1):
        t = i / subdivisions
        curr_point = bezier_recursive(ctrl_points, t)
        total_length += np.linalg.norm(curr_point - prev_point)
        prev_point = curr_point
    return total_length

# Initial calculation
totalLength = calculate_total_length(ctrlPts_flattened)
posAlongPath = 0
velocity = 0.08

# Calculate point and tangent at given parameter
def get_point_and_tangent(t, ctrl_points):
    # Get current point
    point = bezier_recursive(ctrl_points, t)
    
    # Calculate tangent using small delta
    delta = 0.001
    t_next = min(1.0, t + delta)
    next_point = bezier_recursive(ctrl_points, t_next)
    
    tangent = next_point - point
    if np.linalg.norm(tangent) > 0:
        tangent = tangent / np.linalg.norm(tangent)
    
    return point, tangent

# Update orientation based on path tangent
def update_orientation(position, tangent):
    if np.linalg.norm(tangent[:2]) > 0:  # Only use X and Y components
        yaw = np.arctan2(tangent[1], tangent[0])
        orientation_quaternion = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
        sim.setObjectQuaternion(bubbleRobHandle, -1, orientation_quaternion)

# Modified simulation loop with precise path following
def follow_path(ctrl_points):
    global posAlongPath, velocity
    
    # Calculate total length for current control points
    total_length = calculate_total_length(ctrl_points)
    # Reset position along path for new path
    posAlongPath = 0
    
    previousSimulationTime = sim.getSimulationTime()
    
    while posAlongPath < total_length:
        t = sim.getSimulationTime()
        deltaT = t - previousSimulationTime
        
        if deltaT <= 0.0:
            previousSimulationTime = t
            continue
        
        posAlongPath += velocity * deltaT
        
        if posAlongPath >= total_length - 0.001:
            posAlongPath = total_length
            print("Reached the end of the path!")
            break
        
        # Calculate normalized parameter
        t_norm = np.clip(posAlongPath / total_length, 0, 1)
        # Get current position and tangent
        current_pos, tangent = get_point_and_tangent(t_norm, ctrl_points)
        
        # Ensure Z coordinate
        current_pos[2] = initial_z
        
        # Update position and orientation
        sim.setObjectPosition(bubbleRobHandle, -1, current_pos.tolist())
        update_orientation(current_pos, tangent)
        
        previousSimulationTime = t
        sim.step()
        time.sleep(0.05)

# Start simulation
sim.startSimulation()
follow_path(ctrlPts_flattened)
time.sleep(10)
sim.step()
ctrlPts.append([2,0,0.05,0,0,0,1])
ctrlPts_flattened = [coord for point in ctrlPts for coord in point]
pathHandle = sim.createPath(
    ctrlPts_flattened,
    0,  # Options: open path
    100,  # Subdivision for smoothness
    1,  # No smoothness
    0,  # Orientation mode
    [0.0, 0.0, 1.0]  # Up vector
)
sim.step()

# Call follow_path with the new control points
follow_path(ctrlPts_flattened)
