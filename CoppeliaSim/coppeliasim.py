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
ctrlPts = [
    [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0],  # Start point
    [-0.5, 0.75, 0.05, 0.0, 0.0, 0.0, 1.0],  # Second point
    [-1.5, 1.0, 0.05, 0.0, 0.0, 0.0, 1.0],  # Third point
    [-2, 1.5, 0.05, 0.0, 0.0, 0.0, 1.0],  # Fourth point
    [-2.25, 2.0, 0.05, 0.0, 0.0, 0.0, 1.0]   # Fifth point
]

ctrlPts_flattened = [coord for point in ctrlPts for coord in point]

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
    
    point[2] = initial_z  # Set exact Z coordinate
    return point

# Calculate total path length with more precise sampling
def calculate_total_length(ctrlPts_flattened, subdivisions=1000):
    total_length = 0.0
    prev_point = bezier_recursive(ctrlPts_flattened, 0)
    
    for i in range(1, subdivisions + 1):
        t = i / subdivisions
        curr_point = bezier_recursive(ctrlPts_flattened, t)
        total_length += np.linalg.norm(curr_point - prev_point)
        prev_point = curr_point
    
    return total_length

totalLength = calculate_total_length(ctrlPts_flattened)
posAlongPath = 0
velocity = 0.08

# Calculate point and tangent at given parameter
def get_point_and_tangent(t):
    # Get current point
    point = bezier_recursive(ctrlPts_flattened, t)
    
    # Calculate tangent using small delta
    delta = 0.001
    t_next = min(1.0, t + delta)
    next_point = bezier_recursive(ctrlPts_flattened, t_next)
    
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
def follow_path():
    global posAlongPath, velocity, totalLength
    previousSimulationTime = sim.getSimulationTime()
    
    while posAlongPath < totalLength:
        t = sim.getSimulationTime()
        deltaT = t - previousSimulationTime
        
        if deltaT <= 0.0:
            previousSimulationTime = t
            continue
        
        posAlongPath += velocity * deltaT
        
        if posAlongPath >= totalLength - 0.001:
            posAlongPath = totalLength
            print("Reached the end of the path!")
            break
        
        # Calculate normalized parameter
        t_norm = np.clip(posAlongPath / totalLength, 0, 1)
        
        # Get current position and tangent
        current_pos, tangent = get_point_and_tangent(t_norm)
        
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
follow_path()
time.sleep(10)
sim.stopSimulation()