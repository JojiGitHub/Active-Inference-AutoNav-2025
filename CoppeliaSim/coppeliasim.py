from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
from math import comb

# Step 1: Create a client and get handles
client = RemoteAPIClient()
sim = client.getObject('sim')

# Step 2: Get the object handle for the Cuboid
floorHandle = sim.getObject('/Floor')
cuboidHandle = sim.getObject('/Agent')

# Step 3: Define the control points (flattened list with position and quaternion values)
# Add more control points for a longer path
# Control points with more than 3 points
ctrlPts = [
    [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0],  # Start point
    [0.5, 0.75, 0.05, 0.0, 0.0, 0.0, 1.0],  # Second point
    [1.0, 1.0, 0.05, 0.0, 0.0, 0.0, 1.0],  # Third point
    [1.5, 1.5, 0.05, 0.0, 0.0, 0.0, 1.0],  # Fourth point
    [2.0, 2.0, 0.05, 0.0, 0.0, 0.0, 1.0]   # Fifth point (New endpoint)
]

# Flatten the control points into a single list
ctrlPts_flattened = [coord for point in ctrlPts for coord in point]

# Create the path with more than 3 control points
pathHandle = sim.createPath(
    ctrlPts_flattened,  # Flattened control points list
    0,                  # Options (bit0 is not set, meaning path is open)
    100,                # Subdivision (number of points to create along the path for smoother interpolation)
    1,                  # Smoothness
    0,                  # Orientation mode (x-axis along path, y-axis is up)
    [0.0, 0.0, 1.0]     # Up vector for path orientation
)

# Set the path's parent to the floor
sim.setObjectParent(pathHandle, floorHandle, True)

# Define fixed Z value
fixedZ = 0.05  # The fixed Z coordinate value

# Recursive Bezier calculation
def bezier_recursive(ctrlPts, t):
    n = (len(ctrlPts) // 7) - 1  # Calculate based on control points (7 values per point: x, y, z + quaternion)
    point = np.zeros(3)  # We are only interested in the (x, y, z) positions for movement
    
    # Calculate the weighted sum of all control points (only x, y, z for now)
    for i in range(n + 1):
        binomial_coeff = comb(n, i)
        weight = binomial_coeff * ((1 - t) ** (n - i)) * (t ** i)
        point += weight * np.array(ctrlPts[i * 7:i * 7 + 3])  # Only use the position (x, y, z)
    
    # Debug: Print intermediate points calculated by Bezier interpolation
    print(f"Bezier Point at t={t}: {point}")
    
    return point

# New function to calculate the total length of the Bezier curve
# New function to calculate the total length of the Bezier curve
def calculate_total_length(ctrlPts_flattened, subdivisions=100):
    total_length = 0.0
    prev_point = np.array(ctrlPts_flattened[:3])  # Start at the first control point (only x, y, z)
    
    for i in range(1, subdivisions + 1):
        t = i / subdivisions
        curr_point = bezier_recursive(ctrlPts_flattened, t)
        total_length += np.linalg.norm(curr_point - prev_point)  # Add the distance between points
        prev_point = curr_point  # Update prev_point to the current point
    
    return total_length


# Global variables for position and timing
totalLength = 0
posAlongPath = 0
previousSimulationTime = 0

# Initialize function (formerly sysCall_init)
def init():
    global posAlongPath, previousSimulationTime, totalLength

    # Recalculate total length of the path before starting the movement
    totalLength = calculate_total_length(ctrlPts_flattened)  # Use dynamic calculation for total length
    print(f"Total Path Length: {totalLength}")  # Debug: Print total length of the path
    posAlongPath = 0
    previousSimulationTime = sim.getSimulationTime()

# Main loop to simulate path following
def follow_path():
    global posAlongPath, previousSimulationTime
    velocity = 0.04  # m/s

    while not sim.getSimulationStopping():
        t = sim.getSimulationTime()
        deltaT = t - previousSimulationTime

        if deltaT <= 0.0:
            previousSimulationTime = t
            continue

        # Calculate remaining path and adjusted velocity
        remainingPath = totalLength - posAlongPath
        adjustedVelocity = min(velocity, remainingPath / deltaT)
        posAlongPath += adjustedVelocity * deltaT

        if posAlongPath >= totalLength - 0.001:
            posAlongPath = totalLength
            print("Reached the end of the path!")
            break

        # Normalize the position along the path to get t for Bezier interpolation
        t_norm = posAlongPath / totalLength
        interpolatedPos = bezier_recursive(ctrlPts_flattened, t_norm)
        interpolatedPos[2] = fixedZ  # Force Z coordinate to fixed value

        # Debug: Print the interpolated position being set
        print(f"Setting object position to: {interpolatedPos}")

        # Set object position to the interpolated Bezier position
        sim.setObjectPosition(cuboidHandle, -1, interpolatedPos.tolist())
        
        # Keep the orientation fixed to avoid jittering
        sim.setObjectQuaternion(cuboidHandle, -1, [0.0, 0.0, 0.0, 1.0])

        # Update the previous simulation time
        previousSimulationTime = t

        sim.step()
        time.sleep(0.05)

# Initialize path data and other variables before simulation
init()

# Start the simulation
sim.startSimulation()

# Run the path-following function
follow_path()

# Stop the simulation after 10 seconds
time.sleep(10)
sim.stopSimulation()







# def move_to_grid(x, y, z):
#     '''Moves coppelia coordinates (x,y,z) to a 40x40 grid, z coordinate remains constant, outputs coordinate in terms of grid'''
    
#     # Translate x,y coordinate 2.5 up and 2.5 right
#     x = x + 2.5
#     y = y + 2.5
    
#     # Ensure coordinates (x,y) are within (0,0) and (5,5)
#     if x > 5 or x < 0:
#         return "Invalid x coordinate!"
#     elif y > 5 or y < 0:
#         return "Invalid y coordinate!"
    
#     # Convert x, y to grid indices by dividing by 0.05 (since each grid cell is 0.05 wide)
#     x_grid = round(x / 0.125)
#     y_grid = round(y / 0.125)
    
#     # Ensure that the coordinates are within valid grid range (0 to 200)
#     if x_grid > 40 or x_grid < 0:
#         return "Invalid x grid point!"
#     if y_grid > 40 or y_grid < 0:
#         return "Invalid y grid point!"
    
#     # Return the grid indices
#     return (x_grid, y_grid)

    
# def grid_to_coordinates(x_grid, y_grid, z):
#     '''Converts a valid 200x200 grid point back into coppelia (x,y,z) coordinates in the range (x,y) = (0,0)-(5,5), z remains constant'''
    
#     # Ensure the grid points are within valid range (0 to 200)
#     if x_grid > 40 or x_grid < 0:
#         return "Invalid x grid point!"
#     if y_grid > 40 or y_grid < 0:
#         return "Invalid y grid point!"
    
#     # Reverse the grid index conversion by multiplying by 0.05
#     x = x_grid * 0.125
#     y = y_grid * 0.125
    
#     # Return the original (x, y, z) coordinates
#     return (x, y, z)   
    

# def get_object_position(object_name):
#     # Step 2: Get the object handle by name
#     objectHandle = sim.getObject(f'/{object_name}')
    
#     if objectHandle == -1:
#         raise Exception(f"Object '{object_name}' not found.")
    
#     # Step 3: Get the position of the obstacle relative to the world (-1 means world reference)
#     objectPosition = sim.getObjectPosition(objectHandle, -1)
    
#     # Round each element in the position to the nearest thousandth
#     roundedPosition = [round(coord, 3) for coord in objectPosition]
    
#     print(f"Position of {object_name}: {roundedPosition}")
#     return roundedPosition


