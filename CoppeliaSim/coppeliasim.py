# from coppeliasim_zmqremoteapi_client import RemoteAPIClient
# import numpy as np
# import time

# # Step 1: Create a client and get handles
# client = RemoteAPIClient()
# sim = client.getObject('sim')

# # Step 2: Get the object handle for the Cuboid
# floorHandle = sim.getObject('/Floor')
# ctrlPts = [
#     0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0,
#     0.5, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0,  # Start point (position and neutral orientation)
#     1.0, 1.0, 0.05, 0.0, 0.0, 0.0, 1.0   # End point (position and neutral orientation)
# ]

# # Create the path
# pathHandle = sim.createPath(
#     ctrlPts,   # Control points defining the path
#     0,         # Options (bit0 is not set, meaning path is open)
#     100,       # Subdivision (number of points to create along the path for smoother interpolation)
#     1,       # Smoothness (Bezier interpolation is off for a linear path)
#     0,         # Orientation mode (x-axis along path, y-axis is up)
#     [0.0, 0.0, 1.0],  # Up vector for path orientation 
# )

# sim.setObjectParent(pathHandle, floorHandle, True)

# # Get the handle of the previously created path and cuboid
# cuboidHandle = sim.getObject('/Cuboid')

# # Initialize path data
# pathData = sim.unpackDoubleTable(sim.getBufferProperty(pathHandle, 'customData.PATH'))
# m = np.array(pathData).reshape(len(pathData) // 7, 7)
# pathPositions = m[:, :3].flatten().tolist()
# pathQuaternions = m[:, 3:].flatten().tolist()

# # Get path lengths and other variables
# pathLengths, totalLength = sim.getPathLengths(pathPositions, 3)
# velocity = 0.04  # m/s
# posAlongPath = 0
# previousSimulationTime = 0
# fixedZ = 0.05  # The fixed Z coordinate value

# # Initialize function (formerly sysCall_init)
# def init():
#     global posAlongPath, previousSimulationTime
#     posAlongPath = 0
#     previousSimulationTime = sim.getSimulationTime()

# # Main loop to simulate path following (formerly sysCall_thread)
# def follow_path():
#     global posAlongPath, previousSimulationTime
#     while not sim.getSimulationStopping():
#         t = sim.getSimulationTime()
#         deltaT = t - previousSimulationTime

#         # Skip the first step if deltaT is 0
#         if deltaT <= 0.0:
#             previousSimulationTime = t
#             continue

#         # Calculate the remaining distance and adjust velocity
#         remainingPath = totalLength - posAlongPath
#         adjustedVelocity = min(velocity, remainingPath / deltaT)

#         posAlongPath += adjustedVelocity * deltaT

#         if posAlongPath >= totalLength - 0.001:
#             posAlongPath = totalLength
#             print("Reached the end of the path!")
#             break

#         # Interpolate position along the path (x and y) and fix z at fixedZ
#         t_norm = posAlongPath / totalLength
#         startPos = np.array([0.0, 0.0])  # Starting position (x, y)
#         middlePos = np.array([0.5, 0.5])
#         endPos = np.array([1.0, 1.0])    # Ending position (x, y)
#         interpolatedPosXY = (1 - t_norm)**2 * startPos + 2*t_norm*(1 - t_norm)*middlePos + (t_norm**2)*endPos

#         # Combine the interpolated x and y with the fixed z
#         interpolatedPos = np.append(interpolatedPosXY, fixedZ)

#         # Set object position and orientation
#         sim.setObjectPosition(cuboidHandle, -1, interpolatedPos.tolist())  # -1 means relative to world
#         sim.setObjectQuaternion(cuboidHandle, -1, [0.0, 0.0, 0.0, 1.0])

#         # Update the previous simulation time
#         previousSimulationTime = t

#         # Step the simulation
#         sim.step()
#         time.sleep(0.05)

# # Start the simulation
# sim.startSimulation()

# # Initialize and run the path-following function
# init()
# follow_path()

# # Stop the simulation after 10 seconds (adjust as needed)
# time.sleep(10)
# sim.stopSimulation()



from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time

# # Step 1: Create a client and get handles
client = RemoteAPIClient()
sim = client.getObject('sim')

# # Step 2: Get the object handle for the Cuboid
# floorHandle = sim.getObject('/Floor')

# ctrlPts = [
#     0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0,
#     0.5, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0,  # Start point (position and neutral orientation)
#     1.0, 1.0, 0.05, 0.0, 0.0, 0.0, 1.0   # End point (position and neutral orientation)
# ]

# # Create the path
# sim.createPath(
#     ctrlPts,   # Control points defining the path
#     0,         # Options (bit0 is not set, meaning path is open)
#     100,       # Subdivision (number of points to create along the path for smoother interpolation)
#     1,       # Smoothness (Bezier interpolation is off for a linear path)
#     0,         # Orientation mode (x-axis along path, y-axis is up)
#     [0.0, 0.0, 1.0],  # Up vector for path orientation 
# )

# pathHandle = sim.getObject('/Path')

# sim.setObjectParent(pathHandle, floorHandle, True)

# # Get the handle of the cuboid
# cuboidHandle = sim.getObject('/Cuboid')

# print(f"Path Handle: {pathHandle}")
# alias = sim.getObjectAlias(pathHandle)
# print(f"Path Alias: {alias}")

# pathHandle = sim.getObject('/Path')  # Explicitly retrieve path handle
# if pathHandle == -1:
#     print("Error: Path does not exist!")
# else:
#     print(f"Path handle retrieved successfully: {pathHandle}")

# allObjects = sim.getObjectsInTree(sim.handle_scene)
# print("Objects in scene:", [sim.getObjectAlias(obj) for obj in allObjects])

# objType = sim.getObjectType(pathHandle)
# print(f"Object Type: {objType}")

# # Start the simulation
# sim.startSimulation()

# # # Let the cuboid follow the Bezier curve
# sim.followPath(cuboidHandle, pathHandle, 3, 2, 0.2, 0.05)  # Object, Path, Mode, Options, Speed, Accel

# # # Allow time for simulation
# # time.sleep(10)

# # Stop simulation after time
# sim.stopSimulation()

# def move_to_grid(x, y, z, width=0.25, height=0.25):
#     '''Moves Coppelia coordinates (x,y,z) to a 40x40 grid, keeping only valid points.'''
    
#     # Translate x, y coordinate 2.5 up and 2.5 right
#     x = x + 2.5
#     y = y + 2.5
    
#     # Convert x, y to grid indices
#     x_grid = round(x / 0.25)
#     y_grid = round(y / 0.25)

#     # Check if center is within valid grid bounds
#     if 0 <= x_grid <= 40 and 0 <= y_grid <= 40:
#         return [(x_grid, y_grid)]  # Return as a list (single valid center)

#     # Otherwise, check boundary points
#     boundary_points = [
#         (x + width / 2, y + height / 2),
#         (x + width / 2, y - height / 2),
#         (x - width / 2, y + height / 2),
#         (x - width / 2, y - height / 2)
#     ]

#     valid_points = []
#     for bx, by in boundary_points:
#         bx_grid = round(bx / 0.25)
#         by_grid = round(by / 0.25)
#         if 0 <= bx_grid <= 40 and 0 <= by_grid <= 40:
#             valid_points.append((bx_grid, by_grid))

#     return valid_points if valid_points else "Invalid object position!"

# def grid_to_coordinates(x_grid, y_grid, z):
#     '''Converts a valid 40x40 grid point back to Coppelia coordinates.'''
    
#     if 0 <= x_grid <= 40 and 0 <= y_grid <= 40:
#         x = x_grid * 0.25 - 2.5
#         y = y_grid * 0.25 - 2.5
#         return (x, y, z)
    
#     return "Invalid grid point!"

def move_to_grid(x, y, z):
    # Moves Coppelia coordinates (x,y,z) to a 40x40 grid, keeping only valid points.

    # Walls become negative when move directly to grid

    # Make separate function called wall_locations and take two non-negative locations, consider it walls
    pass

def get_obstacle_position(obstacle_name):
    
    obstacleHandle = sim.getObject(f'/{obstacle_name}')

    if obstacleHandle == -1:
        raise Exception(f"Obstacle '{obstacle_name}' not found.")
    
    obstaclePosition = sim.getObjectPosition(obstacleHandle, -1)

    roundedPosition = [round(coord, 3) for coord in obstaclePosition]

    print(f"Position of {obstacle_name}: {roundedPosition}")
    return roundedPosition

# Idea for Beziér curve, interpolate using quaternions to determine shape of curve

    