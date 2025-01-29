from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time

# Step 1: Create a client and get handles
client = RemoteAPIClient()
sim = client.getObject('sim')

# Step 2: Get the object handle for the Cuboid
floorHandle = sim.getObject('/Floor')
ctrlPts = [
    0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0,  # Start point (position and neutral orientation)
    1.0, 1.0, 0.05, 0.0, 0.0, 0.0, 1.0   # End point (position and neutral orientation)
]

# Create the path
pathHandle = sim.createPath(
    ctrlPts,   # Control points defining the path
    0,         # Options (bit0 is not set, meaning path is open)
    100,       # Subdivision (number of points to create along the path for smoother interpolation)
    0.0,       # Smoothness (Bezier interpolation is off for a linear path)
    0,         # Orientation mode (x-axis along path, y-axis is up)
    [0.0, 0.0, 1.0],  # Up vector for path orientation 
)

sim.setObjectParent(pathHandle, floorHandle, True)

# Get the handle of the previously created path and cuboid
cuboidHandle = sim.getObject('/Cuboid')

# Initialize path data
pathData = sim.unpackDoubleTable(sim.getBufferProperty(pathHandle, 'customData.PATH'))
m = np.array(pathData).reshape(len(pathData) // 7, 7)
pathPositions = m[:, :3].flatten().tolist()
pathQuaternions = m[:, 3:].flatten().tolist()

# Get path lengths and other variables
pathLengths, totalLength = sim.getPathLengths(pathPositions, 3)
velocity = 0.04  # m/s
posAlongPath = 0
previousSimulationTime = 0
fixedZ = 0.05  # The fixed Z coordinate value

# Initialize function (formerly sysCall_init)
def init():
    global posAlongPath, previousSimulationTime
    posAlongPath = 0
    previousSimulationTime = sim.getSimulationTime()

# Main loop to simulate path following (formerly sysCall_thread)
def follow_path():
    global posAlongPath, previousSimulationTime
    while not sim.getSimulationStopping():
        t = sim.getSimulationTime()
        deltaT = t - previousSimulationTime

        # Skip the first step if deltaT is 0
        if deltaT <= 0.0:
            previousSimulationTime = t
            continue

        # Calculate the remaining distance and adjust velocity
        remainingPath = totalLength - posAlongPath
        adjustedVelocity = min(velocity, remainingPath / deltaT)

        posAlongPath += adjustedVelocity * deltaT

        if posAlongPath >= totalLength - 0.001:
            posAlongPath = totalLength
            print("Reached the end of the path!")
            break

        # Interpolate position along the path (x and y) and fix z at fixedZ
        t_norm = posAlongPath / totalLength
        startPos = np.array([0.0, 0.0])  # Starting position (x, y)
        endPos = np.array([1.0, 1.0])    # Ending position (x, y)
        interpolatedPosXY = (1 - t_norm) * startPos + t_norm * endPos

        # Combine the interpolated x and y with the fixed z
        interpolatedPos = np.append(interpolatedPosXY, fixedZ)

        # Set object position and orientation
        sim.setObjectPosition(cuboidHandle, -1, interpolatedPos.tolist())  # -1 means relative to world
        sim.setObjectQuaternion(cuboidHandle, -1, [0.0, 0.0, 0.0, 1.0])

        # Update the previous simulation time
        previousSimulationTime = t

        # Step the simulation
        sim.step()
        time.sleep(0.05)

# Start the simulation
sim.startSimulation()

# Initialize and run the path-following function
init()
follow_path()

# Stop the simulation after 10 seconds (adjust as needed)
time.sleep(10)
sim.stopSimulation()
