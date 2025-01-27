from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time

# Step 1: Create a client and get handles
client = RemoteAPIClient()
sim = client.getObject('sim')

# Step 2: Get the object handle for the Cuboid

ctrlPts = [
    0.0, 0.0, 0, 0.0, 0.0, 0.0, 1.0,  # Start point (position and neutral orientation)
    1.0, 1.0, 0, 0.0, 0.0, 0.0, 1.0   # End point (position and neutral orientation)
]

# Create the path
pathHandle = sim.createPath(ctrlPts, 0, 100, 0.0, 0, [0.0, 0.0, 1.0])
print(f"Created path with handle: {pathHandle}")
# Get the handle of the previously created path
pathHandle = sim.getObject('/Path')  # Assuming the path is named '/Path'
cuboidHandle = sim.getObject('/Cuboid')
# Step 3: Define control points for a straight path from (0,0,-0.1) to (1,1,-0.1)


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

# Step 4: Initialize function (formerly sysCall_init)
def init():
    global posAlongPath, previousSimulationTime
    posAlongPath = 0
    previousSimulationTime = sim.getSimulationTime()

# Step 5: Main loop to simulate path following (formerly sysCall_thread)
def follow_path():
    global posAlongPath, previousSimulationTime
    while not sim.getSimulationStopping():
        # Get the current simulation time
        t = sim.getSimulationTime()
        
        # Update position along the path
        posAlongPath += velocity * (t - previousSimulationTime)
        posAlongPath %= totalLength  # Keep position within total path length
        # Below is the attempt at stopping the object at end of path
        # if posAlongPath >= totalLength:
        #     posAlongPath = totalLength  # Clamp position to the end of the path
        #     print("Reached the end of the path!")
        #     sim.stopSimulation()  # Exit the loop to stop the object
        # Interpolate position along the path (manually)
        t_norm = posAlongPath / totalLength  # Normalized position along the path
        startPos = np.array([0.0, 0.0, 0.0])  # Starting position
        endPos = np.array([1.0, 1.0, 0.0])    # Ending position
        
        # Linear interpolation
        interpolatedPos = (1 - t_norm) * startPos + t_norm * endPos
        
        # Set orientation (fixed in this case)
        orientation = [0.0, 0.0, 0.0, 1.0]  # No rotation (neutral quaternion)
        
        # Step 6: Set object position and orientation
        sim.setObjectPosition(cuboidHandle, -1, interpolatedPos.tolist())  # -1 means relative to world
        sim.setObjectQuaternion(cuboidHandle, -1, orientation)
        
        # Update the previous simulation time
        previousSimulationTime = t
        
        # Step the simulation
        sim.step()
        time.sleep(0.05)  # To avoid overload

# Step 7: Start the simulation
sim.startSimulation()

# Step 8: Run the initialization and simulation
init()  # Initialize the script
follow_path()  # Start following the path

# Step 9: Stop the simulation after some time or based on a condition
time.sleep(10)  # Run the simulation for 10 seconds (adjust as needed)
sim.stopSimulation()