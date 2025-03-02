import numpy as np
import random
from ActInfAgent.env import CoppeliaEnv, initialize_environment, clear_environment, move_to_grid, grid_to_coordinates, create_bounding_locations

# Define possible actions
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

def update_vision(current_location, grid_dims, distance):
    """
    Update the agent's field of vision based on the current location and distance
    Returns a list of all grid positions within the vision range
    """
    x, y = current_location
    vision = []
    
    for i in range(max(0, x - distance), min(grid_dims[0], x + distance + 1)):
        for j in range(max(0, y - distance), min(grid_dims[1], y + distance + 1)):
            if (i, j) != current_location:  # Don't include current location
                vision.append((i, j))
    
    return vision

def create_observation(location, red_obs, green_obs, white_obs, grid_dims=(40, 40)):
    """
    Create observation for the active inference agent
    Returns: List of 2 integers:
        - First int is the location ID (flattened grid position)
        - Second int is the state label (0=safe/white, 1=dangerous/red, 2=rewarding/green)
    """
    x, y = location
    location_id = x * grid_dims[1] + y  # Convert (x,y) to flattened index
    
    # Determine state label
    if 'Null' not in red_obs and location in red_obs:
        state_label = 1  # Dangerous
    elif location == green_obs and green_obs != 'Null':
        state_label = 2  # Rewarding
    else:
        state_label = 0  # Safe
        
    return [location_id, state_label]  # Format expected by pymdp agent

def update_vision_matrix(vision_matrix, obs):
    """
    Update the observation matrix based on an observation
    """
    location_id, state_label = obs
    
    # Create one-hot encoding for the state
    one_hot_obs = [0, 0, 0]
    one_hot_obs[state_label] = 1
    
    # Update vision matrix at the location
    vision_matrix[location_id] = one_hot_obs
    
    return vision_matrix

def generate_redspots_from_positions(flat_positions, obstacle_dimensions):
    """Helper function to consistently generate redspots from obstacle positions"""
    redspots = set()  # Use set to prevent duplicates
    
    # Process each obstacle position
    for i in range(0, len(flat_positions), 2):
        if i+1 < len(flat_positions):
            obstacle_pos = (flat_positions[i], flat_positions[i+1])
            
            # Get bounding locations for this obstacle
            bounding_locs = create_bounding_locations(obstacle_pos, obstacle_dimensions)
            
            # Convert all boundary points to grid positions
            for location in bounding_locs:
                bx, by = location
                grid_position = move_to_grid(bx, by)
                
                # Only add valid grid positions
                if isinstance(grid_position, tuple):
                    redspots.add(grid_position)  # Using set.add() to automatically handle duplicates
    
    return list(redspots)

def is_path_to_goal_exists(start_pos, goal_pos, redspots, grid_dims):
    """Check if there exists a valid path from start to goal avoiding redspots"""
    if start_pos == goal_pos:
        return True
        
    # Using BFS to find path
    queue = [(start_pos, [start_pos])]
    visited = {start_pos}
    
    while queue:
        (x, y), path = queue.pop(0)
        
        # Check all possible moves
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            next_pos = (next_x, next_y)
            
            if (0 <= next_x < grid_dims[0] and 
                0 <= next_y < grid_dims[1] and 
                next_pos not in visited and 
                next_pos not in redspots):
                
                if next_pos == goal_pos:
                    return True
                    
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
    
    return False

def test_active_inference(random_seed, agent, grid_dims=(40, 40), max_steps=50):
    """
    Test the active inference agent in the Coppelia environment.
    """
    # Initialize environment with the random seed
    flat_positions, goal_position, obstacle_handles, goal_handle, bubbleRob_position, redspots = initialize_environment(seed=random_seed)
    
    # Convert positions to grid coordinates with validation
    start_pos = move_to_grid(bubbleRob_position[0], bubbleRob_position[1])
    goal_grid_pos = move_to_grid(goal_position[0], goal_position[1])
    
    if not isinstance(start_pos, tuple) or not isinstance(goal_grid_pos, tuple):
        raise ValueError("Invalid grid positions generated for start or goal location")
        
    print(f"Starting position: {bubbleRob_position} -> grid: {start_pos}")
    print(f"Goal position: {goal_position} -> grid: {goal_grid_pos}")
    
    # Generate redspots if not provided
    if not redspots:
        obstacle_dimensions = [0.3, 0.3, 0.8]
        redspots = set()  # Use a set to avoid duplicates
        
        # Process each obstacle position
        for i in range(0, len(flat_positions), 2):
            if i+1 < len(flat_positions):
                obstacle_pos = (flat_positions[i], flat_positions[i+1])
                # Get grid position for obstacle center first
                grid_pos = move_to_grid(obstacle_pos[0], obstacle_pos[1])
                if isinstance(grid_pos, tuple):
                    redspots.add(grid_pos)
                
                # Get boundary points for comprehensive coverage
                bounding_locs = create_bounding_locations(obstacle_pos, obstacle_dimensions)
                for location in bounding_locs:
                    grid_position = move_to_grid(location[0], location[1])
                    if isinstance(grid_position, tuple):
                        redspots.add(grid_position)
        
        redspots = list(redspots)  # Convert back to list
        
    print(f"Number of redspots: {len(redspots)}")
    
    # Verify goal position is not in redspots
    if goal_grid_pos in redspots:
        redspots.remove(goal_grid_pos)
        print("Removed goal position from redspots")
    
    # Initialize environment class
    env = CoppeliaEnv(
        starting_loc=start_pos, 
        redspots=redspots,
        goal=goal_grid_pos,
        grid_dims=grid_dims
    )
    
    # Reset environment and get initial observation
    loc_obs, green_obs, white_obs, red_obs, agent_reward, distance = env.reset()
    
    # Initialize results tracking
    results = {
        'history_of_locs': [loc_obs],
        'history_of_rewards': [agent_reward],
        'history_of_distances': [distance],
        'reached_goal': False,
        'total_steps': 0,
        'total_reward': agent_reward,
        'total_distance': distance
    }
    
    # Main simulation loop
    for t in range(max_steps):
        # Create current observation
        obs = create_observation(loc_obs, red_obs, green_obs, white_obs, grid_dims)
        
        # Update agent's beliefs
        qs = agent.infer_states(obs)
        
        # Update observation matrix using surrounding locations
        surrounding_locs = update_vision(loc_obs, grid_dims, 2)
        for loc in surrounding_locs:
            obs = create_observation(loc, red_obs, green_obs, white_obs, grid_dims)
        
        # Get agent's action
        agent.infer_policies()
        chosen_action_id = agent.sample_action()
        movement_id = int(chosen_action_id[0])
        choice_action = actions[movement_id]
        
        # Take step in environment
        loc_obs, green_obs, white_obs, red_obs, agent_reward, distance = env.step(choice_action)
        
        # Update results
        results['history_of_locs'].append(loc_obs)
        results['history_of_rewards'].append(agent_reward)
        results['history_of_distances'].append(distance)
        results['total_reward'] += agent_reward
        results['total_distance'] = distance
        results['total_steps'] = t + 1
        
        # Check if goal reached
        if env.current_location == goal_grid_pos:
            results['reached_goal'] = True
            print(f"Goal reached at step {t}!")
            break
    
    # Clean up environment
    clear_environment(obstacle_handles, goal_handle)
    
    return results