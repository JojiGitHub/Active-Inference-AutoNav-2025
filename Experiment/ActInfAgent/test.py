import numpy as np
import pandas as pd
import random
from datetime import datetime
try:
    from ActInfAgent.agent import create_redspot_agent
    from ActInfAgent.env import initialize_environment, CoppeliaEnv, move_to_grid, get_object_position
except:
    from agent import create_redspot_agent
    from env import initialize_environment, CoppeliaEnv, move_to_grid, get_object_position

def initialize_experiment(random_seed=42):
    """
    Initialize the experiment with a given random seed.
    
    Args:
        random_seed (int): Seed for random number generation to ensure reproducibility
        
    Returns:
        tuple: (agent, env, grid_locations, actions, agent_pos, goal_location, metrics_dict)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Initialize environment and get initial positions
    print("Initializing CoppeliaSim environment...")
    obstacle_positions, goal_position, obstacle_handles, goal_handle, bubbleRob_position, redspots = initialize_environment(random_seed)
    
    if not all([goal_position, bubbleRob_position]):
        raise RuntimeError("Failed to initialize environment: Could not place all objects")
    
    # Convert CoppeliaSim positions to grid coordinates
    agent_pos = move_to_grid(bubbleRob_position[0], bubbleRob_position[1])
    goal_location = move_to_grid(goal_position[0], goal_position[1])
    
    if not all([agent_pos, goal_location]):
        raise RuntimeError("Failed to convert positions to grid coordinates")
    
    print(f"Agent starting position: {agent_pos}")
    print(f"Goal location: {goal_location}")
    print(f"Number of red spots: {len(redspots)}")
    
    # Initialize Active Inference agent
    agent, grid_locations, actions = create_redspot_agent(
        agent_pos=agent_pos,
        goal_location=goal_location,
        policy_len=3
    )
    
    # Initialize CoppeliaSim environment wrapper
    env = CoppeliaEnv(
        redspots=redspots,
        starting_loc=agent_pos,
        goal=goal_location
    )
    
    # Initialize metrics dictionary for data collection
    metrics_dict = {
        'random_seed': random_seed,
        'steps': [],
        'agent_positions': [],
        'actions_taken': [],
        'rewards': [],
        'shannon_entropy': [],
        'goal_distance': [],
        'sim_positions': []  # Track actual CoppeliaSim positions
    }
    
    return agent, env, grid_locations, actions, agent_pos, goal_location, metrics_dict

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def calculate_shannon_entropy(q):
    """
    Calculate Shannon entropy of a probability distribution with proper numerical stability
    
    Args:
        q: Probability distribution (numpy array or tuple)
        
    Returns:
        float: Shannon entropy value
    """
    # Convert tuple to numpy array if needed
    if isinstance(q, tuple):
        q_array = np.array(q)
    else:
        q_array = q
        
    # Ensure probabilities are valid (non-negative and sum to 1)
    q_array = np.clip(q_array, 0, 1)
    q_array = q_array / (q_array.sum() + 1e-15)  # Normalize to ensure sum is 1
    
    # Calculate entropy only for non-zero probabilities
    nonzero_mask = q_array > 0
    if not nonzero_mask.any():
        return 0.0
    
    q_nonzero = q_array[nonzero_mask]
    return -np.sum(q_nonzero * np.log(q_nonzero))

def run_episode(agent, env, grid_locations, actions, goal_location, metrics_dict, max_steps=100):
    """
    Run a single episode of the Active Inference agent
    
    Args:
        agent: The Active Inference agent instance
        env: The environment instance
        grid_locations: List of grid locations
        actions: List of possible actions
        goal_location: Target location
        metrics_dict: Dictionary to store metrics
        max_steps (int): Maximum number of steps per episode
        
    Returns:
        dict: Updated metrics dictionary with episode data
    """
    # Reset environment
    loc_obs, green_obs, white_obs, red_obs, reward = env.reset()
    
    # Run episode
    for step in range(max_steps):
        # Create observation for agent
        obs = [grid_locations.index(loc_obs), create_color_observation(loc_obs, red_obs, green_obs, white_obs)]
        
        # Get agent's action
        qs = agent.infer_states(obs)
        q = agent.infer_policies()
        action_idx = agent.sample_action()
        
        # Handle multi-dimensional action index
        if isinstance(action_idx, np.ndarray):
            if action_idx.size > 1:
                # Take the first action if multiple are returned
                action_idx = action_idx[0]
            action_idx = int(action_idx)
        
        action = actions[action_idx]
        
        # Calculate metrics with improved numerical stability
        shannon_entropy = calculate_shannon_entropy(q)
        goal_distance = calculate_distance(loc_obs, goal_location)
        
        # Store metrics
        metrics_dict['steps'].append(step)
        metrics_dict['agent_positions'].append(loc_obs)
        metrics_dict['actions_taken'].append(action)
        metrics_dict['rewards'].append(reward)
        metrics_dict['shannon_entropy'].append(shannon_entropy)
        metrics_dict['goal_distance'].append(goal_distance)
        
        # Take action in environment
        loc_obs, green_obs, white_obs, red_obs, reward = env.step(action)
        
        # Check if goal reached
        if loc_obs == goal_location:
            print(f"Goal reached in {step+1} steps!")
            break
    
    # Add final step data if goal wasn't reached
    if loc_obs != goal_location and step == max_steps-1:
        print(f"Goal not reached after {max_steps} steps.")
        
    # Store final outcome
    metrics_dict['goal_reached'] = loc_obs == goal_location
    metrics_dict['total_steps'] = step + 1
    metrics_dict['total_reward'] = sum(metrics_dict['rewards'])
    metrics_dict['avg_shannon_entropy'] = np.mean(metrics_dict['shannon_entropy'])
    
    return metrics_dict

def create_color_observation(position, red_obs, green_obs, white_obs):
    """Determine the color observation for a given position"""
    if red_obs != ['Null'] and position in red_obs:
        return 1  # RED
    elif green_obs == position:
        return 2  # GREEN
    else:
        return 0  # WHITE (default case for unknown or white positions)

def run_experiment(random_seed=42, max_steps=100, save_data=True):
    """
    Run a complete experiment with data collection
    
    Args:
        random_seed (int): Random seed for reproducibility
        max_steps (int): Maximum steps per episode
        save_data (bool): Whether to save data to CSV
        
    Returns:
        pd.DataFrame: DataFrame containing experiment results
    """
    # Initialize experiment
    agent, env, grid_locations, actions, agent_pos, goal_location, metrics_dict = initialize_experiment(random_seed)
    
    # Run episode
    metrics = run_episode(agent, env, grid_locations, actions, goal_location, metrics_dict, max_steps)
    
    # Convert metrics to DataFrame
    df = pd.DataFrame({
        'step': metrics['steps'],
        'position_x': [pos[0] for pos in metrics['agent_positions']],
        'position_y': [pos[1] for pos in metrics['agent_positions']],
        'action': metrics['actions_taken'],
        'reward': metrics['rewards'],
        'shannon_entropy': metrics['shannon_entropy'],
        'goal_distance': metrics['goal_distance'],
    })
    
    # Add experiment metadata
    df['random_seed'] = random_seed
    df['goal_reached'] = metrics['goal_reached']
    df['total_steps'] = metrics['total_steps']
    df['total_reward'] = metrics['total_reward']
    df['avg_shannon_entropy'] = metrics['avg_shannon_entropy']
    
    # Save data if requested
    if save_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"actinf_experiment_seed{random_seed}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    return df

def test_active_inference(random_seed=42, num_steps=100, policy_len=3):
    """
    Run a single episode of the Active Inference agent and collect performance data
    This function provides backward compatibility with the previous implementation
    
    Args:
        random_seed (int): Random seed for reproducibility
        num_steps (int): Maximum number of steps
        policy_len (int): Length of policy for the agent
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Initialize experiment
    agent, env, grid_locations, actions, agent_pos, goal_location, metrics_dict = initialize_experiment(random_seed)
    
    # Run episode
    metrics = run_episode(agent, env, grid_locations, actions, goal_location, metrics_dict, num_steps)
    
    # Return summarized results (compatible with previous interface)
    results = {
        'total_distance': sum(metrics['goal_distance']),
        'total_steps': metrics['total_steps'],
        'avg_shannon_entropy': metrics['avg_shannon_entropy'],
        'revisited_squares': len(set(metrics['agent_positions'])),
        'goal_reached': metrics['goal_reached']
    }
    
    return results

if __name__ == "__main__":
    # Run test with a specific random seed
    df = run_experiment(random_seed=41, max_steps=100, save_data=True)
    print("\nExperiment complete. Summary statistics:")
    print(df.describe())