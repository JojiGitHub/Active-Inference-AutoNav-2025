import torch
import numpy as np
from RL_Agent.env import CoppeliaSim, COPPELIA_AVAILABLE
from RL_Agent.agent import ImprovedDQNAgent
import time
from typing import Dict, Any

def run_environment(random_seed) -> Dict[str, Any]:
    """
    Run one episode in the CoppeliaSim environment with the given random seed.
    
    Args:
        random_seed (int): Seed for random number generation
        
    Returns:
        dict: Results containing episode statistics and final state
    """

    # Load the best model and verify settings
    state_dim = 21  # Updated from 17 to 21 to include agent and goal coordinates
    action_dim = 5
    agent = ImprovedDQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    try:
        # Load the model exactly as saved
        checkpoint = torch.load("RL_Agent/best_model.pth")
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']  # Use the saved epsilon
        # Keep networks in evaluation mode
        agent.q_network.eval()
        agent.target_network.eval()
        print("\nModel Loading Verification:")
        print(f"Loaded model from episode {checkpoint['episode']}")
        print(f"Original training reward: {checkpoint['reward']:.2f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure to train the model first by running train.py")
        return None

    # Check if CoppeliaSim is available
    if not COPPELIA_AVAILABLE:
        print("CoppeliaSim is not available. Cannot run the test.")
        return None
    
    # Set the number of obstacles
    random.seed(random_seed)
    num_obstacles = random.randint(20, 50)

    # Initialize the CoppeliaSim environment
    print("\nInitializing CoppeliaSim environment...")
    env = CoppeliaSim(random_seed=random_seed)
    env.initialize_environment()

    print(f"Generated random environment with {num_obstacles} obstacles using seed {random_seed}:")
    print(f"Red zone positions: {env.red_zone_positions}")
    print(f"Goal position: {env.goal_position}")
    env.render()

    # Run one episode
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    episode_history = []
    
    print(f"\nStarting episode with seed {random_seed}")
    print(f"Initial Position: {env.agent_grid_position}")
    print(f"Goal Position: {env.goal_position}")
    print(f"Red Zone Positions: {env.red_zone_positions}")
    
    env.render()
    
    while not done:
        # Agent selects action with testing=True
        action = agent.select_action(state, testing=True)
        action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][action]
        
        current_pos = env.agent_grid_position
        next_state, reward, done, info = env.step(action)
        new_pos = env.agent_grid_position
        
        # Store step information with distance metrics
        step_info = {
            'step': step_count,
            'current_pos': current_pos,
            'new_pos': new_pos,
            'action': action_name,
            'reward': reward,
            'total_reward': total_reward,
            'distance_traveled': info['step_distance'],
            'total_distance': info['total_distance'],
            'state': {
                'visible_area': state[:17].tolist(),
                'coordinates': state[17:].tolist()
            }
        }
        episode_history.append(step_info)
        
        print("\n" + "="*50)
        print(f"Step {step_count}:")
        print(f"Current Position: {current_pos} → New Position: {new_pos}")
        print(f"Action Taken: {action_name} (Action value: {action})")
        print(f"Reward: {reward:.2f}")
        print(f"Total Reward So Far: {total_reward:.2f}")
        print(f"Distance this step: {info['step_distance']:.4f}")
        print(f"Total distance traveled: {info['total_distance']:.4f}")
        
        # Display state information
        visible_area = state[:17]
        coordinates = state[17:]
        
        print("\nVisible Area Colors:")
        for i, color in enumerate(visible_area):
            if color > 0:
                color_type = {1.0: "RED", 2.0: "GREEN (GOAL)"}.get(color, "UNKNOWN")
                print(f"Cell {i}: {color_type}")
                
        print("\nCoordinate Information:")
        print(f"Agent position (x, y): ({coordinates[0]}, {coordinates[1]})")
        print(f"Goal position (x, y): ({coordinates[2]}, {coordinates[3]})")
        print(f"Done: {done}")
        print("="*50 + "\n")
        
        state = next_state
        total_reward += reward
        step_count += 1
        
        env.render()
        time.sleep(0.5)
    
    # Determine episode end reason
    if step_count >= env.max_steps:
        end_reason = "Maximum steps reached"
    elif new_pos in env.red_zone_positions:
        end_reason = "Agent hit a red zone"
    elif new_pos == env.goal_position:
        end_reason = "Goal reached!"
    else:
        end_reason = "Unknown"
        
    print(f"\nEpisode Results:")
    print(f"Total Steps: {step_count}")
    print(f"Episode Reward: {total_reward:.2f}")
    print(f"Episode ended because: {end_reason}")

    # Stop CoppeliaSim simulation
    env.close()
    
    # Return comprehensive results with new metrics
    results = {
        'random_seed': random_seed,
        'num_obstacles': num_obstacles,
        'total_steps': step_count,
        'total_reward': total_reward,
        'total_distance': env.total_distance,
        'end_reason': end_reason,
        'final_position': new_pos,
        'goal_position': env.goal_position,
        'red_zone_positions': env.red_zone_positions,
        'episode_history': episode_history,
        'performance_metrics': {
            'total_timesteps': step_count,
            'total_distance': env.total_distance,
            'average_distance_per_step': env.total_distance / step_count if step_count > 0 else 0,
            'goal_reached': end_reason == "Goal reached!"
        }
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    results = run_environment(random_seed=42, num_obstacles=20)
    if results:
        print("\nRun completed successfully!")
        print(f"Final reward: {results['total_reward']:.2f}")
        print(f"Total timesteps: {results['total_steps']}")
        print(f"Total distance traveled: {results['total_distance']:.4f}")
        print(f"Average distance per step: {results['performance_metrics']['average_distance_per_step']:.4f}")
        print(f"End reason: {results['end_reason']}")
