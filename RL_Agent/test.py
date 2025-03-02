import torch
import numpy as np
from env import CoppeliaSim, COPPELIA_AVAILABLE
from agent import ImprovedDQNAgent
import time

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
    exit(1)

# Check if CoppeliaSim is available
if not COPPELIA_AVAILABLE:
    print("CoppeliaSim is not available. Cannot run the test.")
    exit(1)

# Initialize the CoppeliaSim environment with random obstacles
print("\nInitializing CoppeliaSim environment...")
random_seed = 42  # Using a fixed seed for reproducibility 
num_obstacles = 20  # Setting number of obstacles to 20

# Initialize CoppeliaSim with our hyperparameters
env = CoppeliaSim(random_seed=random_seed, num_obstacles=num_obstacles)

# Generate random environment in CoppeliaSim
env.initialize_environment()

print(f"Generated random environment with {num_obstacles} obstacles using seed {random_seed}:")
print(f"Red zone positions: {env.red_zone_positions}")
print(f"Goal position: {env.goal_position}")

# Visualize the environment
env.render()

# Run test episodes
num_test_episodes = 5
all_rewards = []
best_episode_reward = float('-inf')

for test_episode in range(num_test_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    print(f"\nStarting Test Episode {test_episode + 1}/{num_test_episodes}")
    print(f"Initial Position: {env.agent_grid_position}")
    print(f"Goal Position: {env.goal_position}")
    print(f"Red Zone Positions: {env.red_zone_positions}")
    
    # Visualize initial state
    env.render()
    
    while not done:
        # Agent selects action with testing=True
        action = agent.select_action(state, testing=True)
        action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][action]
        
        current_pos = env.agent_grid_position
        next_state, reward, done, info = env.step(action)
        new_pos = env.agent_grid_position
        
        print("\n" + "="*50)
        print(f"Step {step_count}:")
        print(f"Current Position: {current_pos} → New Position: {new_pos}")
        print(f"Action Taken: {action_name} (Action value: {action})")
        print(f"Reward: {reward:.2f}")
        print(f"Total Reward So Far: {total_reward:.2f}")
        
        # Updated to split the state representation
        visible_area = state[:17]  # First 17 elements are the visible cells
        coordinates = state[17:]   # Last 4 elements are coordinates
        
        print("\nVisible Area Colors:")
        for i, color in enumerate(visible_area):
            if color > 0:
                color_type = {
                    1.0: "RED",
                    2.0: "GREEN (GOAL)"
                }.get(color, "UNKNOWN")
                print(f"Cell {i}: {color_type}")
                
        print("\nCoordinate Information:")
        print(f"Agent position (x, y): ({coordinates[0]}, {coordinates[1]})")
        print(f"Goal position (x, y): ({coordinates[2]}, {coordinates[3]})")
        
        print(f"Done: {done}")
        print("="*50 + "\n")
        
        state = next_state
        total_reward += reward
        step_count += 1
        
        # Render the environment
        env.render()
        time.sleep(0.5)  # Add delay to observe the visualization
    
    all_rewards.append(total_reward)
    if total_reward > best_episode_reward:
        best_episode_reward = total_reward
    
    print(f"\nTest Episode {test_episode + 1} Results:")
    print(f"Total Steps: {step_count}")
    print(f"Episode Reward: {total_reward:.2f}")
    
    # Provide episode end reason
    if step_count >= env.max_steps:
        print("Episode ended because: Maximum steps reached")
    elif new_pos in env.red_zone_positions:
        print("Episode ended because: Agent hit a red zone")
    elif new_pos == env.goal_position:
        print("Episode ended because: Goal reached!")

print("\nOverall Test Results:")
print(f"Average Reward over {num_test_episodes} episodes: {np.mean(all_rewards):.2f}")
print(f"Best Episode Reward: {best_episode_reward:.2f}")
print(f"Reward Standard Deviation: {np.std(all_rewards):.2f}")
print(f"All Episode Rewards: {[f'{r:.2f}' for r in all_rewards]}")

# Stop CoppeliaSim simulation
env.close()
