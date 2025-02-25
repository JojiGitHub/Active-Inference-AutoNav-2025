import torch
import numpy as np
from env import GridWorldEnv
from agent import ImprovedDQNAgent

# Load the best model and verify settings
state_dim = 21  # Updated from 17 to 21 to include agent and goal coordinates
action_dim = 5
agent = ImprovedDQNAgent(state_dim=state_dim, action_dim=action_dim)

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

# Initialize environment with same settings as training
env = GridWorldEnv(
    initial_position=(0, 0),
    red_zone_positions=[(1, 2), (3, 2), (4, 4), (6, 1)],
    goal_position=(6, 4)
)

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
    print(f"Initial Position: ({env.agent_x}, {env.agent_y})")
    print(f"Goal Position: {env.goal_position}")
    
    while not done:
        # Agent selects action with testing=True
        action = agent.select_action(state, testing=True)
        action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][action]
        
        current_pos = (env.agent_x, env.agent_y)
        next_state, reward, done, _ = env.step(action)
        new_pos = (env.agent_x, env.agent_y)
        
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
        
        env.render(show=True)
    
    all_rewards.append(total_reward)
    if total_reward > best_episode_reward:
        best_episode_reward = total_reward
    
    print(f"\nTest Episode {test_episode + 1} Results:")
    print(f"Total Steps: {step_count}")
    print(f"Episode Reward: {total_reward:.2f}")
    if done:
        print("Episode ended because:", end=" ")
        if step_count >= env.max_steps:
            print("Maximum steps reached")
        elif new_pos in env.red_zone_positions:
            print("Agent hit a red zone")
        elif new_pos == env.goal_position:
            print("Goal reached!")

print("\nOverall Test Results:")
print(f"Average Reward over {num_test_episodes} episodes: {np.mean(all_rewards):.2f}")
print(f"Best Episode Reward: {best_episode_reward:.2f}")
print(f"Reward Standard Deviation: {np.std(all_rewards):.2f}")
print(f"All Episode Rewards: {[f'{r:.2f}' for r in all_rewards]}")
