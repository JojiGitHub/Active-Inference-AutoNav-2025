import torch
import numpy as np
from agent import ImprovedDQNAgent, PrioritizedReplayBuffer
from env import CoppeliaSim, COPPELIA_AVAILABLE
import time
import math

# Helper function to calculate Manhattan distance
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Helper function to calculate path efficiency
def calculate_path_efficiency(start_pos, end_pos, actual_steps):
    optimal_distance = manhattan_distance(start_pos, end_pos)
    return optimal_distance / max(actual_steps, optimal_distance)

# Hyperparameters
state_dim = 21  # 21 includes agent and goal coordinates
action_dim = 5
batch_size = 64
lr = 0.001
gamma = 0.9
num_episodes = 1000
max_timesteps = 80  # Increased to match the environment's max_steps for the larger environment

# Initialize CoppeliaSim environment
if not COPPELIA_AVAILABLE:
    print("CoppeliaSim is not available. Cannot train without simulation.")
    exit(1)

print("\nInitializing CoppeliaSim environment for training...")
# Create a random seed for reproducibility
random_seed = int(time.time())  # Use time as seed for variety during training
# The environment will generate between 20-50 obstacles

# Initialize CoppeliaSim with our hyperparameters
env = CoppeliaSim(random_seed=random_seed)

# Generate random environment in CoppeliaSim - using environment's num_obstacles
red_zone_positions, goal_position = env.initialize_environment()
print(f"Generated random environment with {env.num_obstacles} obstacles using seed {random_seed}")
print(f"Red zone positions: {red_zone_positions}")
print(f"Goal position: {goal_position}")

# Visualize the initial environment
print("\nInitial CoppeliaSim Environment:")
env.render()

# Initialize agent and replay buffer
agent = ImprovedDQNAgent(state_dim=state_dim, action_dim=action_dim, lr=lr)
buffer = PrioritizedReplayBuffer(capacity=10000)

# Initialize tracking variables
best_reward = float('-inf')
episode_rewards = []
episode_steps = []
successful_runs = []  # Track episodes that reached the goal

# Training loop
for episode in range(num_episodes):
    state = env.reset()  # Reset environment and get initial state
    done = False
    total_reward = 0
    steps = 0
    reached_goal = False
    start_position = env.agent_grid_position
    path_positions = [start_position]  # Track the path taken
    
    while not done and steps < max_timesteps:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        
        if buffer.position > batch_size:
            agent.update(buffer, batch_size)
        
        state = next_state
        total_reward += reward
        steps += 1
        path_positions.append(env.agent_grid_position)
        
        # Check if agent reached goal
        if env.agent_grid_position == env.goal_position:
            reached_goal = True
    
    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    
    if reached_goal:
        successful_runs.append((total_reward, steps, episode))
    
    # Calculate averages
    recent_avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
    recent_avg_steps = np.mean(episode_steps[-10:]) if len(episode_steps) >= 10 else np.mean(episode_steps)
    
    print(f"Episode {episode + 1}/{num_episodes}")
    print(f"Steps: {steps}, Avg Steps: {recent_avg_steps:.1f}")
    print(f"Total Reward: {total_reward:.2f}, Avg Reward: {recent_avg_reward:.2f}")
    print(f"Reached Goal: {reached_goal}")
    
    # Visualize progress occasionally
    if episode % 10 == 0:
        env.render()
    
    # Save only if we reach the goal and meet performance criteria
    if reached_goal:
        # Calculate performance metrics
        initial_distance = manhattan_distance(start_position, env.goal_position)
        path_efficiency = calculate_path_efficiency(start_position, env.goal_position, steps)
        obstacle_density = len(env.red_zone_positions) / (env.grid_size[0] * env.grid_size[1])
        
        # Quick validation with 3 episodes
        val_metrics = []
        val_successes = 0
        
        for _ in range(3):
            val_state = env.reset()
            val_done = False
            val_reward = 0
            val_step_count = 0
            val_reached_goal = False
            val_start_pos = env.agent_grid_position
            
            while not val_done and val_step_count < max_timesteps:
                val_action = agent.select_action(val_state)
                val_next_state, val_rew, val_done, _ = env.step(val_action)
                
                val_reward += val_rew
                val_state = val_next_state
                val_step_count += 1
                
                if env.agent_grid_position == env.goal_position:
                    val_reached_goal = True
                    val_efficiency = calculate_path_efficiency(val_start_pos, env.goal_position, val_step_count)
                    val_metrics.append((val_reward, val_efficiency))
                    break
            
            if val_reached_goal:
                val_successes += 1
        
        # Calculate composite score considering multiple factors
        if val_successes >= 2:  # Must reach goal in at least 2/3 validation runs
            val_avg_reward = np.mean([m[0] for m in val_metrics])
            val_avg_efficiency = np.mean([m[1] for m in val_metrics])
            
            # Composite score considers:
            # - Validation average reward
            # - Path efficiency in both training and validation
            # - Initial distance to goal (to avoid lucky short-distance runs)
            # - Obstacle density (reward more complex environments)
            composite_score = (
                val_avg_reward * 0.3 +  # Reward still matters
                path_efficiency * 0.2 +  # Training episode efficiency
                val_avg_efficiency * 0.2 +  # Validation efficiency
                initial_distance * 0.15 +  # Prefer longer distances
                obstacle_density * 0.15  # Prefer complex environments
            )
            
            if composite_score > best_reward:
                best_reward = composite_score
                torch.save({
                    'q_network_state_dict': agent.q_network.state_dict(),
                    'target_network_state_dict': agent.target_network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'episode': episode,
                    'reward': total_reward,
                    'steps': steps,
                    'reached_goal': True,
                    'path_efficiency': path_efficiency,
                    'initial_distance': initial_distance,
                    'obstacle_density': obstacle_density,
                    'composite_score': composite_score
                }, "best_model.pth")
                print(f"\nNew best model saved!")
                print(f"Reward: {total_reward:.2f}, Steps: {steps}")
                print(f"Path Efficiency: {path_efficiency:.2f}")
                print(f"Initial Distance: {initial_distance}")
                print(f"Obstacle Density: {obstacle_density:.2f}")
                print(f"Composite Score: {composite_score:.2f}")
                print(f"Validation success rate: {val_successes}/3")
    
    # Early stopping if we consistently reach goal
    if len(episode_rewards) >= 10:
        recent_successes = sum(1 for r in episode_rewards[-10:] if r > 40)  # Reward > 40 indicates goal reached
        if recent_successes >= 8:  # If we reach goal in 8/10 recent episodes
            print("Early stopping - Found consistently successful path")
            break

print(f"\nTraining completed:")
print(f"Best composite score: {best_reward:.2f}")
print(f"Total successful runs: {len(successful_runs)}")
if successful_runs:
    best_success = max(successful_runs, key=lambda x: x[0])
    print(f"Best successful run: Reward={best_success[0]:.2f}, Steps={best_success[1]}, Episode={best_success[2]}")

# Stop CoppeliaSim simulation
env.close()