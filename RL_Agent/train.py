import torch
import numpy as np
from agent import ImprovedDQNAgent, PrioritizedReplayBuffer
from env import GridWorldEnv  # Import your environment class

# Hyperparameters
state_dim = 17
action_dim = 5
batch_size = 64
lr = 0.001
gamma = 0.9
num_episodes = 1000
max_timesteps = 12  # Matching env's timeout

# Initialize environment and agent
env = GridWorldEnv()
agent = ImprovedDQNAgent(state_dim=state_dim, action_dim=action_dim, lr=lr)
buffer = PrioritizedReplayBuffer(capacity=10000)

# Initialize tracking variables
best_reward = float('-inf')
episode_rewards = []
episode_steps = []
successful_runs = []  # Track episodes that reached the goal

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    reached_goal = False
    
    while not done and steps < max_timesteps:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        
        if buffer.position > batch_size:
            agent.update(buffer, batch_size)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        # Check if agent reached goal
        if (env.agent_x, env.agent_y) == env.goal_position:
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
    
    # Save only if we reach the goal and get better performance
    if reached_goal:
        # Quick validation with 3 episodes
        val_rewards = []
        val_successes = 0
        for _ in range(3):
            val_state = env.reset()
            val_done = False
            val_reward = 0
            val_step_count = 0
            val_reached_goal = False
            
            while not val_done and val_step_count < max_timesteps:
                val_action = agent.select_action(val_state)
                val_next_state, val_rew, val_done, _ = env.step(val_action)
                val_reward += val_rew
                val_state = val_next_state
                val_step_count += 1
                
                if (env.agent_x, env.agent_y) == env.goal_position:
                    val_reached_goal = True
            
            val_rewards.append(val_reward)
            if val_reached_goal:
                val_successes += 1
        
        val_avg_reward = np.mean(val_rewards)
        
        # Save if validation shows consistent goal-reaching
        if val_successes >= 2 and val_avg_reward > best_reward:  # Must reach goal in at least 2/3 validation runs
            best_reward = total_reward
            torch.save({
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'episode': episode,
                'reward': total_reward,
                'steps': steps,
                'reached_goal': True
            }, "best_model.pth")
            print(f"New best model saved! Reward: {total_reward:.2f}, Steps: {steps}")
            print(f"Validation success rate: {val_successes}/3")
    
    # Early stopping if we consistently reach goal
    if len(episode_rewards) >= 10:
        recent_successes = sum(1 for r in episode_rewards[-10:] if r > 40)  # Reward > 40 indicates goal reached
        if recent_successes >= 8:  # If we reach goal in 8/10 recent episodes
            print("Early stopping - Found consistently successful path")
            break

print(f"\nTraining completed:")
print(f"Best reward: {best_reward:.2f}")
print(f"Total successful runs: {len(successful_runs)}")
if successful_runs:
    best_success = max(successful_runs, key=lambda x: x[0])
    print(f"Best successful run: Reward={best_success[0]:.2f}, Steps={best_success[1]}, Episode={best_success[2]}")