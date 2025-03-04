import torch
import numpy as np
from agent import ImprovedDQNAgent, PrioritizedReplayBuffer
from env import GridWorldEnv
import os
import time
import math
import random

# Create directory for saving models if it doesn't exist
os.makedirs("models", exist_ok=True)

# Updated training configuration to prevent overfitting
TRAINING_CONFIG = {
    'state_dim': 29,           # 25 vision cells + agent_x, agent_y + goal_x, goal_y
    'action_dim': 5,           # UP, DOWN, LEFT, RIGHT, STAY
    'batch_size': 64,          # Batch size for updates
    'buffer_size': 100000,     # Larger buffer to store diverse experiences
    'learning_rate': 0.0005,   # Reduced learning rate (was 0.005)
    'gamma': 0.99,
    'num_episodes': 2000,     # Keep high episode count
    'eval_interval': 200,      # Less frequent evaluation to save time
    'max_steps': 75,           # Increased max steps to allow solving larger environments
    'initial_epsilon': 0.5,    # Start with balanced exploration
    'min_epsilon': 0.05,
    'weight_decay': 1e-5       # L2 regularization
}

# Expanded training environment configurations with curriculum learning
CURRICULUM = [
    # Stage 1: Easy environments (first 5000 episodes)
    {
        "grid_sizes": [[8, 8], [10, 10]],
        "obstacles_range": [(2, 5)],
        "max_distance_range": [4, 6],
        "seeds": list(range(1, 101)),  # 100 different seeds
        "episodes": 5000
    },
    # Stage 2: Medium environments (next 5000 episodes)
    {
        "grid_sizes": [[10, 10], [12, 12], [15, 15]],
        "obstacles_range": [(5, 8), (8, 12)],
        "max_distance_range": [5, 7, 9],
        "seeds": list(range(101, 201)),  # Different 100 seeds
        "episodes": 5000
    },
    # Stage 3: Challenging environments (final 10000 episodes)
    {
        "grid_sizes": [[15, 15], [18, 18], [20, 20]],
        "obstacles_range": [(10, 15), (15, 20)],
        "max_distance_range": [8, 10, 12],
        "seeds": list(range(201, 501)),  # 300 different seeds
        "episodes": 10000
    }
]

# Test configurations for consistent evaluation
EVAL_CONFIGS = [
    {"grid_size": [10, 10], "obstacles": 5, "max_dist": 5, "name": "Easy"},
    {"grid_size": [15, 15], "obstacles": 10, "max_dist": 8, "name": "Medium"},
    {"grid_size": [20, 20], "obstacles": 15, "max_dist": 10, "name": "Hard"}
]

def create_env_with_close_goal(grid_size, num_obstacles, max_distance=None, seed=None):
    """Create environment with goal closer to agent for easier learning"""
    # Convert seed to int if it's not None to avoid TypeError
    if seed is not None:
        # Make sure seed is always an integer
        if not isinstance(seed, int):
            seed = int(hash(str(seed)) % 1000000)
        np.random.seed(seed)
    
    # Always use an integer seed for GridWorldEnv
    env_seed = seed if isinstance(seed, int) else np.random.randint(0, 10000)
    
    env = GridWorldEnv(
        random_seed=env_seed,
        grid_dimensions=grid_size,
        max_steps=TRAINING_CONFIG['max_steps']
    )
    
    # Override the default initialization to create an easier environment
    env.redspots = []
    
    # Place fewer obstacles
    for _ in range(num_obstacles):
        while True:
            x = np.random.randint(0, grid_size[0])
            y = np.random.randint(0, grid_size[1])
            # Make sure we're not placing obstacles in the middle area
            if (x, y) not in env.redspots and (x < grid_size[0]//4 or x > 3*grid_size[0]//4 or 
                y < grid_size[1]//4 or y > 3*grid_size[1]//4):
                env.redspots.append((x, y))
                break
    
    # Place agent first
    agent_x = np.random.randint(grid_size[0]//4, 3*grid_size[0]//4)
    agent_y = np.random.randint(grid_size[1]//4, 3*grid_size[1]//4)
    env.x = agent_x
    env.y = agent_y
    env.init_loc = (agent_x, agent_y)
    env.current_location = (agent_x, agent_y)
    
    # Place goal at a controlled distance from the agent
    while True:
        # Choose a distance (not too close, not too far)
        if max_distance:
            target_distance = np.random.randint(2, max_distance + 1)
        else:
            target_distance = np.random.randint(2, min(grid_size) // 2)
            
        # Generate potential moves to reach target distance
        dx = np.random.randint(-target_distance, target_distance + 1)
        dy = target_distance - abs(dx) if np.random.random() > 0.5 else -(target_distance - abs(dx))
        
        goal_x = agent_x + dx
        goal_y = agent_y + dy
        
        # Ensure goal is within bounds and not on an obstacle
        if (0 <= goal_x < grid_size[0] and 0 <= goal_y < grid_size[1] and
            (goal_x, goal_y) not in env.redspots and
            (goal_x, goal_y) != env.current_location):
            env.goal = (goal_x, goal_y)
            break
            
    return env

def evaluate_agent(agent, num_episodes=5):
    """Evaluate agent on test environments"""
    successes = 0
    total_rewards = []
    results_by_difficulty = {config["name"]: {"successes": 0, "rewards": []} for config in EVAL_CONFIGS}
    
    print("\nEvaluating agent...")
    
    for config in EVAL_CONFIGS:
        config_successes = 0
        
        for _ in range(num_episodes):
            env = create_env_with_close_goal(
                config["grid_size"], 
                config["obstacles"], 
                config["max_dist"]
            )
            
            state = env.reset()
            done = False
            episode_reward = 0
            success = False
            steps = 0
            
            while not done and steps < TRAINING_CONFIG['max_steps'] and episode_reward > -10:
                action = agent.select_action(state, testing=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
                
                if info['manhattan_distance'] == 0:  # Reached goal
                    success = True
                    break
                    
            if success:
                successes += 1
                config_successes += 1
                
            total_rewards.append(episode_reward)
            results_by_difficulty[config["name"]]["rewards"].append(episode_reward)
        
        results_by_difficulty[config["name"]]["successes"] = config_successes
        results_by_difficulty[config["name"]]["success_rate"] = config_successes / num_episodes
        print(f"  {config['name']} environments: {config_successes}/{num_episodes} successes ({config_successes/num_episodes:.2f})")
    
    total_eval_episodes = len(EVAL_CONFIGS) * num_episodes
    success_rate = successes / total_eval_episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"Overall - Success rate: {success_rate:.2f}, Avg reward: {avg_reward:.2f}")
    return success_rate, avg_reward, results_by_difficulty

def save_model(agent, metrics, filename):
    """Save the trained model"""
    data = {
        'q_network_state': agent.q_network.state_dict(),
        'target_network_state': agent.target_network.state_dict(),
        'metrics': metrics,
        'timestamp': time.strftime("%Y%m%d-%H%M%S")
    }
    torch.save(data, filename)
    print(f"Model saved to {filename}")

def get_curriculum_config(episode):
    """Get environment configuration based on curriculum stage"""
    total_episodes = 0
    for stage in CURRICULUM:
        total_episodes += stage["episodes"]
        if episode < total_episodes:
            return stage
    return CURRICULUM[-1]  # Default to the final stage

def main():
    print("=== Starting Curriculum RL Training for Grid World ===")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # Initialize agent with regularization
    agent = ImprovedDQNAgent(
        state_dim=TRAINING_CONFIG['state_dim'],
        action_dim=TRAINING_CONFIG['action_dim'],
        lr=TRAINING_CONFIG['learning_rate'],
        initial_epsilon=TRAINING_CONFIG['initial_epsilon']
    )
    
    # Replay buffer with capacity
    buffer = PrioritizedReplayBuffer(capacity=TRAINING_CONFIG['buffer_size'])
    
    # Progress tracking variables
    rewards_history = []
    successes_history = []
    best_success_rate = 0.0
    current_stage = 0
    
    # Main training loop
    for episode in range(TRAINING_CONFIG['num_episodes']):
        # Get current curriculum stage
        stage_config = get_curriculum_config(episode)
        
        # Track curriculum stage changes
        stage_idx = CURRICULUM.index(stage_config)
        if stage_idx > current_stage:
            current_stage = stage_idx
            print(f"\n=== Moving to curriculum stage {current_stage + 1} ===")
            # Save a checkpoint at each curriculum change
            metrics = {
                'stage': current_stage,
                'episode': episode,
                'success_rate': best_success_rate
            }
            save_model(agent, metrics, f"models/stage{current_stage}_model.pth")
        
        # Create environment based on current curriculum stage
        grid_size = random.choice(stage_config["grid_sizes"])
        obstacles_range = random.choice(stage_config["obstacles_range"])
        num_obstacles = random.randint(*obstacles_range)
        max_distance = random.choice(stage_config["max_distance_range"])
        seed = random.choice(stage_config["seeds"])
        
        env = create_env_with_close_goal(
            grid_size=grid_size,
            num_obstacles=num_obstacles,
            max_distance=max_distance,
            seed=seed
        )
        
        # Run episode
        state = env.reset()
        done = False
        episode_reward = 0
        reached_goal = False
        steps = 0
        
        # Adaptive exploration - decrease epsilon faster as training progresses
        if episode % 100 == 0 and episode > 0:
            if agent.epsilon > TRAINING_CONFIG['min_epsilon']:
                agent.epsilon = max(
                    TRAINING_CONFIG['min_epsilon'], 
                    agent.epsilon * 0.97  # Slightly slower decay (was 0.95)
                )
                print(f"Adjusted epsilon to {agent.epsilon:.3f}")
        
        # Run single episode
        while not done and steps < TRAINING_CONFIG['max_steps']:
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer
            buffer.add(state, action, reward, next_state, done)
            
            # Update agent if enough samples in buffer
            if buffer.position >= TRAINING_CONFIG['batch_size']:
                agent.update(buffer, TRAINING_CONFIG['batch_size'])
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Check if goal reached
            if info['manhattan_distance'] == 0:
                reached_goal = True
                break
        
        # Track progress
        rewards_history.append(episode_reward)
        successes_history.append(1 if reached_goal else 0)
        
        # Display progress
        if (episode + 1) % 20 == 0:
            recent_rewards = rewards_history[-20:]
            recent_successes = successes_history[-20:]
            success_rate = sum(recent_successes) / len(recent_successes)
            print(f"Episode {episode+1}/{TRAINING_CONFIG['num_episodes']} - "
                  f"Stage {current_stage+1} - "
                  f"Grid {grid_size} - "
                  f"Obstacles {num_obstacles} - "
                  f"Reward: {np.mean(recent_rewards):.2f}, "
                  f"Success rate: {success_rate:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Periodically evaluate and save model
        if (episode + 1) % TRAINING_CONFIG['eval_interval'] == 0:
            success_rate, avg_reward, difficulty_results = evaluate_agent(agent, num_episodes=5)
            
            # Save checkpoint periodically regardless of performance
            if (episode + 1) % 1000 == 0:
                checkpoint_metrics = {
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'episode': episode + 1,
                    'difficulty_results': difficulty_results
                }
                save_model(agent, checkpoint_metrics, f"models/checkpoint_ep{episode+1}.pth")
            
            # Save if improved
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                metrics = {
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'episode': episode + 1,
                    'difficulty_results': difficulty_results
                }
                save_model(agent, metrics, "models/best_model.pth")
                # Also save to root directory
                save_model(agent, metrics, "best_model.pth")
                print(f"New best model with success rate {success_rate:.2f}")
    
    # Final evaluation with more episodes
    final_success_rate, final_avg_reward, final_difficulty_results = evaluate_agent(agent, num_episodes=10)
    print("\n=== Training Complete ===")
    print(f"Final success rate: {final_success_rate:.2f}")
    print(f"Final average reward: {final_avg_reward:.2f}")
    print(f"Best success rate achieved: {best_success_rate:.2f}")
    
    # Detailed difficulty breakdown
    print("\nPerformance by difficulty:")
    for name, results in final_difficulty_results.items():
        print(f"  {name}: {results['success_rate']:.2f} success rate, avg reward: {np.mean(results['rewards']):.2f}")
    
    # Save final model
    final_metrics = {
        'success_rate': final_success_rate,
        'avg_reward': final_avg_reward,
        'episode': TRAINING_CONFIG['num_episodes'],
        'difficulty_results': final_difficulty_results
    }
    save_model(agent, final_metrics, "models/final_model.pth")

if __name__ == "__main__":
    main()