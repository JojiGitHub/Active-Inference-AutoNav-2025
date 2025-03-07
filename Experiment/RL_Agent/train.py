import torch
import numpy as np
from agent import ImprovedDQNAgent, PrioritizedReplayBuffer, device
from env import GridWorldEnv
import os
import time
import math
import random
import argparse
import glob
import re
from datetime import datetime

# Create directory for saving models if it doesn't exist
os.makedirs("models", exist_ok=True)

# Updated training configuration to prevent overfitting
TRAINING_CONFIG = {
    'state_dim': 29,           # 25 vision cells + agent_x, agent_y + goal_x, goal_y
    'action_dim': 5,           # UP, DOWN, LEFT, RIGHT, STAY
    'batch_size': 128,         # Increased batch size for better generalization
    'buffer_size': 500000,     # Much larger buffer for more diverse experiences
    'learning_rate': 0.0003,   # Further reduced learning rate for better stability
    'gamma': 0.99,
    'num_episodes': 2000,      # Reduced from 10000 to 2000
    'eval_interval': 100,      # More frequent evaluation (reduced from 250)
    'max_steps': 150,          # Increased max steps to allow solving complex environments
    'initial_epsilon': 0.7,    # Higher initial exploration
    'min_epsilon': 0.05,
    'weight_decay': 1e-5       # L2 regularization
}

# Condensed curriculum with fewer stages and episodes
CURRICULUM = [
    # Stage 1: Easy environments (first 500 episodes)
    {
        "grid_sizes": [[8, 8], [10, 10], [12, 12]],
        "obstacles_range": [(2, 4), (3, 5)],
        "max_distance_range": [4, 5, 6],
        "seeds": list(range(1, 201)),  # 200 different seeds
        "episodes": 500,
        "patterns": ["random", "clusters"]  # Start with simpler patterns
    },
    # Stage 2: Medium environments (next 750 episodes)
    {
        "grid_sizes": [[12, 12], [15, 15], [18, 18]],
        "obstacles_range": [(4, 8), (6, 10)],
        "max_distance_range": [6, 8, 10],
        "seeds": list(range(201, 501)),  # 300 different seeds
        "episodes": 750,
        "patterns": ["random", "clusters", "walls"]  # Add wall patterns
    },
    # Stage 3: Challenging environments (final 750 episodes)
    {
        "grid_sizes": [[15, 15], [18, 18], [20, 20]],
        "obstacles_range": [(8, 12), (10, 15)],
        "max_distance_range": [8, 10, 12],
        "seeds": list(range(501, 1001)),  # 500 different seeds
        "episodes": 750,
        "patterns": ["random", "clusters", "walls", "maze"]  # All patterns
    }
]

# Simplified test configurations
EVAL_CONFIGS = [
    {"grid_size": [10, 10], "obstacles": 4, "max_dist": 5, "name": "Easy"},
    {"grid_size": [15, 15], "obstacles": 8, "max_dist": 8, "name": "Medium"},
    {"grid_size": [20, 20], "obstacles": 12, "max_dist": 10, "name": "Hard"}
]

def create_env_with_close_goal(grid_size, num_obstacles, max_distance=None, seed=None, obstacle_pattern="random"):
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
        max_steps=TRAINING_CONFIG['max_steps'],
        use_coppeliasim=False  # Explicitly disable CoppeliaSim
    )
    
    # Override the default initialization to create an easier environment
    env.redspots = []
    
    # Place obstacles based on the selected pattern
    if obstacle_pattern == "random":
        for _ in range(num_obstacles):
            while True:
                x = np.random.randint(0, grid_size[0])
                y = np.random.randint(0, grid_size[1])
                if (x, y) not in env.redspots and (x, y) != env.init_loc:
                    env.redspots.append((x, y))
                    break
    elif obstacle_pattern == "clusters":
        cluster_centers = [(np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])) for _ in range(num_obstacles // 3)]
        for center in cluster_centers:
            for _ in range(3):
                while True:
                    x = center[0] + np.random.randint(-1, 2)
                    y = center[1] + np.random.randint(-1, 2)
                    if (x, y) not in env.redspots and (x, y) != env.init_loc and 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                        env.redspots.append((x, y))
                        break
    elif obstacle_pattern == "walls":
        for _ in range(num_obstacles // 5):
            orientation = np.random.choice(["horizontal", "vertical"])
            if orientation == "horizontal":
                x = np.random.randint(0, grid_size[0])
                y_start = np.random.randint(0, grid_size[1] - 4)
                for y in range(y_start, y_start + 5):
                    if (x, y) not in env.redspots and (x, y) != env.init_loc:
                        env.redspots.append((x, y))
            else:
                y = np.random.randint(0, grid_size[1])
                x_start = np.random.randint(0, grid_size[0] - 4)
                for x in range(x_start, x_start + 5):
                    if (x, y) not in env.redspots and (x, y) != env.init_loc:
                        env.redspots.append((x, y))
    elif obstacle_pattern == "maze":
        for _ in range(num_obstacles // 2):
            x = np.random.randint(0, grid_size[0])
            y = np.random.randint(0, grid_size[1])
            if (x, y) not in env.redspots and (x, y) != env.init_loc:
                env.redspots.append((x, y))
                for _ in range(3):
                    direction = np.random.choice(["up", "down", "left", "right"])
                    if direction == "up" and x > 0:
                        x -= 1
                    elif direction == "down" and x < grid_size[0] - 1:
                        x += 1
                    elif direction == "left" and y > 0:
                        y -= 1
                    elif direction == "right" and y < grid_size[1] - 1:
                        y += 1
                    if (x, y) not in env.redspots and (x, y) != env.init_loc:
                        env.redspots.append((x, y))
    
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
        'optimizer_state': agent.optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': time.strftime("%Y%m%d-%H%M%S"),
        'epsilon': agent.epsilon,
        'episode': metrics.get('episode', 0),
        'best_success_rate': metrics.get('success_rate', 0.0)
    }
    torch.save(data, filename)
    print(f"Model saved to {filename}")

def load_checkpoint(path):
    """Load a checkpoint for resuming training"""
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # Extract training progress information
    episode = checkpoint.get('episode', 0)
    best_success_rate = checkpoint.get('best_success_rate', 0.0)
    epsilon = checkpoint.get('epsilon', TRAINING_CONFIG['min_epsilon'])
    timestamp = checkpoint.get('timestamp', 'unknown')
    
    if timestamp != 'unknown':
        try:
            # Try to parse the timestamp to show when the model was last saved
            save_time = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
            print(f"Checkpoint was saved on: {save_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"Checkpoint was saved with timestamp: {timestamp}")
    
    print(f"Resuming from episode {episode}/{TRAINING_CONFIG['num_episodes']} with success rate {best_success_rate:.4f}")
    return checkpoint, episode, best_success_rate, epsilon

def find_latest_checkpoint():
    """Find the most recent checkpoint file"""
    checkpoints = glob.glob("models/checkpoint_ep*.pth") + ["models/best_model.pth"]
    if not checkpoints:
        return None
    
    # Get the modification time for each checkpoint file
    checkpoint_times = [(path, os.path.getmtime(path)) for path in checkpoints if os.path.exists(path)]
    if not checkpoint_times:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_times.sort(key=lambda x: x[1], reverse=True)
    latest_checkpoint = checkpoint_times[0][0]
    mod_time = datetime.fromtimestamp(checkpoint_times[0][1]).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Found latest checkpoint: {latest_checkpoint} (last modified: {mod_time})")
    return latest_checkpoint

def get_curriculum_config(episode):
    """Get environment configuration based on curriculum stage"""
    total_episodes = 0
    for stage in CURRICULUM:
        total_episodes += stage["episodes"]
        if episode < total_episodes:
            return stage
    return CURRICULUM[-1]  # Default to the final stage

def main(resume=False, specific_checkpoint=None):
    print("=== Starting Training ===")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    random.seed(42)
    
    # Initialize agent
    agent = ImprovedDQNAgent(
        state_dim=TRAINING_CONFIG['state_dim'],
        action_dim=TRAINING_CONFIG['action_dim'],
        lr=TRAINING_CONFIG['learning_rate'],
        initial_epsilon=TRAINING_CONFIG['initial_epsilon']
    )
    
    # Initialize replay buffer
    buffer = PrioritizedReplayBuffer(capacity=TRAINING_CONFIG['buffer_size'])
    
    # Progress tracking variables
    rewards_history = []
    successes_history = []
    best_success_rate = 0.0
    current_stage = 0
    last_save_time = time.time()
    start_episode = 0
    
    # Optionally resume from checkpoint
    if resume:
        checkpoint_path = specific_checkpoint if specific_checkpoint else find_latest_checkpoint()
        if checkpoint_path:
            checkpoint, start_episode, best_success_rate, agent.epsilon = load_checkpoint(checkpoint_path)
            
            # Load model weights and optimizer state
            agent.q_network.load_state_dict(checkpoint['q_network_state'])
            agent.target_network.load_state_dict(checkpoint['target_network_state'])
            
            # Try to load optimizer state if available
            if 'optimizer_state' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Update training progress
            print(f"Resuming training from episode {start_episode+1}/{TRAINING_CONFIG['num_episodes']}")
            print(f"Current epsilon: {agent.epsilon:.4f}, Best success rate: {best_success_rate:.4f}")
            
            # Update current stage based on episode number
            total_episodes = 0
            for i, stage in enumerate(CURRICULUM):
                total_episodes += stage["episodes"]
                if start_episode < total_episodes:
                    current_stage = i
                    break
            
            print(f"Current curriculum stage: {current_stage + 1}")
        else:
            print("No checkpoint found. Starting training from scratch.")
    
    # Main training loop
    for episode in range(start_episode, TRAINING_CONFIG['num_episodes']):
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
        
        # Create environment with random pattern from current stage
        grid_size = random.choice(stage_config["grid_sizes"])
        obstacles_range = random.choice(stage_config["obstacles_range"])
        num_obstacles = random.randint(*obstacles_range)
        max_distance = random.choice(stage_config["max_distance_range"])
        seed = random.choice(stage_config["seeds"])
        pattern = random.choice(stage_config["patterns"])
        
        env = create_env_with_close_goal(
            grid_size=grid_size,
            num_obstacles=num_obstacles,
            max_distance=max_distance,
            seed=seed,
            obstacle_pattern=pattern
        )
        
        # Run episode
        state = env.reset()
        done = False
        episode_reward = 0
        reached_goal = False
        steps = 0
        
        # More gradual epsilon decay
        if episode % 100 == 0 and episode > 0:
            if agent.epsilon > TRAINING_CONFIG['min_epsilon']:
                agent.epsilon = max(
                    TRAINING_CONFIG['min_epsilon'],
                    agent.epsilon * 0.98  # Even slower decay
                )
                print(f"Adjusted epsilon to {agent.epsilon:.3f}")
        
        # Run single episode (no logging of individual actions)
        while not done and steps < TRAINING_CONFIG['max_steps']:
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer
            buffer.add(state, action, reward, next_state, done)
            
            # Update agent if enough samples in buffer
            if buffer.position >= TRAINING_CONFIG['batch_size']:
                # Multiple updates per step for better learning
                num_updates = 2 if episode > 5000 else 1
                for _ in range(num_updates):
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
        
        # Display progress every 20 episodes - condensed output format
        if (episode + 1) % 20 == 0:
            recent_rewards = rewards_history[-20:]
            recent_successes = successes_history[-20:]
            success_rate = sum(recent_successes) / len(recent_successes)
            print(f"Episode {episode+1}/{TRAINING_CONFIG['num_episodes']} - "
                  f"R: {np.mean(recent_rewards):.2f}, "
                  f"S: {success_rate:.2f}, "
                  f"ε: {agent.epsilon:.3f}")
        
        # Periodically evaluate and save model
        current_time = time.time()
        time_since_last_save = current_time - last_save_time
        
        if ((episode + 1) % TRAINING_CONFIG['eval_interval'] == 0) or time_since_last_save >= 1800:  # 30 minutes
            print("\nEvaluating...")
            success_rate, avg_reward, difficulty_results = evaluate_agent(agent, num_episodes=30)
            last_save_time = current_time
            
            # Save checkpoint periodically
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
                save_model(agent, metrics, "../best_model.pth")  # Also save to root directory
                print(f"New best model with success rate {success_rate:.2f}")

        # Clean up environment
        env.close()
    
    # Final evaluation with more episodes
    final_success_rate, final_avg_reward, final_difficulty_results = evaluate_agent(agent, num_episodes=50)
    print("\n=== Training Complete ===")
    print(f"Final success rate: {final_success_rate:.2f}")
    print(f"Final average reward: {final_avg_reward:.2f}")
    print(f"Best success rate achieved: {best_success_rate:.2f}")
    
    # Save final model
    final_metrics = {
        'success_rate': final_success_rate,
        'avg_reward': final_avg_reward,
        'episode': TRAINING_CONFIG['num_episodes'],
        'difficulty_results': final_difficulty_results
    }
    save_model(agent, final_metrics, "models/final_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agent for grid navigation')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a specific checkpoint file to resume from')
    args = parser.parse_args()
    
    # If a specific checkpoint is provided, use that instead of finding the latest
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Using specified checkpoint: {args.checkpoint}")
            main(resume=True, specific_checkpoint=args.checkpoint)
        else:
            print(f"Checkpoint {args.checkpoint} not found. Starting from scratch.")
            main(resume=False)
    else:
        main(resume=args.resume)