import torch
import numpy as np
from agent import ImprovedDQNAgent
from env import GridWorldEnv
import time
import matplotlib.pyplot as plt
import sys
import os

# Add CoppeliaSim directory to path to import the module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CoppeliaSim'))
try:
    from CoppeliaSim.coppeliasim import sim
    COPPELIA_AVAILABLE = True
    print("CoppeliaSim module found.")
except ImportError:
    COPPELIA_AVAILABLE = False
    print("Warning: CoppeliaSim module not found. Simulation functionality may be limited.")

def evaluate_episode(agent, env, max_steps=100, render=False):
    """Run a single evaluation episode"""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    reached_goal = False
    
    while not done and steps < max_steps:
        action = agent.select_action(state, testing=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1
        
        if render and steps % 5 == 0:  # Render every 5 steps
            env.render()
            time.sleep(0.1)  # Slow down visualization
            
        if info['manhattan_distance'] == 0:
            reached_goal = True
            break
    
    return {
        'reward': total_reward,
        'steps': steps,
        'reached_goal': reached_goal,
        'final_distance': info['manhattan_distance']
    }

def run_evaluation(model_path, num_episodes=50, test_seeds=None):
    """Comprehensive model evaluation across multiple seeds"""
    if test_seeds is None:
        test_seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]  # 10 different seeds
    
    # Load model
    state_dim = 29  # Updated state dimension (25 vision + 4 coordinates)
    action_dim = 5
    agent = ImprovedDQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    checkpoint = torch.load(model_path)
    # Fix: Use the correct key for the state dictionary
    agent.q_network.load_state_dict(checkpoint['q_network_state'])
    agent.epsilon = 0.05  # Low exploration for testing
    
    results = []
    
    print(f"\nEvaluating model: {model_path}")
    print(f"Testing across {len(test_seeds)} different seeds...")
    
    for seed in test_seeds:
        env = GridWorldEnv(random_seed=seed, max_steps=100, use_coppeliasim=False)
        seed_results = []
        
        for episode in range(num_episodes):
            result = evaluate_episode(agent, env)
            seed_results.append(result)
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                success_rate = sum(r['reached_goal'] for r in seed_results) / len(seed_results)
                print(f"Seed {seed} - Episode {episode + 1}/{num_episodes}, "
                      f"Success Rate: {success_rate:.2f}")
        
        # Compute metrics for this seed
        successes = sum(r['reached_goal'] for r in seed_results)
        success_rate = successes / num_episodes
        avg_reward = np.mean([r['reward'] for r in seed_results])
        avg_steps = np.mean([r['steps'] for r in seed_results])
        success_steps = np.mean([r['steps'] for r in seed_results if r['reached_goal']]) if successes > 0 else float('inf')
        
        results.append({
            'seed': seed,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_steps': success_steps,
            'individual_results': seed_results
        })
        
        print(f"\nSeed {seed} Summary:")
        print(f"Success Rate: {success_rate:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Steps (Successful): {success_steps:.1f}" if successes > 0 else "No successful episodes")
        
    # Overall results
    overall_success_rate = np.mean([r['success_rate'] for r in results])
    overall_reward = np.mean([r['avg_reward'] for r in results])
    overall_steps = np.mean([r['avg_steps'] for r in results])
    
    print("\nOverall Performance:")
    print(f"Average Success Rate: {overall_success_rate:.2f}")
    print(f"Average Reward: {overall_reward:.2f}")
    print(f"Average Steps: {overall_steps:.1f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    success_rates = [r['success_rate'] for r in results]
    plt.bar(range(len(test_seeds)), success_rates)
    plt.xlabel('Seed Index')
    plt.ylabel('Success Rate')
    plt.title('Model Performance Across Different Seeds')
    plt.axhline(y=overall_success_rate, color='r', linestyle='--', label='Average')
    plt.legend()
    plt.show()
    
    return results, {
        'overall_success_rate': overall_success_rate,
        'overall_reward': overall_reward,
        'overall_steps': overall_steps
    }

def check_coppelia_connection():
    """Check if CoppeliaSim is available and connected"""
    if not COPPELIA_AVAILABLE:
        return False
    
    try:
        # Try to get the simulation state to see if CoppeliaSim is connected
        state = sim.getSimulationState()
        return True
    except Exception as e:
        print(f"Error connecting to CoppeliaSim: {e}")
        return False

def run_environment(random_seed=42, use_coppeliasim=False):
    """
    Run a single episode of the RL agent in a compatible interface for data collection
    
    Args:
        random_seed (int): Seed for random number generation
        use_coppeliasim (bool): Whether to use CoppeliaSim
        
    Returns:
        dict: Dictionary containing performance metrics and episode data
    """
    # Check if CoppeliaSim should be used and is available
    can_use_coppeliasim = use_coppeliasim and check_coppelia_connection()
    if use_coppeliasim and not can_use_coppeliasim:
        print("CoppeliaSim requested but not available. Falling back to grid-only mode.")
    
    # Create environment
    env = GridWorldEnv(random_seed=random_seed, max_steps=100, use_coppeliasim=can_use_coppeliasim)
    
    # Create agent and load model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ImprovedDQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    try:
        checkpoint = torch.load("best_model.pth")
        # Fix: Use the correct key for the state dictionary
        agent.q_network.load_state_dict(checkpoint['q_network_state'])
        print("Loaded model from best_model.pth")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Using random policy")
    
    agent.epsilon = 0.05  # Low exploration for testing
    
    # Run episode
    result = evaluate_episode(agent, env, render=can_use_coppeliasim)
    
    # Format results consistently with the Active Inference agent
    performance_metrics = {
        'total_timesteps': result['steps'],
        'total_reward': result['reward'],
        'total_distance': result['final_distance'],
        'goal_reached': result['reached_goal'],
        'random_seed': random_seed
    }
    
    return {
        'performance_metrics': performance_metrics,
        'episode_data': {'actions': [], 'states': [], 'rewards': []}  # Placeholder for compatibility
    }

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test RL agent in GridWorld')
    parser.add_argument('--use_coppeliasim', action='store_true', help='Use CoppeliaSim for visualization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()
    
    # Check if CoppeliaSim connection is available when requested
    if args.use_coppeliasim:
        coppelia_connected = check_coppelia_connection()
        if coppelia_connected:
            print("CoppeliaSim connected successfully.")
        else:
            print("Could not connect to CoppeliaSim. Running in grid-only mode.")
            args.use_coppeliasim = False
    
    # Create agent instance for visualization
    state_dim = 29  # Updated state dimension (25 vision + 4 coordinates)
    action_dim = 5
    vis_agent = ImprovedDQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    try:
        checkpoint = torch.load("best_model.pth")
        # Fix: Use the correct key for the state dictionary
        vis_agent.q_network.load_state_dict(checkpoint['q_network_state'])
        vis_agent.epsilon = 0.05  # Low exploration for testing
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Will proceed with randomly initialized model")
    
    # Visualize episodes with the model
    print("\nVisualizing episodes with model...")
    env = GridWorldEnv(random_seed=args.seed, use_coppeliasim=args.use_coppeliasim)
    
    for i in range(3):
        print(f"\nVisualization Episode {i+1}")
        result = evaluate_episode(vis_agent, env, render=args.render)
        print(f"Episode Result - Reward: {result['reward']:.2f}, "
              f"Steps: {result['steps']}, Goal Reached: {result['reached_goal']}")
    
    env.close()
