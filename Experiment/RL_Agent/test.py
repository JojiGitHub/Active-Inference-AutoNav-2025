import torch
import numpy as np
try:
    from RL_Agent.agent import ImprovedDQNAgent
except:
    from agent import ImprovedDQNAgent
try:
    from env import GridWorldEnv
except:
    from RL_Agent.env import GridWorldEnv
import time
import matplotlib.pyplot as plt
import sys
import os

# Add CoppeliaSim directory to path to import the module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CoppeliaSim'))
try:
    # Import directly from the coppeliasim_zmqremoteapi_client like in ActInfAgent
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    # Create a client and get the sim object
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    COPPELIA_AVAILABLE = True
    print("Successfully connected to CoppeliaSim Remote API")
except ImportError:
    print("Failed to import coppeliasim_zmqremoteapi_client. Make sure the CoppeliaSim Remote API client is properly installed.")
    print("You may need to install it with: pip install coppeliasim-zmqremoteapi-client")
    try:
        from CoppeliaSim.coppeliasim import sim
        COPPELIA_AVAILABLE = True
        print("Warning: Using fallback CoppeliaSim module.")
    except ImportError:
        COPPELIA_AVAILABLE = False
        print("Warning: CoppeliaSim module not found. Simulation functionality may be limited.")
        # Create a dummy sim object
        class DummySim:
            def __getattr__(self, name):
                def dummy_method(*args, **kwargs):
                    print(f"Dummy sim.{name} called with args: {args}, kwargs: {kwargs}")
                    return -1
                return dummy_method
        sim = DummySim()

def evaluate_episode(agent, env, max_steps=100, render=False, step_delay=0.0):
    """Run a single evaluation episode"""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    reached_goal = False
    hit_obstacle = False
    hit_wall = False
    
    while not done and steps < max_steps:
        action = agent.select_action(state, testing=True)
        
        # Add delay before step to slow down the simulation
        if step_delay > 0:
            print(f"Waiting {step_delay} seconds before next step...")
            time.sleep(step_delay)
            
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1
        
        # Track if we hit a wall
        if ((action == 0 and env.y == 0) or  # UP
            (action == 1 and env.y == env.grid_dimensions[1] - 1) or  # DOWN
            (action == 2 and env.x == 0) or  # LEFT
            (action == 3 and env.x == env.grid_dimensions[0] - 1)):  # RIGHT
            hit_wall = True
        
        # If reward is very negative, we likely hit an obstacle
        if reward <= -10:
            hit_obstacle = True
        
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
        'hit_obstacle': hit_obstacle,
        'hit_wall': hit_wall,
        'final_distance': info['manhattan_distance']
    }

def run_evaluation(model_path, num_episodes=50, test_seeds=None):
    """Comprehensive model evaluation across multiple seeds"""
    if test_seeds is None:
        test_seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]  # 10 different seeds
    
    # Load model with correct state dimensions
    state_dim = 29  # 25 vision cells + 4 coordinates (keeping original dimension)
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
                obstacle_hits = sum(r['hit_obstacle'] for r in seed_results)
                wall_hits = sum(r['hit_wall'] for r in seed_results)
                print(f"Seed {seed} - Episode {episode + 1}/{num_episodes}, "
                      f"Success Rate: {success_rate:.2f}, "
                      f"Obstacle collisions: {obstacle_hits}, "
                      f"Wall collisions: {wall_hits}")
        
        # Compute metrics for this seed
        successes = sum(r['reached_goal'] for r in seed_results)
        obstacle_hits = sum(r['hit_obstacle'] for r in seed_results)
        wall_hits = sum(r['hit_wall'] for r in seed_results)
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
            'obstacle_hits': obstacle_hits,
            'wall_hits': wall_hits,
            'individual_results': seed_results
        })
        
        print(f"\nSeed {seed} Summary:")
        print(f"Success Rate: {success_rate:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Obstacle Collisions: {obstacle_hits}/{num_episodes} ({obstacle_hits/num_episodes:.2%})")
        print(f"Wall Collisions: {wall_hits}/{num_episodes} ({wall_hits/num_episodes:.2%})")
        print(f"Average Steps (Successful): {success_steps:.1f}" if successes > 0 else "No successful episodes")
        
        # Clean up environment
        env.close()
        
    # Overall results
    overall_success_rate = np.mean([r['success_rate'] for r in results])
    overall_reward = np.mean([r['avg_reward'] for r in results])
    overall_steps = np.mean([r['avg_steps'] for r in results])
    total_obstacle_hits = sum([r['obstacle_hits'] for r in results])
    total_wall_hits = sum([r['wall_hits'] for r in results])
    total_episodes = len(test_seeds) * num_episodes
    
    print("\nOverall Performance:")
    print(f"Average Success Rate: {overall_success_rate:.2f}")
    print(f"Average Reward: {overall_reward:.2f}")
    print(f"Average Steps: {overall_steps:.1f}")
    print(f"Total Obstacle Collisions: {total_obstacle_hits}/{total_episodes} ({total_obstacle_hits/total_episodes:.2%})")
    print(f"Total Wall Collisions: {total_wall_hits}/{total_episodes} ({total_wall_hits/total_episodes:.2%})")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Success rates subplot
    plt.subplot(2, 1, 1)
    success_rates = [r['success_rate'] for r in results]
    plt.bar(range(len(test_seeds)), success_rates)
    plt.xlabel('Seed Index')
    plt.ylabel('Success Rate')
    plt.title('Model Success Rate Across Different Seeds')
    plt.axhline(y=overall_success_rate, color='r', linestyle='--', label='Average')
    plt.legend()
    
    # Collision rates subplot
    plt.subplot(2, 1, 2)
    obstacle_rates = [r['obstacle_hits'] / num_episodes for r in results]
    wall_rates = [r['wall_hits'] / num_episodes for r in results]
    
    x = np.arange(len(test_seeds))
    width = 0.35
    plt.bar(x - width/2, obstacle_rates, width, label='Obstacle Collisions')
    plt.bar(x + width/2, wall_rates, width, label='Wall Collisions')
    plt.xlabel('Seed Index')
    plt.ylabel('Collision Rate')
    plt.title('Collision Rates Across Different Seeds')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results, {
        'overall_success_rate': overall_success_rate,
        'overall_reward': overall_reward,
        'overall_steps': overall_steps,
        'obstacle_collision_rate': total_obstacle_hits/total_episodes,
        'wall_collision_rate': total_wall_hits/total_episodes
    }

def check_coppelia_connection():
    """Check if CoppeliaSim is available and connected"""
    if not COPPELIA_AVAILABLE:
        return False
    
    try:
        # First try to stop any running simulation
        sim_state = sim.getSimulationState()
        if (sim_state != sim.simulation_stopped):
            print("Stopping existing simulation...")
            sim.stopSimulation()
            time.sleep(1.0)  # Wait for simulation to fully stop
            
        # Try to get the simulation state to see if CoppeliaSim is connected
        sim_state = sim.getSimulationState()
        
        # Try explicitly starting the simulation to verify it's working
        try:
            print("Starting new CoppeliaSim simulation...")
            sim.startSimulation()
            time.sleep(0.5)  # Give it time to initialize
            print("Successfully started CoppeliaSim simulation")
            return True
        except Exception as e:
            print(f"Error starting simulation: {e}")
            return False
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
    
    # Clean up resources
    env.close()
    
    return {
        'performance_metrics': performance_metrics,
        'episode_data': {'actions': [], 'states': [], 'rewards': []}  # Placeholder for compatibility
    }

def main(use_coppeliasim=False, random_seed=42, render_visualization=False, sim_speed=0.0):
    """
    Main function to run the RL Agent test with a simple hyperparameter for CoppeliaSim activation
    
    Args:
        use_coppeliasim (bool): Whether to use CoppeliaSim for visualization
        random_seed (int): Random seed for reproducibility
        render_visualization (bool): Whether to render the environment
        sim_speed (float): Delay in seconds between steps to slow down the simulation (0.0 = no delay)
    """
    # Check if CoppeliaSim connection is available when requested
    if use_coppeliasim:
        print("Checking CoppeliaSim connection...")
        coppelia_connected = check_coppelia_connection()
        if coppelia_connected:
            print("CoppeliaSim connected successfully.")
        else:
            print("Could not connect to CoppeliaSim. Running in grid-only mode.")
            use_coppeliasim = False
    
    # Create agent instance for visualization with correct state dimensions
    state_dim = 29  # 25 vision cells + 4 coordinates (keeping original dimension)
    action_dim = 5
    vis_agent = ImprovedDQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Try to load the trained model
    try:
        # First try to load from models directory (standard location)
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'best_model.pth')
        if not os.path.exists(model_path):
            # Try current directory
            model_path = "best_model.pth"
        
        checkpoint = torch.load(model_path)
        vis_agent.q_network.load_state_dict(checkpoint['q_network_state'])
        vis_agent.epsilon = 0.05  # Low exploration for testing
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Will proceed with randomly initialized model")
    
    # Print simulation speed info
    if sim_speed > 0:
        print(f"\nRunning with simulation speed delay: {sim_speed} seconds between steps")
    
    # Visualize episodes with the model
    print("\nVisualizing episodes with model...")
    env = GridWorldEnv(random_seed=random_seed, use_coppeliasim=use_coppeliasim)
    
    # If using CoppeliaSim, also modify the path following delay in the environment
    if use_coppeliasim and hasattr(env, 'follow_path'):
        # Set the path following delay to slow down the robot movement
        # This is modifying the environment's internal time.sleep value
        print("Setting environment path following delay to 0.1 seconds")
        env.path_follow_delay = 0.1  # Set longer delay for path following
    
    # Run a more comprehensive test with 5 visualization episodes
    obstacle_collisions = 0
    wall_collisions = 0
    successes = 0
    
    try:
        for i in range(5):  # Increased from 3 to 5 episodes
            print(f"\nVisualization Episode {i+1}")
            result = evaluate_episode(
                vis_agent, 
                env, 
                render=render_visualization, 
                step_delay=sim_speed
            )
            print(f"Episode Result - Reward: {result['reward']:.2f}, "
                  f"Steps: {result['steps']}, Goal Reached: {result['reached_goal']}")
            
            if result['hit_obstacle']:
                obstacle_collisions += 1
                print("WARNING: Agent hit an obstacle!")
                
            if result['hit_wall']:
                wall_collisions += 1
                print("WARNING: Agent hit a wall!")
                
            if result['reached_goal']:
                successes += 1
                
        # Print summary
        print("\nTest Summary:")
        print(f"Success Rate: {successes/5:.2f}")
        print(f"Obstacle Collisions: {obstacle_collisions}/5")
        print(f"Wall Collisions: {wall_collisions}/5")
        
    finally:
        # Make sure to clean up even if there's an error
        print("Cleaning up environment...")
        env.close()
        
    print("Testing complete")

if __name__ == "__main__":
    # Simple hyperparameters for testing
    USE_COPPELIASIM = False  # Set this to False to disable CoppeliaSim
    RANDOM_SEED = 42        # Change this to use a different random seed
    RENDER = True           # Set this to True to render the grid visualization 
    SIM_SPEED = 0.5         # Delay in seconds between steps (0.5 = half second delay)
    
    # Run the main function with the specified parameters
    main(
        use_coppeliasim=USE_COPPELIASIM, 
        random_seed=RANDOM_SEED, 
        render_visualization=RENDER,
        sim_speed=SIM_SPEED
    )
