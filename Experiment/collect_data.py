import numpy as np
import pandas as pd
import os
import time
import argparse
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Import from both agent types
from ActInfAgent.test import run_experiment as run_actinf_experiment
from ActInfAgent.test import test_active_inference
from RL_Agent.test import run_environment as run_rl_experiment
from RL_Agent.test import check_coppelia_connection

def collect_experiment_data(agent_type='actinf', num_episodes=100, num_steps=100, 
                            policy_len=3, start_seed=1, use_coppeliasim=False, simulation_mode=True):
    """
    Collect experimental data from multiple episodes with simplified metrics
    
    Args:
        agent_type (str): 'actinf' for Active Inference or 'rl' for Reinforcement Learning
        num_episodes (int): Number of episodes to run
        num_steps (int): Maximum number of steps per episode
        policy_len (int): Length of policy for the Active Inference agent
        start_seed (int): Starting random seed
        use_coppeliasim (bool): Whether to use CoppeliaSim for visualization
        simulation_mode (bool): Whether to run in simulation mode (without requiring CoppeliaSim)
    
    Returns:
        pd.DataFrame: DataFrame containing collected data points
    """
    # Check if CoppeliaSim is connected when needed
    if use_coppeliasim:
        coppelia_connected = check_coppelia_connection()
        if coppelia_connected:
            print("CoppeliaSim connected successfully.")
        else:
            print("Could not connect to CoppeliaSim. Running in grid-only mode.")
            use_coppeliasim = False
            simulation_mode = True

    # Set simulation mode flag for ActInfAgent if needed
    if simulation_mode and agent_type == 'actinf':
        # Add a flag to the environment indicating we want to use simulation mode
        try:
            import ActInfAgent.env
            ActInfAgent.env.SIMULATION_MODE = True
            print("Running in simulation mode (no CoppeliaSim connection required)")
        except ImportError:
            print("Warning: Could not import ActInfAgent.env to set simulation mode")
    
    # Create directory for storing experiment data
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_type_str = "ActInf" if agent_type == "actinf" else "RL"
    experiment_dir = f"{agent_type_str}_data_{experiment_time}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize results storage
    collected_data = []
    
    print(f"Starting {agent_type_str} data collection with {num_episodes} episodes")
    start_time = time.time()
    
    # Run episodes with different random seeds
    for episode in range(num_episodes):
        seed = start_seed + episode
        
        try:
            print(f"\nRunning trial {episode+1}/{num_episodes} with seed {seed}")
            
            # Run experiment based on agent type
            if agent_type == 'actinf':
                # For Active Inference agent
                if use_coppeliasim:
                    # Detailed data collection
                    df = run_actinf_experiment(
                        random_seed=seed,
                        max_steps=num_steps,
                        save_data=False
                    )
                    
                    # Extract required metrics
                    data_point = {
                        'trial_number': seed,
                        'total_distance': df['goal_distance'].sum(),
                        'total_steps': len(df),
                        'shannon_entropy': df['shannon_entropy'].mean(),
                        'goal_reached': df['goal_reached'].iloc[-1]
                    }
                else:
                    # Summary statistics
                    results = test_active_inference(
                        random_seed=seed,
                        num_steps=num_steps,
                        policy_len=policy_len
                    )
                    
                    # Extract required metrics
                    data_point = {
                        'trial_number': seed,
                        'total_distance': results.get('total_distance', 0),
                        'total_steps': results.get('total_steps', 0),
                        'shannon_entropy': results.get('avg_shannon_entropy', 0),
                        'goal_reached': results.get('goal_reached', False)
                    }
            else:
                # For RL agent
                full_results = run_rl_experiment(
                    random_seed=seed,
                    use_coppeliasim=use_coppeliasim
                )
                
                # Extract required metrics
                metrics = full_results['performance_metrics']
                data_point = {
                    'trial_number': seed,
                    'total_distance': metrics.get('total_distance', 0),
                    'total_steps': metrics.get('total_timesteps', 0),
                    'shannon_entropy': 'N/A',  # RL agent doesn't compute entropy
                    'goal_reached': metrics.get('goal_reached', False)
                }
            
            # Store data point
            collected_data.append(data_point)
            
            # Print data point
            print(f"Trial {seed}:")
            print(f"  Total distance: {data_point['total_distance']}")
            print(f"  Total steps: {data_point['total_steps']}")
            print(f"  Shannon entropy: {data_point['shannon_entropy']}")
            print(f"  Goal reached: {data_point['goal_reached']}")
            
        except Exception as e:
            print(f"Error in trial {episode+1} with seed {seed}: {str(e)}")
            # Continue with the next trial
            continue
        
        # Print progress periodically
        if (episode + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Completed {episode + 1}/{num_episodes} trials ({elapsed:.1f}s elapsed)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(collected_data)
    
    # Save results
    csv_file = os.path.join(experiment_dir, f"{agent_type_str}_results.csv")
    results_df.to_csv(csv_file, index=False)
    
    print("\nData collection complete!")
    print(f"Total trials: {len(results_df)}")
    print(f"Results saved to {csv_file}")
    
    return results_df

def save_comparison_data(actinf_df, rl_df, output_file='agent_comparison.csv'):
    """
    Save data from both agent types to a single CSV file with additional calculated metrics
    
    Args:
        actinf_df (pd.DataFrame): DataFrame containing Active Inference agent results
        rl_df (pd.DataFrame): DataFrame containing RL agent results
        output_file (str): Path to output CSV file
    
    Returns:
        pd.DataFrame: Combined DataFrame with additional metrics
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Create copies to avoid modifying originals
    actinf_data = actinf_df.copy()
    rl_data = rl_df.copy()
    
    # Add agent type column
    actinf_data['agent_type'] = 'Active Inference'
    rl_data['agent_type'] = 'Reinforcement Learning'
    
    # Add efficiency metrics
    if 'total_steps' in actinf_data.columns and 'goal_reached' in actinf_data.columns:
        actinf_data['path_efficiency'] = np.where(
            actinf_data['goal_reached'], 
            actinf_data['total_distance'] / actinf_data['total_steps'], 
            0
        )
        
    if 'total_steps' in rl_data.columns and 'goal_reached' in rl_data.columns:
        rl_data['path_efficiency'] = np.where(
            rl_data['goal_reached'], 
            rl_data['total_distance'] / rl_data['total_steps'], 
            0
        )
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actinf_data['timestamp'] = timestamp
    rl_data['timestamp'] = timestamp
    
    # Combine datasets
    combined_df = pd.concat([actinf_data, rl_data], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    
    return combined_df

def export_performance_summary(actinf_df, rl_df, output_file='performance_summary.csv'):
    """
    Create and export a summary of performance metrics for both agent types
    
    Args:
        actinf_df (pd.DataFrame): DataFrame containing Active Inference agent results
        rl_df (pd.DataFrame): DataFrame containing RL agent results
        output_file (str): Path to output CSV file
        
    Returns:
        pd.DataFrame: Summary DataFrame with performance metrics
    """
    # Initialize summary dictionary
    summary = {
        'Metric': [
            'Trials Run', 
            'Success Rate (%)', 
            'Avg Steps',
            'Avg Steps (Successful Only)', 
            'Avg Distance', 
            'Path Efficiency',
        ],
        'Active Inference': [],
        'Reinforcement Learning': []
    }
    
    # Active Inference metrics
    summary['Active Inference'] = [
        len(actinf_df),
        round(actinf_df['goal_reached'].mean() * 100, 2),
        round(actinf_df['total_steps'].mean(), 2),
        round(actinf_df[actinf_df['goal_reached']]['total_steps'].mean() 
              if not actinf_df[actinf_df['goal_reached']].empty else 0, 2),
        round(actinf_df['total_distance'].mean(), 2),
        round(actinf_df[actinf_df['goal_reached']]['path_efficiency'].mean() 
              if 'path_efficiency' in actinf_df.columns and not actinf_df[actinf_df['goal_reached']].empty else 0, 2)
    ]
    
    # RL metrics
    summary['Reinforcement Learning'] = [
        len(rl_df),
        round(rl_df['goal_reached'].mean() * 100, 2),
        round(rl_df['total_steps'].mean(), 2),
        round(rl_df[rl_df['goal_reached']]['total_steps'].mean() 
              if not rl_df[rl_df['goal_reached']].empty else 0, 2),
        round(rl_df['total_distance'].mean(), 2),
        round(rl_df[rl_df['goal_reached']]['path_efficiency'].mean() 
              if 'path_efficiency' in rl_df.columns and not rl_df[rl_df['goal_reached']].empty else 0, 2)
    ]
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"Performance summary saved to {output_file}")
    
    return summary_df

def run_comparison_experiments(episodes=20, steps=100, policy_len=3, start_seed=42, 
                              use_coppeliasim=False, simulation_mode=True, 
                              output_dir='experiment_results'):
    """
    Run experiments for both agent types and save comparison data
    
    Args:
        episodes (int): Number of episodes to run
        steps (int): Maximum steps per episode
        policy_len (int): Policy length for Active Inference agent
        start_seed (int): Starting random seed
        use_coppeliasim (bool): Whether to use CoppeliaSim
        simulation_mode (bool): Run in simulation mode (without CoppeliaSim)
        output_dir (str): Directory to save results
        
    Returns:
        tuple: (actinf_df, rl_df, combined_df, summary_df)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check CoppeliaSim connection if requested
    if use_coppeliasim:
        coppelia_connected = check_coppelia_connection()
        if coppelia_connected:
            print("CoppeliaSim connected successfully.")
        else:
            print("Could not connect to CoppeliaSim. Running in grid-only mode.")
            use_coppeliasim = False
            simulation_mode = True
    
    print("Running Active Inference experiments...")
    actinf_df = collect_experiment_data(
        agent_type='actinf',
        num_episodes=episodes,
        num_steps=steps,
        policy_len=policy_len,
        start_seed=start_seed,
        use_coppeliasim=use_coppeliasim,
        simulation_mode=simulation_mode
    )
    
    print("\nRunning Reinforcement Learning experiments...")
    rl_df = collect_experiment_data(
        agent_type='rl',
        num_episodes=episodes,
        num_steps=steps,
        start_seed=start_seed,
        use_coppeliasim=use_coppeliasim,
        simulation_mode=simulation_mode
    )
    
    # Add path efficiency metrics
    if 'total_steps' in actinf_df.columns and 'goal_reached' in actinf_df.columns:
        actinf_df['path_efficiency'] = np.where(
            actinf_df['goal_reached'], 
            actinf_df['total_distance'] / actinf_df['total_steps'], 
            0
        )
        
    if 'total_steps' in rl_df.columns and 'goal_reached' in rl_df.columns:
        rl_df['path_efficiency'] = np.where(
            rl_df['goal_reached'], 
            rl_df['total_distance'] / rl_df['total_steps'], 
            0
        )
    
    # Save data
    combined_file = os.path.join(output_dir, f"combined_results_{timestamp}.csv")
    summary_file = os.path.join(output_dir, f"performance_summary_{timestamp}.csv")
    
    combined_df = save_comparison_data(actinf_df, rl_df, combined_file)
    summary_df = export_performance_summary(actinf_df, rl_df, summary_file)
    
    # Create and save visualization
    create_comparison_plots(actinf_df, rl_df, output_dir, timestamp)
    
    print(f"\nAll results saved to directory: {output_dir}")
    return actinf_df, rl_df, combined_df, summary_df

def create_comparison_plots(actinf_df, rl_df, output_dir, timestamp):
    """
    Create and save comparison plots for both agent types
    
    Args:
        actinf_df (pd.DataFrame): DataFrame containing Active Inference agent results
        rl_df (pd.DataFrame): DataFrame containing RL agent results
        output_dir (str): Directory to save plots
        timestamp (str): Timestamp for filenames
    """
    # Success rate comparison
    plt.figure(figsize=(10, 6))
    actinf_success = actinf_df['goal_reached'].mean() * 100
    rl_success = rl_df['goal_reached'].mean() * 100
    
    bars = plt.bar(['Active Inference', 'Reinforcement Learning'], [actinf_success, rl_success])
    plt.title('Success Rate Comparison')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, f"success_rate_comparison_{timestamp}.png"))
    
    # Average steps comparison
    plt.figure(figsize=(10, 6))
    actinf_steps = actinf_df['total_steps'].mean()
    rl_steps = rl_df['total_steps'].mean()
    
    actinf_steps_success = actinf_df[actinf_df['goal_reached']]['total_steps'].mean() if not actinf_df[actinf_df['goal_reached']].empty else 0
    rl_steps_success = rl_df[rl_df['goal_reached']]['total_steps'].mean() if not rl_df[rl_df['goal_reached']].empty else 0
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, [actinf_steps, actinf_steps_success], width, label='Active Inference')
    plt.bar(x + width/2, [rl_steps, rl_steps_success], width, label='Reinforcement Learning')
    
    plt.xlabel('Metric')
    plt.ylabel('Average Steps')
    plt.title('Steps Comparison')
    plt.xticks(x, ['All Episodes', 'Successful Episodes'])
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f"steps_comparison_{timestamp}.png"))
    
    # Path efficiency for successful episodes
    plt.figure(figsize=(10, 6))
    
    if 'path_efficiency' in actinf_df.columns and 'path_efficiency' in rl_df.columns:
        actinf_eff = actinf_df[actinf_df['goal_reached']]['path_efficiency'].mean() if not actinf_df[actinf_df['goal_reached']].empty else 0
        rl_eff = rl_df[rl_df['goal_reached']]['path_efficiency'].mean() if not rl_df[rl_df['goal_reached']].empty else 0
        
        bars = plt.bar(['Active Inference', 'Reinforcement Learning'], [actinf_eff, rl_eff])
        plt.title('Path Efficiency for Successful Episodes')
        plt.ylabel('Efficiency (distance/steps)')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')
        
        plt.savefig(os.path.join(output_dir, f"efficiency_comparison_{timestamp}.png"))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Collect data from experiments')
    parser.add_argument('--agent', type=str, choices=['actinf', 'rl', 'both'], default='actinf',
                      help='Which agent to use: actinf, rl, or both (default: actinf)')
    parser.add_argument('--episodes', type=int, default=20, 
                      help='Number of episodes to run (default: 20)')
    parser.add_argument('--steps', type=int, default=100,
                      help='Maximum steps per episode (default: 100)')
    parser.add_argument('--policy_len', type=int, default=3,
                      help='Policy length for Active Inference agent (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Starting random seed (default: 42)')
    parser.add_argument('--use_coppeliasim', action='store_true',
                      help='Use CoppeliaSim for visualization (slower)')
    parser.add_argument('--no_simulation', action='store_true',
                      help='Disable simulation mode - requires running CoppeliaSim')
    parser.add_argument('--output_dir', type=str, default='experiment_results',
                      help='Directory to save results (default: experiment_results)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check CoppeliaSim connection if requested
    if args.use_coppeliasim:
        coppelia_connected = check_coppelia_connection()
        if coppelia_connected:
            print("CoppeliaSim connected successfully.")
        else:
            print("Could not connect to CoppeliaSim. Running in grid-only mode.")
            args.use_coppeliasim = False
            args.no_simulation = False  # Enable simulation mode since CoppeliaSim isn't available
    
    if args.agent == 'both':
        # Run comparison with both agent types
        actinf_df, rl_df, combined_df, summary_df = run_comparison_experiments(
            episodes=args.episodes,
            steps=args.steps,
            policy_len=args.policy_len,
            start_seed=args.seed,
            use_coppeliasim=args.use_coppeliasim,
            simulation_mode=not args.no_simulation,
            output_dir=args.output_dir
        )
        
        print("\nComparison Summary:")
        print(summary_df)
    else:
        # Collect data for single agent type
        results_df = collect_experiment_data(
            agent_type=args.agent,
            num_episodes=args.episodes,
            num_steps=args.steps,
            policy_len=args.policy_len,
            start_seed=args.seed,
            use_coppeliasim=args.use_coppeliasim,
            simulation_mode=not args.no_simulation
        )
        
        # Save detailed results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_type_str = "ActInf" if args.agent == "actinf" else "RL"
        detailed_file = os.path.join(args.output_dir, f"detailed_{agent_type_str}_{timestamp}.csv")
        results_df.to_csv(detailed_file, index=False)
        print(f"Detailed results saved to {detailed_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Average total distance: {results_df['total_distance'].mean():.2f}")
        print(f"Average total steps: {results_df['total_steps'].mean():.2f}")
        
        if args.agent == 'actinf':
            print(f"Average Shannon entropy: {results_df['shannon_entropy'].mean():.4f}")
        
        success_rate = results_df['goal_reached'].mean() * 100
        print(f"Goal reached: {success_rate:.1f}%")