import torch
import numpy as np
import pandas as pd
from ActInfAgent.test import test_active_inference
from ActInfAgent.agent import create_redspot_agent
from RL_Agent.test import run_environment
import random
from datetime import datetime

def collect_data(num_episodes=5, grid_dims=(40, 40)):
    """
    Collect data from both RL and Active Inference agents
    
    Args:
        num_episodes (int): Number of episodes to run for each agent
        grid_dims (tuple): Grid dimensions for the environment
    """
    # Initialize data storage
    data = {
        'episode': [],
        'agent_type': [],
        'random_seed': [],
        'total_steps': [],
        'total_reward': [],
        'total_distance': [],
        'reached_goal': []
    }
    
    # Run episodes
    for i in range(num_episodes):
        random_seed = i  # Use episode number as random seed for reproducibility
        print(f"\nRunning episode {i+1}/{num_episodes}")
        
        # Run RL agent episode
        print("\nRunning RL Agent...")
        rl_results = run_environment(random_seed=random_seed)
        if rl_results:
            data['episode'].append(i)
            data['agent_type'].append('RL')
            data['random_seed'].append(random_seed)
            data['total_steps'].append(rl_results['total_steps'])
            data['total_reward'].append(rl_results['total_reward'])
            data['total_distance'].append(rl_results['total_distance'])
            data['reached_goal'].append(rl_results['end_reason'] == 'goal_reached')
        
        # Run Active Inference agent episode
        print("\nRunning Active Inference Agent...")
        
        # For each episode, create a new active inference agent with appropriate initialization
        # The initialization parameters will be determined by the environment during test_active_inference
        act_inf_agent = create_redspot_agent(
            grid_dims=list(grid_dims),
            # These will be overridden in test_active_inference with actual positions
            agent_pos=(0, 0),
            goal_location=(0, 0),
            redspots=[]
        )
        
        actinf_results = test_active_inference(random_seed=random_seed, agent=act_inf_agent)
        
        data['episode'].append(i)
        data['agent_type'].append('ActiveInference')
        data['random_seed'].append(random_seed)
        data['total_steps'].append(actinf_results['total_steps'])
        data['total_reward'].append(actinf_results['total_reward'])
        data['total_distance'].append(actinf_results['total_distance'])
        data['reached_goal'].append(actinf_results['reached_goal'])
        
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'experiment_results_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\nData collection complete. Results saved to {filename}")
    
    return df

results_df = collect_data(num_episodes=5)

# Print summary statistics
print("\nSummary Statistics:")
print("\nSuccess Rate by Agent Type:")
print(results_df.groupby('agent_type')['reached_goal'].mean())

print("\nAverage Steps by Agent Type:")
print(results_df.groupby('agent_type')['total_steps'].mean())

print("\nAverage Distance by Agent Type:")
print(results_df.groupby('agent_type')['total_distance'].mean())

print("\nAverage Reward by Agent Type:")
print(results_df.groupby('agent_type')['total_reward'].mean())