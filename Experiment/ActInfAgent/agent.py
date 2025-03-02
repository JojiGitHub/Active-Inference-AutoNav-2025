import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymdp
from pymdp import utils, control, maths
from pymdp.agent import Agent
import random
import time
from math import comb

def update_vision_matrix(vision_matrix, obs):
    """
    Update the observation matrix based on an observation
    """
    one_hot_obs = [0, 0, 0]
    one_hot_obs[obs[1]] = 1
    vision_matrix[obs[0]] = one_hot_obs
    return vision_matrix


def visualize_vision_matrix(vision_matrix, grid_dims):
    """
    Visualizes the observation matrix as three separate heatmaps, one for each attribute
    (SAFE, DANGER, REWARDING)
    
    Args:
        vision_matrix: numpy array of shape (num_locations, num_attributes)
        grid_dims: list of [height, width] of the grid
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Labels for each subplot
    titles = ['SAFE', 'DANGER', 'REWARDING']
    
    for idx, title in enumerate(titles):
        # Reshape the matrix to match grid dimensions
        heatmap_data = vision_matrix[:, idx].reshape(grid_dims[0], grid_dims[1])
        
        # Plot heatmap
        sns.heatmap(heatmap_data, 
                   ax=axes[idx], 
                   cmap='YlOrRd',
                   vmin=0, 
                   vmax=1,
                   annot=True,
                   fmt='.2f')
        
        axes[idx].set_title(f'{title} Probability')
        axes[idx].set_xlabel('Y coordinate')
        axes[idx].set_ylabel('X coordinate')
    
    plt.tight_layout()
    plt.show()


def add_noise(matrix, noise_level=0.1):
    """
    Add noise to transition matrix while preserving normalization
    
    Args:
        matrix: Original transition matrix
        noise_level: Amount of noise to add (0-1)
    """
    # Generate random noise
    noise = np.random.uniform(-noise_level, noise_level, size=matrix.shape)
    
    # Add noise to matrix
    noisy_matrix = matrix + noise
    
    # Ensure non-negative
    noisy_matrix = np.maximum(noisy_matrix, 0.0)
    
    # Normalize columns to sum to 1
    noisy_matrix = noisy_matrix / noisy_matrix.sum(axis=0, keepdims=True)
    
    return noisy_matrix


def custom_get_expected_states(qs, B, policy, vision_matrix):
    """
    Compute the expected states under a policy, also known as the posterior predictive density over states

    Parameters
    ----------
    qs: ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at a given timepoint.
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    policy: 2D ``numpy.ndarray``
        Array that stores actions entailed by a policy over time. Shape is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    vision_matrix: numpy.ndarray
        Matrix containing the agent's visual observations

    Returns
    -------
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    """
    n_steps = policy.shape[0]
    n_factors = policy.shape[1]
    
    # Initialize posterior predictive density as a list of beliefs over time, including current posterior beliefs about hidden states as the first element
    qs_pi = [qs] + [utils.obj_array(n_factors) for t in range(n_steps)]
    
    # Get expected states over time
    for t in range(n_steps):
        for control_factor, action in enumerate(policy[t,:]):
            qs_pi[t+1][control_factor] = B[control_factor][:,:,int(action)].dot(qs_pi[t][control_factor])
            qs_pi[t+1][1] = vision_matrix[qs_pi[t+1][0].argmax()]
    
    return qs_pi[1:]


def plot_beliefs(belief_dist, title_str=""):
    """
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    """
    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError("Distribution not normalized! Please normalize")

    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()


def plot_likelihood(matrix, title_str="Likelihood distribution (A)"):
    """
    Plots a 2-D likelihood matrix as a heatmap
    """
    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError("Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)")
    
    fig = plt.figure(figsize=(6, 6))
    ax = sns.heatmap(matrix, cmap='gray', cbar=False, vmin=0.0, vmax=1.0)
    plt.title(title_str)
    plt.show()


def plot_grid(grid_locations, num_x=3, num_y=3):
    """
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate 
    labeled with its linear index (its `state id`)
    """
    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
        y, x = location
        grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar=False, fmt='.0f', cmap='crest')


def create_observation(position, grid_locations, red_obs, green_obs, white_obs):
    """
    Create an observation vector from the agent's current position and its observations
    """
    return [grid_locations.index(position), create_color_observation(position, red_obs, green_obs, white_obs)]


def create_color_observation(position, red_obs, green_obs, white_obs):
    """
    Determine the color observation based on the agent's position
    """
    if red_obs != ['Null']:
        if position in red_obs: 
            return 1  # RED
    if green_obs == position: 
        return 2  # GREEN
    elif white_obs != ['Null']:
        if position in white_obs: 
            return 0  # WHITE
    return 0  # Default to WHITE if no specific observation


def update_vision(current_location, grid_dims, distance):
    """
    Update the agent's field of vision based on the current location and distance
    Returns a list of all grid positions within the vision range
    
    Args:
        current_location (tuple): Current (x,y) position of the agent
        grid_dims (list): Dimensions of the grid [width, height]
        distance (int): Vision range/distance
        
    Returns:
        list: List of (x,y) tuples representing visible grid positions
    """
    x, y = current_location
    x_min = max(0, x - distance)
    x_max = min(grid_dims[0], x + distance + 1)
    y_min = max(0, y - distance)
    y_max = min(grid_dims[1], y + distance + 1)
    
    visible_locations = []
    for y_pos in range(y_min, y_max):
        for x_pos in range(x_min, x_max):
            visible_locations.append((x_pos, y_pos))
            
    return visible_locations


def initialize_generative_model(grid_dims, agent_pos, goal_location, redspots):
    """
    Initialize the generative model for the active inference agent
    
    Args:
        grid_dims (list): Grid dimensions [height, width]
        agent_pos (tuple): Agent's starting position (x, y)
        goal_location (tuple): Goal position (x, y)
        redspots (list): List of dangerous (red) spot positions
        
    Returns:
        A, B, C, D: Matrices for the agent's generative model
        vision_matrix: Vision matrix for the agent
        actions: List of possible actions
    """
    # Generate all grid locations
    grid_locations = []
    for i in range(grid_dims[0]):
        for j in range(grid_dims[1]):
            grid_locations.append((i, j))
    
    # Define states and observations
    # s1 = current location
    # s2 = attribute of current location (SAFE, DANGER, REWARDING)
    current_attribute = ['SAFE', 'DANGER', 'REWARDING']
    num_states = [len(grid_locations), len(current_attribute)]
    
    # o1 = observed current location
    # o2 = color of current location (WHITE, RED, GREEN)
    current_color = ['WHITE', 'RED', 'GREEN']
    num_obs = [len(grid_locations), len(current_color)]
    
    # Initialize vision matrix
    vision_matrix = np.zeros((num_states[0], num_states[1]))
    for loc in range(num_states[0]):
        vision_matrix[loc] = np.array([0.33, 0.33, 0.33])
        vision_matrix[loc] /= vision_matrix[loc].sum()
    
    # Define A Matrix (Observation model)
    A_shapes = []
    for i in num_obs:
        A_shapes.append([i] + num_states)
    
    A = utils.obj_array_zeros(A_shapes)
    
    # Location observation modality A[0]
    for safety_level in range(num_states[1]):
        A[0][:, :, safety_level] = np.eye(num_states[0])
    
    # Color observation modality A[1]
    # Map safety levels to indices
    safety_level_to_index = {state: i for i, state in enumerate(current_attribute)}
    
    # Probabilities for each color given the safety level
    probabilities = {
        "SAFE": [1, 0, 0],        # ['WHITE', 'RED', 'GREEN']
        "DANGER": [0, 1, 0],      # ['WHITE', 'RED', 'GREEN']
        "REWARDING": [0, 0, 1]    # ['WHITE', 'RED', 'GREEN']
    }
    
    # Populate A[1]
    for safety_level, probs in probabilities.items():
        safety_idx = safety_level_to_index[safety_level]
        for loc in range(len(grid_locations)):  # Iterate over grid locations
            for color_idx, prob in enumerate(probs):  # Iterate over colors
                A[1][color_idx, loc, safety_idx] = prob
    
    # Add noise to matrices
    for modality in range(len(A)):
        A[modality] = add_noise(A[modality], noise_level=0)
    
    # Define B Matrix (Transition model)
    actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    num_controls = [len(actions), 1]
    B_f_shapes = [[ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]
    B = utils.obj_array_zeros(B_f_shapes)
    
    # B[0] - Control Factor - Location Transitions
    for action_id, action_label in enumerate(actions):
        for curr_state, (x, y) in enumerate(grid_locations):
            # Calculate next position based on action
            if action_label == "UP":
                next_y = max(0, y - 1)        # Move up (decrease y)
                next_x = x
            elif action_label == "DOWN":
                next_y = min(grid_dims[1]-1, y + 1)  # Move down (increase y)
                next_x = x
            elif action_label == "LEFT":
                next_x = max(0, x - 1)        # Move left (decrease x)
                next_y = y
            elif action_label == "RIGHT":
                next_x = min(grid_dims[0]-1, x + 1)  # Move right (increase x)
                next_y = y
            else:  # STAY
                next_x = x
                next_y = y
            
            # Get the state index for the next position
            next_state = grid_locations.index((next_x, next_y))
            
            # Set transition probability to 1.0
            B[0][next_state, curr_state, action_id] = 1.0
    
    # B[1] - Non-Control Factor - Identity Matrix
    B[1][:,:,0] = np.eye(3)  # Identity matrix for safety level transitions
    
    # Normalize B matrix columns for each action
    for action_id in range(len(actions)):
        # Get slice for current action
        B_action = B[0][..., action_id]
        
        # Replace zero columns with ones in appropriate positions
        zero_cols = (B_action.sum(axis=0) == 0)
        for col in range(B_action.shape[1]):
            if zero_cols[col]:
                # Stay in the same state if no transition is defined
                B_action[col, col] = 1.0
        
        # Normalize columns
        column_sums = B_action.sum(axis=0)
        B[0][..., action_id] = B_action / column_sums[None, :]
    
    # Define C Vector (prior preferences)
    C = utils.obj_array_zeros(num_obs)
    
    # C[0] - Preference for location observations
    C[0] = np.ones(len(grid_locations))
    C[0][grid_locations.index(goal_location)] += 1
    
    # Add distance-based preferences
    for i, loc in enumerate(grid_locations):
        x = ((goal_location[0] - loc[0])**2 + (goal_location[1] - loc[1])**2) ** 0.5
        C[0][i] -= x * 0.1
    
    # Apply softmax to get proper distribution
    C[0] = maths.softmax(C[0])
    
    # C[1] - Preference for color observations (WHITE, RED, GREEN)
    C[1] = np.zeros((num_obs[1],))
    C[1][0] = -0.1    # WHITE
    C[1][1] = -1      # RED
    C[1][2] = 1.1     # GREEN
    
    # Define D Vector (prior beliefs about hidden states)
    D = utils.obj_array_uniform(num_states)
    
    # D[0] - Belief about current location
    D[0] = np.zeros(num_states[0])
    D[0][grid_locations.index(agent_pos)] = 1.0  # One-hot encoding for location
    
    # D[1] - Belief about attribute of current location
    D[1] = np.ones(num_states[1]) / num_states[1]  # Uniform distribution
    
    return A, B, C, D, vision_matrix, actions, grid_locations


class RedspotAgent(Agent):
    def __init__(self, A, B, C, D, policy_len=3, vision_matrix=None, grid_dims=None, grid_locations=None, actions=None):
        super().__init__(A=A, B=B, C=C, D=D, policy_len=policy_len)
        self.vision_matrix = vision_matrix
        self.grid_dims = grid_dims
        self.grid_locations = grid_locations
        self.actions = actions
        
        # Override the get_expected_states function in the control module
        control.get_expected_states = lambda qs, B, policy: custom_get_expected_states(qs, B, policy, self.vision_matrix)
        
    def update_vision(self, obs):
        """Update the agent's vision matrix based on an observation"""
        self.vision_matrix = update_vision_matrix(self.vision_matrix, obs)
        
    def visualize_vision(self):
        """Visualize the agent's current vision matrix"""
        visualize_vision_matrix(self.vision_matrix, self.grid_dims)
        
    def get_action_name(self, action_id):
        """Get the name of an action from its ID"""
        return self.actions[action_id]
        
    def create_observation_from_env(self, position, red_obs, green_obs, white_obs):
        """Create an observation vector from the environment state"""
        return create_observation(position, self.grid_locations, red_obs, green_obs, white_obs)


def create_redspot_agent(grid_dims=[40, 40], agent_pos=(24, 30), goal_location=(14, 16), redspots=None):
    """
    Create a RedspotAgent with initialized matrices
    
    Args:
        grid_dims (list): Grid dimensions [height, width]
        agent_pos (tuple): Agent's starting position (x, y)
        goal_location (tuple): Goal position (x, y)
        redspots (list): List of dangerous (red) spot positions
        
    Returns:
        RedspotAgent: Initialized agent ready to be used
    """
    if redspots is None:
        redspots = []
        
    A, B, C, D, vision_matrix, actions, grid_locations = initialize_generative_model(
        grid_dims, agent_pos, goal_location, redspots
    )
    
    agent = RedspotAgent(
        A=A, B=B, C=C, D=D, 
        policy_len=3,
        vision_matrix=vision_matrix, 
        grid_dims=grid_dims,
        grid_locations=grid_locations,
        actions=actions
    )
    
    return agent


