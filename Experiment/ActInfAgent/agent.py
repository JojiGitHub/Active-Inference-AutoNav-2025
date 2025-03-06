import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymdp import utils
from scipy.special import softmax
from pymdp import control
from pymdp.agent import Agent

def initialize_grid(grid_dims=[40, 40]):
    """Initialize the grid world and return grid locations"""
    grid_locations = []
    for i in range(grid_dims[0]):
        for j in range(grid_dims[1]):
            grid_locations.append((i,j))
    return grid_locations, grid_dims

def initialize_state_space(grid_locations):
    """Initialize the state space parameters"""
    current_attribute = ['SAFE', 'DANGER', 'REWARDING']
    num_states = [len(grid_locations), len(current_attribute)]
    current_color = ['WHITE', 'RED', 'GREEN']
    num_obs = [len(grid_locations), len(current_color)]
    return current_attribute, current_color, num_states, num_obs

def create_vision_matrix(num_states):
    """Create and initialize the vision matrix"""
    vision_matrix = np.zeros((num_states[0], num_states[1]))
    for loc in range(num_states[0]):
        vision_matrix[loc] = np.array([0.33, 0.33, 0.33])
        vision_matrix[loc] /= vision_matrix[loc].sum()
    return vision_matrix

def create_likelihood_matrices(grid_locations, num_states, num_obs, current_attribute):
    """Create A matrices (likelihood matrices)"""
    A_shapes = []
    for i in num_obs:
        A_shapes.append([i] + num_states)
    A = utils.obj_array_zeros(A_shapes)
    
    # Location observation mapping
    for safety_level in range(num_states[1]):
        A[0][:,:,safety_level] = np.eye(num_states[0])

    # Color observation mapping
    safety_level_to_index = {state: i for i, state in enumerate(current_attribute)}
    probabilities = {
        "SAFE": [1, 0, 0],
        "DANGER": [0, 1, 0],
        "REWARDING": [0, 0, 1]
    }
    
    for safety_level, probs in probabilities.items():
        safety_idx = safety_level_to_index[safety_level]
        for loc in range(len(grid_locations)):
            for color_idx, prob in enumerate(probs):
                A[1][color_idx, loc, safety_idx] = prob
                
    # Add minimal noise to observations
    for modality in range(len(A)):
        A[modality] = add_noise(A[modality], noise_level=0)
        
    return A

def create_transition_matrices(grid_locations, grid_dims, num_states):
    """Create B matrices (transition matrices)"""
    actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    num_controls = [5, 1]
    B_f_shapes = [[ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]
    B = utils.obj_array_zeros(B_f_shapes)
    
    for action_id, action_label in enumerate(actions):
        for curr_state, (x, y) in enumerate(grid_locations):
            next_x, next_y = get_next_position(x, y, action_label, grid_dims)
            next_state = grid_locations.index((next_x, next_y))
            B[0][next_state, curr_state, action_id] = 1.0
    
    # Safety level transitions
    B[1][:,:,0] = np.eye(3)
    
    # Normalize transitions
    for action_id in range(len(actions)):
        B_action = B[0][..., action_id]
        zero_cols = (B_action.sum(axis=0) == 0)
        for col in range(B_action.shape[1]):
            if zero_cols[col]:
                B_action[col, col] = 1.0
        column_sums = B_action.sum(axis=0)
        B[0][..., action_id] = B_action / column_sums[None, :]
    
    return B, actions

def get_next_position(x, y, action, grid_dims):
    """Calculate next position based on action"""
    if action == "UP":
        return x, max(0, y - 1)
    elif action == "DOWN":
        return x, min(grid_dims[1]-1, y + 1)
    elif action == "LEFT":
        return max(0, x - 1), y
    elif action == "RIGHT":
        return min(grid_dims[0]-1, x + 1), y
    else:  # STAY
        return x, y

def create_preference_matrices(grid_locations, goal_location, num_obs):
    """Create C matrices (preference matrices)"""
    C = utils.obj_array_zeros(num_obs)
    
    # Location preferences
    C[0] = np.ones(len(grid_locations))
    C[0][grid_locations.index(goal_location)] += 1
    
    for i, loc in enumerate(grid_locations):
        distance = ((goal_location[0] - loc[0])**2 + (goal_location[1] - loc[1])**2) ** 0.5
        C[0][i] -= distance * 0.1
    C[0] = softmax(C[0])
    
    # Color preferences
    C[1] = np.zeros((num_obs[1],))
    C[1][0] = -0.1  # WHITE
    C[1][1] = -1    # RED
    C[1][2] = 1.1   # GREEN
    
    return C

def create_prior_beliefs(grid_locations, agent_pos, num_states):
    """Create D matrices (prior beliefs)"""
    D = utils.obj_array_uniform(num_states)
    D[0] = np.zeros(num_states[0])
    D[0][grid_locations.index(agent_pos)] = 1.0
    D[1] = np.ones(num_states[1]) / num_states[1]
    return D

def visualize_vision_matrix(vision_matrix, grid_dims=(40,40)):
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

def create_redspot_agent(agent_pos, goal_location, grid_dims=[40, 40], policy_len=3):
    """Create and initialize a complete Active Inference agent"""
    global vision_matrix
    
    # Initialize grid world
    grid_locations, grid_dims = initialize_grid(grid_dims)
    
    # Initialize state space
    current_attribute, current_color, num_states, num_obs = initialize_state_space(grid_locations)
    
    # Initialize vision matrix globally
    vision_matrix = create_vision_matrix(num_states)
    
    # Create likelihood matrices (A)
    A = create_likelihood_matrices(grid_locations, num_states, num_obs, current_attribute)
    
    # Create transition matrices (B)
    B, actions = create_transition_matrices(grid_locations, grid_dims, num_states)
    
    # Create preference matrices (C)
    C = create_preference_matrices(grid_locations, goal_location, num_obs)
    
    # Create prior beliefs (D)
    D = create_prior_beliefs(grid_locations, agent_pos, num_states)
    
    # Define a wrapper function that uses module-level vision_matrix
    def get_expected_states_wrapper(qs, B, policy):

        print('vison achieved')
        # visualize_vision_matrix(vision_matrix)


        from sys import modules
        current_module = modules[__name__]
        n_steps = policy.shape[0]
        n_factors = policy.shape[1]
        qs_pi = [qs] + [utils.obj_array(n_factors) for t in range(n_steps)]
        
        for t in range(n_steps):
            for control_factor, action in enumerate(policy[t,:]):
                qs_pi[t+1][control_factor] = B[control_factor][:,:,int(action)].dot(qs_pi[t][control_factor])
                # Only update vision for location changes (control_factor 0)
                if control_factor == 0:
                    max_loc = qs_pi[t+1][0].argmax()
                    qs_pi[t+1][1] = current_module.vision_matrix[max_loc]
        
        return qs_pi[1:]
    
    # Override the control module's function
    control.get_expected_states = get_expected_states_wrapper
    
    # Initialize Active Inference agent
    agent = Agent(A=A, B=B, C=C, D=D, control_fac_idx=[0], policy_len=policy_len)
    
    return agent, grid_locations, actions

def update_vision_matrix(vision_matrix, obs):
    """Update the observation matrix based on an observation"""
    one_hot_obs = [0, 0, 0]
    one_hot_obs[obs[1]] = 1
    vision_matrix[obs[0]] = one_hot_obs
    return vision_matrix

def create_observation(position, red_obs, green_obs, white_obs, grid_locations):
    """Create an observation tuple for the agent"""
    color_obs = create_color_observation(position, red_obs, green_obs, white_obs)
    return [grid_locations.index(position), color_obs]

def create_color_observation(position, red_obs, green_obs, white_obs):
    """Determine the color observation for a given position"""
    if red_obs != ['Null'] and position in red_obs:
        return 1  # RED
    if green_obs == position:
        return 2  # GREEN
    else:
        return 0  # WHITE (default case for unknown or white positions)

def custom_get_expected_states(qs, B, policy, vision_matrix):
    """Custom implementation of expected states computation with vision matrix"""
    n_steps = policy.shape[0]
    n_factors = policy.shape[1]
    qs_pi = [qs] + [utils.obj_array(n_factors) for t in range(n_steps)]
    
    for t in range(n_steps):
        for control_factor, action in enumerate(policy[t,:]):
            qs_pi[t+1][control_factor] = B[control_factor][:,:,int(action)].dot(qs_pi[t][control_factor])
            # Only update vision for location changes (control_factor 0)
            if control_factor == 0:
                max_loc = qs_pi[t+1][0].argmax()
                qs_pi[t+1][1] = vision_matrix[max_loc]
    
    return qs_pi[1:]

def add_noise(matrix, noise_level=0.1):
    """Add noise to a probability matrix while maintaining normalization"""
    noisy_matrix = matrix + np.random.uniform(0, noise_level, size=matrix.shape)
    return noisy_matrix / noisy_matrix.sum(axis=0, keepdims=True)

