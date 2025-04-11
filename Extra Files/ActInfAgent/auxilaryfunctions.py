def plot_grid(grid_locations, num_x = 3, num_y = 3 ):
    """
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate 
    labeled with its linear index (its `state id`)
    """

    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
      y, x = location
      grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar = False, fmt='.0f', cmap='crest')

def plot_likelihood(matrix, title_str = "Likelihood distribution (A)"):
    """
    Plots a 2-D likelihood matrix as a heatmap
    """

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
      raise ValueError("Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)")
    
    fig = plt.figure(figsize = (6,6))
    ax = sns.heatmap(matrix, cmap = 'gray', cbar = False, vmin = 0.0, vmax = 1.0)
    plt.title(title_str)
    plt.show()


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

def add_noise(matrix, noise_level=0.1):
    """
    Add noise to transition matrix while preserving normalization
    
    Args:
        A: Original transition matrix
        noise_level: Amount of noise to add (0-1)
    """
    # Generate random noise
    noise = np.random.uniform(-noise_level, noise_level, size=matrix.shape)

    print(noise) 
    
    # Add noise to matrix
    noisy_matrix = matrix + noise
    
    # Ensure non-negative
    noisy_matrix = np.maximum(noisy_matrix, 0.0)
    
    # Normalize columns to sum to 1
    noisy_matrix = noisy_matrix / noisy_matrix.sum(axis=0, keepdims=True)
    
    return noisy_matrix

def move_to_grid(x, y, z):
    '''Moves coppelia coordinates (x,y,z) to a 40x40 grid, z coordinate remains constant'''
    
    # Translate x,y coordinate 2.5 up and 2.5 right
    x = x + 2.5
    y = y + 2.5
    
    # Convert to 40x40 grid (5 units / 40 = 0.125 units per grid cell)
    x_grid = int(round(x / 0.125))
    y_grid = int(round(y / 0.125))
    
    # Ensure coordinates are within grid bounds
    if x_grid < 0 or x_grid >= 40 or y_grid < 0 or y_grid >= 40:
        return None
    
    return (x_grid, y_grid)
    
def grid_to_coordinates(x_grid, y_grid, z):
    '''Converts a 40x40 grid point back into coppelia coordinates'''
    
    # Ensure coordinates are within grid bounds
    if x_grid < 0 or x_grid >= 40 or y_grid < 0 or y_grid >= 40:
        return None
    
    # Convert grid coordinates back to CoppeliaSim world coordinates
    x = (x_grid * 0.125) - 2.5
    y = (y_grid * 0.125) - 2.5
    
    return (x, y, z)