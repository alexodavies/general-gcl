import torch

def add_continuous_feature_noise(data, t):
    """
    Add Gaussian noise to node and edge features in a diffusion-like process.
    This is appropriate for continuous node and edge features.
    
    Parameters:
    - data: PyTorch Geometric data object containing node features (x) and edge attributes (edge_attr).
    - t: Current time step in the diffusion process, normalised by total steps, ie range [0,1].
    - node_noise_std: Base standard deviation of Gaussian noise for node features.
    - edge_noise_std: Base standard deviation of Gaussian noise for edge features.
    
    Returns:
    - Noisy data object with modified node features and edge attributes at time step t.
    """
    # Diffusion schedule (linearly increasing noise std as t increases)
    noise_scale = torch.sqrt(torch.tensor(t))
    
    
    
    # Add noise to node features
    if data.x is not None:
        data.x = data.x.to(torch.float32)
        # Normalize node features
        node_mean = data.x.mean(dim=0, keepdim=True)
        node_std = data.x.std(dim=0, keepdim=True) + 1e-6
        data.x = (data.x - node_mean) / node_std
        
        # Add Gaussian noise based on current time step
        noise = torch.randn_like(data.x) * 0.1 * noise_scale
        data.x += noise

        # Re-normalize after adding noise
        data.x = (data.x - data.x.mean(dim=0, keepdim=True)) / (data.x.std(dim=0, keepdim=True) + 1e-6)

    # Add noise to edge features
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(torch.float32)
        # Normalize edge features
        edge_mean = data.edge_attr.mean(dim=0, keepdim=True)
        edge_std = data.edge_attr.std(dim=0, keepdim=True) + 1e-6
        data.edge_attr = (data.edge_attr - edge_mean) / edge_std
        
        # Add Gaussian noise based on current time step
        noise = torch.randn_like(data.edge_attr) * 0.1 *  noise_scale
        data.edge_attr += noise

        # Re-normalize after adding noise
        data.edge_attr = (data.edge_attr - data.edge_attr.mean(dim=0, keepdim=True)) / (data.edge_attr.std(dim=0, keepdim=True) + 1e-6)

    return data


def add_discrete_feature_noise(data, t):
    """
    Apply discrete noise to bit vector node and edge features (0 or 1) in a diffusion-like process.
    This is appropriate for binary node and edge features.
    
    Parameters:
    - data: PyTorch Geometric data object containing node features (x) and edge attributes (edge_attr).
    - t: Current time step in the diffusion process, normalized by total steps, i.e., range [0,1].
    
    Returns:
    - Noisy data object with modified node features and edge attributes at time step t.
    """
    # Diffusion schedule: linearly increasing probability of flipping bits as t increases
    flip_prob = t / 2  # Probability of flipping a bit is directly proportional to t (range [0, 1])

    # Add noise to node features
    if data.x is not None:
        # Ensure the node features are integers (0 or 1)
        data.x = data.x.to(torch.int32)
        
        # Create a probability matrix of the same size as `data.x`, but with floating-point values
        flip_probs = torch.full_like(data.x, flip_prob, dtype=torch.float32)
        
        # Sample random flips with probability flip_prob
        random_flips = torch.bernoulli(flip_probs).to(torch.int32)
        
        # Apply bit flips (XOR operation to flip the bits)
        data.x = data.x ^ random_flips

    # Add noise to edge features
    if data.edge_attr is not None:
        # Ensure the edge features are integers (0 or 1)
        data.edge_attr = data.edge_attr.to(torch.int32)
        
        # Create a probability matrix of the same size as `data.edge_attr`, but with floating-point values
        flip_probs = torch.full_like(data.edge_attr, flip_prob, dtype=torch.float32)
        
        # Sample random flips with probability flip_prob
        random_flips = torch.bernoulli(flip_probs).to(torch.int32)
        
        # Apply bit flips (XOR operation to flip the bits)
        data.edge_attr = data.edge_attr ^ random_flips

    return data