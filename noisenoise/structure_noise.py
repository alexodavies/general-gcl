import torch

def add_structure_noise(data, t):
    """
    Add edge noise in a diffusion-like process: edges are randomly dropped and added based on the diffusion time step.
    
    Parameters:
    - data: PyTorch Geometric data object containing edge_index and num_nodes.
    - t: Current time step in the diffusion process, normalised by total steps, ie range [0,1].
    
    Returns:
    - Noisy data object with modified edge_index.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Scale probabilities based on the diffusion time step
    drop_prob = t #(t / max_t)
    add_prob = t #(t / max_t)

    # Original edges
    row, col = edge_index

    # Randomly drop edges
    mask = torch.rand(row.size(0)) > drop_prob
    row, col = row[mask], col[mask]

    n_dropped = mask.size(0) - row.size(0)

    # Randomly add new edges
    num_new_edges = n_dropped # int(add_prob * row.size(0))
    new_row = torch.randint(0, num_nodes, (num_new_edges,))
    new_col = torch.randint(0, num_nodes, (num_new_edges,))

    # Combine the original and new edges
    row = torch.cat([row, new_row], dim=0)
    col = torch.cat([col, new_col], dim=0)

    # Update edge_index in the data object
    data.edge_index = torch.stack([row, col], dim=0)
    
    return data

