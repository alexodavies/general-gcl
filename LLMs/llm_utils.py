import numpy as np

def format_features(features, key_list):
    """Helper function to format the features with their corresponding keys."""
    return ", ".join([f"{key}: {value}" for key, value in zip(key_list, features)])

def edge_list_to_text(edge_list, condense = True):
    if isinstance(edge_list, list):
        edge_list = np.array(edge_list)

    if edge_list.shape[1] != 2:
        edge_list = edge_list.T
    assert edge_list.shape[1] == 2, "Each edge should have 2 nodes."
    
    connections = []
    edges = []
    nodes = [str(node) for node in np.unique(edge_list)]

    for edge in edge_list:
        node1, node2 = edge
        if (node2, node1) not in edges:
            if condense:
                connections.append(f"{node1} <-> {node2}")
            else:
                connections.append(f"Node {node1} is connected to node {node2}.")
            edges.append((node1, node2))

    
    # Join connections and node info
    return "nodes: " + ", ".join(nodes) + " edges: " +  ", ".join(connections)