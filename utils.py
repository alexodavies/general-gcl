import networkx as nx
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
import torch


def vis_from_pyg(data, filename = None, ax = None):
    """
    Visualise a pytorch_geometric.data.Data object
    Args:
        data: pytorch_geometric.data.Data object
        filename: if passed, this is the filename for the saved image. Ignored if ax is not None
        ax: matplotlib axis object, which is returned if passed

    Returns:

    """
    g, labels = better_to_nx(data)
    if ax is None:
        fig, ax = plt.subplots(figsize = (6,6))
        ax_was_none = True
    else:
        ax_was_none = False

    pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    if np.unique(labels).shape[0] != 1:
        nx.draw_networkx_nodes(g, pos=pos, node_color=labels, cmap="tab20", node_size=64,
                               vmin=0, vmax=20, ax=ax)

    ax.axis('off')
    ax.set_title(f"|V|: {g.order()}, |E|: {g.number_of_edges()}")

    plt.tight_layout()

    if not ax_was_none:
        return ax
    elif filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

    plt.close()



def vis_grid(datalist, filename):
    """
    Visualise a set of graphs, from pytorch_geometric.data.Data objects
    Args:
        datalist: list of pyg.data.Data objects
        filename: the visualised grid is saved to this path

    Returns:
        None
    """

    # Trim to square root to ensure square grid
    grid_dim = int(np.sqrt(len(datalist)))

    fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(8,8))

    # Unpack axes
    axes = [num for sublist in axes for num in sublist]

    for i_axis, ax in enumerate(axes):
        ax = vis_from_pyg(datalist[i_axis], ax = ax)

    plt.savefig(filename)

def better_to_nx(data):
    """
    Converts a pytorch_geometric.data.Data object to a networkx graph,
    robust to nodes with no edges, unlike the original pytorch_geometric version

    Args:
        data: pytorch_geometric.data.Data object

    Returns:
        g: a networkx.Graph graph
        labels: torch.Tensor of node labels
    """
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

    for ilabel in range(labels.shape[0]):
        if ilabel not in np.unique(edges):
            g.add_node(ilabel)

    return g, labels

if __name__ == "__main__":
    g = nx.erdos_renyi_graph(100, 0.1)

    data = torch_geometric.utils.from_networkx(g)
    data.x = torch.Tensor(np.random.randint(size=(g.order(), 5), high=5, low = 0))


    vis_from_pyg(data)


