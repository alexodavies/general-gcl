import networkx as nx
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
import torch


def vis_from_pyg(data, filename = None):
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

    fig, ax = plt.subplots(figsize = (6,6))

    pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx_edges(g, pos = pos)
    nx.draw_networkx_nodes(g, pos = pos, node_color=labels, cmap="tab20")

    ax.axis('off')

    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    g = nx.erdos_renyi_graph(100, 0.1)

    data = torch_geometric.utils.from_networkx(g)
    data.x = torch.Tensor(np.random.randint(size=(g.order(), 5), high=5, low = 0))


    vis_from_pyg(data)


