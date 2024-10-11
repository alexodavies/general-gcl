import networkx as nx
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from datetime import datetime
from littleballoffur.exploration_sampling import *
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.utils import remove_self_loops
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

def get_total_mol_onehot_dims():
    return np.sum(full_atom_feature_dims), np.sum(full_bond_feature_dims)

def wandb_cfg_to_actual_cfg(original_cfg, wandb_cfg):
    """
    Retrive wandb config from saved file
    Args:
        original_cfg: the config from this run
        wandb_cfg: the saved config from the training run

    Returns:
        a config with values updated to those from the saved training run
    """
    original_keys = list(vars(original_cfg).keys())
    wandb_keys = list(wandb_cfg.keys())

    for key in original_keys:
        if key not in wandb_keys:
            continue

        vars(original_cfg)[key] = wandb_cfg[key]['value']

    return original_cfg


def nx_to_rdkit(graph, labels):
    m = Chem.MolFromSmiles('')
    mw = Chem.RWMol(m)
    atom_index = {}
    for n, d in graph.nodes(data=True):
        label = labels[n]
        atom_index[n] = mw.AddAtom(Chem.Atom(int(label) + 1))
    for a, b, d in graph.edges(data=True):
        start = atom_index[a]
        end = atom_index[b]
        mw.AddBond(start, end, Chem.BondType.SINGLE)


    mol = mw.GetMol()
    return mol


def vis_molecule(molecule):
    im = Chem.Draw.MolToImage(molecule, size=(600, 600))

    return im


def vis_from_pyg(data, filename = None, ax = None, save = True):
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
        fig, ax = plt.subplots(figsize = (2,2))
        ax_was_none = True
    else:
        ax_was_none = False

    if "ogbg" not in filename:
        pos = nx.kamada_kawai_layout(g)

        nx.draw_networkx_edges(g, pos = pos, ax = ax)
        if np.unique(labels).shape[0] != 1:
            nx.draw_networkx_nodes(g, pos=pos, node_color=labels,
                                   edgecolors="black",
                                   cmap="Dark2", node_size=64,
                                   vmin=0, vmax=10, ax=ax)
    else:
        im = vis_molecule(nx_to_rdkit(g, labels))
        ax.imshow(im)

    ax.axis('off')

    # ax.axis('off')
    # ax.set_title(f"|V|: {g.order()}, |E|: {g.number_of_edges()}")

    plt.tight_layout()

    if not ax_was_none:
        return ax
    elif filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi = 300)
        plt.close()

    plt.close()

def lowres_vis_from_pyg(data, filename = None, ax = None, save = True):
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
        fig, ax = plt.subplots(figsize = (2,2))
        ax_was_none = True
    else:
        ax_was_none = False

    if "ogbg" not in filename:
        pos = nx.kamada_kawai_layout(g)

        nx.draw_networkx_edges(g, pos = pos, ax = ax)
        if np.unique(labels).shape[0] != 1:
            nx.draw_networkx_nodes(g, pos=pos, node_color=labels,
                                   edgecolors="black",
                                   cmap="Dark2", node_size=64,
                                   vmin=0, vmax=10, ax=ax)
    else:
        im = vis_molecule(nx_to_rdkit(g, labels))
        ax.imshow(im)

    ax.axis('off')
    # ax.set_title(f"|V|: {g.order()}, |E|: {g.number_of_edges()}")

    plt.tight_layout()

    if not ax_was_none:
        return ax
    elif filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi = 50)
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
        ax = vis_from_pyg(datalist[i_axis], ax = ax, filename=filename, save = False)

    plt.savefig(filename)
    plt.close()

def visualize_grid_with_labels(datalist, filename, names):
    """
    Visualize a set of graphs, from PyTorch Geometric Data objects, with labeled rows and columns.

    Args:
        datalist: list of pyg.data.Data objects.
        filename: the path where the visualized grid is saved.
        names: list of names to use for row and column labels.
    
    Returns:
        None
    """

    # Calculate grid dimensions
    grid_dim = int(np.ceil(np.sqrt(len(datalist))))

    # Create subplots
    fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(10, 10))

    # Unpack axes if there's more than one subplot
    axes = [num for sublist in axes for num in sublist]

    # Iterate through each axis and data point to visualize the graphs
    for i_axis, ax in enumerate(axes):
        if i_axis < len(datalist):
            name_idx = i_axis % grid_dim
            # Visualize the data using vis_from_pyg function (assumed to be defined elsewhere)
            vis_from_pyg(datalist[i_axis], ax=ax, filename=names[name_idx], save=False)
        else:
            ax.set_visible(False)  # Hide extra subplots if datalist is not a perfect square

    # Set labels for the rows and columns
    for i in range(grid_dim):
        if i < len(names):
            print(f"Setting axis label: {names[i]}")
            # Set row labels (Y-axis, left side)
            plt.setp(axes[i * grid_dim].yaxis, label_position='left')
            axes[i * grid_dim].set_ylabel(names[i])

            # Set column labels (X-axis, top)
            plt.setp(axes[i].xaxis, label_position='top')
            axes[i].set_xlabel(names[i])

    # Adjust layout to prevent overlap of labels and plots
    plt.tight_layout(pad = 1.15)

    # Save the figure
    plt.savefig(filename)
    plt.close()

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
    edges, _ = remove_self_loops(data.edge_index)
    edges = edges.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

    for ilabel in range(labels.shape[0]):
        if ilabel not in np.unique(edges):
            g.add_node(ilabel)

    return g, labels

def setup_wandb(cfg, offline = False, name = None):
    """
    Uses a config dictionary to initialise wandb to track sampling.
    Requires a wandb account, https://wandb.ai/

    params: cfg: argparse Namespace

    returns:
    param: cfg: same config
    """

    kwargs = {'name': name if name is not None else 'all' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'gcl-validation-reiteration', 'config': cfg,
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'entity':'hierarchical-diffusion',
              'mode':'online' if offline else 'online'}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    return cfg


def summarize_model(model):
    """
    Prints the layers of the model, total trainable parameters, and model size in MB.
    
    Args:
    model (torch.nn.Module): The PyTorch model to summarize.
    """
    
    # List model layers
    print("Model Layers:")
    for name, layer in model.named_children():
        print(f"{name}: {layer.__class__.__name__}")

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model size (in MB)
    model_size = total_params * 4 / (1024 ** 2)  # Assuming 32-bit (4 bytes per float)

    print(f"\nModel Size: {model_size:.2f} MB")
    print(f"Trainable Parameters: {trainable_params}")


def ESWR(graph, n_graphs, size):

    # possible_samplers = inspect.getmembers(samplers, inspect.isclass)
    #
    # possible_samplers = [item[1] for item in possible_samplers]
    possible_samplers = [MetropolisHastingsRandomWalkSampler, DiffusionSampler, ForestFireSampler]
    sampler_list = []
    for sampler in possible_samplers:
        for i in range(24,96):
            sampler_list.append(sampler(i))
    # # selected_sampler = possible_samplers[np.random.randint(len(possible_samplers))]
    #
    #
    # print(f"Sampling {n_graphs} graphs from {graph}")
    # graphs = []
    # for i in tqdm(range(n_graphs), leave = False):
    #     selected_sampler = possible_samplers[np.random.randint(len(possible_samplers))]
    #     sampler = selected_sampler(number_of_nodes=np.random.randint(12, 48))
    #     graphs.append(nx.convert_node_labels_to_integers(sampler.sample(graph)))
    # sampler = selected_sampler(number_of_nodes=np.random.randint(12, 36))
    # sampler = MetropolisHastingsRandomWalkSampler(48)
    graphs = []
    for i in tqdm(range(n_graphs)):
        sampler = sampler_list[np.random.randint(len(sampler_list))]
        g = nx.convert_node_labels_to_integers(sampler.sample(graph))
        graphs.append(g)
    # graphs = [nx.convert_node_labels_to_integers(sampler.sample(graph)) for i in tqdm(range(n_graphs))]

    return graphs

if __name__ == "__main__":
    g = nx.erdos_renyi_graph(100, 0.1)

    data = torch_geometric.utils.from_networkx(g)
    data.x = torch.Tensor(np.random.randint(size=(g.order(), 5), high=5, low = 0))


    vis_from_pyg(data)


