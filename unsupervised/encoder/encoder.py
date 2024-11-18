import numpy as np
import torch
import copy
from abc import ABC
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from unsupervised.convs import GINEConv, GPSConv
from torch_geometric.nn import GATv2Conv, GCNConv

from typing import Any, Dict, Optional
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import get_laplacian

from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)

class LaplacianEigenvectorPEBatch:
	def __init__(self, num_eigenvectors=5):
		self.num_eigenvectors = num_eigenvectors

	def compute_positional_encoding(self, edge_index, batch, num_nodes, edge_weight=None):
		device = edge_index.device
		pe_batch = torch.zeros((num_nodes, self.num_eigenvectors), device=device)

		# Iterate through each graph in the batch
		for i in range(batch.max().item() + 1):
			graph_mask = (batch == i)
			graph_nodes = torch.where(graph_mask)[0]
			num_graph_nodes = len(graph_nodes)

			if num_graph_nodes < 2:  # Skip small graphs
				continue

			# Get the subgraph's edge_index and edge_weight
			edge_mask = graph_mask[edge_index[0]] & graph_mask[edge_index[1]]

			sub_edge_index = edge_index[:, edge_mask]

			# The removal of edges is encoded through edge weight
			weight_based_mask = edge_weight[edge_mask] < 0.5 if edge_weight is not None else torch.ones(sub_edge_index.shape[1]).to(torch.bool)
			sub_edge_index = sub_edge_index[:, weight_based_mask]
			# sub_edge_weight = edge_weight[edge_mask] if edge_weight is not None else None

			# Convert to a dense adjacency matrix
			# adj_matrix = to_dense_adj(sub_edge_index, max_num_nodes=num_graph_nodes, edge_attr=sub_edge_weight).squeeze(0)
			adj_matrix = to_dense_adj(sub_edge_index, max_num_nodes=num_graph_nodes).squeeze(0)

			# Compute the degree matrix (D) and Laplacian (L = D - A)
			degree_matrix = torch.diag(adj_matrix.sum(dim=1))
			laplacian = degree_matrix - adj_matrix

			# Regularize the Laplacian to prevent numerical issues
			eps = 1e-3
			laplacian = laplacian + eps * torch.eye(num_graph_nodes, device=device)
			assert torch.isfinite(laplacian).all(), "Laplacian matrix contains NaNs or Infs"

			# Compute eigenvectors using a dense Laplacian
			laplacian_cpu = laplacian.to('cpu')  # Perform eigen decomposition on CPU
			
			eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_cpu)

			# Select the smallest non-zero eigenvectors for positional encoding
			pe = eigenvectors[:, 1:self.num_eigenvectors + 1]

			# Normalize the positional encoding to have zero mean and unit variance
			pe = (pe - pe.mean(dim=0)) / pe.std(dim=0)

			if pe.size(1) < self.num_eigenvectors:
				padding = torch.zeros(pe.size(0), self.num_eigenvectors - pe.size(1), device=pe.device)
				pe = torch.cat([pe, padding], dim=1)
			# Assign positional encodings to the correct nodes in the batch
			pe_batch[graph_mask] = pe.to(device)

		return pe_batch


class AddRandomWalkPE:
	def __init__(self, walk_length=5, attr_name='random_walk_pe'):
		"""
		Initialize the RandomWalkPositionalEncoding class for batched graphs.

		Args:
			walk_length (int): Number of random walk steps.
			attr_name (str): Attribute name for positional encodings.
		"""
		self.walk_length = walk_length
		self.attr_name = attr_name

	def compute_positional_encoding(self, x, edge_index, edge_weight=None, batch=None):
		"""
		Compute Random Walk Positional Encoding (RWPE) for a batch of graphs.

		Args:
			x (torch.Tensor): Node features (shape: [num_nodes, num_features]).
			edge_index (torch.Tensor): Edge indices in COO format (shape: [2, num_edges]).
			edge_weight (torch.Tensor, optional): Edge weights, if any (shape: [num_edges]). Default is None.
			batch (torch.Tensor): Batch vector that maps each node to a specific graph (shape: [num_nodes]).

		Returns:
			torch.Tensor: The random walk positional encoding for all graphs in the batch.
		"""
		device = edge_index.device
		batch_size = batch.max().item() + 1  # Number of graphs in the batch
		num_nodes = x.size(0)  # Total number of nodes across all graphs

		# Create dense adjacency matrices (including edge weights if provided)
		adj_matrices = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)

		# Initialize a tensor to store positional encodings for each node in the batch
		pe_batch = torch.zeros((num_nodes, self.walk_length), device=device)

		# Iterate through each graph in the batch
		for i in range(batch_size):
			adj = adj_matrices[i]  # Get the adjacency matrix for the i-th graph
			row_sum = adj.sum(dim=1, keepdim=True)
			adj = adj / row_sum.clamp(min=1e-9)  # Normalize rows to sum to 1	

			adj = adj + torch.eye(adj.shape[0], device=device)

			# Identify nodes belonging to this graph
			graph_mask = (batch == i)
			num_graph_nodes = graph_mask.sum().item()  # Number of nodes in this graph

			# If the graph has no nodes, skip it
			if num_graph_nodes == 0:
				continue

			# Create identity matrix for self-loops
			loop_index = torch.arange(num_graph_nodes, device=device)

			# Function to extract diagonal elements (self-loop probabilities)
			def get_pe(out):
				return out[loop_index, loop_index]

			# Initialize the positional encoding for this graph
			out = adj
			out = torch.clamp(out, min=-1e6, max=1e6)
			pe_list = [get_pe(out)]  # Start with step 0

			# Perform random walks
			for _ in range(self.walk_length - 1):
				out = out @ adj  # Multiply adjacency matrix with itself
				pe_list.append(get_pe(out))  # Get the diagonal
			# pe = torch.where(graph_mask, pe, torch.zeros_like(pe))  # Zero encoding for isolated nodes
			# Stack the positional encodings for the current graph
			pe = torch.stack(pe_list, dim=-1)  # Shape: (num_graph_nodes, walk_length)
			

			# Assign the result to the corresponding nodes in the batch
			pe_batch[graph_mask] = pe
		assert torch.isfinite(pe_batch).all(), "NaNs or Infs detected in positional encoding"
		return pe_batch  # Shape: (num_nodes, walk_length)

# GraphGPS results are not listed in current works
# Only used for attention with GPSConv
class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class Encoder(torch.nn.Module):
	"""
	Encoder module for graph neural networks.

	Args:
		emb_dim (int): The dimensionality of the node embeddings.
		num_gc_layers (int): The number of graph convolutional layers.
		drop_ratio (float): The dropout ratio.
		pooling_type (str): The type of graph pooling to use.
		is_infograph (bool): Whether to use Infograph pooling.
		convolution (str): The type of graph convolutional operation to use.
		edge_dim (int): The dimensionality of the edge embeddings.

	Attributes:
		pooling_type (str): The type of graph pooling being used.
		emb_dim (int): The dimensionality of the node embeddings.
		num_gc_layers (int): The number of graph convolutional layers.
		drop_ratio (float): The dropout ratio.
		is_infograph (bool): Whether to use Infograph pooling.
		out_node_dim (int): The output dimensionality of the node embeddings.
		out_graph_dim (int): The output dimensionality of the graph embeddings.
		convs (torch.nn.ModuleList): List of graph convolutional layers.
		bns (torch.nn.ModuleList): List of batch normalization layers.
		atom_encoder (AtomEncoder): Atom encoder module.
		bond_encoder (BondEncoder): Bond encoder module.
		edge_dim (int): The dimensionality of the edge embeddings.
		convolution (type): The type of graph convolutional operation being used.

	Methods:
		init_emb(): Initializes the node embeddings.
		forward(batch, x, edge_index, edge_attr, edge_weight=None): Performs forward pass through the encoder.
		get_embeddings(loader, device, is_rand_label=False, every=1, node_features=False): Computes embeddings for a given data loader.

	"""

	def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0,
				 pooling_type="standard", is_infograph=False,
				 convolution="gin", edge_dim=1,
				 pos_features = 4, pe_dim = 8):
		super(Encoder, self).__init__()

		self.pooling_type = pooling_type

		if convolution == "gps":	
			emb_dim = np.around(emb_dim / 4).astype(int) * 4
			self.pe_transform = AddRandomWalkPE(walk_length=pos_features)
			self.pe_lin = Linear(pos_features, pe_dim)
			self.pe_norm = torch.nn.BatchNorm1d(pos_features)
		
		self.emb_dim = emb_dim
		self.num_gc_layers = num_gc_layers
		self.drop_ratio = drop_ratio
		self.is_infograph = is_infograph



		self.out_node_dim = self.emb_dim
		if self.pooling_type == "standard":
			self.out_graph_dim = self.emb_dim
		elif self.pooling_type == "layerwise":
			self.out_graph_dim = self.emb_dim * self.num_gc_layers
		else:
			raise NotImplementedError

		self.convs = torch.nn.ModuleList()
		self.bns = torch.nn.ModuleList()

		# self.atom_encoder = AtomEncoder(emb_dim)
		if convolution != "gps":
			self.atom_encoder = AtomEncoder(emb_dim)
		else:
			self.atom_encoder = AtomEncoder(emb_dim - pe_dim)

		self.bond_encoder = BondEncoder(emb_dim)
		self.edge_dim = edge_dim

		if convolution == "gin":
			# print(f"Using GIN backbone for {num_gc_layers} layers")
			self.convolution = GINEConv

			for i in range(num_gc_layers):
				nn = Sequential(Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), ReLU(),
								Linear(2 * emb_dim, emb_dim))
				conv = GINEConv(nn)
				bn = torch.nn.BatchNorm1d(emb_dim)
				self.convs.append(conv)
				self.bns.append(bn)

		elif convolution == "gcn":
			print(f"Using GCN backbone for {num_gc_layers} layers")
			self.convolution = GCNConv

			for i in range(num_gc_layers):
				self.convs.append(GCNConv(emb_dim, emb_dim))
				bn = torch.nn.BatchNorm1d(emb_dim)
				self.bns.append(bn)

		elif convolution == "gat":
			print(f"Using GAT backbone for {num_gc_layers} layers")
			self.convolution = GATv2Conv

			for i in range(num_gc_layers):
				self.convs.append(GATv2Conv(emb_dim, emb_dim))
				bn = torch.nn.BatchNorm1d(emb_dim)
				self.bns.append(bn)

		elif convolution == "gps":
			print(f"Using GPS with GIN backbone for {num_gc_layers} layers")
			self.convolution = GPSConv

			for i in range(num_gc_layers):
				nn = Sequential(Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), ReLU(),
								Linear(2 * emb_dim, emb_dim))
				conv = GPSConv(emb_dim,
				   				GINEConv(nn),
								heads = 4,
								attn_type="performer",
								act = "leaky_relu",  # to allow small gradients even for negative inputs
								act_kwargs = {"negative_slope": 0.01},  # if using LeakyReLU)
								# dropout=0.3
								)
				bn = torch.nn.BatchNorm1d(emb_dim)
				self.convs.append(conv)
				self.bns.append(bn)

			self.redraw_projection = RedrawProjection(
										self.convs,
										redraw_interval=20000)

		else:	
			raise NotImplementedError

		self.init_emb()

	def init_emb(self):
		"""
		Initializes the node embeddings.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)
			# if isinstance(m, PerformerAttention):
			# # Custom initialization for attention weights if necessary
			# 	torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)



	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
		"""
		Performs forward pass through the encoder.

		Args:
			batch (Tensor): The batch tensor.
			x (Tensor): The node feature tensor.
			edge_index (LongTensor): The edge index tensor.
			edge_attr (Tensor): The edge attribute tensor.
			edge_weight (Tensor, optional): The edge weight tensor. Defaults to None.

		Returns:
			Tuple[Tensor, Tensor]: The graph embedding and node embedding tensors.

		"""

		assert torch.isfinite(x).all(), "Input node features contain NaNs or Infs"
		assert torch.isfinite(edge_attr).all(), "Input edge attributes contain NaNs or Infs"

		# print(x, x.shape)
		x = self.atom_encoder(x.to(torch.int))
		if self.convolution == GPSConv:
			# self, edge_index, batch, num_nodes, edge_weight=None
			pe_x = self.pe_transform.compute_positional_encoding(x, edge_index, edge_weight=edge_weight, batch=batch)
			pe_x = self.pe_norm(pe_x)
			pe_x = self.pe_lin(pe_x)
			assert torch.isfinite(pe_x).all(), "Positional encoding pe_x contains NaNs or Infs"
			assert torch.sum(torch.isnan(x)) == 0, f"X contain NaNs: {pe_x}, num nans: {torch.sum(torch.isnan(pe_x))}"
			assert torch.sum(torch.isnan(x)) == 0, f"X contain NaNs: {x}, num nans: {torch.sum(torch.isnan(x))}"
			x = torch.cat([x, pe_x], dim=1)

		edge_attr = self.bond_encoder(edge_attr.to(torch.int))
		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):
			
			# for ix in range(x.shape[0]):
				# assert torch.sum(torch.isnan(x[ix, :])) == 0 , f"X contain NaNs: {x[ix, :]}, num nans: {torch.sum(torch.isnan(x[ix, :]))}, layer {i}, Item {ix}"
			if edge_weight is None:
				edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)

			if self.convolution == GINEConv:
				x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			elif self.convolution == GCNConv:
				x = self.convs[i](x, edge_index, edge_weight)
			elif self.convolution == GATv2Conv:
				x = self.convs[i](x, edge_index)
			elif self.convolution == GPSConv:
				x = self.convs[i](x, edge_index, batch, edge_attr = edge_attr, edge_weight = edge_weight)

				# assert torch.sum(torch.isnan(x)) == 0, f"X contain NaNs: {x}, num nans: {torch.sum(torch.isnan(x))}, layer {i}, {torch.min(x)}, {torch.max(x)}"

			
			x = self.bns[i](x)

			if i == self.num_gc_layers - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			
			# assert torch.sum(torch.isnan(x)) == 0, f"X contain NaNs: {x}, num nans: {torch.sum(torch.isnan(x))}, layer {i}"
			xs.append(x)
		
		# compute graph embedding using pooling
		if self.pooling_type == "standard":
			xpool = global_add_pool(x, batch)
			return xpool, x

		elif self.pooling_type == "layerwise":
			xpool = [global_add_pool(x, batch) for x in xs]
			xpool = torch.cat(xpool, 1)
			if self.is_infograph:
				return xpool, torch.cat(xs, 1)
			else:
				return xpool, x
		else:
			raise NotImplementedError

	def get_embeddings(self, loader, device, is_rand_label=False, every=1, node_features=False):
		"""
		Computes embeddings for a given data loader.

		Args:
			loader (DataLoader): The data loader.
			device (torch.device): The device to perform computations on.
			is_rand_label (bool, optional): Whether to use random labels. Defaults to False.
			every (int, optional): The interval at which to compute embeddings. Defaults to 1.
			node_features (bool, optional): Whether to use node features. Defaults to False.

		Returns:
			Tuple[np.ndarray, np.ndarray]: The computed embeddings and labels.

		"""
		ret = []
		y = []
		with torch.no_grad():
			for i, data in enumerate(loader):
				if i % every != 0:
					continue

				if isinstance(data, list):
					data = data[0].to(device)

				data = data.to(device)
				batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr

				# Hard coding for now - should find a smarter way of doing this during evaluation
				x = x[:, 0].reshape(-1, 1)
				edge_attr = edge_attr[:, 0].reshape(-1, 1)

				if not node_features:
					x = torch.ones((x.shape[0], 1)).to(device)
					edge_attr = torch.ones((edge_attr.shape[0], 1)).to(device)

				edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

				if x is None:
					x = torch.ones((batch.shape[0], 1)).to(device)

				x, _ = self.forward(batch, x, edge_index, edge_attr, edge_weight)

				ret.append(x.cpu().numpy())

				try:
					if is_rand_label:
						y.append(data.rand_label.cpu().numpy())
					else:
						y.append(data.y.cpu().numpy())
				except AttributeError:
					y.append(torch.ones(x.shape[0]).to(torch.float))
		ret = np.concatenate(ret, 0)
		y = np.concatenate(y, 0)
		return ret, y




class NodeEncoder(torch.nn.Module):
	"""
	NodeEncoder is a module that performs node encoding in a graph neural network.

	Args:
		emb_dim (int): The dimensionality of the node embeddings.
		num_gc_layers (int): The number of graph convolutional layers.
		drop_ratio (float): The dropout ratio.
		pooling_type (str): The type of pooling to use for graph embedding.
		is_infograph (bool): Whether to use Infograph pooling.
		convolution (torch.nn.Module): The graph convolutional layer to use.

	Attributes:
		pooling_type (str): The type of pooling used for graph embedding.
		emb_dim (int): The dimensionality of the node embeddings.
		num_gc_layers (int): The number of graph convolutional layers.
		drop_ratio (float): The dropout ratio.
		is_infograph (bool): Whether to use Infograph pooling.
		out_node_dim (int): The output dimensionality of the node embeddings.
		out_graph_dim (int): The output dimensionality of the graph embeddings.
		convs (torch.nn.ModuleList): The list of graph convolutional layers.
		bns (torch.nn.ModuleList): The list of batch normalization layers.
		atom_encoder (AtomEncoder): The atom encoder module.
		bond_encoder (BondEncoder): The bond encoder module.

	Methods:
		init_emb(): Initializes the node embeddings.
		forward(batch, x, edge_index, edge_attr, edge_weight=None): Performs forward pass through the module.
		get_embeddings(loader, device, is_rand_label=False, every=1, node_features=False): Computes node embeddings.

	"""

	def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0,
				 pooling_type="standard", is_infograph=False, convolution=GINEConv):
		super(NodeEncoder, self).__init__()

		self.pooling_type = pooling_type
		self.emb_dim = emb_dim
		self.num_gc_layers = num_gc_layers
		self.drop_ratio = drop_ratio
		self.is_infograph = is_infograph

		self.out_node_dim = self.emb_dim
		if self.pooling_type == "standard":
			self.out_graph_dim = self.emb_dim
		elif self.pooling_type == "layerwise":
			self.out_graph_dim = self.emb_dim * self.num_gc_layers
		else:
			raise NotImplementedError

		self.convs = torch.nn.ModuleList()
		self.bns = torch.nn.ModuleList()

		self.atom_encoder = AtomEncoder(emb_dim)
		self.bond_encoder = BondEncoder(emb_dim)

		if convolution != GATv2Conv:
			for i in range(num_gc_layers):
				nn = Sequential(Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), ReLU(), Linear(2*emb_dim, emb_dim))
				conv = convolution(nn)
				bn = torch.nn.BatchNorm1d(emb_dim)
				self.convs.append(conv)
				self.bns.append(bn)
		else:
			for i in range(num_gc_layers):
				conv = convolution(in_channels=self.emb_dim,
								   out_channels=self.out_node_dim,
								   heads=1)
				self.convs.append(conv)
				bn = torch.nn.BatchNorm1d(emb_dim)
				self.bns.append(bn)

		self.init_emb()

	def init_emb(self):
		"""
		Initializes the node embeddings.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
		"""
		Performs forward pass through the module.

		Args:
			batch (torch.Tensor): The batch tensor.
			x (torch.Tensor): The node feature tensor.
			edge_index (torch.Tensor): The edge index tensor.
			edge_attr (torch.Tensor): The edge attribute tensor.
			edge_weight (torch.Tensor, optional): The edge weight tensor.

		Returns:
			torch.Tensor: The graph embedding tensor.
			torch.Tensor: The node embedding tensor.
		"""
		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))

		xs = []
		for i in range(self.num_gc_layers):
			x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			x = self.bns[i](x)
			if i == self.num_gc_layers - 1:
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		if self.pooling_type == "standard":
			xpool = global_add_pool(x, batch)
			return xpool, x
		elif self.pooling_type == "layerwise":
			xpool = [global_add_pool(x, batch) for x in xs]
			xpool = torch.cat(xpool, 1)
			if self.is_infograph:
				return xpool, torch.cat(xs, 1)
			else:
				return xpool, x
		else:
			raise NotImplementedError

	def get_embeddings(self, loader, device, is_rand_label=False, every=1, node_features=False):
		"""
		Computes node embeddings.

		Args:
			loader (torch.utils.data.DataLoader): The data loader.
			device (torch.device): The device to use for computation.
			is_rand_label (bool, optional): Whether to use random labels.
			every (int, optional): The interval for computing embeddings.
			node_features (bool, optional): Whether to use node features.

		Returns:
			numpy.ndarray: The computed node embeddings.
			numpy.ndarray: The corresponding labels.
		"""
		ret = []
		y = []
		with torch.no_grad():
			for i, data in enumerate(loader):
				if i % every != 0:
					continue

				if isinstance(data, list):
					data = data[0].to(device)

				data = data.to(device)
				batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr

				if not node_features:
					x = torch.ones_like(x)

				edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

				if x is None:
					x = torch.ones((batch.shape[0], 1)).to(device)
				xpool, x = self.forward(batch.to(device), x.to(device), edge_index.to(device), edge_attr.to(device), edge_weight.to(device))

				ret.append(x.cpu().numpy())
				try:
					if is_rand_label:
						y.append(data.rand_label.cpu().numpy())
					else:
						y.append(data.y.cpu().numpy())
				except AttributeError:
					y.append(torch.ones(x.shape[0]).to(torch.float))
		ret = np.concatenate(ret, 0)
		y = np.concatenate(y, 0)
		return ret, y
