import numpy as np
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from unsupervised.convs import GINEConv
from torch_geometric.nn import GATv2Conv, GCNConv


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
				 convolution="gin", edge_dim=1):
		super(Encoder, self).__init__()

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
		self.edge_dim = edge_dim

		if convolution == "gin":
			print(f"Using GIN backbone for {num_gc_layers} layers")
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
		# print(x, x.shape)
		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))
		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):

			if edge_weight is None:
				edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)

			if self.convolution == GINEConv:
				x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			elif self.convolution == GCNConv:
				x = self.convs[i](x, edge_index, edge_weight)
			elif self.convolution == GATv2Conv:
				x = self.convs[i](x, edge_index)

			x = self.bns[i](x)
			if i == self.num_gc_layers - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
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
