import numpy as np
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from unsupervised.convs import GINEConv, SAGEConv
from torch_geometric.nn import GATv2Conv

class GenericEdgeEncoder(torch.nn.Module):

	def __init__(self, emb_dim, feat_dim, n_layers = 1):
		super(GenericEdgeEncoder, self).__init__()

		self.layers = []  # torch.nn.ModuleList()

		layer_sizes = [feat_dim] + [emb_dim] * (n_layers)
		for i in range(n_layers):
			lin = Linear(layer_sizes[i], layer_sizes[i + 1])

			# lin = Linear(feat_dim, emb_dim)
			torch.nn.init.xavier_uniform_(lin.weight.data)
			self.layers.append(lin)

			if i != n_layers:
				self.layers.append(ReLU())

		self.model = Sequential(*self.layers)

	# self.model = Sequential(lin)


	def forward(self, x):
		return self.model(x.float())

class GenericNodeEncoder(torch.nn.Module):

	def __init__(self, emb_dim, feat_dim, n_layers = 1):
		super(GenericNodeEncoder, self).__init__()


		self.layers = []  # torch.nn.ModuleList()

		layer_sizes = [feat_dim] + [emb_dim] * (n_layers)
		for i in range(n_layers):
			lin = Linear(layer_sizes[i], layer_sizes[i + 1])

			# lin = Linear(feat_dim, emb_dim)
			torch.nn.init.xavier_uniform_(lin.weight.data)
			self.layers.append(lin)

			if i != n_layers:
				self.layers.append(ReLU())

		self.model = Sequential(*self.layers)

	def forward(self, x):
		return self.model(x.float())


class FeatureEncoder(torch.nn.Module):
	def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0,
				 pooling_type="standard", is_infograph=False,
				 convolution = GINEConv, node_feature_dim = 512, edge_feature_dim = 512,
				  input_head_layers = 3):
		super(FeatureEncoder, self).__init__()

		self.pooling_type = pooling_type
		self.emb_dim = emb_dim
		self.num_gc_layers = num_gc_layers
		self.drop_ratio = drop_ratio
		self.is_infograph = is_infograph
		self.node_feature_dim = node_feature_dim
		self.edge_feature_dim = edge_feature_dim
		self.input_head_layers = input_head_layers

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
		# self.bond_encoder = BondEncoder(emb_dim)
		# self.atom_encoder = NodeEncoder(emb_dim, feature_dim)
		# self.bond_encoder = EdgeEncoder(emb_dim, feature_dim)
		self.atom_encoder = GenericNodeEncoder(emb_dim, node_feature_dim, n_layers = self.input_head_layers)
		self.bond_encoder = GenericEdgeEncoder(emb_dim, edge_feature_dim, n_layers = self.input_head_layers)
		self.convolution = convolution

		# if convolution != GATv2Conv:

		for i in range(num_gc_layers):
			nn = Sequential(Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), ReLU(), Linear(2*emb_dim, emb_dim))
			conv = convolution(nn)
			bn = torch.nn.BatchNorm1d(emb_dim)
			self.convs.append(conv)
			self.bns.append(bn)
		# else:
		#
		# 	for i in range(num_gc_layers):
		# 		conv = convolution(in_channels=emb_dim,
		# 						   out_channels=emb_dim,
		# 						   edge_dim=edge_dim)
		# 		self.convs.append(conv)
		# 		bn = torch.nn.BatchNorm1d(emb_dim)
		# 		self.bns.append(bn)

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
		x = F.pad(x, (0, self.feature_dim - x.shape[1], 0, 0), "constant", 1)
		x = self.atom_encoder(x)

		edge_attr = F.pad(edge_attr, (0, self.feature_dim - edge_attr.shape[1], 0, 0), "constant", 1)
		edge_attr = self.bond_encoder(edge_attr)
		edge_start_shape = edge_attr.shape[1]

		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):

			if edge_weight is None:
				edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)
			x = self.convs[i](x, edge_index, edge_attr, edge_weight)
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

	def get_embeddings(self, loader, device, is_rand_label=False, every = 1,  node_features = False):
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
#
# class FeatureEncoder(torch.nn.Module):
# 	def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0,
# 				 pooling_type="standard", is_infograph=False,
# 				 convolution = GINEConv, edge_dim = 1, feature_dim = 512):
# 		super(FeatureEncoder, self).__init__()
#
# 		self.pooling_type = pooling_type
# 		self.emb_dim = emb_dim
# 		self.num_gc_layers = num_gc_layers
# 		self.drop_ratio = drop_ratio
# 		self.is_infograph = is_infograph
# 		self.feature_dim = feature_dim
#
# 		self.out_node_dim = self.emb_dim
# 		if self.pooling_type == "standard":
# 			self.out_graph_dim = self.emb_dim
# 		elif self.pooling_type == "layerwise":
# 			self.out_graph_dim = self.emb_dim * self.num_gc_layers
# 		else:
# 			raise NotImplementedError
#
# 		self.convs = torch.nn.ModuleList()
# 		self.bns = torch.nn.ModuleList()
#
# 		# self.atom_encoder = AtomEncoder(emb_dim)
# 		# self.bond_encoder = BondEncoder(emb_dim)
# 		# self.atom_encoder = NodeEncoder(emb_dim, feature_dim)
# 		# self.bond_encoder = EdgeEncoder(emb_dim, feature_dim)
# 		self.atom_encoder = GenericNodeEncoder(emb_dim, feature_dim)
# 		self.bond_encoder = GenericEdgeEncoder(emb_dim, feature_dim)
# 		self.convolution = convolution
#
# 		if convolution != GATv2Conv:
#
# 			for i in range(num_gc_layers):
# 				nn = Sequential(Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), ReLU(), Linear(2*emb_dim, emb_dim))
# 				conv = convolution(nn)
# 				bn = torch.nn.BatchNorm1d(emb_dim)
# 				self.convs.append(conv)
# 				self.bns.append(bn)
# 		else:
#
# 			for i in range(num_gc_layers):
# 				conv = convolution(in_channels=emb_dim,
# 								   out_channels=emb_dim,
# 								   edge_dim=edge_dim)
# 				self.convs.append(conv)
# 				bn = torch.nn.BatchNorm1d(emb_dim)
# 				self.bns.append(bn)
#
# 		self.init_emb()
#
# 	def init_emb(self):
# 		for m in self.layers():
# 			if isinstance(m, Linear):
# 				torch.nn.init.xavier_uniform_(m.weight.data)
# 				if m.bias is not None:
# 					m.bias.data.fill_(0.0)
#
# 	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
# 		x = F.pad(x, (0, self.feature_dim - x.shape[1], 0, 0), "constant", 1)
# 		x = self.atom_encoder(x)
#
# 		edge_attr = F.pad(edge_attr, (0, self.feature_dim - edge_attr.shape[1], 0, 0), "constant", 1)
# 		edge_attr = self.bond_encoder(edge_attr)
# 		edge_start_shape = edge_attr.shape[1]
#
# 		# compute node embeddings using GNN
# 		xs = []
# 		for i in range(self.num_gc_layers):
#
# 			if edge_weight is None:
# 				edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)
# 			x = self.convs[i](x, edge_index, edge_attr, edge_weight)
# 			x = self.bns[i](x)
# 			if i == self.num_gc_layers - 1:
# 				# remove relu for the last layer
# 				x = F.dropout(x, self.drop_ratio, training=self.training)
# 			else:
# 				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
# 			xs.append(x)
#
# 		# compute graph embedding using pooling
# 		if self.pooling_type == "standard":
# 			xpool = global_add_pool(x, batch)
# 			return xpool, x
#
# 		elif self.pooling_type == "layerwise":
# 			xpool = [global_add_pool(x, batch) for x in xs]
# 			xpool = torch.cat(xpool, 1)
# 			if self.is_infograph:
# 				return xpool, torch.cat(xs, 1)
# 			else:
# 				return xpool, x
# 		else:
# 			raise NotImplementedError
#
# 	def get_embeddings(self, loader, device, is_rand_label=False, every = 1,  node_features = False):
# 		ret = []
# 		y = []
# 		with torch.no_grad():
# 			for i, data in enumerate(loader):
# 				if i % every != 0:
# 					continue
#
# 				if isinstance(data, list):
# 					data = data[0].to(device)
#
# 				data = data.to(device)
# 				batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr
#
# 				if not node_features:
# 					x = torch.ones_like(x)
#
# 				edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
#
# 				if x is None:
# 					x = torch.ones((batch.shape[0], 1)).to(device)
# 				x, _ = self.forward(batch, x, edge_index, edge_attr, edge_weight)
#
# 				ret.append(x.cpu().numpy())
#
# 				try:
# 					if is_rand_label:
# 						y.append(data.rand_label.cpu().numpy())
# 					else:
# 						y.append(data.y.cpu().numpy())
# 				except AttributeError:
# 					y.append(torch.ones(x.shape[0]).to(torch.float))
# 		ret = np.concatenate(ret, 0)
# 		y = np.concatenate(y, 0)
# 		return ret, y