import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
import numpy as np


class GenericEdgeEncoder(torch.nn.Module):

	def __init__(self, emb_dim, feat_dim, n_layers = 1,
				 node_feature_dim=512, edge_feature_dim=512,
				 input_head_layers=3
				 ):
		super(GenericEdgeEncoder, self).__init__()
		self.layers = []
		spread_layers = [min(emb_dim, feat_dim) + np.abs(feat_dim - emb_dim) * i for i in range(n_layers - 1)]

		layer_sizes = [feat_dim] + spread_layers + [emb_dim]
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

		spread_layers = [min(emb_dim, feat_dim) + np.abs(feat_dim - emb_dim) * i for i in range(n_layers - 1)]

		layer_sizes = [feat_dim] + spread_layers + [emb_dim]
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

class FeaturedTransferModel(torch.nn.Module):
	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features = False,
				 node_feature_dim=512, edge_feature_dim=512, input_head_layers=3):
		super(FeaturedTransferModel, self).__init__()

		self.encoder = encoder
		self.num_gc_layers = encoder.num_gc_layers
		self.convs = encoder.convs
		self.bns = encoder.bns
		self.drop_ratio = encoder.drop_ratio
		self.input_proj_dim = self.encoder.out_graph_dim
		self.features = features
		self.node_feature_dim = node_feature_dim
		self.edge_feature_dim = edge_feature_dim
		self.input_head_layers = input_head_layers

		self.atom_encoder = GenericNodeEncoder(proj_hidden_dim, node_feature_dim, n_layers = self.input_head_layers)
		self.bond_encoder = GenericEdgeEncoder(proj_hidden_dim, edge_feature_dim, n_layers = self.input_head_layers)
		self.output_layer = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
									   Linear(proj_hidden_dim, output_dim))

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
		if not self.features:
			x = torch.ones_like(x)
		# print(x)
		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))
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
		# if self.pooling_type == "standard":
		node_emb = x
		z = global_add_pool(x, batch)

		# elif self.pooling_type == "layerwise":
		# 	xpool = [global_add_pool(x, batch) for x in xs]
		# 	xpool = torch.cat(xpool, 1)
		# else:
		# 	raise NotImplementedError

		# z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

		z = self.output_layer(z)
		# z shape -> Batch x proj_hidden_dim
		return z, node_emb

