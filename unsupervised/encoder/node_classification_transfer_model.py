import torch
from torch.nn import Sequential, Linear, ReLU, Softmax
from torch_geometric.nn import global_add_pool, GCNConv, GATv2Conv
import torch.nn.functional as F
import numpy as np
from unsupervised.convs import GINEConv, SAGEConv


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
				self.layers.append(torch.nn.Dropout(p=0.2))

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
			self.layers.append(torch.nn.Dropout(p=0.2))
			if i != n_layers:
				self.layers.append(ReLU())


		self.model = Sequential(*self.layers)

	def forward(self, x):
		return self.model(x.float())



class NodeClassificationTransferModel(torch.nn.Module):
	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features = False,
				 node_feature_dim=512, edge_feature_dim=512, input_head_layers=1, layers_for_output = 2):
		super(NodeClassificationTransferModel, self).__init__()

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
		self.layers_for_output = layers_for_output

		self.atom_encoder = GenericNodeEncoder(proj_hidden_dim, node_feature_dim, n_layers = self.input_head_layers)
		self.bond_encoder = GenericEdgeEncoder(proj_hidden_dim, edge_feature_dim, n_layers = self.input_head_layers)

		# self.input_head = torch.nn.ModuleList()
		# self.input_bns = torch.nn.ModuleList()
		# spread_layers = [min(node_feature_dim, proj_hidden_dim) + np.abs(node_feature_dim - proj_hidden_dim) * i for i in range(input_head_layers - 1)]
		# layer_sizes = [node_feature_dim] + spread_layers + [proj_hidden_dim]
		# for i in range(input_head_layers):
		# 	in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
		# 	nn = Sequential(Linear(in_size, 2 * in_size), torch.nn.BatchNorm1d(2 * in_size), ReLU(),
		# 					Linear(2 * in_size, out_size))
		# 	conv = GINEConv(nn)
		# 	bn = torch.nn.BatchNorm1d(in_size)
		# 	self.input_head.append(conv)
		# 	self.input_bns.append(bn)



		# self.output_layer = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
		# 							   Linear(proj_hidden_dim, output_dim))

		# self.output_layer = out_convolution(in_channels = proj_hidden_dim,
		# 							 out_channels = output_dim,
		# 							 edge_dim = encoder.edge_dim)
		self.softmax = Softmax()
		# self.output_layer = GCNConv(proj_hidden_dim, output_dim)
		# self.output_layer = GINEConv(in_channels = proj_hidden_dim,
		# 							out_channels = output_dim,
		# 							edge_dim = encoder.edge_dim)



		# nn = Sequential(Linear(proj_hidden_dim, 2 * proj_hidden_dim), torch.nn.BatchNorm1d(2 * proj_hidden_dim), ReLU(),
		# 				Linear(2 * proj_hidden_dim, output_dim))
		# self.output_layer = GINEConv(nn)

		self.convolution = encoder.convolution

		# if self.convolution == GINEConv:
		# 	nn = Sequential(Linear(proj_hidden_dim, 2 * proj_hidden_dim), torch.nn.BatchNorm1d(2 * proj_hidden_dim),
		# 					ReLU(),
		# 					Linear(2 * proj_hidden_dim, output_dim))
		# 	self.output_layer = GINEConv(nn)
		#
		# elif self.convolution == GCNConv:
		# 	self.output_layer = GCNConv(proj_hidden_dim, output_dim)
		#
		# elif self.convolution == GATv2Conv:
		# 	self.output_layer = GATv2Conv(proj_hidden_dim, output_dim)
		#
		# else:
		# 	raise NotImplementedError

		self.output_layer = Linear(proj_hidden_dim, output_dim)


		# bn = torch.nn.BatchNorm1d(emb_dim)


		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, x, edge_index, edge_attr, edge_weight=None):
		if not self.features:
			x = torch.ones_like(x)
		# print(x)

		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))

		# for i in range(self.input_head_layers):
		#
		# 	if edge_weight is None:
		# 		edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)
		#
		# 	x = self.input_head[i](x, edge_index, edge_attr, edge_weight)
		# 	x = self.input_bns[i](x)
		# 	if i == self.input_head_layers - 1:
		# 		# remove relu for the last layer
		# 		x = F.dropout(x, self.drop_ratio, training=self.training)
		# 	else:
		# 		x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):

			if edge_weight is None:
				edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)

			# x = self.convs[i](x, edge_index, edge_attr, edge_weight)

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

		node_emb = x
		z = self.output_layer(x)
		# z = global_add_pool(x, batch)

		# if self.convolution == GINEConv:
		# 	z = self.output_layer(x, edge_index, edge_attr, edge_weight)
		# elif self.convolution == GCNConv:
		# 	z = self.output_layer(x, edge_index, edge_weight)
		# elif self.convolution == GATv2Conv:
		# 	z = self.output_layer(x, edge_index)

		# z = self.output_layer(x, edge_index, edge_attr, edge_weight)
		# z shape -> Batch x proj_hidden_dim
		return self.softmax(z), node_emb

