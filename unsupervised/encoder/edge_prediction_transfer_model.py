import torch
from torch.nn import Sequential, Linear, ReLU, Softmax, Sigmoid
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
import numpy as np
from unsupervised.convs import GINEConv, SAGEConv
from torch_geometric.nn import GCNConv, GATv2Conv


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



class EdgePredictionTransferModel(torch.nn.Module):
	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features = False,
				 node_feature_dim=512, edge_feature_dim=512, input_head_layers=3, layers_for_output = None):
		super(EdgePredictionTransferModel, self).__init__()

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
		self.layers_for_output = layers_for_output if layers_for_output is not None else len(self.convs)

		self.atom_encoder = GenericNodeEncoder(proj_hidden_dim, node_feature_dim, n_layers = self.input_head_layers)
		self.bond_encoder = GenericEdgeEncoder(proj_hidden_dim, edge_feature_dim, n_layers = self.input_head_layers)

		self.softmax = Softmax()

		self.convolution = encoder.convolution

		# nn = Sequential(Linear(proj_hidden_dim, 2 * proj_hidden_dim), torch.nn.BatchNorm1d(2 * proj_hidden_dim), ReLU(),
		# 				Linear(2 * proj_hidden_dim, output_dim))
		# self.output_layer = GINEConv(nn)

		self.output_layer = Linear(2 * proj_hidden_dim, 1)
		self.sigmoid_out = Sigmoid()


		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, data):
		device = "cuda" if torch.cuda.is_available() else "cpu"
		if data.edge_attr is None:
			data.edge_attr = torch.ones(data.edge_index.shape[1]).reshape(-1, 1).to(device)

		node_embedding = self.embedding_forward(data.x, data.edge_index, data.edge_attr)

		interested_edges = data.edge_label_index
		x1 = node_embedding[interested_edges[0]]
		x2 = node_embedding[interested_edges[1]]



		to_mlp = torch.cat([x1, x2], dim = 1)

		preds = self.output_layer(to_mlp)
		return self.sigmoid_out(preds)



	def embedding_forward(self, x, edge_index, edge_attr, edge_weight=None, mask = None):
		if not self.features:
			x = torch.ones_like(x)[:,0].reshape(-1,1)

		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))


		# compute node embeddings using GNN
		xs = []
		for i in range(self.layers_for_output):

			if edge_weight is None:
				edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)

			if self.convolution == GINEConv:
				x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			elif self.convolution == GCNConv:
				x = self.convs[i](x, edge_index, edge_weight)
			elif self.convolution == GATv2Conv:
				x = self.convs[i](x, edge_index)

			# x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			x = self.bns[i](x)
			if i == self.layers_for_output - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		node_emb = x

		return node_emb





		# z = global_add_pool(x, batch)




		# z = self.output_layer(x, edge_index, edge_attr, edge_weight)
		# z shape -> Batch x proj_hidden_dim
		# return self.softmax(z), node_emb

