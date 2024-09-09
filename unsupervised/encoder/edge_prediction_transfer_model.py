import torch
from torch.nn import Sequential, Linear, ReLU, Softmax, Sigmoid
import torch.nn.functional as F
import numpy as np
from unsupervised.convs import GINEConv
from torch_geometric.nn import GCNConv, GATv2Conv
from unsupervised.encoder import GenericEdgeEncoder, GenericNodeEncoder


class EdgePredictionTransferModel(torch.nn.Module):
	"""
	EdgePredictionTransferModel is a PyTorch module for edge prediction transfer learning.

	Args:
		encoder (torch.nn.Module): The encoder module used for node and edge feature encoding.
		proj_hidden_dim (int): The hidden dimension size for the projection layer. Default is 300.
		output_dim (int): The output dimension size. Default is 300.
		features (bool): Whether to use node and edge features. Default is False.
		node_feature_dim (int): The dimension size of node features. Default is 512.
		edge_feature_dim (int): The dimension size of edge features. Default is 512.
		input_head_layers (int): The number of layers in the input head. Default is 3.
		layers_for_output (int): The number of layers used for output. If None, it is set to the number of convolutional layers.

	Attributes:
		encoder (torch.nn.Module): The encoder module used for node and edge feature encoding.
		num_gc_layers (int): The number of graph convolutional layers in the encoder.
		convs (list): List of graph convolutional layers in the encoder.
		bns (list): List of batch normalization layers in the encoder.
		drop_ratio (float): The dropout ratio used in the encoder.
		input_proj_dim (int): The dimension size of the encoder's output graph.
		atom_encoder (GenericNodeEncoder): The node encoder module.
		bond_encoder (GenericEdgeEncoder): The edge encoder module.
		softmax (Softmax): The softmax activation function.
		convolution (type): The type of graph convolution used in the encoder.
		output_layer (torch.nn.Module): The output layer of the model.
		sigmoid_out (Sigmoid): The sigmoid activation function.

	Methods:
		init_emb(): Initializes the embeddings of the model.
		forward(data): Performs a forward pass of the model.
		embedding_forward(x, edge_index, edge_attr, edge_weight=None, mask=None): Computes node embeddings using the encoder.

	"""

	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features=False,
				 node_feature_dim=512, edge_feature_dim=512, input_head_layers=3, layers_for_output=None):
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

		self.atom_encoder = GenericNodeEncoder(proj_hidden_dim, node_feature_dim, n_layers=self.input_head_layers)
		self.bond_encoder = GenericEdgeEncoder(proj_hidden_dim, edge_feature_dim, n_layers=self.input_head_layers)

		self.softmax = Softmax()

		self.convolution = encoder.convolution

		self.output_layer = Sequential(Linear(2 * self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
									   Linear(proj_hidden_dim, 1))

		self.sigmoid_out = Sigmoid()

		self.init_emb()

	def init_emb(self):
		"""
		Initializes the embeddings of the model.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, data):
		"""
		Performs a forward pass of the model.

		Args:
			data (torch_geometric.data.Data): The input data.

		Returns:
			torch.Tensor: The predicted edge probabilities.
		"""
		device = "cuda" if torch.cuda.is_available() else "cpu"
		if data.edge_attr is None:
			data.edge_attr = torch.ones(data.edge_index.shape[1]).reshape(-1, 1).to(device)

		node_embedding = self.embedding_forward(data.x, data.edge_index, data.edge_attr)

		interested_edges = data.edge_label_index
		x1 = node_embedding[interested_edges[0]]
		x2 = node_embedding[interested_edges[1]]

		to_mlp = torch.cat([x1, x2], dim=1)

		preds = self.output_layer(to_mlp)
		return self.sigmoid_out(preds)

	def embedding_forward(self, x, edge_index, edge_attr, edge_weight=None, mask=None):
		"""
		Computes node embeddings using the encoder.

		Args:
			x (torch.Tensor): The node features.
			edge_index (torch.Tensor): The edge indices.
			edge_attr (torch.Tensor): The edge features.
			edge_weight (torch.Tensor, optional): The edge weights. Defaults to None.
			mask (torch.Tensor, optional): The mask tensor. Defaults to None.

		Returns:
			torch.Tensor: The computed node embeddings.
		"""
		if not self.features:
			x = torch.ones_like(x)[:, 0].reshape(-1, 1)

		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))

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

			x = self.bns[i](x)
			if i == self.layers_for_output - 1:
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		node_emb = x

		return node_emb

