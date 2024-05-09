import torch
from torch.nn import Sequential, Linear, ReLU, Softmax
from torch_geometric.nn import  GCNConv, GATv2Conv
import torch.nn.functional as F
from unsupervised.convs import GINEConv, SAGEConv
from unsupervised.encoder import GenericEdgeEncoder, GenericNodeEncoder


class NodeClassificationTransferModel(torch.nn.Module):
	"""
	NodeClassificationTransferModel is a PyTorch module for node classification transfer learning.

	Args:
		encoder (torch.nn.Module): The encoder module used for node feature encoding.
		proj_hidden_dim (int): The hidden dimension size for the projection layer. Default is 300.
		output_dim (int): The output dimension size. Default is 300.
		features (bool): Whether to use node features. Default is False.
		node_feature_dim (int): The dimension size of node features. Default is 512.
		edge_feature_dim (int): The dimension size of edge features. Default is 512.
		input_head_layers (int): The number of layers in the input head. Default is 3.
		layers_for_output (int): The number of layers to use for output. If None, all layers will be used. Default is None.
	"""

	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features=False,
				 node_feature_dim=512, edge_feature_dim=512, input_head_layers=3, layers_for_output=None):
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
		self.layers_for_output = layers_for_output if layers_for_output is not None else len(self.convs)

		self.atom_encoder = GenericNodeEncoder(proj_hidden_dim, node_feature_dim, n_layers=self.input_head_layers)
		self.bond_encoder = GenericEdgeEncoder(proj_hidden_dim, edge_feature_dim, n_layers=self.input_head_layers)

		self.softmax = Softmax()

		self.convolution = encoder.convolution

		self.output_layer = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
									   Linear(proj_hidden_dim, output_dim))

		self.init_emb()

	def init_emb(self):
		"""
		Initialize the embeddings of the linear layers.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, x, edge_index, edge_attr, edge_weight=None):
		"""
		Forward pass of the NodeClassificationTransferModel.

		Args:
			x (torch.Tensor): The input node features.
			edge_index (torch.Tensor): The edge indices.
			edge_attr (torch.Tensor): The edge attributes.
			edge_weight (torch.Tensor): The edge weights. Default is None.

		Returns:
			tuple: A tuple containing the softmax output and the node embeddings.
		"""
		if not self.features:
			x = torch.ones_like(x)[:, 0].reshape(-1, 1)

		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))

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
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		node_emb = x
		z = self.output_layer(x)

		return self.softmax(z), node_emb

