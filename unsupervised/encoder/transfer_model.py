import torch
from torch.nn import Sequential, Linear, ReLU


class TransferModel(torch.nn.Module):
	"""
	TransferModel class represents a transfer learning model.

	Args:
		encoder (torch.nn.Module): The encoder module used for feature extraction.
		proj_hidden_dim (int, optional): The hidden dimension of the projection layer. Defaults to 300.
		output_dim (int, optional): The output dimension of the model. Defaults to 300.
		features (bool, optional): Flag indicating whether input features are used. Defaults to False.
	"""

	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features=False):
		super(TransferModel, self).__init__()

		self.encoder = encoder
		self.input_proj_dim = self.encoder.out_graph_dim
		self.features = features

		self.output_layer = Sequential(
			Linear(self.input_proj_dim, proj_hidden_dim),
			ReLU(inplace=True),
			Linear(proj_hidden_dim, output_dim)
		)

		self.init_emb()

	def init_emb(self):
		"""
		Initialize the embedding layers with Xavier uniform initialization.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
		"""
		Forward pass of the TransferModel.

		Args:
			batch: The batch tensor.
			x: The input tensor.
			edge_index: The edge index tensor.
			edge_attr: The edge attribute tensor.
			edge_weight: The edge weight tensor (optional).

		Returns:
			z: The output tensor.
			node_emb: The node embedding tensor.
		"""
		if not self.features:
			x = torch.ones_like(x)

		if x.shape[1] > 1:
			x = x[:, 0].reshape(-1, 1)
		if edge_attr.shape[1] > 1:
			edge_attr = edge_attr[:, 0].reshape(-1, 1)

		z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

		z = self.output_layer(z)
		# z shape -> Batch x proj_hidden_dim
		return z, node_emb

