import torch
from torch.nn import Sequential, Linear, ReLU


class TransferModel(torch.nn.Module):
	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features = False):
		super(TransferModel, self).__init__()

		self.encoder = encoder
		self.input_proj_dim = self.encoder.out_graph_dim
		self.features = features

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

		if x.shape[1] > 1:
			x = x[:, 0].reshape(-1,1)
		if edge_attr.shape[1] > 1:
			edge_attr = edge_attr[:,0].reshape(-1,1)

		z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

		z = self.output_layer(z)
		# z shape -> Batch x proj_hidden_dim
		return z, node_emb

