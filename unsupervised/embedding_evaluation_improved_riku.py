import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
# from umap import UMAP
from torch_geometric.data import DataLoader
import wandb


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def get_emb_y(loader, encoder, device, dtype='numpy', is_rand_label=False, every = 1):
	x, y = encoder.get_embeddings(loader, device, is_rand_label, every = every)
	if dtype == 'numpy':
		return x,y
	elif dtype == 'torch':
		return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
	else:
		raise NotImplementedError

class EmbeddingEvaluation():
	def __init__(self, base_classifier, evaluator, task_type, num_tasks, device, params_dict=None, param_search=True,is_rand_label=False):
		self.is_rand_label = is_rand_label
		self.base_classifier = base_classifier
		self.evaluator = evaluator
		self.eval_metric = evaluator.eval_metric
		self.task_type = task_type
		self.num_tasks = num_tasks
		self.device = device
		self.param_search = param_search
		self.params_dict = params_dict
		if self.eval_metric == 'rmse':
			self.gscv_scoring_name = 'neg_root_mean_squared_error'
		elif self.eval_metric == 'mae':
			self.gscv_scoring_name = 'neg_mean_absolute_error'
		elif self.eval_metric == 'rocauc':
			self.gscv_scoring_name = 'roc_auc'
		elif self.eval_metric == 'accuracy':
			self.gscv_scoring_name = 'accuracy'
		else:
			raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

		self.classifier = None
	def scorer(self, y_true, y_raw):
		input_dict = {"y_true": y_true, "y_pred": y_raw}
		score = self.evaluator.eval(input_dict)[self.eval_metric]
		return score

	def ee_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
		if self.param_search:
			params_dict = {'C': [0.001, 0.01,0.1,1,10,100,1000]}
			self.classifier = make_pipeline(StandardScaler(),
			                                GridSearchCV(self.base_classifier, params_dict, cv=5, scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
			                                )
		else:
			self.classifier = make_pipeline(StandardScaler(), self.base_classifier)


		self.classifier.fit(train_emb, np.squeeze(train_y))

		if self.eval_metric == 'accuracy':
			train_raw = self.classifier.predict(train_emb)
			val_raw = self.classifier.predict(val_emb)
			test_raw = self.classifier.predict(test_emb)
		else:
			train_raw = self.classifier.predict_proba(train_emb)[:, 1]
			val_raw = self.classifier.predict_proba(val_emb)[:, 1]
			test_raw = self.classifier.predict_proba(test_emb)[:, 1]

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

	def ee_multioutput_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):

		params_dict = {
			'multioutputclassifier__estimator__C': [1e-1, 1e0, 1e1, 1e2]}
		self.classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(
			self.base_classifier, n_jobs=-1))
		
		if np.isnan(train_y).any():
			print("Has NaNs ... ignoring them")
			train_y = np.nan_to_num(train_y)
		self.classifier.fit(train_emb, train_y)

		train_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(train_emb)])
		val_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(val_emb)])
		test_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(test_emb)])

		return train_raw, val_raw, test_raw

	def ee_regression(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
		if self.param_search:
			params_dict = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
# 			params_dict = {'alpha': [500, 50, 5, 0.5, 0.05, 0.005, 0.0005]}
			self.classifier = GridSearchCV(self.base_classifier, params_dict, cv=5,
			                          scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
		else:
			self.classifier = self.base_classifier

		self.classifier.fit(train_emb, np.squeeze(train_y))

		train_raw = self.classifier.predict(train_emb)
		val_raw = self.classifier.predict(val_emb)
		test_raw = self.classifier.predict(test_emb)

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

	def vis(self, train_emb, val_emb, test_emb):
		embedder = PCA(n_components=2).fit(train_emb)
		# embedder = UMAP(n_components=2, n_jobs=4).fit(train_emb)
		proj_train, proj_val, proj_test = embedder.transform(train_emb), embedder.transform(val_emb), embedder.transform(test_emb)

		fig, ax = plt.subplots(figsize=(6,6))

		ax.scatter(proj_train[:,0], proj_train[:,1], label = "train", marker = "x")
		ax.scatter(proj_val[:,0], proj_val[:,1], label = "val", marker = "+")
		ax.scatter(proj_test[:,0], proj_test[:,1], label = "test", marker = "*")

		ax.legend(shadow=True)

		plt.savefig("outputs/embedding.png")
		wandb.log({"Embedding":wandb.Image("outputs/embedding.png")})

		# plt.show()


	def embedding_evaluation(self, encoder, train_loader, valid_loader, test_loader, vis = False):
		encoder.eval()
		train_emb, train_y = get_emb_y(train_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		val_emb, val_y = get_emb_y(valid_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		test_emb, test_y = get_emb_y(test_loader, encoder, self.device, is_rand_label=self.is_rand_label)

		# if vis:
		# 	self.vis(train_emb, val_emb, test_emb)

		if 'classification' in self.task_type:

			if self.num_tasks == 1:
				train_raw, val_raw, test_raw = self.ee_binary_classification(train_emb, train_y, val_emb, val_y, test_emb,
				                                                        test_y)
			elif self.num_tasks > 1:
				train_raw, val_raw, test_raw = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb, val_y,
				                                                                    test_emb, test_y)
			else:
				raise NotImplementedError
		else:
			if self.num_tasks == 1:
				train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y)
			else:
				raise NotImplementedError

		train_score = self.scorer(train_y, train_raw)

		val_score = self.scorer(val_y, val_raw)

		test_score = self.scorer(test_y, test_raw)

		return train_score, val_score, test_score

	def kf_embedding_evaluation(self, encoder, dataset, folds=10, batch_size=128):
		kf_train = []
		kf_val = []
		kf_test = []

		kf = KFold(n_splits=folds, shuffle=True, random_state=None)
		for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset)):
			test_dataset = [dataset[int(i)] for i in list(test_index)]
			train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=None)

			train_dataset = [dataset[int(i)] for i in list(train_index)]
			val_dataset = [dataset[int(i)] for i in list(val_index)]

			train_loader = DataLoader(train_dataset, batch_size=batch_size)
			valid_loader = DataLoader(val_dataset, batch_size=batch_size)
			test_loader = DataLoader(test_dataset, batch_size=batch_size)

			train_score, val_score, test_score = self.embedding_evaluation(encoder, train_loader, valid_loader, test_loader)

			kf_train.append(train_score)
			kf_val.append(val_score)
			kf_test.append(test_score)

		return np.array(kf_train).mean(), np.array(kf_val).mean(), np.array(kf_test).mean()


class GeneralEmbeddingEvaluation():
	def __init__(self):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def embedding_evaluation(self, encoder, loaders, names):
		all_embeddings, separate_embeddings = self.get_embeddings(encoder, loaders)
		self.centroid_similarities(separate_embeddings, names)
		self.vis(all_embeddings, separate_embeddings, names)

	def get_embeddings(self, encoder, loaders):
		encoder.eval()
		all_embeddings = None
		separate_embeddings = []
		# colours = []
		for i, loader in enumerate(loaders):
			train_emb, train_y = get_emb_y(loader, encoder, self.device, is_rand_label=False, every=5)
			separate_embeddings.append(train_emb)
			if all_embeddings is None:
				all_embeddings = train_emb
			else:
				all_embeddings = np.concatenate((all_embeddings, train_emb))
			# colours += [i for n in range(train_emb.shape[0])]

		return all_embeddings, separate_embeddings
	def vis(self, all_embeddings, separate_embeddings, names):
		# embedder = PCA(n_components=2).fit(embeddings)
		embedder = UMAP(n_components=2, n_jobs=4).fit(all_embeddings[::10, :])
		# proj_train, proj_val, proj_test = embedder.transform(train_emb), embedder.transform(
		# 	val_emb), embedder.transform(test_emb)

		fig, ax = plt.subplots(figsize=(9, 9))

		for i, emb in enumerate(separate_embeddings):
			proj = embedder.transform(emb[::10, :])
			ax.scatter(proj[:, 0], proj[:, 1],
					   alpha= 1 - proj.shape[0] / all_embeddings.shape[0], s = 5,
					   label=f"{names[i]} - {proj.shape[0]} graphs")

		ax.legend(shadow=True)

		plt.savefig("outputs/embedding.png")
		wandb.log({"Embedding": wandb.Image("outputs/embedding.png")})

	def centroid_similarities(self, embeddings, names):
		embed_dim = embeddings[0].shape[1]
		centroids = np.zeros((len(embeddings), embed_dim))

		for i, embedding in enumerate(embeddings):
			centroids[i, :] = np.mean(embedding, axis = 0)

		pairwise_similarities = cosine_similarity(centroids)
		print(pairwise_similarities)

		fig, ax = plt.subplots(figsize=(7,6))

		im = ax.imshow(pairwise_similarities, cmap = "binary", vmin = -1, vmax = 1)

		ax.set_xticks(np.arange(len(names)), labels = names)
		ax.set_yticks(np.arange(len(names)), labels = names)

		# for i1 in range(len(names)):
		# 	for i2 in range(len(names)):
		# 		text = ax.text(i2, i1, np.around(pairwise_similarities[i1, i2], decimals = 3),
		# 					   ha="center", va="center", color="w")

		annotate_heatmap(im, valfmt="{x:.3f}")

		plt.savefig("outputs/pairwise-similarity.png")
		wandb.log({"Pairwise Dataset Similarities": wandb.Image("outputs/pairwise-similarity.png")})


		pairwise_sum = 0
		for i1 in range(pairwise_similarities.shape[0]):
			for i2 in range(pairwise_similarities.shape[1]):
				if i2 <= i1:
					pass
				else:
					pairwise_sum += pairwise_similarities[i1, i2]

		mean_separation = pairwise_sum / ((pairwise_similarities.shape[0]**2)/2 - pairwise_similarities.shape[0])

		wandb.log({"Mean Cosine Dataset Separation":mean_separation})
	# plt.show()
