# FoToM

Repository for the paper **Towards Generalised Pre-Training of Graph Models**, presenting the **T**opology **O**nly **P**re-training (**ToP**).

### Abstract:

Models**, presenting the **T**opology **O**nly **P**re-training (**ToP**).

### Abstract:

The principal benefit of unsupervised representation learning is that a pre-trained model can be fine-tuned where data or labels are scarce.
Existing approaches for graph representation learning are domain specific, maintaining consistent node and edge attributes across the pre-training and target datasets.
This has precluded transfer to multiple domains.
<!-- A model capable of positive transfer on arbitrary tasks and domains would represent the first foundation graph model. -->

In this work we present **T**opology **O**nly **P**re-Training (**ToP**), a graph pre-training method based on node and edge feature exclusion.
We use **ToP** with graph contrastive learning to pre-train models over multiple graph domains.
We demonstrate positive transfer on evaluation datasets from multiple domains, including domains not present in pre-training data.
On 75% of experiments, **ToP** performs significantly better than a supervised baseline, with an 8 to 40% reduction in error at 95% confidence. 
The remaining cases show equal performance to the baseline. 
Contrary to other research, pre-training with **ToP** on a dataset with the target domain excluded leads us to better performance than pre-training on a dataset from only the target domain.
The multi-domain model at worst, matches, and on 75% of tasks, significantly outperforms single-domain (p <= 0.01).

## Code Usage

The environmental setup is fairly minimal.
The broad requirements are standard for graph deep-learning:

 - pytorch-geometric
 - pytorch (shocking I know)
 - numpy, scipy, matplotlib, pandas, etc.
 - ogb (open graph benchmark)

Apologies for this being a rather brief description. 
Pytorch-geometric is still a little volatile, and often doesn't play well with others depending on your hardware, so we leave it up to the user.

### Training
`train.py --args` can be used to train new FoToM models.

Most arguments have descriptions accessible via `--help`.
Arguments specific to FoToM are as follows:

 - `-f --node-features`       whether to use node labels during evaluation
 - `-c --no-molecules`        whether to exclude molecules from training data
 - `-s --no-socials`          whether to exclude non-molecules from training data
 - `-rn --randon-node-views`  whether to switch from adversarial augmentations to random node dropping
 - `-re --randon-edge-views`  whether to switch from adversarial augmentations to random edge dropping
 - `--dropped`                if using random edge/node dropping, what proportion to drop
 - `--backbone`               which GNN backbone to use (default GIN, optionally GCN, GAT)
 - `--exclude`                option to exclude individual non-molecular datasets, primarily for ablation studies ("facebook_large", "twitch_egos", "cora", "roads", "fruit_fly")

## Models

Models can be downloaded through `download_models.sh`.
We include several pre-trained models:

|-|-|-|-|
| Checkpoint name | Backbone | Contrastive Method | Pre-Train Data |
|-|-|-|-|
| all-100.pt | GIN | AD-GCL | All |
| social-100.pt | GIN | AD-GCL | Non-Molecules |
| chem-100.pt | GIN | AD-GCL | Molecules |
|-|-|-|-|
| gat-all.pt | GAT | AD-GCL | All |
| gat-social.pt | GAT | AD-GCL | Non-Molecules |
| gat-chem.pt | GAT | AD-GCL | Molecules |
|-|-|-|-|
| gcn-all.pt | GCN | AD-GCL | All |
| gcn-social.pt | GCN | AD-GCL | Non-Molecules |
| gcn-chem.pt | GCN | AD-GCL | Molecules |
|-|-|-|-|
| edge-views-all.pt | GIN | GraphCL (edges) | All |
| edge-views-social.pt | GIN | GraphCL (edges) | Non-Molecules |
| edge-views-chem.pt | GIN | GraphCL (edges) | Molecules |
|-|-|-|-|
| node-views-all.pt | GIN | GraphCL (nodes) | All |
| node-views-social.pt | GIN | GraphCL (nodes) | Non-Molecules |
| node-views-chem.pt | GIN | GraphCL (nodes) | Molecules |
|-|-|-|-|
| random.pt | GIN | AD-GCL| Random Graphs|
|-|-|-|-|
| Available on request | GIN | AD-GCL | Non-Molecules, with component dataset excluded|


### Transfer

`transfer.py --args` can be used to fine-tune FoToM models, optinally with node labels.
Please note that it does this on all of our validation datasets.
Pre-trained models can be downloaded with `download_models.sh`.

 - `-f --node-features`       whether to use node labels during evaluation
 - `--num` the maximum number of samples to include in the test sets (doubled for validation)
 - `--checkpoint` the model checkpoint to fine-tune. We include the following:
 - - `all-100.pt` a model trained for 100 epochs on all the training data
 - - `chem-100.pt` a model trained for 100 epochs on only molecules
 - - `social-100.pt` a model trained for 100 epochs on on non-molecules
 - - `untrained` as a checkpoint will not load a checkpoint, instead training a model from scratch

#### Linear Transfer

`linear_transfer.py` functions in essentially the same way, except that it uses a linear model in place of fine-tuning a FoToM model.

## Transfer with Features

`features_transfer.py --args` can be used to fine-tune FoToM models, optionally with complete node and edge features.
Currently this is limited to our chemical benchmark dataserts.
Pre-trained models can be downloaded with `download_models.sh`, see above.

 - `-f --node-features`       whether to use node (and edge) features during evaluation
 - `--num` the maximum number of samples to include in the test sets (doubled for validation)
 - `--checkpoint` the model checkpoint to fine-tune. We include the following:
 - - `all-100.pt` a model trained for 100 epochs on all the training data
 - - `chem-100.pt` a model trained for 100 epochs on only molecules
 - - `social-100.pt` a model trained for 100 epochs on on non-molecules
 - - `untrained` as a checkpoint will not load a checkpoint, instead training a model from scratch
 - `--backbone` Model backbone to use (gin, gcn, gat)

 ### Node and Edge Transfer

`node_classification_transfer.py` and `edge_prediction_transfer.py` perform transfer runs for node classification and edge prediction.
Arguments are the same as for transfer with features.

#### Further code

Dataset code can be found under `/datasets/`.
Loaders can be found here under `loaders.py`, and other datasets have their own respective processing files.
`from_ogb_dataset.py` converts an OGB dataset into a standard pytorch-geometric dataset.





