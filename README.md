# ToP

Repository for the paper **Towards Generalised Pre-Training of Graph Models**, presenting the **T**opology **O**nly **P**re-training (**ToP**).

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

### Reproducing our Results

Experimental details can be found in the ToP paper.
Default parameters in-code match those used in our experiments, and key-word arguments (see below) can be used to change experimental setup if necessary.

Training the ToP-All model requires this command:

``
python train.py --epochs 100
``

Having pre-trained a model, the resulting checkpoints and config files can be found under `wandb/`.
To perform transfer, first move `checkpoint.pt` and `config.yaml` to `outputs/`, and rename `config.yaml` to `checkpoint.yaml`.

The transfer command is then (see below for further arguments):

``
python transfer.py --checkpoint checkpoint.pt
``

The same for transfer on molecular benchmarks with features:

``
python features_transfer.py --checkpoint checkpoint.pt -f
``

and the same syntax for node classification and edge classification.

## Code Usage

The environmental setup is fairly minimal.
The broad requirements are standard for graph deep-learning:

 - Python=3.11
 - pytorch-geometric
 - pytorch (shocking I know)
 - numpy, scipy, matplotlib, pandas, etc.
 - ogb (open graph benchmark)


#### Environment

We include an environment for our working computer:

- Linux
- Ubuntu 22.02
- Nvidia driver version 555.42.06
- Cuda version 12.5

Follow these steps to create a Conda env with the required packages:

```
conda create -n pyg-top python=3.11
>>> ...are you sure you want to ... [y/n]
conda activate pyg-top
pip install -r requirements.txt
```

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

Models can be downloaded through https://drive.google.com/file/d/1Ionm2UsVLNpPmQdOiBBzGq_YjELllGeC/view?usp=sharing.
This zip archive should be placed in the root directory, then unpacked with `unzip_models.sh`.

We include several pre-trained models:

| Checkpoint name | Backbone | Contrastive Method | Pre-Train Data |
|-|-|-|-|
| untrained | Specified with ``--backbone`` | None | None |
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
 - `--checkpoint` the model checkpoint to fine-tune. See above for included models.

#### Linear Transfer

`linear_transfer.py` functions in essentially the same way, except that it uses a linear model in place of fine-tuning a FoToM model.

## Transfer with Features

`features_transfer.py --args` can be used to fine-tune FoToM models, optionally with complete node and edge features.
Currently this is limited to our chemical benchmark dataserts.
Pre-trained models can be downloaded with `download_models.sh`, see above.

 - `-f --node-features`       whether to use node (and edge) features during evaluation
 - `--num` the maximum number of samples to include in the test sets (doubled for validation)
 - `--checkpoint` the model checkpoint to fine-tune. See above for included models.
 - `--backbone` Model backbone to use (gin, gcn, gat)

 ### Node and Edge Transfer

`node_classification_transfer.py` and `edge_prediction_transfer.py` perform transfer runs for node classification and edge prediction.
Arguments are the same as for transfer with features.

#### Noise/Noise Analysis

Our code for testing the information in features vs structure can be found in `general-gcl/noisenoise`.

The code is executed in `features-vs-structure-lines.py`, with model names hard-coded, so simply run:

```
    python features-vs-structure-lines.py
```

The resulting figures are then placed under `outputs/noise-noise/.`

#### Further code

Dataset code can be found under `/datasets/`.
Loaders can be found here under `loaders.py`, and other datasets have their own respective processing files.
`from_ogb_dataset.py` converts an OGB dataset into a standard pytorch-geometric dataset.




