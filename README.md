# FoToM

Repository for the paper **A Foundation Graph Model** in submission at ICML as of 01/02/24.

### Abstract:

The principal benefit of unsupervised graph representation learning is that a pre-trained model can be fine-tuned where data or labels are scarce.
Current methods maintain consistent node and edge attributes across the pretraining and target datasets. 
This precludes transfer to other domains.
A model capable of positive transfer on an arbitrary task and domain represents the first foundation graph model.

![Domain transfer diagram](https://github.com/neutralpronoun/general-gcl/blob/main/figures/domains.drawio.png)

In this work we use adversarial contrastive learning to present FOTOM, the first foundation graph model. 
By excluding node and edge attributes we are able to pre-train over varied graph domains.
We demonstrate positive transfer on evaluation datasets from multiple domains compared to a non-pretrained model. 
This includes domains not present in pre-training data. 
Contrary to other research, pre-training on a dataset with the target domain excluded leads us to better performance than pre-training on a dataset from only the target domain. 
This includes when node labels are used in evaluation, where performance is consistently superior to single-domain or non-pre-trained models.

## Code Usage

The environmental setup is fairly minimal.
The broad requirements are standard for graph deep-learning:

 - pytorch-geometric
 - pytorch (shocking I know)
 - numpy, scipy, matplotlib, pandas, etc.
 - ogb (open graph benchmark)

Apologies for this being a rather brief description. 
Pytorch-geometric is still a little volatile, and often doesn't play well with others depending on your hardware, so we leave it up to the user to get out of versioning hell.

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

### Transfer

`transfer.py --args` can be used to fine-tune FoToM models.
Please not that it does this on all of our validation datasets.

 - `--num` the maximum number of samples to include in the test sets (doubled for validation)
 - `--checkpoint` the model checkpoint to fine-tune. We include the following:
 - - `all-100.pt` a model trained for 100 epochs on all the training data
 - - `chem-100.pt` a model trained for 100 epochs on only molecules
 - - `social-100.pt` a model trained for 100 epochs on on non-molecules
 - `untrained` as a checkpoint will not load a checkpoint, instead training a model from scratch

`linear_transfer.py` functions in essentially the same way, except that it uses a linear model in place of fine-tuning a FoToM model.

#### Further code

Dataset code can be found under `/datasets/`.
Loaders can be found here under `loaders.py`, and other datasets have their own respective processing files.
`from_ogb_dataset.py` converts an OGB dataset into a standard pytorch-geometric dataset.



