# FoToM

Repository for the paper **A Foundation Graph Model** in submission at ICML as of 01/02/24.

### Abstract:

The principal benefit of unsupervised graph representation learning is that a pre-trained model can be fine-tuned where data or labels are scarce.
Current methods maintain consistent node and edge attributes across the pretraining and target datasets. 
This precludes transfer to other domains.
A model capable of positive transfer on an arbitrary task and domain represents the first foundation graph model.

In this work we use adversarial contrastive learning to present FOTOM, the first foundation graph model. 
By excluding node and edge attributes we are able to pre-train over varied graph domains.
We demonstrate positive transfer on evaluation datasets from multiple domains compared to a non-pretrained model. 
This includes domains not present in pre-training data. 
Contrary to other research, pre-training on a dataset with the target domain excluded leads us to better performance than pre-training on a dataset from only the target domain. 
This includes when node labels are used in evaluation, where performance is consistently superior to single-domain or non-pre-trained models.




