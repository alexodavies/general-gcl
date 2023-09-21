# MuD-GR
### *Mu*lti-*D*omain *G*raph *R*epresentations

Repository for our (ongoing) work on using adversarial graph constrastive learning to produce domain-agnostic graph representations.

An interactive version of the embedding space, through PCA, can be explored in-browser through `bokeh-embedding-dashboard.html`

Code can be run through gcl_transfer.py, with arguments:

 - `--model-lr` learning rate for encoder
 - `--view_lr` learning rate for view learner
 - `--num gc layers` number of GNN layers before pooling
 - `--pooling_type` GNN pooling type, Standard/Layerwise
 - `--emb_dim` embedding dimension (from projection head)
 - `--batch_size` batch_size
 - `--drop_ratio` dropout probability
 - `--reg_lambda` regularisation strength
 - `--epochs` number of train epochs

