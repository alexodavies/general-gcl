import argparse
import logging
import random

import wandb
from datetime import datetime
from tqdm import tqdm
import os

import numpy as np
import torch
from ogb.graphproppred import Evaluator
from torch_geometric.transforms import Compose
from torch_scatter import scatter

from utils import setup_wandb
from datasets.loaders import get_train_loader, get_val_loaders, get_test_loaders


from unsupervised.embedding_evaluation import EmbeddingEvaluation, GeneralEmbeddingEvaluation, DummyEmbeddingEvaluation
from unsupervised.encoder import Encoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight
from unsupervised.view_learner import ViewLearner


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def cl_loss(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss


def train_epoch_random_edges(dataloader,
                      model, model_optimizer,
                      model_loss_all,
                      drop_proportion = 0.2):
    """
    single train epoch for encoder with random edge dropping as augmentation
    Args:
        dataloader: dataloader (see get_train_loader)
        model: encoder
        model_optimizer: optimizer for model
        model_loss_all: stores losses for each batch

    Returns:
        model_loss_all: loss for epoch
        view_loss_all: None - placeholder for code consistency with AD-GCL
        reg_all: None - placeholder for code consistency with AD-GCL
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in tqdm(dataloader, leave = False):
        # set up
        batch = batch.to(device)
        # # train (model) to minimize contrastive loss
        model.train()
        # view_learner.eval()
        model.zero_grad()

        edge_weights_1 = torch.bernoulli((1 - drop_proportion) * torch.ones(batch.edge_index.shape[1])).to(device)
        edge_weights_2 = torch.bernoulli((1 - drop_proportion) * torch.ones(batch.edge_index.shape[1])).to(device)

        x_aug_1, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, edge_weights_1)
        x_aug_2, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, edge_weights_2)

        model_loss = cl_loss(x_aug_1, x_aug_2) #
        # model_loss = model.calc_loss(x_aug_1, x_aug_2)
        model_loss_all += model_loss.item() * batch.num_graphs

        # standard gradient descent formulation
        model_loss.backward()
        model_optimizer.step()

    return model_loss_all, -1., -1.



def train_epoch_random_nodes(dataloader,
                      model, model_optimizer,
                      model_loss_all,
                      drop_proportion = 0.2):
    """
    single train epoch for encoder with random node dropping as augmentation
    Args:
        dataloader: dataloader (see get_train_loader)
        model: encoder
        model_optimizer: optimizer for model
        model_loss_all: stores losses for each batch

    Returns:
        model_loss_all: loss for epoch
        view_loss_all: None - placeholder for code consistency with AD-GCL
        reg_all: None - placeholder for code consistency with AD-GCL
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in tqdm(dataloader, leave = False):
        # set up
        batch = batch.to(device)
        # # train (model) to minimize contrastive loss
        model.train()
        # view_learner.eval()
        model.zero_grad()


        nodes_dropped_1 = torch.bernoulli(drop_proportion * torch.ones(batch.x.shape[0])).to(bool).to(device)
        nodes_dropped_2 = torch.bernoulli(drop_proportion * torch.ones(batch.x.shape[0])).to(bool).to(device)

        mask_1 = torch.ones(batch.edge_index.shape[1], dtype=bool).to(device)
        mask_2 = torch.ones(batch.edge_index.shape[1], dtype=bool).to(device)

        # Find edges connected to dropped nodes and update the mask
        edges_dropped_1 = torch.any(batch.edge_index == torch.nonzero(nodes_dropped_1).unsqueeze(1), dim=0).any(dim=0).to(device)
        edges_dropped_2 = torch.any(batch.edge_index == torch.nonzero(nodes_dropped_2).unsqueeze(1), dim=0).any(dim=0).to(device)

        mask_1[edges_dropped_1] = False
        mask_2[edges_dropped_2] = False

        mask_1 = mask_1.to(batch.x.dtype).to(device)
        mask_2 = mask_2.to(batch.x.dtype).to(device)


        x_aug_1, _ = model(batch.batch.to(device), batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), mask_1.to(device))
        x_aug_2, _ = model(batch.batch.to(device), batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), mask_2.to(device))

        model_loss = cl_loss(x_aug_1, x_aug_2) #
        # model_loss = model.calc_loss(x_aug_1, x_aug_2)
        model_loss_all += model_loss.item() * batch.num_graphs

        # standard gradient descent formulation
        model_loss.backward()
        model_optimizer.step()

    return model_loss_all, -1., -1.

def train_epoch_adgcl(dataloader,
                      model, model_optimizer,
                      view_learner, view_optimizer,
                      model_loss_all, view_loss_all, reg_all):
    """
    single train epoch for encoder and view learner

    Args:
        dataloader: dataloader (see get_train_loader)
        model: encoder
        model_optimizer: optimizer for model
        view_learner: view learner
        view_optimizer: optimizer for view learner
        model_loss_all: stores losses for each batch
        view_loss_all: stores view losses for each batch
        reg_all: stores regularization losses for each batch

    Returns:
        model_loss_all: loss for epoch
        view_loss_all: view loss for epoch
        reg_all: regularization loss for epoch
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in tqdm(dataloader, leave = False):
        # set up
        batch = batch.to(device)

        # train view to maximize contrastive loss
        view_learner.train()
        view_learner.zero_grad()
        model.eval()

        x, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)

        edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)

        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        x_aug, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

        # regularization

        row, col = batch.edge_index
        edge_batch = batch.batch[row]
        edge_drop_out_prob = 1 - batch_aug_edge_weight

        uni, edge_batch_num = edge_batch.unique(return_counts=True)
        sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

        reg = []
        for b_id in range(args.batch_size):
            if b_id in uni:
                num_edges = edge_batch_num[uni.tolist().index(b_id)]
                reg.append(sum_pe[b_id] / num_edges)
            else:
                # means no edges in that graph. So don't include.
                pass
        num_graph_with_edges = len(reg)
        reg = torch.stack(reg)
        reg = reg.mean()

        view_loss = model.calc_loss(x, x_aug) - (args.reg_lambda * reg)
        view_loss_all += view_loss.item() * batch.num_graphs
        reg_all += reg.item()
        # gradient ascent formulation
        (-view_loss).backward()
        view_optimizer.step()

        # train (model) to minimize contrastive loss
        model.train()
        view_learner.eval()
        model.zero_grad()

        x, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
        edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

        x_aug, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

        model_loss = model.calc_loss(x, x_aug)
        model_loss_all += model_loss.item() * batch.num_graphs

        # standard gradient descent formulation
        model_loss.backward()
        model_optimizer.step()

    return model_loss_all, view_loss_all, reg_all

def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    # setup_seed(args.seed)

    if "original_datasets" not in os.listdir():
        os.mkdir("original_datasets")

    random_node_dropping = args.random_node_views
    random_edge_dropping = args.random_edge_views

    random_dropping = True if random_edge_dropping or random_node_dropping else False
    drop_proportion = args.dropped

    evaluation_node_features = args.node_features
    molecules = not args.no_molecules
    socials = not args.no_socials
    print(f"Chemicals: {molecules}, Socials: {socials}")

    my_transforms = Compose([initialize_edge_weight])

    dataset_subset = ["chemical" if molecules else "dummy",
                      "social" if socials else "dummy"]

    print(f"Passing subset: {dataset_subset}")
    dataloader = get_train_loader(args.batch_size, my_transforms, subset=dataset_subset)

    val_loaders, names = get_val_loaders(args.batch_size, my_transforms)
    test_loaders, names = get_test_loaders(args.batch_size, my_transforms)

    model = GInfoMinMax(Encoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                        proj_hidden_dim=args.proj_dim).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    if not random_dropping:
        view_learner = ViewLearner(Encoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                                   mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
        view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)


    general_ee = GeneralEmbeddingEvaluation()

    model.eval()

    general_ee.embedding_evaluation(model.encoder, val_loaders, test_loaders, names, node_features = evaluation_node_features)

    model_losses = []
    view_losses = []
    view_regs = []

    for epoch in tqdm(range(1, args.epochs + 1)):
        fin_model_loss = 0.
        fin_view_loss = 0.
        fin_reg = 0.

        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0


        if not random_dropping:
            model_loss_all, view_loss_all, reg_all = train_epoch_adgcl(dataloader,
                                                                       model, model_optimizer,
                                                                       view_learner, view_optimizer,
                                                                       model_loss_all, view_loss_all, reg_all)
        else:
            if random_edge_dropping:
                print(f"\n\nUSING RANDOM EDGE DROPPING\n\n")
                model_loss_all, view_loss_all, reg_all = train_epoch_random_edges(dataloader,
                                                                           model, model_optimizer,
                                                                           model_loss_all,
                                                                           drop_proportion=drop_proportion)
            elif random_node_dropping:
                print(f"\n\nUSING RANDOM NODE DROPPING\n\n")
                model_loss_all, view_loss_all, reg_all = train_epoch_random_nodes(dataloader,
                                                                           model, model_optimizer,
                                                                           model_loss_all,
                                                                           drop_proportion=drop_proportion)

        fin_model_loss += model_loss_all / len(dataloader)
        fin_view_loss += view_loss_all / len(dataloader)
        fin_reg += reg_all / len(dataloader)

        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)

        wandb.log({"Model Loss": fin_model_loss,
                   "View Loss": fin_view_loss,
                   "Reg Loss": fin_reg})

        if epoch % 10 == 0:

            if args.proj_dim != 1:
                general_ee.embedding_evaluation(model.encoder, val_loaders, test_loaders, names, node_features = evaluation_node_features)

            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.state_dict(),
                'encoder_optimizer_state_dict': model_optimizer.state_dict(),
                'view_state_dict': None if random_dropping else view_learner.state_dict(),
                'view_optimizer_state_dict': None if random_dropping else view_optimizer.state_dict()},
                f"{wandb.run.dir}/Sweep-emb-{args.emb_dim}-epoch-{epoch}{'-random_dropping' if random_dropping else ''}")


def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL')

    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=6,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')

    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--proj_dim', type=int, default=300,
                        help='projection head dimension')

    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=2.0, help='View Learner Edge Perturb Regularization Strength')


    parser.add_argument(
        '-f',
        '--node-features',
        action='store_true',
        help='Whether to include node features (labels) in evaluation',
    )


    parser.add_argument(
        '-c',
        '--no-molecules',
        action='store_true',
        help='Whether to include molecules in training data',
    )

    parser.add_argument(
        '-s',
        '--no-socials',
        action='store_true',
        help='Whether to include social (ie all other) graphs in training data',
    )

    parser.add_argument(
        '-re',
        '--random-edge-views',
        action='store_true',
        help='Whether to use random edge dropping',
    )

    parser.add_argument(
        '-rn',
        '--random-node-views',
        action='store_true',
        help='Whether to use random node dropping',
    )

    parser.add_argument(
        '--dropped', type=float, default=0.2,
        help='Proportion of edges dropped during random edge dropping',
    )

    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    args = setup_wandb(args, offline=True)
    run(args)

