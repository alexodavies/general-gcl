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
from ogb.graphproppred import PygGraphPropPredDataset, GraphPropPredDataset
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from torch_geometric.data import DataLoader, Data
from torch_geometric.transforms import Compose
from torch_scatter import scatter
import matplotlib.pyplot as plt

from datasets.facebook_dataset import get_fb_dataset, FacebookDataset
from datasets.ego_dataset import get_deezer, EgoDataset
from datasets.community_dataset import get_community_dataset
from datasets.cora_dataset import get_cora_dataset, CoraDataset
from datasets.random_dataset import get_random_dataset
from datasets.from_ogb_dataset import FromOGBDataset

from unsupervised.embedding_evaluation import EmbeddingEvaluation, GeneralEmbeddingEvaluation
from unsupervised.encoder import MoleculeEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight
from unsupervised.view_learner import ViewLearner


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def setup_wandb(cfg):
    """
    Uses a config dictionary to initialise wandb to track sampling.
    Requires a wandb account, https://wandb.ai/

    params: cfg: argparse Namespace

    returns:
    param: cfg: same config
    """
    # config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    kwargs = {'name': f"{cfg.dataset}-" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'gcl_{cfg.dataset}', 'config': cfg,
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'entity':'hierarchical-diffusion'}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    wandb.log({"Type":"Sampling"})

    return cfg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def from_ogb_dataset(dataset, target_y = torch.Tensor([[0,0]])):

    graph_list = []

    for item in dataset:
        data = Data(x=item.x, edge_index=item.edge_index, edge_attr=item.edge_attr, y=target_y)
        graph_list.append(data)

    return

def get_big_dataset(dataset, batch_size, transforms, num_social = 50000):
    names = ["ogbg-molclintox", "ogbg-molpcba"]

    datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in names]

    split_idx = [data.get_idx_split() for data in datasets]

    datasets = [data[split_idx[i]["train"]] for i, data in enumerate(datasets)]

    datasets = [FromOGBDataset(os.getcwd()+'/original_datasets/'+names[i], data) for i, data in enumerate(datasets)]

    combined = FromOGBDataset(os.getcwd()+'/original_datasets/'+'ogbg-molesol', dataset)

    for data in datasets:
        combined += data


    for item in combined:
        # print(item.y)
        if item.y != None:
            print(item)
        # if item.y.shape[1] != 2:
        #     print(item.y)
        #     item.y = torch.Tensor([[0,0]])
        #     print(item.y)

    social_datasets = [transforms(FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large', num=num_social)),
                       transforms(EgoDataset(os.getcwd()+'/original_datasets/'+'twitch_egos', num=num_social)),
                       transforms(CoraDataset(os.getcwd()+'/original_datasets/'+'cora', num=num_social))]

    for data in social_datasets:
        combined += data

    # for item in combined:
    #     if item
        # if item.y.shape[1] != 2:
        #     print(item.y)
            # item.y = torch.Tensor([[0,0]])

    # combined = transforms(combined)
    print(combined)


    return DataLoader(combined, batch_size=batch_size, shuffle=True), "Dummy"

    #
    # datasets = datasets + [get_fb_dataset(num = num_social),
    #                   get_deezer(num=2 * num_social),
    #                   get_cora_dataset(batch_size, num=num_social)]#,
    #                   # get_community_dataset(batch_size, num = num_social),
    #                   # get_random_dataset(batch_size, num = num_social)
    #                   # ]

    # large_dataset =

    # return [DataLoader(data, batch_size=batch_size) for data in datasets + [dataset]] +  social_loaders,\
    #        names +  ["Molesol (target)", "Facebook", "Egos", "Cora"]#, "Communities", "Random"]

    # out = torch.utils.data.ConcatDataset([datasets])
    #
    # return out

def get_val_loaders(dataset, batch_size, transforms, num_social = 5000):
    names = ["ogbg-molclintox", "ogbg-molpcba"]

    datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in names]
    split_idx = [data.get_idx_split() for data in datasets]

    datasets = [data[split_idx[i]["valid"]] for i, data in enumerate(datasets)]

    dataset_lengths = [len(data) for data in datasets]

    for i, data in enumerate(datasets):
        if dataset_lengths[i] > num_social:
            datasets[i] = data[:num_social]


    combined = dataset

    for data in datasets:
        combined += data

    social_datasets = [transforms(FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large', stage = "val", num=num_social)),
                       transforms(EgoDataset(os.getcwd()+'/original_datasets/'+'twitch_egos', stage = "val", num=num_social)),
                       transforms(CoraDataset(os.getcwd()+'/original_datasets/'+'cora', stage = "val", num=num_social))]

    all_datasets = datasets + [dataset] + social_datasets

    datasets = [DataLoader(data, batch_size=batch_size) for data in all_datasets]
    print(datasets)
    # for data in social_datasets:
    #     combined += data
    #
    # # combined = transforms(combined)
    # print(combined)

    return datasets, names + ["Molesol (target)", "Facebook", "Egos", "Cora"]

    #
    # datasets = datasets + [get_fb_dataset(num = num_social),
    #                   get_deezer(num=2 * num_social),
    #                   get_cora_dataset(batch_size, num=num_social)]#,
    #                   # get_community_dataset(batch_size, num = num_social),
    #                   # get_random_dataset(batch_size, num = num_social)
    #                   # ]

    # large_dataset =

    # return [DataLoader(data, batch_size=batch_size) for data in datasets + [dataset]] +  social_loaders,\
    #        names +  ["Molesol (target)", "Facebook", "Egos", "Cora"]#, "Communities", "Random"]

    # out = torch.utils.data.ConcatDataset([datasets])
    #
    # return out

def train_epoch(dataloader,
                model, model_optimizer,
                view_learner, view_optimizer,
                model_loss_all, view_loss_all, reg_all):

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
    setup_seed(args.seed)

    evaluator = Evaluator(name=args.dataset)
    my_transforms = Compose([initialize_edge_weight])
    dataset = PygGraphPropPredDataset(name=args.dataset, root='./original_datasets/', transform=my_transforms)

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=512, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=512, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=512, shuffle=False)


    # big_dataset = get_big_dataset(dataset[split_idx["train"]], my_transforms)

    dataloader, names = get_big_dataset(dataset[split_idx["train"]], args.batch_size, my_transforms)
    val_loaders, names = get_val_loaders(dataset[split_idx["train"]], args.batch_size, my_transforms)

    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # extra_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='./original_datasets/', transform=my_transforms)
    #
    # extra_split_idx = extra_dataset.get_idx_split()
    # extra_train_loader = DataLoader(extra_dataset[extra_split_idx["train"]], batch_size=512, shuffle=True)
    # extra_valid_loader = DataLoader(extra_dataset[extra_split_idx["valid"]], batch_size=512, shuffle=False)
    # extra_test_loader = DataLoader(extra_dataset[extra_split_idx["test"]], batch_size=512, shuffle=False)
    #
    # print(dataset[split_idx["train"]] + extra_dataset[extra_split_idx["train"]])
    #
    # extra_dataloader = DataLoader(extra_dataset, batch_size=args.batch_size, shuffle=True)

    model = GInfoMinMax(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                    proj_hidden_dim=args.emb_dim).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)


    view_learner = ViewLearner(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)

    if 'classification' in dataset.task_type:
        ee = EmbeddingEvaluation(LogisticRegression(dual=False, fit_intercept=True, max_iter=10000),
                                 evaluator, dataset.task_type, dataset.num_tasks, device, params_dict=None,
                                 param_search=True)
        # ee = EmbeddingEvaluation(MLPClassifier(max_iter=2000),
        #                          evaluator, dataset.task_type, dataset.num_tasks, device, params_dict=None,
        #                          param_search=True)
    elif 'regression' in dataset.task_type:
        ee = EmbeddingEvaluation(Ridge(fit_intercept=True, copy_X=True, max_iter=10000),
                                 evaluator, dataset.task_type, dataset.num_tasks, device, params_dict=None,
                                 param_search=True)
        # ee = EmbeddingEvaluation(MLPRegressor(max_iter=2000),
        #                          evaluator, dataset.task_type, dataset.num_tasks, device, params_dict=None,
        #                          param_search=True)
    else:
        raise NotImplementedError

    general_ee = GeneralEmbeddingEvaluation()

    model.eval()
    train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader, test_loader)
    general_ee.embedding_evaluation(model.encoder, val_loaders, names)
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score,
                                                                                         test_score))


    model_losses = []
    view_losses = []
    view_regs = []
    valid_curve = []
    test_curve = []
    train_curve = []

    best_val = 1.0e4

    for epoch in tqdm(range(1, args.epochs + 1)):

        fin_model_loss = 0.
        fin_view_loss = 0.
        fin_reg = 0.

        # for dataloader in dataloaders:

        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0

        model_loss_all, view_loss_all, reg_all = train_epoch(dataloader,
                    model, model_optimizer,
                    view_learner, view_optimizer,
                    model_loss_all, view_loss_all, reg_all)

        fin_model_loss += model_loss_all / len(dataloader)
        fin_view_loss += view_loss_all / len(dataloader)
        fin_reg += reg_all / len(dataloader)



        # logging.info('Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(epoch, fin_model_loss, fin_view_loss, fin_reg))
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)

        wandb.log({"Model Loss": fin_model_loss,
                   "View Loss": fin_view_loss,
                   "Reg Loss": fin_reg})

        if epoch % 5 == 0:
            model.eval()
            train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader,
                                                                         test_loader, vis=True)

            general_ee.embedding_evaluation(model.encoder, val_loaders, names)

            wandb.log({"Train Score": train_score,
                       "Val Score": val_score,
                       "Test Score": test_curve})

        # logging.info(
        #     "Metric: {} Train: {} Val: {} Test: {}".format(evaluator.eval_metric, train_score, val_score, test_score))

            train_curve.append(train_score)
            valid_curve.append(val_score)
            test_curve.append(test_score)

            if val_score <= best_val:
                best_val = val_score
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': model.state_dict(),
                    'encoder_optimizer_state_dict': model_optimizer.state_dict(),
                    'view_state_dict': view_learner.state_dict(),
                    'view_optimizer_state_dict': view_optimizer.state_dict(),
                    'loss': val_score,
                }, f"{wandb.run.dir}/Checkpoint-{epoch}.pt")

    train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader,
                                                                 test_loader, vis = True)

    general_ee.embedding_evaluation(model.encoder, val_loaders, names)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    # plt.plot(valid_curve, label="validation")
    # plt.plot(train_curve, label = "train")
    # plt.plot(test_curve, label = "test")
    #
    # plt.legend(shadow=True)
    #
    # plt.show()

    logging.info('FinishedTraining!')
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info('BestTrainScore: {}'.format(best_train))
    logging.info('BestValidationScore: {}'.format(valid_curve[best_val_epoch]))
    logging.info('FinalTestScore: {}'.format(test_curve[best_val_epoch]))

    return valid_curve[best_val_epoch], test_curve[best_val_epoch]


def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL ogbg-mol*')

    parser.add_argument('--dataset', type=str, default='ogbg-molesol',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=5,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')

    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    args = setup_wandb(args)
    # print(args.dataset, repr(args))
    # quit()
    run(args)

