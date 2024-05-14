import json
import argparse
from collections import Counter
import numpy as np
import torch

from utils import *
from dataset import get_dataset, get_largest_cc, load_eigen, Transd2Ind, DataGraphSAINT

import agent as agent
import importlib
importlib.reload(agent)
from agent import GraphAgent

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", type=int, default=1, help="gpu id")
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--config", type=str, default='./config/config_distill.json')

parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--dataset", type=str, default="citeseer") # [citeseer, pubmed, ogbn-arxiv, flickr, reddit, squirrel, twitch-gamer]
parser.add_argument("--normalize_features", type=bool, default=True)
parser.add_argument("--reduction_rate", type=float, default=0.5)

parser.add_argument("--evaluate_gnn", type=str, default="GCN")
parser.add_argument("--epoch_gnn", type=int, default=2000)
parser.add_argument("--nlayers", type=float, default=2)
parser.add_argument("--hidden_dim", type=int, default=256)

args = parser.parse_args([])

section=f"{args.dataset}-{str(args.reduction_rate)}"

with open(args.config, "r") as config_file:
    config = json.load(config_file)
if section in config:
    config = config[section]
for key, value in config.items():
    setattr(args, key, value)

print(args)

torch.cuda.set_device(args.gpu_id)
seed_everything(args.seed)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)

else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full)

dataset_dir = f"./data/{args.dataset}"
idx_lcc = np.load(f"{dataset_dir}/idx_lcc.npy")
idx_train_lcc = np.load(f"{dataset_dir}/idx_train_lcc.npy")
idx_map = np.load(f"{dataset_dir}/idx_map.npy")

eigenvals_lcc, eigenvecs_lcc = load_eigen(args.dataset)
eigenvals_lcc = torch.FloatTensor(eigenvals_lcc)
eigenvecs_lcc = torch.FloatTensor(eigenvecs_lcc)

n_syn = int(len(data.idx_train) * args.reduction_rate)
args.eigen_k = args.eigen_k if args.eigen_k < n_syn else n_syn
eigenvals, eigenvecs = get_syn_eigen(real_eigenvals=eigenvals_lcc, real_eigenvecs=eigenvecs_lcc, eigen_k=args.eigen_k, ratio=args.ratio)

co_x_trans_real = get_subspace_covariance_matrix(eigenvecs, data.x_full[idx_lcc]) #kdd
embed_sum = get_embed_sum(eigenvals=eigenvals, eigenvecs=eigenvecs, x=data.x_full[idx_lcc])
embed_sum = embed_sum[idx_map,:]
embed_mean_real = get_embed_mean(embed_sum=embed_sum, label=data.y_full[idx_train_lcc])

data = data.cuda()
eigenvals = eigenvals.cuda()
co_x_trans_real = co_x_trans_real.cuda()
embed_mean_real = embed_mean_real.cuda()

accs = []
for ep in range(args.runs):
    args.expID = ep
    agent = GraphAgent(args, data)
    acc = agent.train(eigenvals, co_x_trans_real, embed_mean_real)
    accs.append(acc)

mean_acc = np.mean(accs)
std_acc = np.std(accs)
print(f"Mean ACC: {mean_acc}\t Std: {std_acc}")