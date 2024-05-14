import os
import argparse
import torch
import numpy as np
import scipy.sparse as sp

from dataset import get_dataset, get_largest_cc, get_eigh, Transd2Ind, DataGraphSAINT
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="citeseer") #[citeseer, pubmed, ogbn-arxiv, flickr, reddit, squirrel, twitch-gamer]
parser.add_argument("--normalize_features", type=bool, default=True)
args = parser.parse_args([])

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)

else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full)

dir = f"./data/{args.dataset}"
if not os.path.isdir(dir):
    os.makedirs(dir)

idx_lcc, adj_norm_lcc, _ = get_largest_cc(data.adj_full, data.num_nodes, args.dataset)
np.save(f"{dir}/idx_lcc.npy", idx_lcc)

L_lcc = sp.eye(len(idx_lcc)) - adj_norm_lcc    
eigenvals_lcc, eigenvecs_lcc = get_eigh(L_lcc, f"{args.dataset}", True)

idx_train_lcc, idx_map = get_train_lcc(idx_lcc=idx_lcc, idx_train=data.idx_train, y_full=data.y_full, num_nodes=data.num_nodes, num_classes=data.num_classes)
np.save(f"{dir}/idx_train_lcc.npy", idx_train_lcc)
np.save(f"{dir}/idx_map.npy", idx_map)
