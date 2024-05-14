import os
import json
from typing import Any
import networkx as nx
import gdown

import pandas as pd
import numpy as np
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh

import scipy.sparse as sp

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork
import torch_geometric.transforms as T
from torch_geometric.utils import convert, to_undirected

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils import normalize_adj


class Transd2Ind:
    """transductive setting to inductive setting"""
    
    def __init__(self, data, **kwargs):

        n = data.num_nodes
        adj = sp.csr_matrix(
            (
                np.ones(data.edge_index.shape[1]),
                (data.edge_index[0], data.edge_index[1]),
            ),
            shape=(n, n),
        )
        features = data.x
        labels = data.y
        if len(labels.shape) == 2 and labels.shape[1] == 1:
            labels = labels.reshape(-1)  # ogb-arxiv needs to reshape

        if hasattr(data, "train_mask"):
            # for fixed split
            idx_train = mask_to_index(data.train_mask, n)
            idx_val = mask_to_index(data.val_mask, n)
            idx_test = mask_to_index(data.test_mask, n)

        if "keep_ratio" in kwargs:
            keep_ratio = kwargs["keep_ratio"]
            if keep_ratio < 1:
                idx_train, _ = train_test_split(
                    idx_train,
                    random_state=None,
                    train_size=keep_ratio,
                    test_size=1 - keep_ratio,
                    stratify=labels[idx_train],
                )
        
        self.num_nodes = data.num_nodes
        self.num_classes = data.num_classes

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]
        
        self.x_train = features[idx_train]
        self.x_val = features[idx_val]
        self.x_test = features[idx_test]

        self.y_train = labels[idx_train]
        self.y_val = labels[idx_val]
        self.y_test = labels[idx_test]

        self.adj_full = adj
        self.x_full = features
        self.y_full = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
    
    def cuda(self):
        self.x_full = self.x_full.cuda()
        self.x_train = self.x_train.cuda()
        self.x_val = self.x_val.cuda()
        self.x_test = self.x_test.cuda()

        self.y_full = self.y_full.cuda()
        self.y_train = self.y_train.cuda()
        self.y_val = self.y_val.cuda()
        self.y_test = self.y_test.cuda()

        return self
    
class DataGraphSAINT:
    """datasets used in GraphSAINT paper"""

    def __init__(self, dataset, **kwargs):
        dataset_str = "./data/" + dataset + "/"
        adj_full = sp.load_npz(dataset_str + "adj_full.npz")
        self.num_nodes = adj_full.shape[0]
        if dataset == "ogbn-arxiv":
            adj_full = adj_full + adj_full.T
            adj_full[adj_full > 1] = 1

        role = json.load(open(dataset_str + "role.json", "r"))
        idx_train = role["tr"]
        idx_test = role["te"]
        idx_val = role["va"]

        if "label_rate" in kwargs:
            label_rate = kwargs["label_rate"]
            if label_rate < 1:
                idx_train = idx_train[: int(label_rate * len(idx_train))]

        self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        self.adj_val = adj_full[np.ix_(idx_val, idx_val)]
        self.adj_test = adj_full[np.ix_(idx_test, idx_test)]

        feat = np.load(dataset_str + "feats.npy")
        # ---- normalize feat ----
        x_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(x_train)
        feat = scaler.transform(feat)
        feat = torch.FloatTensor(feat)

        self.x_train = feat[idx_train]
        self.x_val = feat[idx_val]
        self.x_test = feat[idx_test]

        class_map = json.load(open(dataset_str + "class_map.json", "r"))
        labels = self.process_labels(class_map)
        labels = torch.LongTensor(labels)

        self.y_train = labels[idx_train]
        self.y_val = labels[idx_val]
        self.y_test = labels[idx_test]

        self.adj_full = adj_full
        self.x_full = feat
        self.y_full = labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)
        
    def cuda(self):
        self.x_full = self.x_full.cuda()
        self.x_train = self.x_train.cuda()
        self.x_val = self.x_val.cuda()
        self.x_test = self.x_test.cuda()

        self.y_full = self.y_full.cuda()
        self.y_train = self.y_train.cuda()
        self.y_val = self.y_val.cuda()
        self.y_test = self.y_test.cuda()

        return self

    def process_labels(self, class_map):
        """setup vertex property map for output classests"""

        num_vertices = self.num_nodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.num_classes = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k, v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int64)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.num_classes = max(class_arr) + 1
        return class_arr

class NCDataset(object):
    def __init__(self, name):
        self.data = None
        self.name = name
        self.num_classes = None

    def __getitem__(self, idx):
        assert idx == 0
        return self.data

class Dataset(object):
    def __init__(self, x, y, edge_index, num_nodes):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.num_nodes = num_nodes

def get_dataset(name, normalize_features=True, transform=None):
    path = f"./data/{name}"

    if name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(path, name)
    elif name in ["squirrel"]:
        preProcDs = WikipediaNetwork(path, name, False, T.NormalizeFeatures())
        dataset = WikipediaNetwork(path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    elif name in ["twitch-gamer"]:
        dataset = load_twitch_gamer_dataset(path)
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    data = dataset[0]
    data.num_classes = dataset.num_classes
    if name in ["squirrel"]:
        data.train_mask, data.val_mask, data.test_mask = torch.load(f"{path}/split.pt")    
        data.edge_index = preProcDs[0].edge_index
    if name in ["twitch-gamer"]:
        data.train_mask, data.val_mask, data.test_mask = torch.load(f"{path}/split.pt")    
    return data

def load_twitch_gamer_dataset(path, task="mature", normalize=True):
    dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0'}

    if not os.path.exists(f'{path}/twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
            output=f'{path}/twitch-gamer_feat.csv', quiet=False)
    if not os.path.exists(f'{path}/twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
            output=f'{path}/twitch-gamer_edges.csv', quiet=False)
    
    edges = pd.read_csv(f'{path}/twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{path}/twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    edge_index = to_undirected(edge_index)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)

    label = torch.tensor(label, dtype=torch.int64)
    dataset = NCDataset("twitch-gamer")
    dataset.data = Dataset(x=node_feat, y=label, edge_index=edge_index, num_nodes=num_nodes)
    dataset.num_classes = dataset.data.y.max().item() + 1
    return dataset


def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features


def index_to_mask(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[index] = 1
    return mask


def get_largest_cc(adj, num_nodes, data_name):
    mx = (adj).tocoo()
    edge_index = torch.LongTensor(np.vstack([mx.row, mx.col]))
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    nx_graph = convert.to_networkx(data, to_undirected=True)
    
    largest_cc = max(nx.connected_components(nx_graph), key=len)
    idx_lcc = list(largest_cc)
    largest_cc_graph = nx_graph.subgraph(largest_cc)
    
    if len(idx_lcc) == num_nodes:
        adj_lcc = adj
        adj_norm_lcc = normalize_adj(adj_lcc)
    else:
        adj_lcc = nx.to_scipy_sparse_array(largest_cc_graph)
        adj_norm_lcc = normalize_adj(adj_lcc)
        
    return idx_lcc, adj_norm_lcc, adj_lcc


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def load_eigen(dataset):
    dir = "./data/" + dataset + "/"
    val_file_name = "eigenvalues.npy"
    vec_file_name = "eigenvectors.npy"
    
    eigvals_path = os.path.join(dir, val_file_name)
    eigvecs_path = os.path.join(dir, vec_file_name)
    eigenvalues = np.load(eigvals_path)
    eigenvectors = np.load(eigvecs_path)

    if dataset in [ 'ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
        val_file_name = "eigenvalues_la.npy"
        vec_file_name = "eigenvectors_la.npy"
        eigvals_path = os.path.join(dir, val_file_name)
        eigvecs_path = os.path.join(dir, vec_file_name)    
        eigenvalues_la = np.load(eigvals_path)
        eigenvectors_la = np.load(eigvecs_path)
        
        eigenvalues = np.hstack([eigenvalues, eigenvalues_la])
        eigenvectors = np.hstack([eigenvectors, eigenvectors_la])
    
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]
                     
    return eigenvalues, eigenvectors


def get_eigh(laplacian_matrix, data_name, save=True):

    dir = "./data/" + data_name + "/"
    if not os.path.isdir(dir):
        os.makedirs(dir)

    val_file_name = "eigenvalues.npy"
    vec_file_name = "eigenvectors.npy"

    eigvals_path = os.path.join(dir, val_file_name)
    eigvecs_path = os.path.join(dir, vec_file_name)

    if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path):            
        eigenvalues = np.load(eigvals_path)
        eigenvectors = np.load(eigvecs_path)
    else:
        if data_name in [ 'ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
            eigenvalues, eigenvectors = eigsh(A=laplacian_matrix, k=1000, which="SA", tol=1e-5)        
        else:
            if sp.issparse(laplacian_matrix):
                laplacian_matrix = laplacian_matrix.todense()
            eigenvalues, eigenvectors = eigh(laplacian_matrix)

        if save:
            np.save(eigvals_path, eigenvalues)
            np.save(eigvecs_path, eigenvectors)
    
    if data_name in [ 'ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
        val_file_name = "eigenvalues_la.npy"
        vec_file_name = "eigenvectors_la.npy"
        eigvals_path = os.path.join(dir, val_file_name)
        eigvecs_path = os.path.join(dir, vec_file_name)
        
        if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path):            
            eigenvalues_la = np.load(eigvals_path)
            eigenvectors_la = np.load(eigvecs_path)
        else:
            eigenvalues_la, eigenvectors_la = eigsh(A=laplacian_matrix, k=1000, which="LA", tol=1e-5)
            if save:
                np.save(eigvals_path, eigenvalues_la)
                np.save(eigvecs_path, eigenvectors_la)
        
        eigenvalues = np.hstack([eigenvalues, eigenvalues_la])
        eigenvectors = np.hstack([eigenvectors, eigenvectors_la])
        
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]
                     
    return eigenvalues, eigenvectors

