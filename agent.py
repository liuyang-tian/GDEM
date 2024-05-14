import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import networkx as nx
from sklearn.metrics import accuracy_score

from utils import *
from dataset import get_eigh

import matplotlib.pyplot as plt


from model.sgc import SGC
from model.gcn import GCN
from model.appnp import APPNP
from model.chebnet import ChebNet
from model.chebnetII import ChebNetII
from model.bernnet import BernNet
from model.gprgnn import GPRGNN


class GraphAgent:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.n_syn = int(len(data.idx_train) * args.reduction_rate)
        self.d = (data.x_train).shape[1]
        self.num_classes = data.num_classes
        self.syn_class_indices = {}
        self.class_dict = None
        
        self.x_syn = nn.Parameter(torch.FloatTensor(self.n_syn, self.d).cuda())
        self.eigenvecs_syn = nn.Parameter(
            torch.FloatTensor(self.n_syn, args.eigen_k).cuda()
        )

        y_full = data.y_full
        idx_train = data.idx_train
        self.y_syn = torch.LongTensor(self.generate_labels_syn(y_full[idx_train], args.reduction_rate)).cuda()

        init_syn_feat = self.get_init_syn_feat(dataset=args.dataset, reduction_rate=args.reduction_rate, expID=args.expID)
        init_syn_eigenvecs = self.get_init_syn_eigenvecs(self.n_syn, self.num_classes)
        init_syn_eigenvecs = init_syn_eigenvecs[:, :args.eigen_k]
        self.reset_parameters(init_syn_feat, init_syn_eigenvecs)


    def reset_parameters(self, init_syn_feat, init_syn_eigenvecs):
        self.x_syn.data.copy_(init_syn_feat)
        self.eigenvecs_syn.data.copy_(init_syn_eigenvecs)
        
    
    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.y_syn.cpu().numpy())

        for c in range(data.num_classes):
            tmp = self.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]

        return features

    
    def retrieve_class(self, c, num=256):
        y_train = self.data.y_train.cpu().numpy()
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.data.num_classes):
                self.class_dict['class_%s'%i] = (y_train == i)
        idx = np.arange(len(self.data.idx_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

        
    def train(self, eigenvals_syn, co_x_trans_real, embed_mean_real):
        args = self.args
        data = self.data
        adj_full = data.adj_full
        adj_full = normalize_adj_to_sparse_tensor(adj_full)

        optimizer_feat = torch.optim.Adam(
            [self.x_syn], lr=args.lr_feat
        )
        optimizer_eigenvec = torch.optim.Adam(
            [self.eigenvecs_syn], lr=args.lr_eigenvec
        )
        
        for ep in range(args.epoch):
            loss = 0.0
            x_syn = self.x_syn
            eigenvecs_syn = self.eigenvecs_syn
            
            # eigenbasis match
            co_x_trans_syn = get_subspace_covariance_matrix(eigenvecs=eigenvecs_syn, x=x_syn) # kdd
            eigen_match_loss = F.mse_loss(co_x_trans_syn, co_x_trans_real)
            loss += args.alpha * eigen_match_loss

            # class loss 
            embed_sum_syn = get_embed_sum(eigenvals=eigenvals_syn, eigenvecs=eigenvecs_syn, x=x_syn)
            embed_mean_syn = get_embed_mean(embed_sum=embed_sum_syn, label=self.y_syn)  #cd
            cov_embed = embed_mean_real @ embed_mean_syn.T
            iden = torch.eye(data.num_classes).cuda()
            class_loss = F.mse_loss(cov_embed, iden)
            loss += args.beta * class_loss

            # orthog_norm
            orthog_syn = eigenvecs_syn.T @ eigenvecs_syn
            iden = torch.eye(args.eigen_k).cuda()
            orthog_norm = F.mse_loss(orthog_syn, iden)
            loss += args.gamma * orthog_norm
            
            if (ep == 0) or (ep == (args.epoch - 1)):
                print(f"epoch: {ep}")
                print(f"eigen_match_loss: {eigen_match_loss}")
                print(f"args.alpha * eigen_match_loss: {args.alpha * eigen_match_loss}")
                
                print(f"class_loss: {class_loss}")
                print(f"args.beta * class_loss: {args.beta * class_loss}")

                print(f"orthog_norm: {orthog_norm}")
                print(f"args.gamma * orthog_norm: {args.gamma * orthog_norm}")

            optimizer_eigenvec.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward()
            
            # update U:
            if ep % (args.e1 + args.e2) < args.e1:
                optimizer_eigenvec.step()
            else:
                optimizer_feat.step()

        x_syn, y_syn = self.x_syn.detach(), self.y_syn
        eigenvecs_syn = self.eigenvecs_syn.detach()
        
        acc = self.test_with_val(
            x_syn,
            eigenvals_syn,
            eigenvecs_syn,
            y_syn,
            verbose=False
        )
        
        dir = f"./saved_ours/{args.dataset}-{args.reduction_rate}"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        torch.save(
            eigenvals_syn,
            f"{dir}/eigenvals_syn_{args.expID}.pt",
        )
        torch.save(
            eigenvecs_syn,
            f"{dir}/eigenvecs_syn_{args.expID}.pt",
        )
        torch.save(
            x_syn, f"{dir}/feat_{args.expID}.pt"
        )

        return acc

    
    def test_with_val(
        self,
        x_syn,
        eigenvals_syn,
        eigenvecs_syn,
        y_syn,
        verbose=False
    ):
        args = self.args
        data = self.data
        evaluate_gnn = args.evaluate_gnn
                
        L_syn = eigenvecs_syn @ torch.diag(eigenvals_syn) @ eigenvecs_syn.T
        adj_syn = torch.eye(self.n_syn).cuda() - L_syn
        
        if evaluate_gnn == "SGC":
            model = SGC(
                num_features=self.d,
                num_classes=data.num_classes,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
            ).cuda()
        elif evaluate_gnn == "GCN":
            model = GCN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout
            ).cuda()
        elif evaluate_gnn == "ChebNet":
            model = ChebNet(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                k=args.k,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout,
            ).cuda()
        elif evaluate_gnn == "APPNP":
            model = APPNP(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout,
                alpha=0.1,
            ).cuda()
        elif evaluate_gnn == "ChebNetII":
            model = ChebNetII(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate
            ).cuda()
        elif evaluate_gnn == "BernNet":
            model = BernNet(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate,
            ).cuda()
        elif evaluate_gnn == "GPRGNN":
            model = GPRGNN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate,
            ).cuda()
                
        model.cuda()
        model.fit_with_val(
            x_syn,
            y_syn,
            adj_syn,
            data,
            args.epoch_gnn,
            verbose=verbose
        )

        model.eval()
        
        # Full graph
        idx_test = data.idx_test
        x_full = data.x_full
        y_full = data.y_full
        adj_full = data.adj_full
        adj_full = normalize_adj_to_sparse_tensor(adj_full)
        
        y_test = (y_full[idx_test]).cpu().numpy()
        output = model.predict(x_full, adj_full)
        loss_test = F.nll_loss(output[idx_test], y_full[idx_test])

        pred = output.max(1)[1].cpu().numpy()
        acc_test = accuracy_score(y_test, pred[idx_test])

        print(
            f"(Test set results: loss= {loss_test.item():.4f}, accuracy= {acc_test:.4f}\n"
        )
        
        return acc_test

    def get_eigenspace_embed(self, eigen_vecs, x):
        eigen_vecs = eigen_vecs.unsqueeze(2) # k * n * 1
        eigen_vecs_t = eigen_vecs.permute(0, 2, 1) # k * 1 * n
        eigenspace = torch.bmm(eigen_vecs, eigen_vecs_t) # knn
        embed = torch.matmul(eigenspace, x) # knn*nd=knd
        return embed

    def get_real_embed(self, k, L, x):
        filtered_x = x

        emb_list = []
        for i in range(k):
            filtered_x = L @ filtered_x
            emb_list.append(filtered_x)

        embed = torch.stack(emb_list, dim=0)
        return embed

    def get_syn_embed(self, k, eigenvals, eigen_vecs, x):
        trans_x = eigen_vecs @ x
        filtered_x = trans_x

        emb_list = []
        for i in range(k):
            filtered_x = torch.diag(eigenvals) @ filtered_x
            emb_list.append(eigen_vecs.T @ filtered_x)

        embed = torch.stack(emb_list, dim=0)
        return embed

    
    def get_init_syn_feat(self, dataset, reduction_rate, expID):
        init_syn_x = torch.load(f"./initial_feat/{dataset}/x_init_{reduction_rate}_{expID}.pt", map_location="cpu")
        return init_syn_x
    
        
    def get_init_syn_eigenvecs(self, n_syn, num_classes):
        n_nodes_per_class = n_syn // num_classes
        n_nodes_last = n_syn % num_classes

        size = [n_nodes_per_class for i in range(num_classes - 1)] + (
            [n_syn - (num_classes - 1) * n_nodes_per_class] if n_nodes_last != 0 else [n_nodes_per_class]
        )
        prob_same_community = 1 / num_classes
        prob_diff_community = prob_same_community / 3

        prob = [
            [prob_diff_community for i in range(num_classes)]
            for i in range(num_classes)
        ]
        for idx in range(num_classes):
            prob[idx][idx] = prob_same_community

        syn_graph = nx.stochastic_block_model(size, prob)
        syn_graph_adj = nx.adjacency_matrix(syn_graph)
        syn_graph_L = normalize_adj(syn_graph_adj)
        syn_graph_L = np.eye(n_syn) - syn_graph_L
        _, eigen_vecs = get_eigh(syn_graph_L, "", False)

        return torch.FloatTensor(eigen_vecs).cuda()


    def generate_labels_syn(self, train_label, reduction_rate):
        from collections import Counter

        n = len(train_label)
        counter = Counter(train_label.cpu().numpy())

        num_class_dict = {}

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        y_syn = []
        self.syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * reduction_rate) - sum_
                self.syn_class_indices[c] = [len(y_syn), len(y_syn) + num_class_dict[c]]
                y_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(y_syn), len(y_syn) + num_class_dict[c]]
                y_syn += [c] * num_class_dict[c]

        return y_syn
    