import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.utils import add_self_loops

from copy import deepcopy
from scipy.special import comb
from sklearn.metrics import accuracy_score
import torch_sparse
from torch_sparse import SparseTensor
from utils import normalize_adj_to_sparse_tensor


class BernNet(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        k,
        lr,
        lr_conv,
        weight_decay,
        wd_conv,
        dropout,
        dprate
    ):
        super(BernNet, self).__init__()
        self.lr = lr
        self.lr_conv = lr_conv
        
        self.weight_decay = weight_decay
        self.wd_conv = wd_conv

        self.dropout = dropout
        self.dprate = dprate

        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)
        self.conv = BernConv(k)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.feat_encoder.weight.data)
        if self.feat_encoder.bias is not None:
            self.feat_encoder.bias.data.zero_()
        
        nn.init.xavier_uniform_(self.final_encoder.weight.data)
        if self.final_encoder.bias is not None:
            self.final_encoder.bias.data.zero_()

    def forward(self, x, adj, poly_item):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.feat_encoder(x))
        x = F.dropout(x, self.dropout, self.training)
        x = self.final_encoder(x)
        
        if self.dprate == 0.0:
            x = self.conv(x, adj, poly_item)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.conv(x, adj, poly_item)

        return F.log_softmax(x, dim=1)    
    
    def fit_with_val(
        self,
        x_syn,
        y_syn,
        adj_syn,
        data,
        epochs,
        verbose=False,
    ):
        adj = data.adj_full
        x_real = data.x_full

        adj = normalize_adj_to_sparse_tensor(adj)
        num_nodes = x_real.shape[0]
        laplacian_mat, poly_item = self.get_poly_item(adj, num_nodes)

        idx_val = data.idx_val
        idx_test = data.idx_test
        y_full = data.y_full
        y_val = (data.y_val).cpu().numpy()
        y_test = (data.y_test).cpu().numpy()
        
        num_nodes_syn = x_syn.shape[0]
        laplacian_mat_syn, poly_item_syn = self.get_poly_item(adj_syn, num_nodes_syn)

        if verbose:
            print("=== training bernnet model ===")

        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.feat_encoder.parameters(),
                    "weight_decay": self.weight_decay,
                    "lr": self.lr,
                },
                {
                    "params": self.final_encoder.parameters(),
                    "weight_decay": self.weight_decay,
                    "lr": self.lr,
                },
                {
                    "params": self.conv.parameters(),
                    "weight_decay": self.wd_conv,
                    "lr": self.lr_conv,
                },
            ]
        )

        best_acc_val = 0
        y_val = (y_full[idx_val]).cpu().numpy()
        y_test = (y_full[idx_test]).cpu().numpy()

        lr = self.lr
        for i in range(epochs):
            self.train()

            optimizer.zero_grad()

            output = self.forward(x_syn, laplacian_mat_syn, poly_item_syn)
            loss_train = F.nll_loss(output, y_syn)

            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print("Epoch {}, training loss: {}".format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(x_real, laplacian_mat, poly_item)
                output = output[idx_val]
                    
                loss_val = F.nll_loss(output, y_full[idx_val])

                pred = output.max(1)[1]
                pred = pred.cpu().numpy()
                acc_val = accuracy_score(y_val, pred)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print(
                "=== picking the best model according to the performance on validation ==="
            )
        self.load_state_dict(weights)
    
    
    def get_poly_item(self, adj, num_nodes):
        if isinstance(adj, torch_sparse.SparseTensor):
            adj_storage = adj.storage
            row = adj_storage._row
            col = adj_storage._col
            val = adj_storage._value
            edge_index = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
            
            edge_index1, edge_weight1 = add_self_loops(edge_index=edge_index, edge_attr=-val, fill_value=1., num_nodes=num_nodes)
            laplacian_mat = SparseTensor(row=edge_index1[0], col=edge_index1[1], value=edge_weight1, sparse_sizes=(num_nodes, num_nodes)).coalesce()
            
            edge_index2, edge_weight2 = add_self_loops(edge_index=edge_index, edge_attr=val, fill_value=1., num_nodes=num_nodes)
            poly_item = SparseTensor(row=edge_index2[0], col=edge_index2[1], value=edge_weight2, sparse_sizes=(num_nodes, num_nodes)).coalesce()

        else:
            iden = torch.eye(num_nodes).cuda()
            laplacian_mat = iden - adj
            poly_item = iden + adj
        
        return laplacian_mat, poly_item
    

    @torch.no_grad()
    def predict(self, x, adj, poly_item):
        self.eval()
        return self.forward(x, adj, poly_item)


class BernConv(nn.Module):
    def __init__(self, k):
        super(BernConv, self).__init__()
        self.k = k
        self.filter_param = nn.Parameter(torch.Tensor(k+1, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        self.filter_param.data.fill_(1)

    def forward(self, x, adj, poly_item):
        filter_param = F.relu(self.filter_param)
        y = self.get_bern_poly(adj, poly_item, x, filter_param)
        return y

    def get_bern_poly(self, poly_item1, poly_item2, x, filter_param):
        first_poly_list = [x]
        i_pow_poly = x
        for i in range(self.k):
            i_pow_poly = poly_item1 @ i_pow_poly
            first_poly_list.append(i_pow_poly)
        
        y = 0.
        for i in range(self.k+1):
            filter_poly = first_poly_list[self.k - i]
            for j in range(i):
                filter_poly = poly_item2 @ filter_poly
            y += (comb(self.k, i) / (2**self.k)) * filter_param[i] * filter_poly

        return y