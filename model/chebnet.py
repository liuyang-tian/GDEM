import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim

from torch_geometric.nn.inits import zeros

from copy import deepcopy
from sklearn.metrics import accuracy_score
from utils import normalize_adj_to_sparse_tensor


class ChebNet(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        nlayers=2,
        k=2,
        lr=0.01,
        weight_decay=5e-4,
        dropout=0.5,
        with_relu=True,
        with_bias=True,
        with_bn=False
    ):
        super(ChebNet, self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        
        self.dropout = dropout
        
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.with_bn = with_bn
        
        self.layers = nn.ModuleList([])
        if nlayers == 1:
            self.layers.append(
                ChebConvolution(num_features, num_classes, k=k, with_bias=with_bias)
            )
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(
                ChebConvolution(num_features, hidden_dim, k=k, with_bias=with_bias)
            )
            for i in range(nlayers - 2):
                self.layers.append(
                    ChebConvolution(hidden_dim, hidden_dim, k=k, with_bias=with_bias)
                )
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(
                ChebConvolution(hidden_dim, num_classes, k=k, with_bias=with_bias)
            )
            
        self.reset_parameter()

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for ix, conv in enumerate(self.layers):
            x = conv(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)


    def reset_parameter(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

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

        idx_val = data.idx_val
        idx_test = data.idx_test
        y_full = data.y_full
        y_val = (data.y_val).cpu().numpy()
        y_test = (data.y_test).cpu().numpy()

        if verbose:
            print("=== training chebnet model ===")

        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_acc_val = 0
        y_val = (y_full[idx_val]).cpu().numpy()
        y_test = (y_full[idx_test]).cpu().numpy()

        lr = self.lr
        for i in range(epochs):
            self.train()

            optimizer.zero_grad()

            output = self.forward(x_syn, adj_syn)
            loss_train = F.nll_loss(output, y_syn)

            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print("Epoch {}, training loss: {}".format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(x_real, adj)
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

    @torch.no_grad()
    def predict(self, x, adj):
        self.eval()
        return self.forward(x, adj)


class ChebConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """
    def __init__(
        self, in_features, out_features, k=2, with_bias=True, single_param=False):
        """set single_param to True to alleivate the overfitting issue"""
        super(ChebConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linears = torch.nn.ModuleList(
            [nn.Linear(in_features, out_features, bias=False) for _ in range(k)]
        )
        if with_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.single_param = single_param
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.linears:
            nn.init.xavier_uniform_(lin.weight.data.T)
            if lin.bias is not None:
                lin.bias.data.zero_()                     
        zeros(self.bias)

    def forward(self, x, adj, size=None):
        """ Graph Convolutional Layer forward function
        """
        Tx_0 = x
        Tx_1 = x
        output = self.linears[0](Tx_0)

        if len(self.linears) > 1:
            Tx_1 = adj @ x
            
            if self.single_param:
                output = output + self.linears[0](Tx_1)
            else:
                output = output + self.linears[1](Tx_1)

        for lin in self.linears[2:]:
            if self.single_param:
                lin = self.linears[0]
            Tx_2 = adj @ Tx_1
            Tx_2 = 2.0 * Tx_2 - Tx_0
            output = output + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
