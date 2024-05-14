import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import torch
from copy import deepcopy
from sklearn.metrics import accuracy_score
from scipy.special import comb

import matplotlib.pyplot as plt
import random
import numpy as np
import math
from utils import normalize_adj_to_sparse_tensor


class GPRGNN(nn.Module):
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
        dprate,
        alpha=0.1,
        init_method="PPR",
        Gamma=None,
    ):
        super(GPRGNN, self).__init__()
        self.lr = lr
        self.lr_conv = lr_conv
        self.wd = weight_decay
        self.wd_conv = wd_conv

        self.dropout = dropout
        self.dprate = dprate

        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)
        self.conv = GPRConv(init_method, alpha, k, Gamma)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.feat_encoder.weight.data)
        if self.feat_encoder.bias is not None:
            self.feat_encoder.bias.data.zero_()
        
        nn.init.xavier_uniform_(self.final_encoder.weight.data)
        if self.final_encoder.bias is not None:
            self.final_encoder.bias.data.zero_()
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.feat_encoder(x))
        x = F.dropout(x, self.dropout, self.training)
        x = self.final_encoder(x)

        # for conv in self.layers:
        if self.dprate == 0.0:
            x = self.conv(x, adj)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.conv(x, adj)

        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def predict(self, x, adj):
        self.eval()
        return self.forward(x, adj)

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
            print("=== training gprgnn model ===")

        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.feat_encoder.parameters(),
                    "weight_decay": self.wd,
                    "lr": self.lr,
                },
                {
                    "params": self.final_encoder.parameters(),
                    "weight_decay": self.wd,
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

    def conv_visualize(self, eigenvals):
        eigenvals = eigenvals.cpu().numpy()

        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(1, 1, 1)

        xs = 1 - eigenvals

        for layer in self.layers:
            conv_vals = 1.0
            conv_vals = conv_vals * layer.conv_val(eigenvals)
        ys = conv_vals
        color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
        ax.plot(xs, ys, color=color, alpha=0.8)
        ax.set_xlabel("frequency", size=20)
        ax.set_ylabel("feature chanel", size=20)
        plt.show()


class GPRConv(nn.Module):
    def __init__(self, init_method, alpha, k, Gamma):
        super(GPRConv, self).__init__()
        self.k = k

        assert init_method in ["SGC", "PPR", "NPPR", "Random", "WS"]
        if init_method == "SGC":
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            gamma = 0.0 * np.ones(k + 1)
            gamma[alpha] = 1.0
        elif init_method == "PPR":
            # PPR-like
            gamma = alpha * (1 - alpha) ** np.arange(k + 1)
            gamma[-1] = (1 - alpha) ** k
        elif init_method == "NPPR":
            # Negative PPR
            gamma = (alpha) ** np.arange(k + 1)
            gamma = gamma / np.sum(np.abs(gamma))
        elif init_method == "Random":
            # Random
            bound = np.sqrt(3 / (k + 1))
            gamma = np.random.uniform(-bound, bound, k + 1)
            gamma = gamma / np.sum(np.abs(gamma))
        elif init_method == "WS":
            # Specify Gamma
            gamma = Gamma

        self.gamma = nn.Parameter(torch.Tensor(gamma))
        
    def forward(self, x, adj):
        y = 0.0

        for step in range(self.k + 1):
            if step == 0:
                layer_conv_output = x
            else:
                layer_conv_output = adj @ layer_conv_output

            y += self.gamma[step] * layer_conv_output

        return y

    def conv_val(self, eigenvals):
        gamma = self.gamma.detach().cpu().numpy()
        conv_base = np.ones_like(eigenvals)
        conv_val = gamma[0] * conv_base
        for k in range(self.k):
            conv_base = conv_base * eigenvals
            conv_val += gamma[k + 1] * conv_base
        return conv_val

