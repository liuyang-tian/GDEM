import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from copy import deepcopy
from sklearn.metrics import accuracy_score
from utils import normalize_adj_to_sparse_tensor


class APPNP(nn.Module):

    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        k,
        lr=0.01,
        weight_decay=5e-4,
        dropout=0.5,
        alpha = 0.1
        ):
        super(APPNP, self).__init__()
        
        self.alpha = alpha
        self.k = k
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        
        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)
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
        
        h = x
        for i in range(self.k):
            x = adj @ x
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

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

        idx_val = data.idx_val
        idx_test = data.idx_test
        y_full = data.y_full
        y_val = (data.y_val).cpu().numpy()
        y_test = (data.y_test).cpu().numpy()

        if verbose:
            print("=== training appnp model ===")

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
