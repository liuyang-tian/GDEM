import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

import torch_sparse

from copy import deepcopy
from sklearn.metrics import accuracy_score

from utils import normalize_adj_to_sparse_tensor


class ChebNetII(nn.Module):
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
        super(ChebNetII, self).__init__()

        self.lr = lr
        self.lr_conv = lr_conv
        self.weight_decay = weight_decay
        self.wd_conv = wd_conv

        self.dropout = dropout
        self.dprate = dprate
        
        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)


        # chebynodes_vals
        cheby_nodes=[]
        for j in range(0, k+1):
            x_j = math.cos((k-j+0.5)*math.pi/(k+1))
            cheby_nodes.append(x_j)
        cheby_nodes_val=[]
        for x_j in cheby_nodes:
            chebynode_val=[]
            for j in range(0, k+1):
                if j == 0:
                    chebynode_val.append([1])
                elif j == 1:
                    chebynode_val.append([x_j])
                else:
                    item = 2 * x_j * chebynode_val[j-1][0] - chebynode_val[j-2][0]
                    chebynode_val.append([item])
            chebynode_val = torch.Tensor(chebynode_val)
            cheby_nodes_val.append(chebynode_val)
        chebynodes_vals=torch.cat(cheby_nodes_val, dim=1).cuda()
        
        self.conv = ChebConvII(k, chebynodes_vals)
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

        if self.dprate == 0.0:
            x = self.conv(x, adj)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.conv(x, adj)

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
            print("=== training chebnetII model ===")

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


class ChebConvII(nn.Module):
    def __init__(self, k, chebynodes_vals):
        super(ChebConvII, self).__init__()
        self.k = k
        self.chebynodes_vals = chebynodes_vals
        self.filter_param = nn.Parameter(torch.Tensor(k+1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.filter_param.data.fill_(1.0)
   
    def forward(self, x, laplacian_matrix):
        filter_param = F.relu(self.filter_param)
        filter_param = self.chebynodes_vals @ filter_param    # (k+1)*1
        filter_param = 2 * filter_param / (self.k + 1)
        filter_param[0] = filter_param[0] / 2
        
        if isinstance(laplacian_matrix, torch_sparse.SparseTensor):
            row = laplacian_matrix.storage.row()
            col = laplacian_matrix.storage.col()
            edge_weight = laplacian_matrix.storage.value()
            scaled_laplacian = SparseTensor(row=row, col=col, value=-edge_weight, sparse_sizes=(x.shape[0], x.shape[0])).coalesce()
        else:
            scaled_laplacian = -laplacian_matrix

        
        cheby_poly = self.get_cheby_poly(x, scaled_laplacian)
        cheby_poly = torch.stack(cheby_poly)    # (k+1)*N*d

        filter_param = filter_param.unsqueeze(dim=2) * cheby_poly    # (k+1)*N*d
        y = filter_param.sum(dim=0)    # N*d

        return y
    

    def get_cheby_poly(self, p0, multiplicator):
        # cheby polynomial for Laplacian
        Cheby_poly = []
        for i in range(0, self.k+1):
            if i == 0:
                Cheby_poly.append(p0)
            elif i == 1:
                Cheby_poly.append(multiplicator @ p0)
            else:
                x_i = 2 * (multiplicator @ Cheby_poly[i-1]) - Cheby_poly[i-2]
                Cheby_poly.append(x_i)
                  
        return Cheby_poly
