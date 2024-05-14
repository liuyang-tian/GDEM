import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch_sparse

from copy import deepcopy
from sklearn.metrics import accuracy_score
from utils import normalize_adj_to_sparse_tensor


class SGC(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        nlayers=2,
        lr=0.01,
        weight_decay=5e-4,
        with_relu=True,
        with_bias=True
    ):

        super(SGC, self).__init__()

        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        
        self.nlayers = nlayers
        self.with_relu = with_relu
        self.with_bias = with_bias

        self.conv = GraphConvolution(num_features, num_classes, with_bias=with_bias)


    def forward(self, x, adj):
        weight = self.conv.weight
        bias = self.conv.bias
        x = torch.mm(x, weight)
        
        for i in range(self.nlayers):
            x = adj @ x

        x = x + bias
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
            print("=== training sgc model ===")

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
    
class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data.T)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
