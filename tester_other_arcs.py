import os
import numpy as np
import torch
import torch.nn.functional as F

from utils import normalize_adj_to_sparse_tensor
from sklearn.metrics import accuracy_score

from model.sgc import SGC
from model.gcn import GCN
from model.appnp import APPNP
from model.chebnet import ChebNet
from model.chebnetII import ChebNetII
from model.bernnet import BernNet
from model.gprgnn import GPRGNN


class Evaluator:
    def __init__(self, data, args):
        self.data = data
        self.args = args
        n = int(len(data.idx_train) * args.reduction_rate)
        self.n_syn = n
        self.d = (data.x_full).shape[1]
        self.y_syn = torch.LongTensor(self.generate_labels_syn(data)).cuda()

    def generate_labels_syn(self, data):
        from collections import Counter

        y_train = (data.y_train).cpu().numpy()
        counter = Counter(y_train)
        num_class_dict = {}
        n = len(y_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [
                    len(labels_syn),
                    len(labels_syn) + num_class_dict[c],
                ]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [
                    len(labels_syn),
                    len(labels_syn) + num_class_dict[c],
                ]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def get_syn_data(self, expID):
        args = self.args

        dir = f"./saved_ours/{args.dataset}-{args.reduction_rate}"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        eigenvals_syn = torch.load(
            f"{dir}/eigenvals_syn_{expID}.pt", map_location='cpu'
        )
        eigenvecs_syn = torch.load(
            f"{dir}/eigenvecs_syn_{expID}.pt", map_location='cpu'
        )
        x_syn = torch.load(
            f"{dir}/feat_{expID}.pt", map_location='cpu'
        )

        x_syn = x_syn
        L_syn = eigenvecs_syn @ torch.diag(eigenvals_syn) @ eigenvecs_syn.T

        return x_syn, L_syn

    def test(self, test_model, expID, verbose=True):
        args = self.args
        data = self.data

        x_syn, L_syn = self.get_syn_data(expID)
        x_syn, L_syn = x_syn.cuda(), L_syn.cuda()
        y_syn = self.y_syn

        adj_syn = torch.eye(self.n_syn).cuda() - L_syn

        print("======= testing %s" % test_model)
        if test_model == "SGC":
            model = SGC(
                num_features=self.d,
                num_classes=data.num_classes,
                nlayers=args.nlayers,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ).cuda()
        elif test_model == "GCN":
            model = GCN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
            ).cuda()
        elif test_model == "ChebNet":
            model = ChebNet(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                k=args.k,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
            ).cuda()
        elif test_model == "APPNP":
            model = APPNP(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                alpha=args.alpha,
                k=args.k,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
            ).cuda()
        elif test_model == "ChebNetII":
            model = ChebNetII(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr,
                lr_conv=args.lr_conv,
                weight_decay=args.weight_decay,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate
            ).cuda()
        elif test_model == "BernNet":
            model = BernNet(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr,
                lr_conv=args.lr_conv,
                weight_decay=args.weight_decay,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate,
            ).cuda()
        elif test_model == "GPRGNN":
            model = GPRGNN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                alpha=args.alpha,
                k=args.k,
                lr=args.lr,
                lr_conv=args.lr_conv,
                weight_decay=args.weight_decay,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate,
            ).cuda()
                    
        model.fit_with_val(
            x_syn,
            y_syn,
            adj_syn,
            data,
            args.epochs,
            verbose,
        )

        model.eval()
        
        y_full = data.y_full
        idx_test = data.idx_test
        y_test = (y_full[idx_test]).cpu().numpy()

        x_real_test = data.x_full
        adj_real_test = data.adj_full

        adj_real_test = normalize_adj_to_sparse_tensor(adj_real_test)
        if test_model == "BernNet":
            laplacian_mat, poly_item_test = model.get_poly_item(adj=adj_real_test, num_nodes=x_real_test.shape[0])
            output = model.predict(x_real_test, laplacian_mat, poly_item_test)
        else:
            output = model.predict(x_real_test, adj_real_test)
        pred = output.max(1)[1]
        pred = pred.cpu().numpy()

        loss_test = F.nll_loss(output[idx_test], y_full[idx_test])
        acc_test = accuracy_score(y_test, pred[idx_test])

        if verbose:
            print(
                "Test full set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()),
            )
        return acc_test

    def train(self, verbose=True):
        args = self.args
        runs = args.runs
        test_model = args.test_model

        res = []
        for ep in range(runs):
            expID = ep
            res.append(self.test(test_model=test_model, expID=expID, verbose=verbose))
        res = np.array(res)
        return res
