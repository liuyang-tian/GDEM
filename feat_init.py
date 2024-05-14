import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from sklearn.metrics import accuracy_score

from dataset import get_dataset, Transd2Ind, DataGraphSAINT
from utils import seed_everything
    
class MLP(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        dropout):
        
        super(MLP, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_dim), nn.Linear(hidden_dim, num_classes)])
        self.reset_parameter()
    
    def reset_parameter(self):
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight.data)
            if lin.bias is not None:
                lin.bias.data.zero_()
    
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)


class GraphAgent:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        
        self.n_syn = int(len(data.idx_train) * args.reduction_rate)
        print(self.n_syn)
        self.d = (data.x_train).shape[1]
        self.num_classes = data.num_classes

        self.x_syn = nn.Parameter(torch.FloatTensor(self.n_syn, self.d).cuda())
        self.y_syn = torch.LongTensor(self.generate_labels_syn(data.y_full[data.idx_train], args.reduction_rate)).cuda()
        
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_syn.data)
    
    
    def train(self):
        args = self.args
        data = self.data
        
        optimizer_feat = torch.optim.Adam(
            [self.x_syn], lr=args.lr_feat, weight_decay=args.wd_feat
        )

        model = self.mlp_trainer(args, data, verbose=False)
        model.train()
        
        for i in range(args.epoch):
            output = model(self.x_syn)
            loss = F.nll_loss(output, self.y_syn)
            
            optimizer_feat.zero_grad()
            loss.backward()
            optimizer_feat.step()

        x_syn, y_syn = self.x_syn.detach(), self.y_syn
        
        dir = f"./initial_feat/{args.dataset}"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        torch.save(
            x_syn, f"{dir}/x_init_{args.reduction_rate}_{args.expID}.pt",
        )
        
        acc = self.test_with_val(
            x_syn,
            y_syn
        )

        return acc

    def test_with_val(self, x_syn, y_syn):
        args = self.args
        data = self.data
        x_full = data.x_full
        y_full = data.y_full
        
        idx_train = data.idx_train
        idx_val = data.idx_val
        idx_test = data.idx_test
        
        model = MLP(num_features=self.d, num_classes=self.num_classes, hidden_dim=args.hidden_dim, dropout=args.dropout).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc_val = 0
        y_train = (y_full[idx_train]).cpu().numpy()
        y_val = (y_full[idx_val]).cpu().numpy()
        y_test = (y_full[idx_test]).cpu().numpy()

        epochs = 2000
        lr = args.lr
        for i in range(epochs):
            if i == epochs // 2 and i > 0:
                lr = lr * 0.1
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )

            model.train()
            optimizer.zero_grad()

            output = model.forward(x_syn)
            loss_train = F.nll_loss(output, y_syn)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output = model.forward(data.x_val)
                loss_val = F.nll_loss(output, y_full[idx_val])

                pred = output.max(1)[1]
                pred = pred.cpu().numpy()
                
                acc_val = accuracy_score(y_val, pred)
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    weights = deepcopy(model.state_dict())

        model.load_state_dict(weights)
        
        model.eval()
        output = model.forward(x_full)
        loss_test = F.nll_loss(output[idx_test], y_full[idx_test])

        pred = output.max(1)[1].cpu().numpy()
    
        acc_train = accuracy_score(y_train, pred[idx_train])
        acc_val = accuracy_score(y_val, pred[idx_val])
        acc_test = accuracy_score(y_test, pred[idx_test])

        print(
            f"Test set results: test_loss= {loss_test.item():.4f}, train_acc= {acc_train:.4f}, val_acc= {acc_val:.4f}, test_acc= {acc_test:.4f}\n"
        )
        
        return acc_test

    
    
    def mlp_trainer(self, args, data, verbose):
        x_full = data.x_full
        y_full = data.y_full
        
        idx_train = data.idx_train
        idx_val = data.idx_val
        idx_test = data.idx_test
        
        model = MLP(num_features=x_full.shape[1], num_classes=data.num_classes, hidden_dim=args.hidden_dim, dropout=args.dropout).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_acc_val = 0
        y_train = (y_full[idx_train]).cpu().numpy()
        y_val = (y_full[idx_val]).cpu().numpy()
        y_test = (y_full[idx_test]).cpu().numpy()
        
        lr = args.lr
        for i in range(args.epoch_mlp):
            if i == args.epoch_mlp // 2 and i > 0:
                lr = lr * 0.1
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
            
            model.train()
            optimizer.zero_grad()

            output = model.forward(data.x_train)
            loss_train = F.nll_loss(output, y_full[idx_train])

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output = model.forward(data.x_val)
                loss_val = F.nll_loss(output, y_full[idx_val])

                pred = output.max(1)[1]
                pred = pred.cpu().numpy()
                acc_val = accuracy_score(y_val, pred)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    weights = deepcopy(model.state_dict())
            
        model.load_state_dict(weights)        

        if verbose:
            model.eval()
            output = model.forward(x_full)
            loss_test = F.nll_loss(output[idx_test], y_full[idx_test])
            pred = output.max(1)[1].cpu().numpy()
            
            acc_train = accuracy_score(y_train, pred[idx_train])
            acc_val = accuracy_score(y_val, pred[idx_val])
            acc_test = accuracy_score(y_test, pred[idx_test])

            print(
                f"Test set results: test_loss= {loss_test.item():.4f}, train_acc= {acc_train:.4f}, val_acc= {acc_val:.4f}, test_acc= {acc_test:.4f}"
            )
        
        return model
    
    
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

        self.num_class_dict = num_class_dict
        return y_syn
    


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=1, help="gpu id")
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--config", type=str, default='./config/config_init.json')

parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--expID", type=int, default=0)

parser.add_argument("--dataset", type=str, default="citeseer")  # [citeseer, pubmed, ogbn-arxiv, flickr, reddit, squirrel, twitch-gamer]
parser.add_argument("--reduction_rate", type=float, default=0.5)
parser.add_argument("--normalize_features", type=bool, default=True)

parser.add_argument("--hidden_dim", type=int, default=256)

args = parser.parse_args([])

with open(args.config, "r") as config_file:
    config = json.load(config_file)

if args.dataset in config:
    config = config[args.dataset]

for key, value in config.items():
    setattr(args, key, value)

torch.cuda.set_device(args.gpu_id)
seed_everything(args.seed)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)

else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full)

data = data.cuda()

accs = []
for ep in range(args.runs):
    args.expID = ep
    agent = GraphAgent(args=args, data=data)
    acc = agent.train()
    accs.append(acc)

print(accs)
mean_acc = np.mean(accs)
std_acc = np.std(accs)
print(f"Mean ACC: {mean_acc}\t Std: {std_acc}")

