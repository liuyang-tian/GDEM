import argparse
import torch

from utils import seed_everything
from dataset import get_dataset, Transd2Ind, DataGraphSAINT

from tester_other_arcs import Evaluator


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--expID", type=int, default=0)

parser.add_argument('--dataset', type=str, default='citeseer')
parser.add_argument('--reduction_rate', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=2000)

parser.add_argument('--test_model', type=str, default='GCN')
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)

args = parser.parse_args()
print(args)

torch.cuda.set_device(args.gpu_id)
seed_everything(args.seed)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)

else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full)

data = data.cuda()

agent = Evaluator(data, args)
outs = agent.train()
print(outs)
print("Test Mean Accuracy:", repr([outs.mean(0), outs.std(0)]))

