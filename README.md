# Graph Distillation with Eigenbasis Matching (GDEM)

This is the Pytorch implementation for ["Graph Distillation with Eigenbasis Matching"](https://arxiv.org/pdf/2310.09202).

![](https://github.com/liuyang-tian/GDEM/blob/main/EM.png)

### Requirements
```
deeprobust==0.2.9
gdown==4.7.3
networkx==3.2.1
numpy==1.26.3
ogb==1.3.6
pandas==2.1.4
scikit-learn==1.3.2
scipy==1.11.4
torch==2.1.2
torch_geometric==2.4.0
torch-sparse==0.6.18
```

## Download Datasets
For Citeseer Pubmed and Squirrel, the code will directly download them.
For Reddit, Flickr, and Ogbn-arXiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). They are available on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (the links are provided by GraphSAINT team). 
For twitch-gamer, you can access it at [Twitch-Gamer](https://drive.google.com/file/d/11Xas4r6oBvDzDzqHT-cEd35nX9X3q3yf/view?usp=sharing).
Download the files and unzip them to `data` at the root directory. 

## Instructions

(1) Run preprocess.py to preprocess the dataset and conduct the spectral decomposition.

(2) Initialize node features of the synthetic graph by running feat_init.py.

(3) Distill the synthetic graph by running distill.py.

(4) you can evaluate the cross-architecture generalization performance of the synthetic graph on various GNNs (GCN, SGC, PPNP, ChebyNet, BernNet, GPR-GNN) by running test_other_arcs.py.

## Cite
Welcome to kindly cite our work with:
```
@inproceedings{liugraph,
  title={Graph Distillation with Eigenbasis Matching},
  author={Liu, Yang and Bo, Deyu and Shi, Chuan},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
