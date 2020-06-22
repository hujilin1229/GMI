import torch
import torch.nn as nn
from utils import process
from layers import GCN, AvgNeighbor, Discriminator, Discriminator_tg
from torch_geometric.nn import GINConv
from torch_geometric.nn.pool import sag_pool
from torch_scatter import scatter_add, scatter_max


class GMI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GMI, self).__init__()
        self.gcn1 = GCN(n_in, n_h, activation)  # if on citeseer and pubmed, the encoder is 1-layer GCN, you need to modify it
        self.gcn2 = GCN(n_h, n_h, activation)

        self.act = nn.PReLU() if activation == 'prelu' else activation

        self.gcn1_mlp = nn.Sequential(nn.Linear(n_in, n_h), self.act)
        self.gcn1_1 = GINConv(self.gcn1_mlp, eps=0.0)

        self.gcn2_mlp = nn.Sequential(nn.Linear(n_h, n_h), self.act)
        self.gcn2_1 = GINConv(self.gcn2_mlp, eps=0.0)

        self.disc1 = Discriminator_tg(n_in, n_h)
        self.disc2 = Discriminator_tg(n_h, n_h)
        self.avg_neighbor = AvgNeighbor()
        self.prelu = nn.PReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, data, neg_num, samp_bias1, samp_bias2):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        h_1, h_w = self.gcn1(seq1, adj)
        h_2, _ = self.gcn2(h_1, adj)

        h_1_1 = self.gcn1_1(x, edge_index)
        h_2_1 = self.gcn2_1(h_1_1, edge_index)

        # I(h_i; x_i)
        res_mi_pos, res_mi_neg = self.disc1(h_2_1, x, process.negative_sampling_tg(batch, neg_num), samp_bias1, samp_bias2)


        h_neighbor = self.prelu(self.avg_neighbor(h_w, adj_ori))
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_pos, res_mi_neg = self.disc1(h_2, seq1, process.negative_sampling(adj_ori, neg_num), samp_bias1, samp_bias2)



        # I(h_i; x_j) node j is a neighbor
        res_local_pos, res_local_neg = self.disc2(h_neighbor, h_2, process.negative_sampling(adj_ori, neg_num), samp_bias1, samp_bias2)
        """I(w_ij; a_ij)"""
        adj_rebuilt = self.sigm(torch.mm(torch.squeeze(h_2), torch.t(torch.squeeze(h_2))))
        
        return res_mi_pos, res_mi_neg, res_local_pos, res_local_neg, adj_rebuilt

    # detach the return variables
    def embed(self, seq, adj):
        h_1, _ = self.gcn1(seq, adj)
        h_2, _ = self.gcn2(h_1, adj)

        return h_2.detach()

