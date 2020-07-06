import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils import process
from layers import AvgNeighbor, Discriminator, Discriminator_tg
from torch_geometric.nn import GINConv
from torch_geometric.nn.pool import sag_pool
from torch_scatter import scatter_add, scatter_max
from torch_geometric.data import Data
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_sparse import spspmm, coalesce
from layers.sparse_softmax import Sparsemax

class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n)
        value.fill_(0)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class GMI_Pool(nn.Module):
    def __init__(self, n_in, n_h, ratio=0.8, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2):
        super(GMI_Pool, self).__init__()

        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl

        # self.gcn1 = GCN(n_in, n_h, activation)  # if on citeseer and pubmed, the encoder is 1-layer GCN, you need to modify it
        # self.gcn2 = GCN(n_h, n_h, activation)

        # self.act = nn.PReLU() if activation == 'prelu' else activation

        # self.gcn1_mlp = nn.Sequential(nn.Linear(n_in, n_h), self.act)
        # self.gcn1_1 = GINConv(self.gcn1_mlp, eps=0.0)
        #
        # self.gcn2_mlp = nn.Sequential(nn.Linear(n_h, n_h), self.act)
        # self.gcn2_1 = GINConv(self.gcn2_mlp, eps=0.0)

        # self.in_channels = n_in
        self.negative_slop = negative_slop
        self.lamb = lamb
        self.att = Parameter(torch.Tensor(1, n_h * 2))
        nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()

        self.disc1 = Discriminator_tg(n_in, n_h)
        # self.disc2 = Discriminator_tg(n_h, n_h)
        # self.avg_neighbor = AvgNeighbor()
        # self.prelu = nn.PReLU()
        # self.sigm = nn.Sigmoid()

        self.neighbor_augment = TwoHopNeighborhood()

    def forward(self, x, edge_index, edge_attr, batch, h, neg_num, samp_bias1, samp_bias2):
        """

        :param x: node feature after convolution
        :param edge_index:
        :param edge_attr:
        :param batch:
        :param h: node feature before convolution
        :param neg_num:
        :param samp_bias1:
        :param samp_bias2:
        :return:
        """

        # I(h_i; x_i)
        res_mi_pos, res_mi_neg = self.disc1(x, h, process.negative_sampling_tg(batch, neg_num), samp_bias1, samp_bias2)
        mi_jsd_score = process.sp_func(res_mi_pos) + process.sp_func(torch.mean(res_mi_neg, dim=1))

        # Graph Pooling
        original_x = x
        perm = topk(mi_jsd_score, self.ratio, batch)
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=mi_jsd_score.size(0))

        # Discard structure learning layer, directly return
        if self.sl is False:
            return x, induced_edge_index, induced_edge_attr, batch

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each pair of nodes is time consuming.
            # To accelerate this process, we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.
            k_hop = 3
            if edge_attr is None:
                edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)

            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            for _ in range(k_hop - 1):
                hop_data = self.neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, perm, num_nodes=mi_jsd_score.size(0))

            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            adj[row, col] = weights
            new_edge_index, weights = dense_to_sparse(adj)
            row, col = new_edge_index
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del adj
            torch.cuda.empty_cache()
        else:
            # Learning the possible edge weights between each pair of nodes in the pooled subgraph, relative slower.
            if edge_attr is None:
                induced_edge_attr = torch.ones((induced_edge_index.size(1),), dtype=x.dtype,
                                               device=induced_edge_index.device)
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
            cum_num_nodes = num_nodes.cumsum(dim=0)
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            # Construct batch fully connected graph in block diagonal matirx format
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[idx_i:idx_j, idx_i:idx_j] = 1.0
            new_edge_index, _ = dense_to_sparse(adj)
            row, col = new_edge_index

            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            adj[row, col] = weights
            induced_row, induced_col = induced_edge_index

            adj[induced_row, induced_col] += induced_edge_attr * self.lamb
            weights = adj[row, col]
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del adj
            torch.cuda.empty_cache()

        return x, new_edge_index, new_edge_attr, batch

        # h_neighbor = self.prelu(self.avg_neighbor(h_w, adj_ori))
        # """FMI (X_i consists of the node i itself and its neighbors)"""
        # # I(h_i; x_i)
        # res_mi_pos, res_mi_neg = self.disc1(h_2, seq1, process.negative_sampling(adj_ori, neg_num), samp_bias1, samp_bias2)
        #
        #
        #
        # # I(h_i; x_j) node j is a neighbor
        # res_local_pos, res_local_neg = self.disc2(h_neighbor, h_2, process.negative_sampling(adj_ori, neg_num), samp_bias1, samp_bias2)
        # """I(w_ij; a_ij)"""
        # adj_rebuilt = self.sigm(torch.mm(torch.squeeze(h_2), torch.t(torch.squeeze(h_2))))
        #
        # return res_mi_pos, res_mi_neg, res_local_pos, res_local_neg, adj_rebuilt

    # detach the return variables
    def embed(self, seq, adj):
        h_1, _ = self.gcn1(seq, adj)
        h_2, _ = self.gcn2(h_1, adj)

        return h_2.detach()

