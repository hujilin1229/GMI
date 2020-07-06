import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from layers.layers import GCN
from models.gmi_pooling import GMI_Pool

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        # (n_in, n_h, ratio=0.8, sample=False, sparse=False, sl=True):
        self.pool1 = GMI_Pool(self.num_features, self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl)
        self.pool2 = GMI_Pool(self.nhid, self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data, neg_num, samp_bias1, samp_bias2):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        original_x = x
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # inputs: x, edge_index, edge_attr, batch, h, neg_num, samp_bias1, samp_bias2
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch, original_x,
                                                     neg_num, samp_bias1, samp_bias2)

        original_x = x
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch, original_x,
                                                     neg_num, samp_bias1, samp_bias2)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
