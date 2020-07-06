import torch
import torch.nn as nn

class Discriminator_tg(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator_tg, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_c, h_pl, sample_tensor, s_bias1=None, s_bias2=None):
        """
        calculate discriminator results

        :param h_c: num_nodes x n_h2
        :param h_pl: num_nodes x n_h1
        :param sample_tensor: negative sampling
        :param s_bias1: bias1
        :param s_bias2: bias2
        :return: num_nodes, num_nodes x num_neg
        """

        x = self.f_k(h_pl, h_c)
        # print(x.shape)
        sc_1 = torch.squeeze(x, 1)
        sc_1 = self.act(sc_1)

        # h_pl, h_c: num_nodes x num_feature
        # num_nodes x num_neg x num_feature
        h_mi = h_pl[sample_tensor]
        h_c_rep = torch.unsqueeze(h_c, dim=1).repeat([1, sample_tensor.size(1), 1])
        sc_2_stack = torch.squeeze(self.f_k(h_mi, h_c_rep), 2)
        sc_2 = self.act(sc_2_stack)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        return sc_1, sc_2