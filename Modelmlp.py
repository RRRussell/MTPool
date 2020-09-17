import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils import euclidean_dist, normalize, output_conv_size, dump_embedding
import numpy as np
from layer import Attention,dense_diff_pool
from gnn_layer import *

class TapNet(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
                 use_att=False, use_ss=False, use_metric=False, use_muse=False, use_lstm=False, use_cnn=True):
        super(TapNet, self).__init__()
        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_muse = use_muse
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_group, self.rp_dim = rp_params

        kernel_ = [2,3,5]
        channel = 8
        self.c1 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[0]), stride=1)
        self.c2 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[1]), stride=1)
        self.c3 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[2]), stride=1)
        # self.c4 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[3]), stride=1)
        d = (len(kernel_) * (51) - sum(kernel_) + len(kernel_)) * channel

        self.decoder = 'GNN'
        self.pm = 'diffpool'
        self.n_pooling = 3
        self.hid1 = 512
        self.hid2 = 128

        if self.decoder == 'GIN':
            ginnn = nn.Sequential(
                nn.Linear(d,self.hid1),
                nn.Tanh(),
                # nn.Linear(args.hid1, args.hid2),
                # nn.ReLU(True),
                nn.Linear(self.hid1,self.hid2),
                # nn.ReLU(True)
            )
            self.gin = DeGINConv(ginnn)

        if self.decoder == 'GNN':
        # self.gnn0 = DenseGraphConv(d, h0)
            self.gnn1 = DenseGraphConv(d, self.hid1)
            self.gnn2 = DenseGraphConv(self.hid1, self.hid2)

            self.hid3 = 64
            self.gnn3 = DenseGraphConv(self.hid2, self.hid3)
            self.hid4 = 16
            self.gnn4 = DenseGraphConv(self.hid3, self.hid4)

        if self.pm == 'attention':
            self.pooling = Attention(input_dim=self.hid2,
                                     att_times=1,
                                     att_num=1,
                                     att_style='ntn_16',
                                     att_weight=True)

        self.hid_mlp1 = 6
        # self.hid_mlp2 = 32

        self.mlp = nn.Sequential(
            nn.Linear(self.hid4,self.hid_mlp1),
        )



        subgraph_size = 8
        node_dim = 40
        self.device = torch.device('cuda:0')
        tanhalpha = 3

        self.gc = graph_constructor(24, subgraph_size, node_dim, self.device, alpha=tanhalpha,
                                    static_feat=None)
        # print(self.use_rp, self.rp_group, self.rp_dim)

        A = np.ones((24, 24), np.int8)
        A = A / np.sum(A, 0)
        A_new = np.zeros((360, 24, 24), dtype=np.float32)
        for i in range(360):
            A_new[i, :, :] = A
        self.A = torch.from_numpy(A_new).cuda()

        for i in range(180):
            A = np.load('/home/jiangnanyida/Documents/MTS/tapnet-master/dataset/NATOPS/X_train.npy')[i].T
            # print(A[0])
            d = {}
            for i in range(A.shape[0]):
                d[i] = A[i]

            df = pd.DataFrame(d)
            df_corr = df.corr()
            self.A[i] = torch.from_numpy(df_corr.to_numpy()/np.sum(df_corr.to_numpy(), 0)).cuda()

    def forward(self, input,test = False):
        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension
        c = x
        a1 = self.c1(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        a2 = self.c2(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        a3 = self.c3(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        # a4 = self.c4(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        x = F.relu(torch.cat([a1, a2,a3], 2))



        # idx = [0]
        # for i in range(1, 24):
        #     idx.append(i)
        # idx = torch.tensor(idx).to(self.device)
        # adp = self.gc(idx, x)
        #
        # g = F.normalize(adp, p=1, dim=1)
        # self.A = g
        if test == True:
            A = np.ones((24, 24), np.int8)
            A = A / np.sum(A, 0)
            A_new = np.zeros((360, 24, 24), dtype=np.float32)
            for i in range(360):
                A_new[i, :, :] = A
            self.A = torch.from_numpy(A_new).cuda()

            for i in range(180):
                A = np.load('/home/jiangnanyida/Documents/MTS/tapnet-master/dataset/NATOPS/X_test.npy')[i].T
                # print(A[0])
                d = {}
                for i in range(A.shape[0]):
                    d[i] = A[i]

                df = pd.DataFrame(d)
                df_corr = df.corr()
                self.A[i] = torch.from_numpy(df_corr.to_numpy() / np.sum(df_corr.to_numpy(), 0)).cuda()

        if self.decoder == 'GIN':
            # x3 = F.relu(self.gin(x, self.A))
            x3 = self.gin(x, self.A)
            x3 = x3.squeeze()

        if self.decoder == 'GNN':
        # x0 = F.relu(self.gnn0(x_conv,self.A))
            x1 = F.relu(self.gnn1(x,self.A))
            # x1 = self.dropout(x1)
            # x2 = self.gnn2(x1,self.A)

            # x3 = self.gnn3(x2,self.A)
            x3 = x1.squeeze()

        x = x3

        x_conv = []
        # x = x.permute(0, 2, 1)

        if self.pm == 'attention':
            for i in range(x.shape[0]):
                x_conv.append(self.pooling(x[i]))

            x_conv = torch.stack(x_conv).view(x.shape[0],-1)

        ##diffpool
        link_loss = 0
        if self.pm == 'diffpool':

            # batch_size, num_nodes, channels, num_clusters = (360, 24, 20, 12)
            # for i in range(self.n_pooling):
            #     if i <self.n_pooling-1:
            #         batch_size, num_nodes, channels, num_clusters = (360, num_nodes if i == 0 else num_clusters, 16, int(num_clusters/self.n_pooling*(self.n_pooling-i)))
            #         s = torch.randn((360, num_nodes, num_clusters)).cuda()
            #         mask = torch.randint(0, 2, (batch_size, num_nodes), dtype=torch.bool).cuda()
            #         mask = None
            #         x_conv, adj_prime, link_loss, ent_loss = dense_diff_pool(x if i == 0 else x_conv, self.A if i == 0 else adj_prime, s, mask)
            #     else:
            #         batch_size, num_nodes, channels, num_clusters = (360, num_nodes if i == 0 else num_clusters, 16, 1)
            #         s = torch.randn((360, num_nodes, num_clusters)).cuda()
            #         mask = torch.randint(0, 2, (batch_size, num_nodes), dtype=torch.bool).cuda()
            #         mask = None
            #         x_conv, adj_prime, link_loss, ent_loss = dense_diff_pool(x if i == 0 else x_conv, self.A if i == 0 else adj_prime, s,mask)

            batch_size, num_nodes, channels, num_clusters = (360, 24, 20, 12)
            s = torch.randn((360, num_nodes, num_clusters)).cuda()
            mask = None
            x_conv, adj_prime, link_loss, ent_loss = dense_diff_pool(x,self.A, s, mask)
            x2 = F.relu(self.gnn2(x_conv, adj_prime))

            batch_size, num_nodes, channels, num_clusters = (360, 12, 20, 6)
            s = torch.randn((360, num_nodes, num_clusters)).cuda()
            mask = None
            x_conv, adj_prime, link_loss, ent_loss = dense_diff_pool(x2, adj_prime, s, mask)
            x3 = F.relu(self.gnn3(x_conv, adj_prime))

            batch_size, num_nodes, channels, num_clusters = (360, 6, 20, 3)
            s = torch.randn((360, num_nodes, num_clusters)).cuda()
            mask = None
            x_conv, adj_prime, link_loss, ent_loss = dense_diff_pool(x3, adj_prime, s, mask)
            x4 = F.relu(self.gnn4(x_conv, adj_prime))

            batch_size, num_nodes, channels, num_clusters = (360, 3, 20, 1)
            s = torch.randn((360, num_nodes, num_clusters)).cuda()
            mask = None
            x_conv, adj_prime, link_loss, ent_loss = dense_diff_pool(x4, adj_prime, s, mask)

        x = x_conv.squeeze(1)

        dists = self.mlp(x)
        # dists = x
        # dists = F.softmax(dists, dim=1)
        # dists = euclidean_dist(x, x_proto)

        # print(dists.max(1)[1].cpu().numpy())
        # print(dists[9])

        return dists,link_loss


