import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils import config_dataset, corr_matrix
import numpy as np
from layer import dense_diff_pool, graph_constructor, Adaptive_Pooling_Layer, Memory_Pooling_Layer, SAG_Pooling_Layer
from gnn_layer import *
from scipy.sparse import coo_matrix


class MTPool(nn.Module):

    def __init__(self, use_cuda, dataset_path, dataset, graph_method, relation_method, pooling_method):
        super(MTPool, self).__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.graph_method = graph_method
        self.relation_method = relation_method
        self.pooling_method = pooling_method

        self.train_len, self.test_len, self.num_nodes, self.feature_dim, self.nclass = config_dataset(dataset)

        # use cpu or gpu
        self.use_cuda = use_cuda
        if self.use_cuda == 1:
            self.device = torch.device('cuda:0,1')
        else:
            self.device = torch.device('cpu')

        # CNN to extract feature
        kernel_ = [3, 5, 7]
        channel = 1
        self.c1 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[0]), stride=1)
        self.c2 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[1]), stride=1)
        self.c3 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[2]), stride=1)

        d = (len(kernel_) * (self.feature_dim) - sum(kernel_) + len(kernel_)) * channel
        # d = self.feature_dim

        # How to build the graph (corr or dynamic)
        # Corr Graph Adjacency Matrix
        if self.relation_method == "corr":
            self.train_A, self.test_A = corr_matrix(self.train_len, self.test_len, self.num_nodes,
                                                    self.use_cuda, self.dataset_path, self.dataset)
        # Dynamic Graph Adjacency Matrix
        elif self.relation_method == "dynamic":
            self.gc = graph_constructor(self.num_nodes, self.device, self.use_cuda, pool_method=pooling_method)
        elif self.relation_method == "all_one":
            pass
        else:

            raise Exception("Only support these relations...")

        # GNN to extract feature
        self.hid = 128

        if self.graph_method == 'GNN':
            self.gnn = DenseGraphConv(d, self.hid)
        elif self.graph_method == 'GIN':
            ginnn = nn.Sequential(
                nn.Linear(d, self.hid),
                nn.Tanh(),
            )
            self.gin = DeGINConv(ginnn)
        else:
            raise Exception("Only support these GNNs...")

        if self.pooling_method == "CoSimPool":
            adaptive_pooling_layers = []
            # ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=self.num_nodes // 3, Dim_output=self.hid, use_cuda=self.use_cuda)
            # adaptive_pooling_layers.append(ap)
            # ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=self.num_nodes//2, Dim_output=self.hid, use_cuda=self.use_cuda)
            # adaptive_pooling_layers.append(ap)
            ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=1, Dim_output=self.hid,
                                        use_cuda=self.use_cuda)
            adaptive_pooling_layers.append(ap)
            # ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid4, N_output=1, Dim_output=self.hid4)
            # adaptive_pooling_layers.append(ap)
            # D = self.num_nodes//4

            # reduce_factor = 4
            # while D > 1:
            #     D = D // reduce_factor
            #     if D < 1:
            #         D = 1
            #     ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid4, N_output=D, Dim_output=self.hid4)
            #     adaptive_pooling_layers.append(ap)

            self.ap = nn.ModuleList(adaptive_pooling_layers)

        elif self.pooling_method == "DiffPool":

            self.gnn_z = DenseGraphConv(self.hid, self.hid)
            self.gnn_s = DenseGraphConv(self.hid, 1)
            # self.reduce_factor = 2
            # self.gnn_z = []
            # self.gnn_s = []
            #
            # num_nodes = self.num_nodes
            # while (num_nodes >= 2):
            #     num_clusters = num_nodes // self.reduce_factor
            #     z = DenseGraphConv(self.hid, self.hid)
            #     s = DenseGraphConv(self.hid, num_clusters)
            #     num_nodes = num_nodes // self.reduce_factor
            #     self.gnn_z.append(z)
            #     self.gnn_s.append(s)

        elif self.pooling_method == "MemPool":
            memory_pooling_layers = []
            mp = Memory_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=1, Dim_output=self.hid,
                                      use_cuda=self.use_cuda)
            memory_pooling_layers.append(mp)
            self.mp = nn.ModuleList(memory_pooling_layers)
        elif self.pooling_method == "SAGPool":
            sag_pooling_layers = []
            sp = SAG_Pooling_Layer(in_channels=self.hid, ratio=float(1) / float(self.num_nodes))
            sag_pooling_layers.append(sp)
            self.sp = nn.ModuleList(sag_pooling_layers)
        else:
            raise Exception("Only support these pooling methods...")

        self.mlp = nn.Sequential(
            nn.Linear(self.hid, self.hid),
            nn.PReLU(),
            nn.Linear(self.hid, self.nclass),
        )

        self.cnn_act = nn.PReLU()
        self.gnn_act = nn.PReLU()

        self.batch_norm_cnn = nn.BatchNorm1d(self.num_nodes)
        self.batch_norm_gnn = nn.BatchNorm1d(self.num_nodes)
        self.batch_norm_mlp = nn.BatchNorm1d(self.hid)

    def forward(self, input, test=False):
        # Process: input -> CNN -> Graph Adjacency Matrix -> GNN -> Pooling -> MLP -> output

        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension

        if test:
            x = x[idx_test]
        else:
            x = x[idx_train]
        c = x

        # CNN to extract feature
        a1 = self.c1(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        a2 = self.c2(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        a3 = self.c3(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        x = self.cnn_act(torch.cat([a1, a2, a3], 2))
        x = self.batch_norm_cnn(x)
        #
        # Graph Adjacency Matrix
        if self.relation_method == "dynamic":
            # Dynamic Graph Adjacency matrix
            idx = [0]
            for i in range(1, self.num_nodes):
                idx.append(i)
            if self.use_cuda == 1:
                idx = torch.tensor(idx).to(self.device)
            adj = self.gc(idx, c)
            if self.pooling_method != "SAGPool":
                g = F.normalize(adj, p=1, dim=1)
                self.A = g
            else:
                self.A = adj
        elif self.relation_method == "corr":
            if test:
                # self.A = torch.tensor(np.load("./test_A.npy"))
                self.A = self.test_A
            else:
                # self.A = torch.tensor(np.load("./train_A.npy"))
                self.A = self.train_A
        elif self.relation_method == "all_one":
            self.A = torch.ones(x.shape[0], x.shape[1], x.shape[1])
        else:
            raise Exception("Only support these relation methods...")

        if self.use_cuda:
            x = x.cuda()
            self.A = self.A.cuda()
        # GNN
        if self.graph_method == 'GNN':
            # x = self.gnn_act(self.gnn(x,self.A))
            x = self.gnn(x, self.A)
            x = x.squeeze()
        elif self.graph_method == 'GIN':
            x = self.gin(x, self.A)
            x = x.squeeze()
        else:
            raise Exception("Only support these graph methods...")

        x = self.batch_norm_gnn(x)

        # Pooling
        if self.pooling_method == 'CoSimPool':
            A = self.A
            for layer in self.ap:
                x, A = layer(x, A)
        elif self.pooling_method == 'DiffPool':
            # num_nodes = self.num_nodes
            # reduce_factor = self.reduce_factor
            # adj_prime = self.A
            # while (num_nodes >= 2):
            #     num_clusters = num_nodes // reduce_factor
            #     z = self.gnn_z(x, a)
            #     s = F.softmax(self.gnn_s(x, a), dim=1)
            #     # if test:
            #         # s = torch.randn((self.test_len, num_nodes, num_clusters))
            #     # else:
            #         # s = torch.randn((self.train_len, num_nodes, num_clusters))
            #     if self.use_cuda:
            #         s = s.cuda()
            #     x, adj_prime = dense_diff_pool(x, adj_prime, s)
            #     num_nodes = num_nodes // reduce_factor
            adj = self.A
            z = self.gnn_z(x, adj)
            s = self.gnn_s(x, adj)
            # s = fs(x, adj)
            x, adj_prime = dense_diff_pool(z, adj, s)
            # for i in range(len(self.gnn_z)):
            #     # if self.use_cuda:
            #     #     x = x.cuda()
            #     #     adj = adj.cuda()
            #     z = self.gnn_z[i](x, adj)
            #     # z = fz
            #     s = self.gnn_s[i](x, adj)
            #     # s = fs(x, adj)
            #     x, adj_prime = dense_diff_pool(z, adj, s)


        elif self.pooling_method == 'MemPool':
            A = self.A
            for layer in self.mp:
                x, A = layer(x, A)
        elif self.pooling_method == 'SAGPool':
            # Convert from (batch_size, num_nodes, feature_dim) to merged graph (batch_size*num_nodes, feature_dim)
            A = self.A
            temp_edge = A.to_sparse().coalesce().indices()
            edge_index = temp_edge[1:] + temp_edge[0] * self.num_nodes
            # delta = 0
            # A[A>=0.1] = 1
            # for j in range(A.shape[0]):
            #     temp_A = A[j]
            #     # temp_x = x[j]
            #     # temp_A = torch.ones(self.num_nodes,self.num_nodes)
            #     # coo_A = coo_matrix(temp_A.cpu().detach().numpy())
            #     # temp_edge_index = torch.tensor([coo_A.row, coo_A.col],dtype=torch.long)+delta
            #     temp_edge_index = temp_A.to_sparse()
            #     # temp_edge_index = temp_edge_index.coalesce().indices()+delta
            #     if j==0:
            #         edge_index = temp_edge_index
            #     else:
            #         edge_index = torch.cat((edge_index,temp_edge_index),1)
            #     delta = delta+self.num_nodes
            # edge_index = torch.tensor(edge_index,dtype=torch.float,requires_grad=True)
            # edge_index = edge_index.coalesce().indices()

            x = x.reshape(-1, self.hid)

            if test:
                batch = torch.tensor(range(self.test_len)).reshape(-1, 1)
            else:
                batch = torch.tensor(range(self.train_len)).reshape(-1, 1)
            batch = batch.repeat(1, self.num_nodes).reshape(-1)

            if self.use_cuda:
                edge_index = edge_index.cuda()
                x = x.cuda()
                batch = batch.cuda()

            for layer in self.sp:
                x, edge_index, _, batch, _ = layer(x=x, edge_index=edge_index, batch=batch)
        else:
            raise Exception("Only support these pooling methods...")

        x = x.squeeze(1)
        x = self.batch_norm_mlp(x)
        # if test:
        #     torch.save(x,"x.pt")
        y = self.mlp(x)
        y = F.softmax(y, 1)
        # print("y",y.T)

        return y


