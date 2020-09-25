import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils import config_dataset, corr_matrix
import numpy as np
from layer import dense_diff_pool, graph_constructor, Adaptive_Pooling_Layer, Memory_Pooling_Layer
from gnn_layer import *

class MTPool(nn.Module):

    def __init__(self, use_cuda, dataset_path, dataset, graph_method, relation_method, pooling_method):
        super(MTPool, self).__init__()

        self.graph_method = graph_method
        self.relation_method = relation_method
        self.pooling_method = pooling_method
        
        self.train_len, self.test_len, self.num_nodes, self.feature_dim, self.nclass = config_dataset(dataset)

        # use cpu or gpu
        self.use_cuda = use_cuda
        if self.use_cuda == 1:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # CNN to extract feature
        kernel_ = [2,3,5]
        channel = 8
        self.c1 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[0]), stride=1)
        self.c2 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[1]), stride=1)
        self.c3 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[2]), stride=1)
        
        d = (len(kernel_) * (self.feature_dim) - sum(kernel_) + len(kernel_)) * channel
        
        # How to build the graph (corr or dynamic)
        # Corr Graph Adjacency Matrix
        if self.relation_method == "corr":
            self.train_A, self.test_A = corr_matrix(self.train_len, self.test_len, self.num_nodes, 
                                                    self.use_cuda, self.dataset_path, self.dataset)
        # Dynamic Graph Adjacency Matrix
        elif self.relation_method == "dynamic":
            self.gc = graph_constructor(self.num_nodes, self.device, self.use_cuda)  
        else:
            raise Exception("Only support these relations...") 

        # GNN to extract feature
        self.hid = 128

        if self.graph_method == 'GNN':
            self.gnn = DenseGraphConv(d, self.hid)
        elif self.graph_method == 'GIN':
            ginnn = nn.Sequential(
                nn.Linear(d,self.hid),
                nn.Tanh(),
            )
            self.gin = DeGINConv(ginnn)
        else:
            raise Exception("Only support these GNNs...") 

        if self.pooling_method == "CoSimPool":
            adaptive_pooling_layers = []

            ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=1, Dim_output=self.hid, use_cuda=self.use_cuda)
            adaptive_pooling_layers.append(ap)   
            # ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid4, N_output=self.num_nodes//8, Dim_output=self.hid4)
            # adaptive_pooling_layers.append(ap)
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
            pass
        elif self.pooling_method == "MemPool":
            memory_pooling_layers = []
            mp = Memory_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=1, Dim_output=self.hid, use_cuda=self.use_cuda)
            memory_pooling_layers.append(mp) 
            self.mp = nn.ModuleList(memory_pooling_layers)  
        else:
            raise Exception("Only support these pooling methods...") 

        self.mlp = nn.Sequential(
            nn.Linear(self.hid,self.nclass),
        )

    def forward(self, input, test = False):
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
        x = F.relu(torch.cat([a1, a2, a3], 2))

        # Graph Adjacency Matrix
        if self.relation_method == "dynamic":
            # Dynamic Graph Adjacency matrix
            idx = [0]
            for i in range(1, self.num_nodes):
                idx.append(i)
            if self.use_cuda:
                idx = torch.tensor(idx).to(self.device)
            adp = self.gc(idx,c)
            g = F.normalize(adp,p=1,dim=1)
            self.A = g
        elif self.relation_method == "corr":
            if test:
                self.A = self.test_A
            else:
                self.A = self.train_A
        else:
            raise Exception("Only support these relation methods...") 

        # GNN
        if self.graph_method == 'GNN':
            x = F.relu(self.gnn(x,self.A))
            x = x.squeeze()
        elif self.graph_method == 'GIN':
            x = self.gin(x, self.A)
            x = x.squeeze()
        else:
            raise Exception("Only support these graph methods...") 

        # Pooling
        if self.pooling_method == 'CoSimPool':
            A = self.A
            for layer in self.ap:
                x, A = layer(x,A)
        elif self.pooling_method == 'DiffPool':
            num_nodes = self.num_nodes
            reduce_factor = 2
            adj_prime = self.A
            while(num_nodes>2):
                num_clusters = num_nodes//reduce_factor
                if test:
                    s = torch.randn((self.test_len, num_nodes, num_clusters))
                else:
                    s = torch.randn((self.train_len, num_nodes, num_clusters))
                if self.use_cuda:
                    s = s.cuda()
                x, adj_prime = dense_diff_pool(x, adj_prime, s)
                num_nodes = num_nodes//reduce_factor
        elif self.pooling_method == 'MemPool':
            A = self.A
            for layer in self.mp:
                x, A = layer(x,A)
        else:
            raise Exception("Only support these pooling methods...") 

        x = x.squeeze(1)

        y = self.mlp(x)

        return y


