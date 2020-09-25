import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.data import Data as PyGSingleGraphData
from utils_mp import *
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter

EPS = 1e-15

def get_att(x,W,emb_dim,batch_size=180):
    temp = torch.mean(x, 1).view((batch_size, 1, -1))  # (1, D)
    h_avg = torch.tanh(torch.matmul(temp, W))
    att = torch.bmm(x, h_avg.transpose(2,1))
    output = torch.bmm(att.transpose(2,1), x)

    return output

class Adaptive_Pooling_Layer(nn.Module):
    """ Memory_Pooling_Layer introduced in the paper 'MEMORY-BASED GRAPH NETWORKS' by Amir Hosein Khasahmadi, etc"""

    ### This layer is for downsampling a node set from N_input to N_output
    ### Input: [B,N_input,Dim_input]
    ###         B:batch size, N_input: number of input nodes, Dim_input: dimension of features of input nodes
    ### Output:[B,N_output,Dim_output]
    ###         B:batch size, N_output: number of output nodes, Dim_output: dimension of features of output nodes

    def __init__(self, Heads, Dim_input, N_output, Dim_output, use_cuda):
        """
            Heads: number of memory heads
            N_input : number of nodes in input node set
            Dim_input: number of feature dimension of input nodes
            N_output : number of the downsampled output nodes
            Dim_output: number of feature dimension of output nodes
        """
        super(Adaptive_Pooling_Layer, self).__init__()
        self.Heads = Heads
        self.Dim_input = Dim_input
        self.N_output = N_output
        self.Dim_output = Dim_output
        self.use_cuda = use_cuda

        if self.use_cuda:
            FLAGS.device = "cuda:0"
        else:
            FLAGS.device = "cpu"

        # Randomly initialize centroids
        self.centroids = nn.Parameter(2 * torch.rand(self.Heads, self.N_output, Dim_input) - 1)
        self.centroids.requires_grad = True

        hiden_channels = self.Heads
        if self.use_cuda:
            self.input2centroids_weight = torch.nn.Parameter(
                torch.zeros(hiden_channels, 1).float().to(FLAGS.device), requires_grad=True)
            self.input2centroids_bias = torch.nn.Parameter(
                torch.zeros(hiden_channels).float().to(FLAGS.device), requires_grad=True)
        else:
            self.input2centroids_weight = torch.nn.Parameter(
                torch.zeros(hiden_channels, 1).float(), requires_grad=True)
            self.input2centroids_bias = torch.nn.Parameter(
                torch.zeros(hiden_channels).float(), requires_grad=True)

        self.memory_aggregation = nn.Conv2d(self.Heads, 1, [1, 1])

        self.dim_feat_transformation = nn.Linear(self.Dim_input, self.Dim_output)

        self.similarity_compute = torch.nn.CosineSimilarity(dim=4, eps=1e-6)

        self.relu = nn.ReLU()

        self.emb_dim = Dim_input
        self.W_0 = glorot([self.emb_dim, self.emb_dim], self.use_cuda)

    def forward(self, node_set, adj, zero_tensor=torch.tensor([0])):
        """
            node_set: Input node set in form of [batch_size, N_input, Dim_input]
            adj: adjacency matrix for node set x in form of [batch_size, N_input, N_input]
            zero_tensor: zero_tensor of size [1]

            (1): new_node_set = LRelu(C*node_set*W)
            (2): C = softmax(pixel_level_conv(C_heads))
            (3): C_heads = t-distribution(node_set, centroids)
            (4): W is a trainable linear transformation
        """
        node_set_input = node_set
        batch_size, self.N_input, _ = node_set.size()

        # batch_centroids = torch.mean(node_set,dim=1,keepdim=True)

        batch_centroids = get_att(node_set,self.W_0,self.emb_dim,batch_size=batch_size)

        batch_centroids = batch_centroids.permute(0, 2, 1)

        batch_centroids = torch.relu(torch.nn.functional.linear(batch_centroids, self.input2centroids_weight, self.input2centroids_bias))

        batch_centroids = batch_centroids.permute(0, 2, 1).view(node_set.size()[0], self.Heads, self.N_output,
                                                                self.Dim_input)

        # From initial node set [batch_size, N_input, Dim_input] to size [batch_size, Heads, N_output, N_input, Dim_input]
        node_set = torch.unsqueeze(node_set, 1).repeat(1, batch_centroids.shape[1], 1, 1)
        node_set = torch.unsqueeze(node_set, 2).repeat(1, 1, batch_centroids.shape[2], 1, 1)
        # Broadcast centroids to the same size as that of node set [batch_size, Heads, N_output, N_input, Dim_input]
        batch_centroids = torch.unsqueeze(batch_centroids, 3).repeat(1, 1, 1, node_set.shape[3], 1)

        # Compute the distance between original node set to centroids
        # [batch_size, Heads, N_output, N_input]
        C_heads = self.similarity_compute(node_set, batch_centroids)

        normalizer = torch.unsqueeze(torch.sum(C_heads, 2), 2)
        C_heads = C_heads / (normalizer + 1e-10)

        # Apply pixel-level convolution and softmax to C_heads
        # Get C: [batch_size, N_output, N_input]
        C = self.memory_aggregation(C_heads)
        # C = torch.softmax(C,1)

        C = C.squeeze(1)

        # [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
        new_node_set = torch.matmul(C, node_set_input)
        # [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
        new_node_set = self.dim_feat_transformation(new_node_set)

        """
            Calculate new_adj
        """

        # [batch_size, N_output, N_input] * [batch_size, N_input, N_input] --> [batch_size, N_output, N_input]
        q_adj = torch.matmul(C, adj)

        # [batch_size, N_output, N_input] * [batch_size, N_input, N_output] --> [batch_size, N_output, N_output]
        new_adj = self.relu(torch.matmul(q_adj, C.transpose(1, 2)))

        return new_node_set, new_adj

class Memory_Pooling_Layer(nn.Module):
    """ Memory_Pooling_Layer introduced in the paper 'MEMORY-BASED GRAPH NETWORKS' by Amir Hosein Khasahmadi, etc"""

    ### This layer is for downsampling a node set from N_input to N_output
    ### Input: [B,N_input,Dim_input]
    ###         B:batch size, N_input: number of input nodes, Dim_input: dimension of features of input nodes
    ### Output:[B,N_output,Dim_output]
    ###         B:batch size, N_output: number of output nodes, Dim_output: dimension of features of output nodes

    def __init__(self, Heads, Dim_input, N_output, Dim_output, use_cuda, Tau=1):
        """
            Heads: number of memory heads
            N_input : number of nodes in input node set
            Dim_input: number of feature dimension of input nodes
            N_output : number of the downsampled output nodes
            Dim_output: number of feature dimension of output nodes
            Tau: parameter for the student t-distribution mentioned in the paper
        """
        super(Memory_Pooling_Layer, self).__init__()
        self.Heads = Heads
        self.Dim_input = Dim_input
        self.N_output = N_output
        self.Dim_output = Dim_output
        self.Tau = Tau

        # Randomly initialize centroids
        self.centroids = nn.Parameter(2 * torch.rand(self.Heads, self.N_output, Dim_input) - 1)
        self.centroids.requires_grad = True

        self.memory_aggregation = nn.Conv2d(self.Heads, 1, [1, 1])
        self.dim_feat_transformation = nn.Linear(self.Dim_input, self.Dim_output)
        self.lrelu = nn.LeakyReLU()

    def forward(self, node_set, adj, zero_tensor=torch.tensor([0])):
        """
            node_set: Input node set in form of [batch_size, N_input, Dim_input]
            adj: adjacency matrix for node set x in form of [batch_size, N_input, N_input]
            zero_tensor: zero_tensor of size [1]

            (1): new_node_set = LRelu(C*node_set*W)
            (2): C = softmax(pixel_level_conv(C_heads))
            (3): C_heads = t-distribution(node_set, centroids)
            (4): W is a trainable linear transformation
        """
        node_set_input = node_set
        _, self.N_input, _ = node_set.size()

        """
            With (1)(2)(3)(4) we calculate new_node_set
        """

        # Copy centroids and repeat it in batch
        ## [h,N_output,Dim_input] --> [batch_size,Heads,N_output,Dim_input]
        batch_centroids = torch.unsqueeze(self.centroids, 0). \
            repeat(node_set.shape[0], 1, 1, 1)
        # From initial node set [batch_size, N_input, Dim_input] to size [batch_size, Heads, N_output, N_input, Dim_input]
        node_set = torch.unsqueeze(node_set, 1).repeat(1, batch_centroids.shape[1], 1, 1)
        node_set = torch.unsqueeze(node_set, 2).repeat(1, 1, batch_centroids.shape[2], 1, 1)
        # Broadcast centroids to the same size as that of node set [batch_size, Heads, N_output, N_input, Dim_input]
        batch_centroids = torch.unsqueeze(batch_centroids, 3).repeat(1, 1, 1, node_set.shape[3], 1)
        # Compute the distance between original node set to centroids
        # [batch_size, Heads, N_output, N_input]

        dist = torch.sum(torch.abs(node_set - batch_centroids) ** 2, 4)

        # Compute the Matrix C_heads : [batch_size, Heads, N_output, N_input]
        C_heads = torch.pow((1 + dist / self.Tau), -(self.Tau + 1) / 2)

        normalizer = torch.unsqueeze(torch.sum(C_heads, 2), 2)
        C_heads = C_heads / normalizer


        # Apply pixel-level convolution and softmax to C_heads
        # Get C: [batch_size, N_output, N_input]
        C = self.memory_aggregation(C_heads)
        C = torch.softmax(C, 1)

        C = C.squeeze(1)

        # [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
        new_node_set = torch.matmul(C, node_set_input)
        # [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
        new_node_set = self.dim_feat_transformation(new_node_set)
        new_node_set = self.lrelu(new_node_set)

        """
            Calculate new_adj
        """
        # [batch_size, N_output, N_input] * [batch_size, N_input, N_input] --> [batch_size, N_output, N_input]
        q_adj = torch.matmul(C, adj)

        # [batch_size, N_output, N_input] * [batch_size, N_input, N_output] --> [batch_size, N_output, N_output]
        new_adj = torch.matmul(q_adj, C.transpose(1, 2))

        return new_node_set, new_adj


class SAGPool(nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

def glorot(shape, use_cuda):
    """Glorot & Bengio (AISTATS 2010) init."""
    if use_cuda:
        rtn = nn.Parameter(torch.Tensor(*shape).cuda())
    else:
        rtn = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(rtn)
    return rtn

class graph_constructor(nn.Module):
    def __init__(self, nnodes, device, use_cuda, dim=40, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)
        self.weight = glorot([self.nnodes,self.nnodes], use_cuda)#nn.Parameter(torch.Tensor(self.nnodes,self.nnodes))

        self.device = device
        
        self.dim = dim
        self.alpha = 3
        self.static_feat = static_feat

    def forward(self, idx,x = None):
        self.option = 2
        if self.option == 1:

            if self.static_feat is None:
                nodevec1 = self.emb1(idx)
                nodevec2 = self.emb2(idx)
            else:
                nodevec1 = self.static_feat[idx, :]
                nodevec2 = nodevec1

            nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
            nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

            a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
            # a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
            b = torch.nn.functional.normalize(self.alpha * a, p=1, dim=1)
            # adj = F.relu(torch.tanh(self.alpha*a))
            adj = F.relu(torch.tanh(b))
            # mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            # mask.fill_(float('0'))
            # s1,t1 = adj.topk(self.k,1)
            # mask.scatter_(1,t1,s1.fill_(1))
            # adj = adj*mask
            return adj
        if self.option == 2:
            # a = torch.matmul(x.permute(0,2,1),x)
            # # a = torch.matmul(x.permute(0,2,1),x)
            # # a = a-a.transpose(1,0)
            # # a = torch.matmul(self.weight,a)
            # b = a
            # b = torch.nn.functional.normalize(a,p=1,dim=1)
            x1 = F.normalize(x, p=2, dim=2)
            a = torch.matmul(x1, x1.transpose(2,1))
            b = torch.matmul(a, self.weight)
            c = torch.nn.functional.normalize(b, p=1, dim=1)
            # zero = torch.zeros_like(b)
            # b = torch.where(b > 0.5, zero, b)
            adj = F.relu(torch.tanh(c))
#             adj[abs(adj)<0.05] = 0

        if self.option == 3:
            a = torch.matmul(x.transpose(2,1),x)
            # w = torch.nn.functional.normalize(self.weight, p=1, dim=1)
            a = torch.nn.functional.normalize(a,p=1,dim=1)
            adj = F.relu(torch.tanh(self.weight+a))

        # mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        # mask.fill_(float('0'))
        # s1,t1 = adj.topk(self.k,1)
        # mask.scatter_(1,t1,s1.fill_(1))
        # adj = adj*mask
        return adj

def dense_diff_pool(x, adj, s, mask=None):

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    return out, out_adj