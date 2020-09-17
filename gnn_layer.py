import torch.nn as nn

import torch.nn.functional as F
import math
import torch
from torch.nn import Parameter
import math


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)
def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class DeGINConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GINConv`.

    :rtype: :class:`Tensor`
    """
    def __init__(self, nn, eps= 7.5 , train_eps=True):
        super(DeGINConv, self).__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out

        out = self.nn(out)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)




class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class DenseSAGEConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        super(DenseSAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)


    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class DenseGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='mean', bias=True):
        assert aggr in ['add', 'mean', 'max']
        super(DenseGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()


    def forward(self, x, adj, mask=None):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)

        if self.aggr == 'mean':
            out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        elif self.aggr == 'max':
            out = out.max(dim=-1)[0]

        out = out + self.lin(x)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class rkGraphConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GraphConv`.
    """
    def __init__(self, num_adj, in_channels, out_channels, attention_mode, aggr='mean', bias=True):
        assert aggr in ['add', 'mean', 'max']
        super(rkGraphConv, self).__init__()
        self.num_adj = num_adj
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        attention_mode = attention_mode
        # self.weight = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(self.num_adj, in_channels, out_channels)))
        self.weight = Parameter(torch.Tensor(self.num_adj, in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        # self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        # self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        # self.lin3 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.attention_mode = attention_mode
        self.attention = Parameter(nn.init.uniform_(torch.FloatTensor(self.num_adj)))
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(2))
        # self.weight.data.uniform_(-stdv, stdv)
        uniform(self.in_channels, self.weight)

        self.lin.reset_parameters()
        # self.lin1.reset_parameters()
        # self.lin2.reset_parameters()
        # self.lin3.reset_parameters()

    def forward(self, x, adj):
        outputs = []
        for i in range(self.num_adj):
            # adj[i] = adj[i].cuda()
            out = torch.matmul(adj[i], x)
            out = torch.matmul(out, self.weight[i])
            # if i == 0:
            #     out = out + self.lin1(x)
            # if i == 1:
            #     out = out + self.lin2(x)
            # if i == 2:
            #     out = out + self.lin3(x)
            # out = out + self.lin(x)
            if self.aggr == 'mean':
                out = out / adj[i].sum(dim=-1, keepdim=True).clamp(min=1)
            outputs.append(out)
        # outputs_raw = torch.stack(outputs)
        outputs = torch.stack(outputs, 3)

        # print(attention)
        output = F.softmax(self.attention) * outputs

        output = torch.sum(output,3)
        output = output + self.lin(x)
        # output = self.batchnorm(output)
        return output

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
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
        self.weight = Parameter(torch.Tensor(8,8))
        uniform(8, self.weight)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = 3
        self.static_feat = static_feat

    def forward(self, idx,x = None):
        self.option = 1
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
            a = torch.matmul(x.transpose(2,1), x)
            b = torch.nn.functional.normalize(self.alpha * a, p=1, dim=1)

            # zero = torch.zeros_like(b)
            # b = torch.where(b > 0.5, zero, b)
            adj = F.relu(torch.tanh(b))
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

    # def fullA(self, idx,x=None):
    #     if self.static_feat is None:
    #         nodevec1 = self.emb1(idx)
    #         nodevec2 = self.emb2(idx)
    #     else:
    #         nodevec1 = self.static_feat[idx,:]
    #         nodevec2 = nodevec1
    #
    #     nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
    #     nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
    #     # nodevec1 = self.alpha*self.lin1(nodevec1)
    #     # nodevec2 = self.alpha*self.lin2(nodevec2)
    #     a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
    #
    #     adj = F.relu(torch.tanh(self.alpha*a))
    #     # adj = F.relu(self.alpha * a)
    #     return adj
