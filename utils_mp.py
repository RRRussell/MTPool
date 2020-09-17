import re
import torch
import argparse
import time
import numpy as np
import networkx as nx
from torch_scatter import scatter_add
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.transforms import LocalDegreeProfile

parser = argparse.ArgumentParser()
FLAGS = parser.parse_args()

# class Timer(object):
#     def __init__(self):
#         self.t = time()
#
#     def time_msec_and_clear(self):
#         # from time import sleep
#         # sleep(2.5)
#         now = time()
#         duration = now - self.t
#         self.t = now
#         # print(duration)
#         rtn = format_mseconds(duration*1000)
#         # print('@@@', rtn, type(rtn), '@@@')
#         return rtn

def assert_valid_nid(nid, g):
    assert type(nid) is int and (0 <= nid < g.number_of_nodes())

def assert_0_based_nids(g):
    for i, (n, ndata) in enumerate(sorted(g.nodes(data=True))):
        assert_valid_nid(n, g)
        assert i == n  # 0-based consecutive node ids

class NodeFeatureEncoder(object):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        if node_feat_name is None:
            return
        # Go through all the graphs in the entire dataset
        # and create a set of all possible
        # labels so we can one-hot encode them.
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(sorted(inputs_set))}
        self._fit_onehotencoder()

    def _fit_onehotencoder(self):
        self.oe = OneHotEncoder(categories='auto').fit(
            np.array(sorted(self.feat_idx_dic.values())).reshape(-1, 1))

    def encode(self, g):
        assert_0_based_nids(g)  # must be [0, 1, 2, ..., N - 1]
        if self.node_feat_name is None:
            return np.array([[1] for n in sorted(g.nodes())])  # NOTE: this will no longer be called now?
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in sorted(g.nodes())]  # sort nids just to make sure
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)

def _one_hot_encode(dataset, input_dim):
    gs = [g.get_nxgraph() for g in dataset.gs] # TODO: encode image's complete graph
    if len(dataset.natts) > 1:
        node_feat_name = None
        raise ValueError('TODO: handle multiple node features')
    elif len(dataset.natts) == 1:
        node_feat_name = dataset.natts[0]
    else:
        #if no node feat return 1
        for g in gs:
            g.init_x = np.ones((nx.number_of_nodes(g), 1))
        return 1
    nfe = NodeFeatureEncoder(gs, node_feat_name)
    for g in gs:
        x = nfe.encode(g)
        g.init_x = x # assign the initial features
    input_dim += nfe.input_dim()
    return input_dim

def sorted_nicely(l, reverse=False):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn

def get_flags_with_prefix_as_list(prefix):
    rtn = []
    d = vars(FLAGS)
    i_check = 1  # one-based
    for k in sorted_nicely(d.keys()):
        v = d[k]
        sp = k.split(prefix)
        if len(sp) == 2 and sp[0] == '' and sp[1].startswith('_'):
            id = int(sp[1][1:])
            if i_check != id:
                raise ValueError('Wrong flag format {}={} '
                                 '(should start from _1'.format(k, v))
            rtn.append(v)
            i_check += 1
    return rtn

def encode_node_features(dataset=None, pyg_single_g=None):
    if dataset:
        assert pyg_single_g is None
        input_dim = 0
    else:
        assert pyg_single_g is not None
        input_dim = pyg_single_g.x.shape[1]
    node_feat_encoders = get_flags_with_prefix_as_list('node_fe')
    if 'one_hot' not in node_feat_encoders:
        raise ValueError('Must have one hot node feature encoder!')
    for nfe in node_feat_encoders:
        if nfe == 'one_hot':
            if dataset:
                input_dim = _one_hot_encode(dataset, input_dim)
        elif nfe == 'local_degree_profile':
            input_dim += 5
            if pyg_single_g:
                pyg_single_g = LocalDegreeProfile()(pyg_single_g)
        else:
            raise ValueError('Unknown node feature encoder {}'.format(nfe))
    if input_dim <= 0:
        raise ValueError('Must have at least one node feature encoder '
                         'so that input_dim > 0')
    if dataset:
        return dataset, input_dim
    else:
        return pyg_single_g, input_dim

def dense_to_sparse(tensor):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        tensor (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert tensor.dim() == 2
    index = tensor.nonzero().t().contiguous()
    value = tensor[index[0], index[1]]
    return index, value

def to_dense_adj(edge_index, batch=None, edge_attr=None):
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    batch_size = batch[-1].item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = num_nodes.max().item()

    size = [batch_size, max_num_nodes, max_num_nodes]
    size = size if edge_attr is None else size + list(edge_attr.size())[1:]
    dtype = torch.float if edge_attr is None else edge_attr.dtype
    adj = torch.zeros(size, dtype=dtype, device=edge_index.device)

    edge_index_0 = batch[edge_index[0]].view(1, -1)
    edge_index_1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    edge_index_2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if edge_attr is None:
        adj[edge_index_0, edge_index_1, edge_index_2] = 1
    else:
        adj[edge_index_0, edge_index_1, edge_index_2] = edge_attr

    return adj