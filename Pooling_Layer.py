import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from config import FLAGS
import networkx as nx
from torch_scatter import scatter_add
from torch_geometric.data import Data as PyGSingleGraphData
from torch_geometric.nn import GCNConv, GINConv, GATConv , ChebConv
from torch.autograd import Variable
import torch_geometric.data
from node_feat import encode_node_features
import torch_geometric.nn as geo_nn
import torch
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax

class SAGPooling(torch.nn.Module):
	r"""The self-attention pooling operator from the `"Self-Attention Graph
	Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
	Attention and Generalization in Graph Neural Networks"
	<https://arxiv.org/abs/1905.02850>`_ papers

	if :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`:

		.. math::
			\mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

			\mathbf{i} &= \mathrm{top}_k(\mathbf{y})

			\mathbf{X}^{\prime} &= (\mathbf{X} \odot
			\mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

			\mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

	if :obj:`min_score` :math:`\tilde{\alpha}` is a value in [0, 1]:

		.. math::
			\mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

			\mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

			\mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

			\mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

	where nodes are dropped based on a learnable projection score
	:math:`\mathbf{p}`.
	Projections scores are learned based on a graph neural network layer.

	Args:
		in_channels (int): Size of each input sample.
		ratio (float): Graph pooling ratio, which is used to compute
			:math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
			This value is ignored if min_score is not None.
			(default: :obj:`0.5`)
		GNN (torch.nn.Module, optional): A graph neural network layer for
			calculating projection scores (one of
			:class:`torch_geometric.nn.conv.GraphConv`,
			:class:`torch_geometric.nn.conv.GCNConv`,
			:class:`torch_geometric.nn.conv.GATConv` or
			:class:`torch_geometric.nn.conv.SAGEConv`). (default:
			:class:`torch_geometric.nn.conv.GraphConv`)
		min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
			which is used to compute indices of pooled nodes
			:math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
			When this value is not :obj:`None`, the :obj:`ratio` argument is
			ignored. (default: :obj:`None`)
		multiplier (float, optional): Coefficient by which features gets
			multiplied after pooling. This can be useful for large graphs and
			when :obj:`min_score` is used. (default: :obj:`1`)
		nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
			(default: :obj:`torch.tanh`)
		**kwargs (optional): Additional parameters for initializing the graph
			neural network layer.
	"""
	def __init__(self, in_channels, ratio=0.5, GNN=geo_nn.GATConv, min_score=None,
				 multiplier=1, nonlinearity=torch.tanh, **kwargs):
		super(SAGPooling, self).__init__()

		self.in_channels = in_channels
		self.ratio = ratio
		self.hiden_gnn = GNN(in_channels, in_channels, **kwargs)
		self.act = torch.nn.ReLU()
		self.gnn = GNN(in_channels, 1, **kwargs)
		self.min_score = min_score
		self.multiplier = multiplier
		self.nonlinearity = nonlinearity

		self.reset_parameters()

	def reset_parameters(self):
		self.gnn.reset_parameters()


	def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
		""""""
		if batch is None:
			batch = edge_index.new_zeros(x.size(0))

		attn = x if attn is None else attn
		attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn

		attn = self.act(self.hiden_gnn(attn, edge_index))
		score = self.act(self.gnn(attn, edge_index).view(-1))

		if self.min_score is None:
			score = self.nonlinearity(score)
		else:
			score = softmax(score, batch)

		perm = topk(score, self.ratio, batch=batch )
		x = x[perm] * score[perm].view(-1, 1)
		x = self.multiplier * x if self.multiplier != 1 else x

		batch = batch[perm]
		edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
										   num_nodes=score.size(0))

		return x, edge_index, edge_attr, batch, perm, score[perm]


	def __repr__(self):
		return '{}({}, {}, {}={}, multiplier={})'.format(
			self.__class__.__name__, self.gnn.__class__.__name__,
			self.in_channels,
			'ratio' if self.min_score is None else 'min_score',
			self.ratio if self.min_score is None else self.min_score,
			self.multiplier)

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

class Memory_Pooling_Layer(nn.Module):
	""" Memory_Pooling_Layer introduced in the paper 'MEMORY-BASED GRAPH NETWORKS' by Amir Hosein Khasahmadi, etc"""

	### This layer is for downsampling a node set from N_input to N_output
	### Input: [B,N_input,Dim_input]
	### 		B:batch size, N_input: number of input nodes, Dim_input: dimension of features of input nodes
	### Output:[B,N_output,Dim_output]
	### 		B:batch size, N_output: number of output nodes, Dim_output: dimension of features of output nodes

	def __init__(self, Heads, Dim_input, N_output, Dim_output, CosimGNN=True, max_num_nodes=500, with_mask=False, Tau=1,
				 linear_block=False, p2p=False):
		"""
			Heads: number of memory heads
			N_input : number of nodes in input node set
			Dim_input: number of feature dimension of input nodes
			N_output : number of the downsampled output nodes
			Dim_output: number of feature dimension of output nodes
			with_mask : with mask computed by adjacency matrix or not
			Tau: parameter for the student t-distribution mentioned in the paper
			linear_block : Whether use linear transformation between hierarchy blocks
		"""
		super(Memory_Pooling_Layer, self).__init__()
		self.Heads = Heads
		self.Tau = Tau
		self.CosimGNN = CosimGNN
		self.Dim_input = Dim_input
		self.N_output = N_output
		self.Dim_output = Dim_output
		self.with_mask = with_mask
		self.linear_block = linear_block
		self.p2p = p2p
		self.max_num_nodes = max_num_nodes

		# Randomly initialize centroids
		self.centroids = \
			nn.Parameter(2 * torch.rand(
				self.Heads, self.N_output, Dim_input) - 1)
		self.centroids.requires_grad = True
		self.dropout_1 = torch.nn.Dropout(p=0.5)
		if self.Heads * self.N_output // 8 > 0:
			hiden_channels_1 = self.Heads * self.N_output // 8
		else:
			hiden_channels_1 = self.Heads
		self.input2centroids_1_weight = torch.nn.Parameter(
			torch.zeros(hiden_channels_1, 1).float()
			.to(FLAGS.device), requires_grad=True)
		self.input2centroids_1_bias = torch.nn.Parameter(torch.zeros(hiden_channels_1).float()
														 .to(FLAGS.device), requires_grad=True)
		if self.Heads * self.N_output//2 > 0:
			hiden_channels_2 = self.Heads * self.N_output//2
		else:
			hiden_channels_2 = self.Heads
		self.input2centroids_2 = nn.Sequential(nn.Linear(hiden_channels_1,hiden_channels_2),nn.ReLU())
		self.input2centroids_3 = nn.Sequential(nn.Linear(hiden_channels_2, self.Heads * self.N_output),nn.ReLU())
		self.input2centroids_4 = nn.Sequential(nn.Linear(self.Heads * self.N_output, self.Heads * self.N_output),
											   nn.ReLU())
		self.input2centroids_5 = nn.Sequential(nn.Linear(self.Heads * self.N_output, self.Heads * self.N_output),
											   nn.ReLU())


		self.memory_aggregation = nn.Conv2d(self.Heads, 1, [1, 1])
		self.bn_1 = torch.nn.BatchNorm2d(1)
		self.dim_feat_transformation = nn.Linear(self.Dim_input, self.Dim_output)
		#self.relu = nn.tanh()
		self.lrelu = nn.LeakyReLU()
		self.similarity_compute = torch.nn.CosineSimilarity(dim=4, eps=1e-6)
		self.similarity_compute_1 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
		self.relu = nn.ReLU()
	def forward(self, ins, batch_data, model):
		zero_tensor = torch.zeros(1)
		node_set_input = batch_data.split_into_batches(ins)
		single_graph_list = []
		metadata_list = []
		new_ins = []

		# print('start pooling')
		# st = datetime.datetime.now()
		for i in range(0, len(node_set_input), 2):
			node_set = node_set_input[i].unsqueeze(0)
			adj = to_dense_adj(batch_data.single_graph_list[i].edge_index)
			if self.CosimGNN:
				new_node_set, new_adj = self.forward_graph_Memory_based_Pooling_plus(node_set, adj, zero_tensor)
			else:
				new_node_set, new_adj = self.forward_graph_Memory_based_Pooling(node_set, adj, zero_tensor)
			new_x_1 = new_node_set.squeeze(0)
			new_edge_index_1, new_edge_attr_1 = dense_to_sparse(new_adj.squeeze(0))

			data_1 = PyGSingleGraphData(x=new_x_1, edge_index=new_edge_index_1, edge_attr=new_edge_attr_1, y=None)
			data_1, _ = encode_node_features(pyg_single_g=data_1)

			node_set = node_set_input[i + 1].unsqueeze(0)
			adj = to_dense_adj(batch_data.single_graph_list[i + 1].edge_index)
			if self.CosimGNN:
				new_node_set, new_adj = self.forward_graph_Memory_based_Pooling_plus(node_set, adj, zero_tensor)
			else:
				new_node_set, new_adj = self.forward_graph_Memory_based_Pooling(node_set, adj, zero_tensor)
			new_x_2 = new_node_set.squeeze(0)
			new_edge_index_2, new_edge_attr_2 = dense_to_sparse(new_adj.squeeze(0))
			data_2 = PyGSingleGraphData(x=new_x_2, edge_index=new_edge_index_2, edge_attr=new_edge_attr_2, y=None)
			data_2, _ = encode_node_features(pyg_single_g=data_2)

			single_graph_list.extend([data_1, data_2])
		# finish = datetime.datetime.now()
		# print('finish pooling!','pooling time: ',finish-st)
		# print('start update data')
		batch_data.update_data(single_graph_list, metadata_list)

		# print('finish update data')
		new_ins = batch_data.merge_data['merge'].x

		'''
		node_set =torch.zeros(256,100,64)
		adj = torch.zeros(256,100,100)
		for i in range(0,len(node_set_input)):
			node_set[i,0:node_set_input[i].size()[0],0:node_set_input[i].size()[1]] = node_set_input[i]
			adj_i = to_dense_adj(batch_data.single_graph_list[i].edge_index).squeeze()
			adj[i,0:adj_i.size()[0],0:adj_i.size()[1]] = adj_i

		new_node_set , new_adj  = self.forward_one_graph(node_set,adj,zero_tensor)
		finish = datetime.datetime.now()
		new_ins = new_node_set
		batch_data.merge_data['merge'].x = new_ins
		batch_data.merge_data['merge'].edge_index , _ =dense_to_sparse(new_adj.squeeze())
		'''
		return new_ins

	def forward_graph_Memory_based_Pooling(self, node_set, adj, zero_tensor):
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

		if self.with_mask and (not adj is None):
			graph_sizes = node_set.size()[1] + zero_tensor
			# size : [batch_size, N_output, N_input]
			aranger = torch.arange(adj.shape[1]).view(1, 1, -1).repeat(adj.shape[0], self.N_output, 1)
			# size : [batch_size, N_output, N_input]
			graph_broad = graph_sizes.view(1, 1, 1).repeat(adj.shape[0], self.N_output, adj.shape[1])
			self.mask = aranger.float() < graph_broad
			self.mask = self.mask.float()
		else:
			self.mask = None

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
		if self.mask is not None:
			mask_broad = torch.unsqueeze(self.mask, 1).repeat(1, self.Heads, 1, 1)
			dist = dist * mask_broad
		# Compute the Matrix C_heads : [batch_size, Heads, N_output, N_input]
		C_heads = torch.pow((1 + dist / self.Tau), -(self.Tau + 1) / 2)

		normalizer = torch.unsqueeze(torch.sum(C_heads, 2), 2)
		C_heads = C_heads / normalizer
		if self.mask is not None:
			mask_broad = torch.unsqueeze(self.mask, 1).repeat(1, self.Heads, 1, 1)
			C_heads = C_heads * mask_broad

		# Apply pixel-level convolution and softmax to C_heads
		# Get C: [batch_size, N_output, N_input]
		C = self.memory_aggregation(C_heads)
		C = torch.softmax(C, 1)

		if self.mask is not None:
			mask_broad = torch.unsqueeze(self.mask, 1)
			C = C * mask_broad
		C = C.squeeze(1)

		# [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
		new_node_set = torch.matmul(C, node_set_input)
		# [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
		new_node_set = self.dim_feat_transformation(new_node_set)
		new_node_set = self.lrelu(new_node_set)

		"""
			Calculate new_adj
		"""

		if not adj is None:

			# [batch_size, N_output, N_input] * [batch_size, N_input, N_input] --> [batch_size, N_output, N_input]
			q_adj = torch.matmul(C, adj)

			# [batch_size, N_output, N_input] * [batch_size, N_input, N_output] --> [batch_size, N_output, N_output]
			new_adj = torch.matmul(q_adj, C.transpose(1, 2))

			if self.p2p and not (self.N_output == 1):
				dg = torch.diag((zero_tensor + 1).repeat(new_adj.shape[1]))
				new_adj = torch.unsqueeze(dg, 0).repeat(new_adj.shape[0], 1, 1)

			return new_node_set, new_adj

		else:
			return new_node_set

	def forward_graph_Memory_based_Pooling_plus(self, node_set, adj, zero_tensor):
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

		if self.with_mask and not (adj == None):
			graph_sizes = node_set.size()[1] + zero_tensor
			# size : [batch_size, N_output, N_input]
			aranger = torch.arange(adj.shape[1]).view(1, 1, -1).repeat(adj.shape[0], self.N_output, 1)
			# size : [batch_size, N_output, N_input]
			graph_broad = graph_sizes.view(1, 1, 1).repeat(adj.shape[0], self.N_output, adj.shape[1])
			self.mask = aranger < graph_broad
		else:
			self.mask = None

		"""
			With (1)(2)(3)(4) we calculate new_node_set
		"""

		# Copy centroids and repeat it in batch
		## [h,N_output,Dim_input] --> [batch_size,Heads,N_output,Dim_input]

		# Copy centroids and repeat it in batch
		## [h,N_output,Dim_input] --> [batch_size,Heads,N_output,Dim_input]
		# _, distances = k_nn(node_set, self.centroids.unsqueeze(0))
		# distances_sum = torch.sum(distances,dim=2).squeeze()
		# distances_sorted, indexes = torch.sort(distances_sum)

		batch_centroids = torch.mean(node_set,dim=1,keepdim=True)
		batch_centroids = batch_centroids.permute(0, 2, 1)

		batch_centroids = torch.relu(torch.nn.functional.linear(batch_centroids, self.input2centroids_1_weight[:, 0:self.N_input],
													 self.input2centroids_1_bias))
		batch_centroids = self.input2centroids_2(batch_centroids)
		batch_centroids = self.input2centroids_3(batch_centroids)
		#batch_centroids = self.input2centroids_4(batch_centroids)
		#batch_centroids = self.input2centroids_5(batch_centroids)
		#batch_centroids = self.input2centroids_6(batch_centroids)
		batch_centroids = batch_centroids.permute(0, 2, 1).view(node_set.size()[0], self.Heads, self.N_output,
																self.Dim_input)

		# From initial node set [batch_size, N_input, Dim_input] to size [batch_size, Heads, N_output, N_input, Dim_input]
		node_set = torch.unsqueeze(node_set, 1).repeat(1, batch_centroids.shape[1], 1, 1)
		node_set = torch.unsqueeze(node_set, 2).repeat(1, 1, batch_centroids.shape[2], 1, 1)
		# Broadcast centroids to the same size as that of node set [batch_size, Heads, N_output, N_input, Dim_input]
		batch_centroids = torch.unsqueeze(batch_centroids, 3).repeat(1, 1, 1, node_set.shape[3], 1)

		# Compute the distance between original node set to centroids
		# [batch_size, Heads, N_output, N_input]
		C_heads = 1 / (self.similarity_compute(node_set, batch_centroids) + 1e-6)
		# dist = torch.sum(torch.abs(node_set - batch_centroids) ** 2, 4)
		# if self.mask is not None:
		#    mask_broad = torch.unsqueeze(self.mask, 1).repeat(1, self.Heads, 1, 1)
		#    dist = dist * mask_broad
		# Compute the Matrix C_heads : [batch_size, Heads, N_output, N_input]
		# C_heads = torch.pow((1 + dist / self.Tau), -(self.Tau + 1) / 2)

		normalizer = torch.unsqueeze(torch.sum(C_heads, 2), 2)
		C_heads = C_heads / normalizer
		if self.mask is not None:
			mask_broad = torch.unsqueeze(self.mask, 1).repeat(1, self.Heads, 1, 1)
			C_heads = C_heads * mask_broad

		# Apply pixel-level convolution and softmax to C_heads
		# Get C: [batch_size, N_output, N_input]
		C = self.memory_aggregation(C_heads)
		# C = torch.softmax(C,1)

		if self.mask is not None:
			mask_broad = torch.unsqueeze(self.mask, 1)
			C = C * mask_broad

		C = C.squeeze(1)

		# [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
		new_node_set = torch.matmul(C, node_set_input)
		# [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
		new_node_set = self.dim_feat_transformation(new_node_set)
		#new_node_set = self.relu(new_node_set)

		"""
			Calculate new_adj
		"""

		if not adj is None:
			# [batch_size, N_output, N_input] * [batch_size, N_input, N_input] --> [batch_size, N_output, N_input]
			q_adj = torch.matmul(C, adj)

			# [batch_size, N_output, N_input] * [batch_size, N_input, N_output] --> [batch_size, N_output, N_output]
			new_adj = self.relu(torch.matmul(q_adj, C.transpose(1, 2)))

			if self.p2p and not (self.N_output == 1):
				dg = torch.diag((zero_tensor + 1).repeat(new_adj.shape[1]))
				new_adj = torch.unsqueeze(dg, 0).repeat(new_adj.shape[0], 1, 1)

			return new_node_set, new_adj

		else:
			return new_node_set

class Max_AVG_Pooling_Layer(nn.Module):

	def __init__(self,Dim_input,Dim_output=512,max_node_num = 200,end_pooling=False,max_pooling=False,global_pool=False):

		super(Max_AVG_Pooling_Layer, self).__init__()
		self.max_pooling = max_pooling
		self.end_pooling = end_pooling
		self.conv = GCNConv(Dim_input, Dim_input)
		self.bn = torch.nn.BatchNorm1d(Dim_input)
		self.Dim_input = Dim_input
		self.MLP_1 = nn.Sequential(#nn.Dropout(0.5),
								   nn.Linear(Dim_input,Dim_input),
								   nn.ReLU(),)
								   #nn.Dropout(0.5),)
								   #nn.Linear(Dim_input,Dim_input),
								   #nn.LeakyReLU(),
								   #nn.Dropout(0.5),)
								   #nn.Linear(Dim_input//2,Dim_input//2),
								   #nn.ReLU(),
								   #nn.Dropout(0.5),
								   #nn.Linear(Dim_input//2,Dim_input//4),
								   #nn.ReLU(),
								   #nn.Dropout(0.5),
								   #nn.Linear(Dim_input//4,Dim_input//4),
								   #nn.ReLU(),
								   #nn.Dropout(0.5))

		self.MLP_2_weight = torch.nn.Parameter(
			torch.randn(Dim_output//16, max_node_num * Dim_input).float()
				.to(FLAGS.device), requires_grad=True)
		self.MLP_2_bias = torch.nn.Parameter(torch.randn(Dim_output//16).float()
														 .to(FLAGS.device), requires_grad=True)
		self.relu = nn.ReLU()
		self.MLP_3 = nn.Sequential(
								   nn.Linear(Dim_output//16,Dim_output),
								   nn.ReLU(),
								   )
								   #nn.Dropout(0.5),
								   #nn.Linear(Dim_output//2,Dim_output//2),
								   #nn.ReLU())
		self.MLP_3_weight = torch.nn.Parameter(
			torch.randn(Dim_output, Dim_output).float()
				.to(FLAGS.device), requires_grad=True)
		self.MLP_3_bias = torch.nn.Parameter(torch.randn(Dim_output).float()
											 .to(FLAGS.device), requires_grad=True)

		#self.similarity_compute = torch.nn.PairwiseDistance()
		if FLAGS.dos_true == 'sim':
			self.similarity_compute = nn.PairwiseDistance()
		else:
			self.similarity_compute = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
		self.criterion = nn.MSELoss()
		self.global_pool = global_pool
	def forward(self, ins, batch_data, model):
		if not self.end_pooling:

			node_set_input = batch_data.split_into_batches(ins)
			single_graph_list = []
			metadata_list = []

			if not self.global_pool:
				for i in range(0, len(node_set_input)):
					node_set = node_set_input[i]
					edge_index = batch_data.single_graph_list[i].edge_index
					if edge_index.size()[1] > 0:
						data = {}
						data['x'] = node_set
						batch = torch.zeros(node_set.size()[0])
						data['batch'] = batch
						data['edge_index'] = edge_index
						Data_graph = torch_geometric.data.Data.from_dict(data)
						cluster = geo_nn.graclus(edge_index)

						if self.max_pooling:
							pooled_Data_graph = geo_nn.max_pool(cluster, Data_graph)
						else:
							pooled_Data_graph = geo_nn.avg_pool(cluster,Data_graph)
						new_x = pooled_Data_graph.x.squeeze(0)
						if len(new_x.size()) == 1:
							new_x = new_x.unsqueeze(0)
						new_edge_index = pooled_Data_graph.edge_index
					else :
						new_x = node_set
						new_edge_index = edge_index

					pooled_data = PyGSingleGraphData(x=new_x, edge_index=new_edge_index, edge_attr=None, y=None)

					pooled_data, _ = encode_node_features(pyg_single_g=pooled_data)

					single_graph_list.extend([pooled_data])

			else:
				for i in range(0, len(node_set_input)):
					node_set = node_set_input[i]
					edge_index = batch_data.single_graph_list[i].edge_index
					batch = torch.zeros(node_set.size()[0]).long()
					new_x = geo_nn.global_mean_pool(node_set,batch)
					new_edge_index = torch.zeros(2,1)
					pooled_data = PyGSingleGraphData(x=new_x, edge_index=new_edge_index, edge_attr=None, y=None)
					pooled_data, _ = encode_node_features(pyg_single_g=pooled_data)

					single_graph_list.extend([pooled_data])

			batch_data.update_data(single_graph_list, metadata_list)
			new_ins = batch_data.merge_data['merge'].x

			return  new_ins
		else:
			pair_list = batch_data.split_into_pair_list(ins, 'x')

			ind_list = batch_data.merge_data['ind_list']
			pairwise_embeddings = []
			true = torch.zeros(len(pair_list), 1, device=FLAGS.device)

			for i in range(0, len(ind_list), 2):
				g1_ind = i
				g2_ind = i + 1
				x_1 = ins[ind_list[g1_ind][0]: ind_list[g1_ind][1]]
				x_2 = ins[ind_list[g2_ind][0]: ind_list[g2_ind][1]]
				edge_index_1 = batch_data.single_graph_list[i].edge_index
				edge_index_2 = batch_data.single_graph_list[i+1].edge_index
				pair = pair_list[int(i//2)]
				'''
				g_1 , g_2 = pair.g1.nxgraph, pair.g2.nxgraph
				g_1_ = nx.convert_node_labels_to_integers(g_1)
				g_1_ = g_1_.to_directed() if not nx.is_directed(g_1_) else g_1_
				g_2_ = nx.convert_node_labels_to_integers(g_2)
				g_2_ = g_2_.to_directed() if not nx.is_directed(g_2_) else g_2_

				edge_index_1 = torch.tensor(list(g_1_.edges)).t()

				'''
				if edge_index_1.size()[1] > 0 :
					cluster_1 = geo_nn.graclus(edge_index_1)
					data_1 = {}
					data_1['x'] = x_1
					batch = torch.zeros(x_1.size()[0])
					data_1['batch'] = batch
					data_1['edge_index'] = edge_index_1
					g_1_data = torch_geometric.data.Data.from_dict(data_1)
					if self.max_pooling:
						pooled_x_1 = geo_nn.max_pool(cluster_1, g_1_data)
					else:
						pooled_x_1 = geo_nn.avg_pool(cluster_1,g_1_data)
					pooled_x_1 = pooled_x_1.x
				else:
					pooled_x_1 = x_1
				pooled_x_1 = self.MLP_1(pooled_x_1)
				node_num, f_dim = pooled_x_1.size()
				pooled_x_1 = pooled_x_1.view(-1)
				pooled_x_1 = self.relu(torch.nn.functional.linear(pooled_x_1,
															 self.MLP_2_weight[:, 0:node_num * f_dim],
															 self.MLP_2_bias))
				pooled_x_1 = self.MLP_3(pooled_x_1)

				if edge_index_2.size()[1] > 0:
					cluster_2 = geo_nn.graclus(edge_index_2)
					data_2 = {}
					data_2['x'] = x_2
					batch = torch.zeros(x_2.size()[0])
					data_2['batch'] = batch
					data_2['edge_index'] = edge_index_2
					g_2_data = torch_geometric.data.Data.from_dict(data_2)
					if self.max_pooling:
						pooled_x_2 = geo_nn.max_pool(cluster_2, g_2_data)
					else:
						pooled_x_2 = geo_nn.avg_pool(cluster_2, g_2_data)
					pooled_x_2 = pooled_x_2.x
				else:
					pooled_x_2 = x_2
				pooled_x_2 = self.MLP_1(pooled_x_2)
				node_num, f_dim = pooled_x_2.size()
				pooled_x_2 = pooled_x_2.view(-1)
				pooled_x_2 = self.relu(torch.nn.functional.linear(pooled_x_2,
														self.MLP_2_weight[:, 0:node_num * f_dim],
														self.MLP_2_bias))
				#pooled_x_2 = torch.nn.functional.linear(pooled_x_2,
				#										self.MLP_3_weight,
				#										self.MLP_3_bias)
				pooled_x_2 = self.MLP_3(pooled_x_2)

				pred_score =self.similarity_compute(pooled_x_1.unsqueeze(0),pooled_x_2.unsqueeze(0))

				pairwise_embeddings.append(pred_score.view(1,1))
				pair.assign_ds_pred(pred_score)
				true[i//2] = pair.get_ds_true(
					FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)


			# print("pairwise_embeddings before sigmoid")
			# print(pairwise_embeddings)
			# MLPs
			# pairwise_embeddings = torch.sigmoid(torch.cat(pairwise_embeddings, 0))
			#pairwise_embeddings =
			#pairwise_embeddings = torch.tensor(pairwise_embeddings).unsqueeze(1)
			#pairwise_embeddings.requires_grad = True

			# print("pairwise_embeddings after sigmoid")
			# print(pairwise_embeddings)
			# print("-----------------------")
			# for proj_layer in self.proj_layers:
			#     pairwise_embeddings = proj_layer(pairwise_embeddings)
			#print("pairwise_embeddings",pairwise_embeddings)

			pairwise_embeddings = (torch.cat(pairwise_embeddings,0))
			pairwise_scores = pairwise_embeddings

			#pairwise_scores = torch.tensor(pairwise_embeddings)
			#pairwise_scores.requires_grad = True
			#true = true.squeeze()
			# pairwise_scores = torch.sigmoid(pairwise_scores)
			# self.scores = pairwise_scores
			# for pair in pair_list:
			#     pair.assign_ds_pred(pairwise_scores[i])

			#print("True", true.reshape(-1, ))
			#print("Pair", pairwise_scores.reshape(-1, ))
			# assert pairwise_scores.shape == (len(pair_list), 1)
			if FLAGS.dos_true == 'sim':
				true_score = torch.tensor(true).long().float()
				true_score.requires_grad = True
				loss = torch.mean(torch.relu(0.1 - true_score * (1 - pairwise_scores)))
				pos_index = torch.where(pairwise_scores < 0.9)
				neg_index = torch.where(pairwise_scores > 1.1)
				classification = torch.zeros(true_score.size()).long()
				classification[neg_index] = -1
				classification[pos_index] = 1
				auc = classification * true_score.long() - 1
				index = auc.nonzero()
				batch_data.auc_classification = true_score.size()[0] - index.size()[0]
				batch_data.sample_num = true_score.size()[0]
			else:
				loss = self.criterion(pairwise_scores, true)
			return loss

class Diff_Pooling_Layer(nn.Module):

	def __init__(self,pool_type , ratio = 6):

		super(Diff_Pooling_Layer, self).__init__()
		#self.S = torch.nn.Parameter(torch.zeros(1,max_node_num,num_feat).float()
		#												 .to(FLAGS.device), requires_grad=True)
		if pool_type == 'topk':
			self.pool = geo_nn.TopKPooling(64,1/ratio)
		elif pool_type == 'sag':
			self.pool = SAGPooling(64,1/ratio)
	def forward(self, ins, batch_data, model):

		node_set_input = batch_data.split_into_batches(ins)
		single_graph_list = []
		metadata_list = []

		for i in range(0, len(node_set_input)):
			node_set = node_set_input[i]
			edge_index = batch_data.single_graph_list[i].edge_index
			pooled = self.pool(node_set,edge_index)
			new_x = pooled[0]
			new_edge_index = pooled[1]
			#new_x , new_edge_index = self.pool(node_set,edge_index)
			#adj = to_dense_adj(edge_index)
			#new_x , new_adj , _ = geo_nn.dense_diff_pool(node_set.unsqueeze(0),adj.unsqueeze(0),self.S[:,0:node_set.size()[0],:])
			#new_edge_index, _ = dense_to_sparse(new_adj.squeeze())
			new_x = new_x.squeeze()
			pooled_data = PyGSingleGraphData(x=new_x, edge_index=new_edge_index, edge_attr=None, y=None)
			pooled_data, _ = encode_node_features(pyg_single_g=pooled_data)

			single_graph_list.extend([pooled_data])

		batch_data.update_data(single_graph_list, metadata_list)
		new_ins = batch_data.merge_data['merge'].x

		return  new_ins