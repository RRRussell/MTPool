import torch
import torch.nn as nn

from torch_geometric.data import Data as PyGSingleGraphData
from utils_mp import *

EPS = 1e-15

class Memory_Pooling_Layer(nn.Module):
    """ Memory_Pooling_Layer introduced in the paper 'MEMORY-BASED GRAPH NETWORKS' by Amir Hosein Khasahmadi, etc"""

    ### This layer is for downsampling a node set from N_input to N_output
    ### Input: [B,N_input,Dim_input]
    ### 		B:batch size, N_input: number of input nodes, Dim_input: dimension of features of input nodes
    ### Output:[B,N_output,Dim_output]
    ### 		B:batch size, N_output: number of output nodes, Dim_output: dimension of features of output nodes

    def __init__(self, Heads, Dim_input, N_output, Dim_output, max_num_nodes=200, with_mask=False, Tau=1,
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

        self.input2centroids_1_weight = torch.nn.Parameter(
            torch.zeros(self.Heads * self.N_output, self.max_num_nodes).float()
            .to(FLAGS.device), requires_grad=True)
        self.input2centroids_1_bias = torch.nn.Parameter(torch.zeros(self.Heads * self.N_output).float()
                                                         .to(FLAGS.device), requires_grad=True)
        self.input2centroids_2_weight = torch.nn.Parameter(
            torch.zeros(self.Heads * self.N_output, self.Heads * self.N_output).float()
            .to(FLAGS.device), requires_grad=True)
        self.input2centroids_2_bias = torch.nn.Parameter(torch.zeros(self.Heads * self.N_output).float()
                                                         .to(FLAGS.device), requires_grad=True)
        self.input2centroids_3_weight = torch.nn.Parameter(
            torch.zeros(self.Heads * self.N_output, self.Heads * self.N_output).float()
            .to(FLAGS.device), requires_grad=True)
        self.input2centroids_3_bias = torch.nn.Parameter(torch.zeros(self.Heads * self.N_output).float()
                                                         .to(FLAGS.device), requires_grad=True)

        self.memory_aggregation = nn.Conv2d(self.Heads, 1, [1, 1])
        self.dim_feat_transformation = nn.Linear(self.Dim_input, self.Dim_output)
        self.lrelu = nn.LeakyReLU()
        self.similarity_compute = torch.nn.CosineSimilarity(dim=4, eps=1e-6)
        self.similarity_compute_1 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, ins, batch_data, model):
        zero_tensor = torch.zeros
        node_set_input = batch_data.split_into_batches(ins)
        single_graph_list = []
        metadata_list = []
        new_ins = []

        # print('start pooling')
        # st = datetime.datetime.now()
        for i in range(0, len(node_set_input), 2):
            node_set = node_set_input[i].unsqueeze(0)
            adj = to_dense_adj(batch_data.single_graph_list[i].edge_index)
            new_node_set, new_adj = self.forward_graph_Memory_based_Pooling_plus(node_set, adj, zero_tensor)
            new_x_1 = new_node_set.squeeze()
            new_edge_index_1, _ = dense_to_sparse(new_adj.squeeze())
            data_1 = PyGSingleGraphData(x=new_x_1, edge_index=new_edge_index_1, edge_attr=None, y=None)
            data_1, _ = encode_node_features(pyg_single_g=data_1)

            node_set = node_set_input[i + 1].unsqueeze(0)
            adj = to_dense_adj(batch_data.single_graph_list[i + 1].edge_index)
            new_node_set, new_adj = self.forward_graph_Memory_based_Pooling_plus(node_set, adj, zero_tensor)
            new_x_2 = new_node_set.squeeze()
            new_edge_index_2, _ = dense_to_sparse(new_adj.squeeze())
            data_2 = PyGSingleGraphData(x=new_x_2, edge_index=new_edge_index_2, edge_attr=None, y=None)
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
        C = C.squeeze().unsqueeze(0)

        # [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
        new_node_set = torch.matmul(C, node_set_input)
        # [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
        new_node_set = self.dim_feat_transformation(new_node_set)

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
        batch_centroids = node_set.permute(0, 2, 1)
        batch_centroids = torch.nn.functional.linear(batch_centroids, self.input2centroids_1_weight[:, 0:self.N_input],
                                                     self.input2centroids_1_bias)
        batch_centroids = torch.nn.functional.linear(batch_centroids, self.input2centroids_2_weight,
                                                     self.input2centroids_2_bias)
        batch_centroids = torch.nn.functional.linear(batch_centroids, self.input2centroids_3_weight,
                                                     self.input2centroids_3_bias)
        batch_centroids = batch_centroids.permute(0, 2, 1).view(node_set.size()[0], self.Heads, self.N_output,
                                                                self.Dim_input)
        # batch_centroids = torch.unsqueeze(self.centroids, 0).\
        #    repeat(node_set.shape[0], 1, 1, 1)

        # From initial node set [batch_size, N_input, Dim_input] to size [batch_size, Heads, N_output, N_input, Dim_input]
        node_set = torch.unsqueeze(node_set, 1).repeat(1, batch_centroids.shape[1], 1, 1)
        node_set = torch.unsqueeze(node_set, 2).repeat(1, 1, batch_centroids.shape[2], 1, 1)
        # Broadcast centroids to the same size as that of node set [batch_size, Heads, N_output, N_input, Dim_input]
        batch_centroids = torch.unsqueeze(batch_centroids, 3).repeat(1, 1, 1, node_set.shape[3], 1)

        # Compute the distance between original node set to centroids
        # [batch_size, Heads, N_output, N_input]

        # dist = torch.sum(torch.abs(node_set - batch_centroids) ** 2, 4)
        # if self.mask is not None:
        #    mask_broad = torch.unsqueeze(self.mask, 1).repeat(1, self.Heads, 1, 1)
        #    dist = dist * mask_broad
        # Compute the Matrix C_heads : [batch_size, Heads, N_output, N_input]
        # C_heads = 1 / (self.similarity_compute(node_set,batch_centroids)+1e-16)
        dist = torch.sum(torch.abs(node_set - batch_centroids) ** 2, 4)
        if self.mask is not None:
            mask_broad = torch.unsqueeze(self.mask, 1).repeat(1, self.Heads, 1, 1)
            dist = dist * mask_broad
        # Compute the Matrix C_heads : [batch_size, Heads, N_output, N_input]
        C_heads = torch.pow((1 + dist / self.Tau), -(self.Tau + 1) / 2)

        # normalizer = torch.unsqueeze(torch.sum(C_heads, 2), 2)
        # C_heads = C_heads / (normalizer+1e-1)
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
        C = C.squeeze().unsqueeze(0)

        # [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
        new_node_set = torch.matmul(C, node_set_input)
        # [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
        new_node_set = self.dim_feat_transformation(new_node_set)

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

def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    rtn = nn.Parameter(torch.Tensor(*shape).cuda())
    nn.init.xavier_normal_(rtn)
    return rtn

def interact_two_sets_of_vectors(x_1, x_2, interaction_dim, V=None,W=None, b=None, act=None, U=None):
    """
    Calculates the interaction scores between each row of x_1 (a marix)
        and x_2 ( a vector).
    :param x_1: an (N, D) matrix (if N == 1, a (1, D) vector)
    :param x_2: a (1, D) vector
    :param interaction_dim: number of interactions
    :param V:
    :param W:
    :param b:
    :param act:
    :param U:
    :return: if U is not None, interaction results of (N, interaction_dim)
             if U is None, interaction results of (N, 1)
    """
    feature_map = []
    # ii = Timer()
    for i in range(interaction_dim):
        # print("i:",i)
        middle = 0.0
        if V is not None:
            # In case x_2 is a vector but x_1 is a matrix, tile x_2.
            tiled_x_2 = torch.mul(torch.ones_like(x_1, device='cuda:0'), x_2)
            concat = torch.cat((x_1, tiled_x_2), 1)
            v_weight = V[i].view(-1, 1)
            V_out = torch.mm(concat, v_weight)
            middle += V_out
        if W is not None:
            temp = torch.mm(x_1, W[i])
            h = torch.mm(temp, x_2.t())  # h = K.sum(temp*e2,axis=1)
            middle += h
        if b is not None:
            middle += b[i]
        feature_map.append(middle)
        # print("in an layer: ntn (interact)",ii.time_msec_and_clear())
    # print("--------------")
    # print(feature_map)
    output = torch.cat(feature_map, 1)
    # print(output)
    # output = F.normalize(output, p=1, dim=1)  # TODO: check why need this
    # print(output)
    if act is not None:
        output = act(output)
    if U is not None:
        output = torch.mm(output, U)
        # print(output.shape)

    return output

class Attention(nn.Module):
    """ Attention layer."""

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight):
        super(Attention, self).__init__()
        self.emb_dim = input_dim  # same dimension D as input embeddings
        self.att_times = att_times
        self.att_num = att_num
        self.att_style = att_style
        self.att_weight = att_weight
        assert (self.att_times >= 1)
        assert (self.att_num >= 1)
        assert (self.att_style == 'dot' or self.att_style == 'slm' or
                'ntn_' in self.att_style)

        self.vars = {}

        for i in range(self.att_num):
            self.vars['W_' + str(i)] = \
                glorot([self.emb_dim, self.emb_dim])
            if self.att_style == 'slm':
                self.interact_dim = 1
                self.vars['NTN_V_' + str(i)] = \
                    glorot([self.interact_dim, 2 * self.emb_dim])
            if 'ntn_' in self.att_style:
                self.interact_dim = int(self.att_style[4])
                self.vars['NTN_V_' + str(i)] = \
                    glorot([self.interact_dim, 2 * self.emb_dim])
                self.vars['NTN_W_' + str(i)] = \
                    glorot([self.interact_dim, self.emb_dim, self.emb_dim])
                self.vars['NTN_U_' + str(i)] = \
                    glorot([self.interact_dim, 1])
                self.vars['NTN_b_' + str(i)] = \
                    glorot([16,1])
        #nn.Parameter([16])#self.interact_dim])

    def forward(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, inputs):
        outputs = []
        for i in range(self.att_num):
            acts = [inputs]
            assert (self.att_times >= 1)
            output = None
            for _ in range(self.att_times):
                # p = Timer()
                x = acts[-1]  # x is N*D
                temp = torch.mean(x, 0).view((1, -1))  # (1, D)
                h_avg = torch.tanh(torch.mm(temp, self.vars['W_' + str(i)])) if self.att_weight else temp
                self.att = self._gen_att(x, h_avg, i)
                output = torch.mm(self.att.view(1, -1), x)  # (1, D)
                x_new = torch.mul(x, self.att)
                acts.append(x_new)
                # print("in an layer: att",p.time_msec_and_clear())
            outputs.append(output)
        return torch.cat(outputs, 1)

    def _gen_att(self, x, h_avg, i):
        if self.att_style == 'dot':
            return interact_two_sets_of_vectors(
                x, h_avg, 1,  # interact only once
                W=[torch.eye(self.emb_dim, device=FLAGS.device)],
                act=torch.sigmoid)
        elif self.att_style == 'slm':
            # return tf.sigmoid(tf.matmul(concat, self.vars['a_' + str(i)]))
            return interact_two_sets_of_vectors(
                x, h_avg, self.interact_dim,
                V=self.vars['NTN_V_' + str(i)],
                act=torch.sigmoid)
        else:
            assert ('ntn_' in self.att_style)
            return interact_two_sets_of_vectors(
                x, h_avg, self.interact_dim,
                V=self.vars['NTN_V_' + str(i)],
                W=self.vars['NTN_W_' + str(i)],
                b=self.vars['NTN_b_' + str(i)],
                act=torch.sigmoid,
                U=self.vars['NTN_U_' + str(i)])

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

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss