import pgl
import paddle
import numpy as np
from pgl.nn import GCNConv as GraphConv

class NNConv(paddle.nn.Layer):
    def __init__(self, in_size, out_size, edge_network, aggregator='sum'):
        super(NNConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.edge_network = edge_network
        self.aggregator = aggregator

    def forward(self, graph, node_feats, edge_feats):
        edges = paddle.to_tensor(graph.edges, dtype='int64')
        src = edges[:, 0]
        dst = edges[:, 1]
        
        edge_weight = self.edge_network(edge_feats)
        edge_weight = paddle.reshape(edge_weight, [-1, self.out_size, self.in_size])
        
        h_src = paddle.gather(node_feats, src, axis=0).unsqueeze(-1)
        msg = paddle.matmul(edge_weight, h_src).squeeze(-1)
        
        sort_idx = paddle.argsort(dst)
        dst_sorted = paddle.gather(dst, sort_idx, axis=0)
        msg_sorted = paddle.gather(msg, sort_idx, axis=0)
        
        if self.aggregator == 'sum':
            out = pgl.math.segment_sum(msg_sorted, dst_sorted)
        elif self.aggregator == 'mean':
            out = pgl.math.segment_mean(msg_sorted, dst_sorted)
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")
        return out


def get_n_params(model):
    n_params = 0
    for item in list(model.parameters()):
        item_param = 1
        for dim in item.shape:
            item_param = item_param * dim
        n_params += item_param
    return n_params


def get_activation(activation, get_nn=False):
    if activation == None or activation in ["relu", "ReLU", "RELU"]:
        if get_nn: return paddle.nn.ReLU
        return paddle.nn.functional.relu
    elif activation in ["elu", "ELU"]:
        if get_nn: return paddle.nn.ELU
        return paddle.nn.functional.elu
    elif activation in ["LeakyReLU", "leakyrelu"]:
        if get_nn: return paddle.nn.LeakyReLU
        return paddle.nn.functional.leaky_relu
    if get_nn: return paddle.nn.ReLU
    return paddle.nn.functional.relu


class MPNNconv(paddle.nn.Layer):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=128, edge_hidden_feats=32, num_step_message_passing=6, activation="relu"):
        super(MPNNconv, self).__init__()
        self.mpnn_activation = get_activation(activation)
        self.project_node_feats = paddle.nn.Sequential(
            paddle.nn.Linear(node_in_feats, node_out_feats),
            get_activation(activation, get_nn=True)(),
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = paddle.nn.Sequential(
            paddle.nn.Linear(edge_in_feats, edge_hidden_feats),
            get_activation(activation, get_nn=True)(),
            paddle.nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
        )
        self.gnn_layer = NNConv(
            in_size=node_out_feats, out_size=node_out_feats, edge_network=edge_network, aggregator='sum'
        )
        self.gru = paddle.nn.GRU(input_size=node_out_feats, hidden_size=node_out_feats, time_major=True)

    def reset_parameters(self):
        try:
            self.project_node_feats[0].reset_parameters()
            self.gru.reset_parameters()
        except:
            pass

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)
        for _ in range(self.num_step_message_passing):
            node_feats = self.mpnn_activation(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats


class solvgnn_binary(paddle.nn.Layer):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim+1,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1,
                                     activation=mpnn_activation)
        self.mlp_activation = get_activation(mlp_activation)
        self.classify1 = paddle.nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = paddle.nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = paddle.nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1']
        g2 = solvdata['g2']
        
        h1 = g1.node_feat["h"]
        h2 = g2.node_feat["h"]
        solv1x = solvdata["solv1_x"]
        inter_hb = solvdata["inter_hb"]
        intra_hb1 = solvdata["intra_hb1"]
        intra_hb2 = solvdata["intra_hb2"]
        
        h1_temp = paddle.nn.functional.relu(self.conv1(g1, h1))
        h1_temp = paddle.nn.functional.relu(self.conv2(g1, h1_temp))
        h2_temp = paddle.nn.functional.relu(self.conv1(g2, h2))
        h2_temp = paddle.nn.functional.relu(self.conv2(g2, h2_temp))
        
        hg1 = pgl.math.segment_mean(h1_temp, solvdata["g1"].graph_ids)
        hg2 = pgl.math.segment_mean(h2_temp, solvdata["g2"].graph_ids)
        
        hg1 = paddle.concat((hg1, solv1x[:, None]), axis=1)
        hg2 = paddle.concat((hg2, 1 - solv1x[:, None]), axis=1)
        
        edge_feats = paddle.concat((inter_hb.repeat(2), intra_hb1, intra_hb2)).unsqueeze(1)
        hg = self.global_conv1(empty_solvsys, paddle.concat((hg1, hg2), axis=0), edge_feats)
        
        output = self.mlp_activation(self.classify1(hg))
        output = self.mlp_activation(self.classify2(output))
        output = self.classify3(output)
        
        output = paddle.concat(
            (output[0:len(output)//2, :],
            output[len(output)//2:, :]), axis=1)
        
        return output
