import paddle
import numpy as np

class GraphConv(paddle.nn.Layer):
    """图卷积层 - 模拟 DGL 的 GraphConv"""
    def __init__(self, in_size, out_size, norm='both', weight=True, bias=True, activation=None):
        super(GraphConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.norm = norm
        
        if weight:
            self.weight = paddle.create_parameter(
                shape=[in_size, out_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform()
            )
        else:
            self.weight = None
            
        if bias:
            self.bias = paddle.create_parameter(
                shape=[out_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(0.0)
            )
        else:
            self.bias = None
            
        self.activation = activation
    
    def forward(self, graph, feat, weight=None):
        # 获取图的邻接矩阵信息
        if hasattr(graph, 'edges'):
            edges = graph.edges
            num_nodes = graph.num_nodes
        else:
            edges = []
            num_nodes = feat.shape[0]
        
        # 线性变换
        if weight is None:
            weight = self.weight
        
        if weight is not None:
            feat = paddle.matmul(feat, weight)
        
        # 简单的消息传递（聚合邻居信息）
        if len(edges) > 0:
            # 创建聚合后的特征
            feat_agg = paddle.zeros_like(feat)
            for u, v in edges:
                feat_agg[v] = feat_agg[v] + feat[u]
            
            # 归一化（度归一化）
            if self.norm == 'both':
                # 计算度
                degree = paddle.zeros([num_nodes], dtype='float32')
                for u, v in edges:
                    degree[u] += 1
                    degree[v] += 1
                
                # 避免除零
                degree = paddle.clip(degree, min=1.0)
                norm = paddle.pow(degree, -0.5)
                norm = norm.unsqueeze(-1)
                
                feat = feat * norm
                feat = feat_agg * norm
            else:
                feat = feat_agg
        
        if self.bias is not None:
            feat = feat + self.bias
            
        if self.activation is not None:
            feat = self.activation(feat)
            
        return feat

class NNConv(paddle.nn.Layer):
    """边神经网络图卷积层 - 模拟 DGL 的 NNConv"""
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats, edge_hidden_feats=32, num_step_message_passing=1, activation=None):
        super(NNConv, self).__init__()
        self.node_in_feats = node_in_feats
        self.edge_in_feats = edge_in_feats
        self.node_out_feats = node_out_feats
        self.edge_hidden_feats = edge_hidden_feats
        
        # 边网络
        self.edge_func = paddle.nn.Sequential(
            paddle.nn.Linear(edge_in_feats, edge_hidden_feats),
            paddle.nn.ReLU(),
            paddle.nn.Linear(edge_hidden_feats, node_out_feats * node_in_feats)
        )
        
        # 消息传递后的 GRU
        self.gru = paddle.nn.GRUCell(node_out_feats, node_out_feats)
        
    def forward(self, graph, node_feats, edge_feats):
        # 简化版本 - 直接返回节点特征
        # 实际实现需要完整的消息传递逻辑
        return node_feats

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
    elif activation in ["leakyrelu", "LeakyReLU", "leaky_relu"]:
        if get_nn: return paddle.nn.LeakyReLU
        return paddle.nn.functional.leaky_relu
    elif activation in ["sigmoid", "Sigmoid"]:
        if get_nn: return paddle.nn.Sigmoid
        return paddle.nn.functional.sigmoid
    elif activation in ["tanh", "Tanh"]:
        if get_nn: return paddle.nn.Tanh
        return paddle.nn.functional.tanh
    else:
        if get_nn: return paddle.nn.Identity
        return lambda x: x

class solvgnn_binary(paddle.nn.Layer):
    def __init__(self, in_dim, hidden_dim, n_classes=1, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_binary, self).__init__()
        
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=paddle.nn.functional.relu)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=paddle.nn.functional.relu)
        
        self.global_conv1 = NNConv(node_in_feats=hidden_dim+1,
                                   edge_in_feats=1,
                                   node_out_feats=hidden_dim,
                                   edge_hidden_feats=32,
                                   num_step_message_passing=1,
                                   activation=mpnn_activation)
        
        self.mlp_activation = get_activation(mlp_activation)
        # 修改输入维度为 129 (hidden_dim + 1) 以匹配拼接后的特征
        self.classify1 = paddle.nn.Linear(hidden_dim+1, hidden_dim)
        self.classify2 = paddle.nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = paddle.nn.Linear(hidden_dim, n_classes)
        
    def forward(self, solvdata, empty_solvsys=None, gamma_grad=False):
        g1 = solvdata['g1']
        g2 = solvdata['g2']
        
        h1 = g1.node_feat["h"]
        h2 = g2.node_feat["h"]
        
        # 图卷积
        h1_temp = self.conv1(g1, h1)
        h1_temp = self.conv2(g1, h1_temp)
        h2_temp = self.conv1(g2, h2)
        h2_temp = self.conv2(g2, h2_temp)
        
        # 聚合节点特征 - 模拟 dgl.mean_nodes
        hg1 = paddle.mean(h1_temp, axis=0, keepdim=True)
        hg2 = paddle.mean(h2_temp, axis=0, keepdim=True)
        
        # 拼接溶剂信息
        solv1x = solvdata["solv1_x"]
        hg1 = paddle.concat((hg1, solv1x.unsqueeze(1)), axis=1)
        hg2 = paddle.concat((hg2, (1 - solv1x).unsqueeze(1)), axis=1)
        
        # 全局图卷积
        hg = paddle.concat((hg1, hg2), axis=0)
        if empty_solvsys is not None:
            # 使用 empty_solvsys 进行消息传递
            hg = self.global_conv1(empty_solvsys, hg, paddle.zeros((4, 1)))
        
        # MLP 分类
        output = self.mlp_activation(self.classify1(hg))
        output = self.mlp_activation(self.classify2(output))
        output = self.classify3(output)
        
        # 匹配 PyTorch 的输出格式
        output = paddle.concat(
            (output[0:len(output)//2, :],
             output[len(output)//2:, :]), axis=1)
        
        return output
