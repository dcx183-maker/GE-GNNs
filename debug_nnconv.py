import numpy as np
import torch
import paddle
import dgl
from dgl.nn.pytorch import NNConv
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)
paddle.seed(42)
paddle.set_device('cpu')

# 创建测试数据
node_feats = np.random.rand(2, 64).astype(np.float32)
edge_feats = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32).reshape(4, 1)

# PyTorch NNConv
edge_network = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 64 * 64)
)
pt_nnconv = NNConv(64, 64, edge_network, aggregator_type='sum')
pt_nnconv.eval()

# 创建图
g = dgl.graph(([0, 1, 0, 1], [1, 0, 0, 1]), num_nodes=2)

pt_node_feats = torch.tensor(node_feats)
pt_edge_feats = torch.tensor(edge_feats)

with torch.no_grad():
    pt_out = pt_nnconv(g, pt_node_feats, pt_edge_feats)
print(f"PyTorch NNConv 输出:\n{pt_out}")

# 手动实现
class ManualNNConvDebug(paddle.nn.Layer):
    def __init__(self):
        super(ManualNNConvDebug, self).__init__()
    
    def forward(self, node_feats, edge_weights, src_nodes, dst_nodes):
        outputs = []
        for v in range(node_feats.shape[0]):
            msgs_v = []
            for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
                if dst == v:
                    msg = paddle.matmul(node_feats[src].unsqueeze(0), edge_weights[i])
                    msgs_v.append(msg.squeeze(0))
            outputs.append(paddle.stack(msgs_v, axis=0).sum(axis=0) if msgs_v else paddle.zeros([node_feats.shape[1]], dtype=node_feats.dtype))
        return paddle.stack(outputs, axis=0)

# 获取边网络权重
w_edge0 = pt_nnconv.edge_func[0].weight.detach().cpu().numpy().T  # [1, 32]
b_edge0 = pt_nnconv.edge_func[0].bias.detach().cpu().numpy()
w_edge2 = pt_nnconv.edge_func[2].weight.detach().cpu().numpy().T  # [32, 64*64]
b_edge2 = pt_nnconv.edge_func[2].bias.detach().cpu().numpy()

# 计算边权重
pd_edge_feats = paddle.to_tensor(edge_feats)
pd_w0 = paddle.to_tensor(w_edge0)
pd_w2 = paddle.to_tensor(w_edge2)

e_out = paddle.matmul(pd_edge_feats, pd_w0) + paddle.to_tensor(b_edge0)
e_out = paddle.nn.functional.relu(e_out)
e_out = paddle.matmul(e_out, pd_w2) + paddle.to_tensor(b_edge2)
edge_weights = paddle.reshape(e_out, [-1, 64, 64])

print(f"\n边权重形状: {edge_weights.shape}")

# 手动NNConv
manual_nnconv = ManualNNConvDebug()
pd_node_feats = paddle.to_tensor(node_feats)
pd_out = manual_nnconv(pd_node_feats, edge_weights, [0, 1, 0, 1], [1, 0, 0, 1])
print(f"\n手动NNConv输出:\n{pd_out}")

# 比较
diff = np.abs(pt_out.detach().cpu().numpy() - pd_out.numpy()).max()
print(f"\nNNConv最大差值: {diff:.6e}")
