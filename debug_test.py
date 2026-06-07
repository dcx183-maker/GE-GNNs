import numpy as np
import torch
import paddle
import dgl
from dgl.nn.pytorch import GraphConv

np.random.seed(42)
torch.manual_seed(42)
paddle.seed(42)
paddle.set_device('cpu')

# 创建简单测试图
num_nodes = 5
edges_u = np.array([0, 1, 2, 3, 0, 4])
edges_v = np.array([1, 2, 3, 4, 2, 0])

feat = np.random.rand(num_nodes, 64).astype(np.float32)

pt_g = dgl.graph((edges_u, edges_v), num_nodes=num_nodes)
pt_g.ndata['h'] = torch.tensor(feat)

# PyTorch GraphConv
torch_conv = GraphConv(64, 64)
torch_conv.eval()
with torch.no_grad():
    torch_out = torch_conv(pt_g, pt_g.ndata['h'].float())
print(f"PyTorch GraphConv 输出:\n{torch_out}")
print(f"PyTorch 输出形状: {torch_out.shape}")

# 查看权重形状
for name, param in torch_conv.named_parameters():
    print(f"{name}: {param.shape}")

# 手动实现
class ManualGCNDebug(paddle.nn.Layer):
    def __init__(self):
        super(ManualGCNDebug, self).__init__()
    
    def forward(self, x, weight, src_nodes, dst_nodes):
        # DGL GraphConv: h = x @ weight, 然后消息传递
        h = paddle.matmul(x, weight)
        
        # 计算入度
        in_deg = {}
        for dst in dst_nodes:
            in_deg[dst] = in_deg.get(dst, 0) + 1
        
        outputs = []
        for v in range(h.shape[0]):
            msgs = [h[u] for u, dst in zip(src_nodes, dst_nodes) if dst == v]
            outputs.append(paddle.stack(msgs, axis=0).sum(axis=0) if msgs else paddle.zeros([h.shape[1]], dtype=h.dtype))
        out = paddle.stack(outputs, axis=0)
        
        # DGL使用的归一化方式
        deg_tensor = paddle.to_tensor([1.0 / np.sqrt(in_deg.get(i, 1)) for i in range(out.shape[0])], dtype='float32').unsqueeze(1)
        return out * deg_tensor

# 使用PyTorch的权重
weight_np = torch_conv.weight.detach().cpu().numpy()
print(f"\n权重形状: {weight_np.shape}")

# 添加自环
self_loop_u, self_loop_v = np.arange(num_nodes), np.arange(num_nodes)
full_u = np.concatenate([edges_u, self_loop_u])
full_v = np.concatenate([edges_v, self_loop_v])

paddle_x = paddle.to_tensor(feat)
paddle_weight = paddle.to_tensor(weight_np)

manual_gcn = ManualGCNDebug()
paddle_out = manual_gcn(paddle_x, paddle_weight, full_u.tolist(), full_v.tolist())
print(f"\n手动实现输出:\n{paddle_out}")

diff = np.abs(torch_out.detach().cpu().numpy() - paddle_out.numpy()).max()
print(f"\n最大差值: {diff:.6e}")
