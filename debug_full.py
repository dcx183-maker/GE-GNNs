import numpy as np
import torch
import paddle
import dgl
from dgl.nn.pytorch import GraphConv

np.random.seed(42)
torch.manual_seed(42)
paddle.seed(42)
paddle.set_device('cpu')

# 创建测试数据
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

# 手动实现
class ManualGCNDebug(paddle.nn.Layer):
    def __init__(self):
        super(ManualGCNDebug, self).__init__()
    
    def forward(self, x, weight, src_nodes, dst_nodes):
        h = paddle.matmul(x, weight)
        
        in_deg = {}
        out_deg = {}
        for src, dst in zip(src_nodes, dst_nodes):
            in_deg[dst] = in_deg.get(dst, 0) + 1
            out_deg[src] = out_deg.get(src, 0) + 1
        
        outputs = []
        for v in range(h.shape[0]):
            msgs = []
            for i, (u, dst) in enumerate(zip(src_nodes, dst_nodes)):
                if dst == v:
                    scale = 1.0 / (np.sqrt(out_deg.get(u, 1)) * np.sqrt(in_deg.get(v, 1)))
                    msgs.append(h[u] * scale)
            outputs.append(paddle.stack(msgs, axis=0).sum(axis=0) if msgs else paddle.zeros([h.shape[1]], dtype=h.dtype))
        out = paddle.stack(outputs, axis=0)
        return out

weight_np = torch_conv.weight.detach().cpu().numpy()
# DGL的GraphConv不会自动添加自环（除非有0入度节点），所以不添加自环
full_u = edges_u
full_v = edges_v

paddle_x = paddle.to_tensor(feat)
paddle_weight = paddle.to_tensor(weight_np)

manual_gcn = ManualGCNDebug()
paddle_out = manual_gcn(paddle_x, paddle_weight, full_u.tolist(), full_v.tolist())

print(f"PyTorch GraphConv 输出 (第一个节点):\n{torch_out[0]}")
print(f"\n手动实现输出 (第一个节点):\n{paddle_out[0]}")

diff = np.abs(torch_out.detach().cpu().numpy() - paddle_out.numpy())
print(f"\n最大差值: {diff.max():.6e}")
print(f"平均差值: {diff.mean():.6e}")

# 检查DGL图的内部表示
print("\nDGL图信息:")
print(f"  节点数: {pt_g.number_of_nodes()}")
print(f"  边数: {pt_g.number_of_edges()}")
print(f"  入度: {pt_g.in_degrees().tolist()}")
print(f"  出度: {pt_g.out_degrees().tolist()}")

# 添加自环后的信息
pt_g_with_self = dgl.add_self_loop(pt_g)
print(f"\n添加自环后的DGL图:")
print(f"  边数: {pt_g_with_self.number_of_edges()}")
print(f"  入度: {pt_g_with_self.in_degrees().tolist()}")
print(f"  出度: {pt_g_with_self.out_degrees().tolist()}")
