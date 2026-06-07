import numpy as np
import torch
import dgl
from dgl.nn.pytorch import GraphConv

np.random.seed(42)
torch.manual_seed(42)

# 创建测试数据
num_nodes = 5
edges_u = np.array([0, 1, 2, 3, 0, 4])
edges_v = np.array([1, 2, 3, 4, 2, 0])
feat = np.random.rand(num_nodes, 64).astype(np.float32)

# 测试1: 不添加自环的图
pt_g_no_self = dgl.graph((edges_u, edges_v), num_nodes=num_nodes)
pt_g_no_self.ndata['h'] = torch.tensor(feat)

torch_conv = GraphConv(64, 64, add_self_loop=False)
torch_conv.eval()

with torch.no_grad():
    torch_out_no_self = torch_conv(pt_g_no_self, pt_g_no_self.ndata['h'].float())

print("DGL GraphConv (add_self_loop=False):")
print(f"  输出 (节点0): {torch_out_no_self[0][:5]}")

# 测试2: 添加自环的图
pt_g_with_self = dgl.add_self_loop(pt_g_no_self)
pt_g_with_self.ndata['h'] = torch.tensor(feat)

torch_conv_self = GraphConv(64, 64, add_self_loop=False)  # 手动添加自环
torch_conv_self.eval()

with torch.no_grad():
    torch_out_with_self = torch_conv_self(pt_g_with_self, pt_g_with_self.ndata['h'].float())

print(f"\nDGL GraphConv (手动添加自环):")
print(f"  输出 (节点0): {torch_out_with_self[0][:5]}")

# 测试3: 默认添加自环
pt_g_default = dgl.graph((edges_u, edges_v), num_nodes=num_nodes)
pt_g_default.ndata['h'] = torch.tensor(feat)

torch_conv_default = GraphConv(64, 64)  # 默认 add_self_loop=True
torch_conv_default.eval()

with torch.no_grad():
    torch_out_default = torch_conv_default(pt_g_default, pt_g_default.ndata['h'].float())

print(f"\nDGL GraphConv (默认 add_self_loop=True):")
print(f"  输出 (节点0): {torch_out_default[0][:5]}")

# 测试4: 检查权重形状
print(f"\n权重形状: {torch_conv.weight.shape}")

# 手动计算验证
print("\n手动验证计算:")
weight = torch_conv.weight.detach().cpu().numpy()  # [64, 64]
x = feat  # [5, 64]

# h = x @ weight
h = x @ weight
print(f"h.shape: {h.shape}")

# 添加自环后的边
full_edges_u = np.concatenate([edges_u, np.arange(num_nodes)])
full_edges_v = np.concatenate([edges_v, np.arange(num_nodes)])

# 计算入度和出度
in_deg = np.bincount(full_edges_v, minlength=num_nodes)
out_deg = np.bincount(full_edges_u, minlength=num_nodes)

print(f"入度: {in_deg}")
print(f"出度: {out_deg}")

# 手动计算GCN输出
out = np.zeros_like(h)
for v in range(num_nodes):
    neighbors = full_edges_u[full_edges_v == v]
    for u in neighbors:
        scale = 1.0 / (np.sqrt(out_deg[u]) * np.sqrt(in_deg[v]))
        out[v] += h[u] * scale

print(f"\n手动计算 (节点0): {out[0][:5]}")
