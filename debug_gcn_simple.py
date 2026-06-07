import numpy as np
import torch
import dgl
from dgl.nn.pytorch import GraphConv

np.random.seed(42)
torch.manual_seed(42)

# 创建测试数据
num_nodes = 3
edges_u = np.array([0, 1])
edges_v = np.array([1, 2])
feat = np.eye(num_nodes, 3).astype(np.float32)  # 简单的单位矩阵

# 创建图并添加自环
pt_g = dgl.graph((edges_u, edges_v), num_nodes=num_nodes)
pt_g = dgl.add_self_loop(pt_g)
pt_g.ndata['h'] = torch.tensor(feat)

# DGL GraphConv
torch_conv = GraphConv(3, 3)
torch_conv.weight.data = torch.eye(3)  # 单位矩阵权重
torch_conv.bias.data = torch.zeros(3)  # 零偏置
torch_conv.eval()

with torch.no_grad():
    torch_out = torch_conv(pt_g, pt_g.ndata['h'].float())

print("DGL GraphConv 输出:")
print(torch_out)

print("\n输入特征:")
print(feat)

# 手动计算
print("\n手动计算验证:")
print("边:", list(zip(edges_u, edges_v)))

# 添加自环
self_loop_u = np.arange(num_nodes)
self_loop_v = np.arange(num_nodes)
full_u = np.concatenate([edges_u, self_loop_u])
full_v = np.concatenate([edges_v, self_loop_v])

print("添加自环后的边:", list(zip(full_u, full_v)))

# 计算度
in_deg = np.bincount(full_v, minlength=num_nodes)
out_deg = np.bincount(full_u, minlength=num_nodes)

print(f"入度: {in_deg}")
print(f"出度: {out_deg}")

# 对称归一化计算
h = feat @ np.eye(3)  # 单位矩阵权重
out = np.zeros_like(h)

for v in range(num_nodes):
    neighbors = full_u[full_v == v]
    for u in neighbors:
        scale = 1.0 / (np.sqrt(out_deg[u]) * np.sqrt(in_deg[v]))
        out[v] += h[u] * scale

print("\n手动对称归一化输出:")
print(out)

# 另一种归一化方式: 只除以sqrt(in_deg)
out2 = np.zeros_like(h)
for v in range(num_nodes):
    neighbors = full_u[full_v == v]
    for u in neighbors:
        scale = 1.0 / np.sqrt(in_deg[v])
        out2[v] += h[u] * scale

print("\n只除以sqrt(in_deg):")
print(out2)

# 另一种归一化方式: 只除以sqrt(out_deg)
out3 = np.zeros_like(h)
for v in range(num_nodes):
    neighbors = full_u[full_v == v]
    for u in neighbors:
        scale = 1.0 / np.sqrt(out_deg[u])
        out3[v] += h[u] * scale

print("\n只除以sqrt(out_deg):")
print(out3)

# 比较
print("\n比较:")
print(f"DGL输出 - 节点0: {torch_out[0]}")
print(f"对称归一化 - 节点0: {out[0]}")
print(f"仅入度归一化 - 节点0: {out2[0]}")
print(f"仅出度归一化 - 节点0: {out3[0]}")
