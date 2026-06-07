import numpy as np
import torch
import paddle

np.random.seed(42)
torch.manual_seed(42)
paddle.seed(42)
paddle.set_device('cpu')

# 创建测试数据
node_feats = np.random.rand(2, 64).astype(np.float32)

# PyTorch GRU
pt_gru = torch.nn.GRU(64, 64)
pt_gru.eval()

pt_input = torch.tensor(node_feats).unsqueeze(0)  # [1, 2, 64]
pt_hidden = torch.tensor(node_feats).unsqueeze(0)  # [1, 2, 64]

with torch.no_grad():
    pt_out, pt_hidden_out = pt_gru(pt_input, pt_hidden)

print(f"PyTorch GRU 输出:\n{pt_out.squeeze(0)}")

# 获取权重
w_ih = pt_gru.weight_ih_l0.detach().cpu().numpy()  # [192, 64]
w_hh = pt_gru.weight_hh_l0.detach().cpu().numpy()  # [192, 64]
b_ih = pt_gru.bias_ih_l0.detach().cpu().numpy()    # [192]
b_hh = pt_gru.bias_hh_l0.detach().cpu().numpy()    # [192]

print(f"\nPyTorch GRU 权重形状:")
print(f"  w_ih: {w_ih.shape}, w_hh: {w_hh.shape}")
print(f"  b_ih: {b_ih.shape}, b_hh: {b_hh.shape}")

# 手动实现GRU
pd_input = paddle.to_tensor(node_feats)  # [2, 64]
pd_hidden = paddle.to_tensor(node_feats)  # [2, 64]

# 转置权重: PyTorch [3*hidden, hidden] -> Paddle [hidden, 3*hidden]
pd_w_ih = paddle.to_tensor(w_ih.T)  # [64, 192]
pd_w_hh = paddle.to_tensor(w_hh.T)  # [64, 192]
pd_b_ih = paddle.to_tensor(b_ih)
pd_b_hh = paddle.to_tensor(b_hh)

# GRU计算
gates_x = paddle.matmul(pd_input, pd_w_ih) + pd_b_ih  # [2, 192]
gates_h = paddle.matmul(pd_hidden, pd_w_hh) + pd_b_hh  # [2, 192]

x_r, x_z, x_n = gates_x[:, :64], gates_x[:, 64:128], gates_x[:, 128:]
h_r, h_z, h_n = gates_h[:, :64], gates_h[:, 64:128], gates_h[:, 128:]

r = paddle.nn.functional.sigmoid(x_r + h_r)
z = paddle.nn.functional.sigmoid(x_z + h_z)
n = paddle.tanh(x_n + r * h_n)
pd_out = (1 - z) * n + z * pd_hidden

print(f"\n手动GRU输出:\n{pd_out}")

diff = np.abs(pt_out.squeeze(0).detach().cpu().numpy() - pd_out.numpy()).max()
print(f"\nGRU最大差值: {diff:.6e}")
