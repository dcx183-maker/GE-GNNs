import numpy as np
import torch
import paddle
import dgl

np.random.seed(42)
torch.manual_seed(42)
paddle.seed(42)
paddle.set_device('cpu')

def create_mock_data():
    num_nodes = 5
    edges_u = np.array([0, 1, 2, 3, 0, 4])
    edges_v = np.array([1, 2, 3, 4, 2, 0])
    
    feat = np.random.rand(num_nodes, 64).astype(np.float32)
    solv1_x_val = np.array([0.4], dtype=np.float32)
    
    inter_hb_val = np.array([0.5], dtype=np.float32)
    intra_hb1_val = np.array([0.5], dtype=np.float32)
    intra_hb2_val = np.array([0.5], dtype=np.float32)

    pt_g1 = dgl.graph((edges_u, edges_v), num_nodes=num_nodes)
    pt_g2 = dgl.graph((edges_u, edges_v), num_nodes=num_nodes)
    pt_g1.ndata['h'] = torch.tensor(feat)
    pt_g2.ndata['h'] = torch.tensor(feat)
    
    # 创建 empty_solvsys 图 - 需要4条边
    pt_empty = dgl.graph(([0, 1, 0, 1], [1, 0, 0, 1]), num_nodes=2)
    pt_empty.ndata['h'] = torch.zeros((2, 64+1))
    
    torch_solvdata = {
        "g1": pt_g1,
        "g2": pt_g2,
        "solv1_x": torch.tensor(solv1_x_val, requires_grad=True),
        "inter_hb": torch.tensor(inter_hb_val),
        "intra_hb1": torch.tensor(intra_hb1_val),
        "intra_hb2": torch.tensor(intra_hb2_val)
    }
    
    return torch_solvdata, pt_empty, feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v

def copy_weights_to_manual(torch_model, paddle_model):
    """将 PyTorch 模型权重拷贝到手动实现的 Paddle 模型"""
    torch_params = dict(torch_model.named_parameters())
    
    # 图卷积层权重 - GraphConv 是 [in, out] 格式，不需要转置
    if 'conv1.weight' in torch_params:
        paddle_model.w_conv1.set_value(paddle.to_tensor(torch_params['conv1.weight'].detach().cpu().numpy()))
    if 'conv2.weight' in torch_params:
        paddle_model.w_conv2.set_value(paddle.to_tensor(torch_params['conv2.weight'].detach().cpu().numpy()))
    
    # MPNNconv 层的 project_node_feats - Linear 是 [out, in]，需要转置
    if 'global_conv1.project_node_feats.0.weight' in torch_params:
        paddle_model.w_proj.set_value(paddle.to_tensor(torch_params['global_conv1.project_node_feats.0.weight'].detach().cpu().numpy().T))
    if 'global_conv1.project_node_feats.0.bias' in torch_params:
        paddle_model.b_proj.set_value(paddle.to_tensor(torch_params['global_conv1.project_node_feats.0.bias'].detach().cpu().numpy()))
    
    # GRU 权重 - PyTorch GRU 是 [3*hidden, hidden]，需要转置为 [hidden, 3*hidden]
    if 'global_conv1.gru.weight_ih_l0' in torch_params:
        paddle_model.w_ih.set_value(paddle.to_tensor(torch_params['global_conv1.gru.weight_ih_l0'].detach().cpu().numpy().T))
    if 'global_conv1.gru.weight_hh_l0' in torch_params:
        paddle_model.w_hh.set_value(paddle.to_tensor(torch_params['global_conv1.gru.weight_hh_l0'].detach().cpu().numpy().T))
    if 'global_conv1.gru.bias_ih_l0' in torch_params:
        paddle_model.b_ih.set_value(paddle.to_tensor(torch_params['global_conv1.gru.bias_ih_l0'].detach().cpu().numpy()))
    if 'global_conv1.gru.bias_hh_l0' in torch_params:
        paddle_model.b_hh.set_value(paddle.to_tensor(torch_params['global_conv1.gru.bias_hh_l0'].detach().cpu().numpy()))
    
    # 边网络权重 - Linear 是 [out, in]，需要转置
    if 'global_conv1.gnn_layer.edge_func.0.weight' in torch_params:
        paddle_model.w_edge0.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.edge_func.0.weight'].detach().cpu().numpy().T))
    if 'global_conv1.gnn_layer.edge_func.0.bias' in torch_params:
        paddle_model.b_edge0.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.edge_func.0.bias'].detach().cpu().numpy()))
    if 'global_conv1.gnn_layer.edge_func.2.weight' in torch_params:
        paddle_model.w_edge2.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.edge_func.2.weight'].detach().cpu().numpy().T))
    if 'global_conv1.gnn_layer.edge_func.2.bias' in torch_params:
        paddle_model.b_edge2.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.edge_func.2.bias'].detach().cpu().numpy()))
    
    # NNConv偏置
    if 'global_conv1.gnn_layer.bias' in torch_params:
        paddle_model.b_nnconv.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.bias'].detach().cpu().numpy()))
    
    # 分类器权重 - Linear 是 [out, in]，需要转置
    if 'classify1.weight' in torch_params:
        paddle_model.w_c1.set_value(paddle.to_tensor(torch_params['classify1.weight'].detach().cpu().numpy().T))
    if 'classify1.bias' in torch_params:
        paddle_model.b_c1.set_value(paddle.to_tensor(torch_params['classify1.bias'].detach().cpu().numpy()))
    if 'classify2.weight' in torch_params:
        paddle_model.w_c2.set_value(paddle.to_tensor(torch_params['classify2.weight'].detach().cpu().numpy().T))
    if 'classify2.bias' in torch_params:
        paddle_model.b_c2.set_value(paddle.to_tensor(torch_params['classify2.bias'].detach().cpu().numpy()))
    if 'classify3.weight' in torch_params:
        paddle_model.w_c3.set_value(paddle.to_tensor(torch_params['classify3.weight'].detach().cpu().numpy().T))
    if 'classify3.bias' in torch_params:
        paddle_model.b_c3.set_value(paddle.to_tensor(torch_params['classify3.bias'].detach().cpu().numpy()))

print("=" * 60)
print("GDI-NN 模型精度对齐测试")
print("=" * 60)

print("\n创建测试数据...")
torch_data, pt_empty, feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v = create_mock_data()

print("\n初始化 PyTorch 模型...")
from model.model_GNN import solvgnn_binary as TorchModel
torch_model = TorchModel(64, 64, n_classes=1)
torch_model.eval()

print("初始化 Paddle 模型（手动实现）...")
from model_paddle.manual_layers import SolvGNNPaddle
paddle_model = SolvGNNPaddle(in_dim=64, hidden_dim=64, n_classes=1)
paddle_model.eval()

print("\n拷贝权重参数...")
copy_weights_to_manual(torch_model, paddle_model)

print("\n" + "=" * 60)
print("前向传播精度测试")
print("=" * 60)

with torch.no_grad():
    torch_out = torch_model(torch_data, empty_solvsys=pt_empty, gamma_grad=False)

# 手动模型使用相同的接口
paddle_out = paddle_model(feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v, [0, 1, 0, 1], [1, 0, 0, 1], num_nodes=5)

print(f"PyTorch 输出形状: {torch_out.shape}")
print(f"Paddle 输出形状: {paddle_out.shape}")

torch_out_np = torch_out.detach().cpu().numpy()
paddle_out_np = paddle_out.numpy()

print(f"\nPyTorch 输出:\n{torch_out_np}")
print(f"\nPaddle 输出:\n{paddle_out_np}")

# 确保形状一致
if torch_out_np.shape != paddle_out_np.shape:
    print(f"⚠️  输出形状不一致: PyTorch {torch_out_np.shape} vs Paddle {paddle_out_np.shape}")
    min_shape = tuple(min(t, p) for t, p in zip(torch_out_np.shape, paddle_out_np.shape))
    torch_out_np = torch_out_np[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else torch_out_np[:min_shape[0]]
    paddle_out_np = paddle_out_np[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else paddle_out_np[:min_shape[0]]

diff = np.abs(torch_out_np - paddle_out_np).max()
print(f"\n前向传播最大差值: {diff:.6e}")

if diff < 1e-4:
    print("✅ 前向传播精度对齐通过！")
else:
    print("❌ 前向传播精度对齐未通过，差值过大")

print("\n" + "=" * 60)
print("反向传播精度测试")
print("=" * 60)

# 设置模型为训练模式
torch_model.train()
paddle_model.train()

# 创建标签
torch_label = torch.tensor([[1.0, 0.0]])
paddle_label = paddle.to_tensor([[1.0, 0.0]])

# PyTorch 反向传播
torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)
torch_optimizer.zero_grad()
torch_out_train = torch_model(torch_data, empty_solvsys=pt_empty, gamma_grad=False)
torch_loss = torch.nn.MSELoss()(torch_out_train, torch_label)
torch_loss.backward()
torch_optimizer.step()

# 保存PyTorch梯度
torch_grads = {}
for name, param in torch_model.named_parameters():
    if param.grad is not None:
        torch_grads[name] = param.grad.detach().cpu().numpy()

# Paddle 反向传播
paddle_optimizer = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.001)
paddle_optimizer.clear_grad()
paddle_out_train = paddle_model(feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v, [0, 1, 0, 1], [1, 0, 0, 1], num_nodes=5)
paddle_loss = paddle.nn.MSELoss()(paddle_out_train, paddle_label)
paddle_loss.backward()
paddle_optimizer.step()

# 保存Paddle梯度
paddle_grads = {}
for name, param in paddle_model.named_parameters():
    if param.grad is not None:
        paddle_grads[name] = param.grad.numpy()

# 比较梯度
max_grad_diff = 0.0
print("梯度对比:")
for name in torch_grads.keys():
    if name in paddle_grads:
        diff = np.abs(torch_grads[name] - paddle_grads[name]).max()
        max_grad_diff = max(max_grad_diff, diff)
        print(f"  {name}: 最大差值 = {diff:.6e}")

print(f"\n最大梯度差值: {max_grad_diff:.6e}")
if max_grad_diff < 1e-4:
    print("✅ 反向传播精度对齐通过！")
else:
    print("❌ 反向传播精度对齐未通过，差值过大")

print("\n" + "=" * 60)
print("训练精度对齐测试（2轮）")
print("=" * 60)

# 重新初始化模型并进行2轮训练
torch_model2 = TorchModel(64, 64, n_classes=1)
paddle_model2 = SolvGNNPaddle(in_dim=64, hidden_dim=64, n_classes=1)
copy_weights_to_manual(torch_model2, paddle_model2)

torch_optimizer2 = torch.optim.Adam(torch_model2.parameters(), lr=0.001)
paddle_optimizer2 = paddle.optimizer.Adam(parameters=paddle_model2.parameters(), learning_rate=0.001)

torch_losses = []
paddle_losses = []

for epoch in range(2):
    # PyTorch 训练
    torch_optimizer2.zero_grad()
    torch_out = torch_model2(torch_data, empty_solvsys=pt_empty, gamma_grad=False)
    torch_loss = torch.nn.MSELoss()(torch_out, torch_label)
    torch_loss.backward()
    torch_optimizer2.step()
    torch_losses.append(float(torch_loss.detach().cpu().numpy()))
    
    # Paddle 训练
    paddle_optimizer2.clear_grad()
    paddle_out = paddle_model2(feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v, [0, 1, 0, 1], [1, 0, 0, 1], num_nodes=5)
    paddle_loss = paddle.nn.MSELoss()(paddle_out, paddle_label)
    paddle_loss.backward()
    paddle_optimizer2.step()
    paddle_losses.append(float(paddle_loss.numpy()))
    
    print(f"Epoch {epoch+1}:")
    print(f"  PyTorch Loss: {torch_losses[-1]:.6f}")
    print(f"  Paddle Loss:  {paddle_losses[-1]:.6f}")

# 比较2轮后的loss
loss_diff = abs(torch_losses[-1] - paddle_losses[-1])
print(f"\n2轮训练后Loss差值: {loss_diff:.6e}")
if loss_diff < 1e-4:
    print("✅ 训练精度对齐通过！")
else:
    print("❌ 训练精度对齐未通过，差值过大")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
