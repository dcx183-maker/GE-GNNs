import numpy as np
import torch
import paddle
import dgl
import time

np.random.seed(42)
torch.manual_seed(42)
paddle.seed(42)

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
    
    pt_empty = dgl.graph(([0, 1, 0, 1], [1, 0, 0, 1]), num_nodes=2)
    pt_empty.ndata['h'] = torch.zeros((2, 64+1))
    
    torch_solvdata = {
        "g1": pt_g1,
        "g2": pt_g2,
        "solv1_x": torch.tensor(solv1_x_val),
        "inter_hb": torch.tensor(inter_hb_val),
        "intra_hb1": torch.tensor(intra_hb1_val),
        "intra_hb2": torch.tensor(intra_hb2_val)
    }
    
    return torch_solvdata, pt_empty, feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v

def copy_weights_to_manual(torch_model, paddle_model):
    torch_params = dict(torch_model.named_parameters())
    
    if 'conv1.weight' in torch_params:
        paddle_model.w_conv1.set_value(paddle.to_tensor(torch_params['conv1.weight'].detach().cpu().numpy()))
    if 'conv2.weight' in torch_params:
        paddle_model.w_conv2.set_value(paddle.to_tensor(torch_params['conv2.weight'].detach().cpu().numpy()))
    
    if 'global_conv1.project_node_feats.0.weight' in torch_params:
        paddle_model.w_proj.set_value(paddle.to_tensor(torch_params['global_conv1.project_node_feats.0.weight'].detach().cpu().numpy().T))
    if 'global_conv1.project_node_feats.0.bias' in torch_params:
        paddle_model.b_proj.set_value(paddle.to_tensor(torch_params['global_conv1.project_node_feats.0.bias'].detach().cpu().numpy()))
    
    if 'global_conv1.gru.weight_ih_l0' in torch_params:
        paddle_model.w_ih.set_value(paddle.to_tensor(torch_params['global_conv1.gru.weight_ih_l0'].detach().cpu().numpy().T))
    if 'global_conv1.gru.weight_hh_l0' in torch_params:
        paddle_model.w_hh.set_value(paddle.to_tensor(torch_params['global_conv1.gru.weight_hh_l0'].detach().cpu().numpy().T))
    if 'global_conv1.gru.bias_ih_l0' in torch_params:
        paddle_model.b_ih.set_value(paddle.to_tensor(torch_params['global_conv1.gru.bias_ih_l0'].detach().cpu().numpy()))
    if 'global_conv1.gru.bias_hh_l0' in torch_params:
        paddle_model.b_hh.set_value(paddle.to_tensor(torch_params['global_conv1.gru.bias_hh_l0'].detach().cpu().numpy()))
    
    if 'global_conv1.gnn_layer.edge_func.0.weight' in torch_params:
        paddle_model.w_edge0.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.edge_func.0.weight'].detach().cpu().numpy().T))
    if 'global_conv1.gnn_layer.edge_func.0.bias' in torch_params:
        paddle_model.b_edge0.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.edge_func.0.bias'].detach().cpu().numpy()))
    if 'global_conv1.gnn_layer.edge_func.2.weight' in torch_params:
        paddle_model.w_edge2.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.edge_func.2.weight'].detach().cpu().numpy().T))
    if 'global_conv1.gnn_layer.edge_func.2.bias' in torch_params:
        paddle_model.b_edge2.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.edge_func.2.bias'].detach().cpu().numpy()))
    
    if 'global_conv1.gnn_layer.bias' in torch_params:
        paddle_model.b_nnconv.set_value(paddle.to_tensor(torch_params['global_conv1.gnn_layer.bias'].detach().cpu().numpy()))
    
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
print("GDI-NN 模型性能测试")
print("=" * 60)

print("\n创建测试数据...")
torch_data, pt_empty, feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v = create_mock_data()

print("初始化模型...")
from model.model_GNN import solvgnn_binary as TorchModel
from model_paddle.manual_layers import SolvGNNPaddle

torch_model = TorchModel(64, 64, n_classes=1)
torch_model.eval()

paddle_model = SolvGNNPaddle(in_dim=64, hidden_dim=64, n_classes=1)
paddle_model.eval()

copy_weights_to_manual(torch_model, paddle_model)

# 预热
print("\n预热模型...")
for _ in range(10):
    with torch.no_grad():
        torch_out = torch_model(torch_data, empty_solvsys=pt_empty, gamma_grad=False)
    paddle_out = paddle_model(feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v, [0, 1, 0, 1], [1, 0, 0, 1], num_nodes=5)

# PyTorch性能测试
print("\n" + "=" * 60)
print("PyTorch 性能测试")
print("=" * 60)
torch_times = []
for i in range(100):
    start = time.time()
    with torch.no_grad():
        torch_out = torch_model(torch_data, empty_solvsys=pt_empty, gamma_grad=False)
    end = time.time()
    torch_times.append(end - start)

torch_mean = np.mean(torch_times) * 1000
torch_std = np.std(torch_times) * 1000
print(f"PyTorch 平均耗时: {torch_mean:.2f} ms")
print(f"PyTorch 标准差:   {torch_std:.2f} ms")

# Paddle性能测试（无JIT）
print("\n" + "=" * 60)
print("Paddle 性能测试（无JIT）")
print("=" * 60)
paddle_times = []
for i in range(100):
    start = time.time()
    paddle_out = paddle_model(feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u, edges_v, [0, 1, 0, 1], [1, 0, 0, 1], num_nodes=5)
    end = time.time()
    paddle_times.append(end - start)

paddle_mean = np.mean(paddle_times) * 1000
paddle_std = np.std(paddle_times) * 1000
print(f"Paddle 平均耗时: {paddle_mean:.2f} ms")
print(f"Paddle 标准差:   {paddle_std:.2f} ms")

# Paddle JIT性能测试
print("\n" + "=" * 60)
print("Paddle JIT 性能测试")
print("=" * 60)

# 简化的JIT函数，不指定input_spec
@paddle.jit.to_static
def paddle_jit_model(feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val):
    return paddle_model(feat, solv1_x_val, inter_hb_val, intra_hb1_val, intra_hb2_val, edges_u.tolist(), edges_v.tolist(), [0, 1, 0, 1], [1, 0, 0, 1], num_nodes=5)

# JIT预热
for _ in range(5):
    paddle_jit_out = paddle_jit_model(
        paddle.to_tensor(feat),
        paddle.to_tensor(solv1_x_val),
        paddle.to_tensor(inter_hb_val),
        paddle.to_tensor(intra_hb1_val),
        paddle.to_tensor(intra_hb2_val)
    )

paddle_jit_times = []
for i in range(100):
    start = time.time()
    paddle_jit_out = paddle_jit_model(
        paddle.to_tensor(feat),
        paddle.to_tensor(solv1_x_val),
        paddle.to_tensor(inter_hb_val),
        paddle.to_tensor(intra_hb1_val),
        paddle.to_tensor(intra_hb2_val)
    )
    end = time.time()
    paddle_jit_times.append(end - start)

paddle_jit_mean = np.mean(paddle_jit_times) * 1000
paddle_jit_std = np.std(paddle_jit_times) * 1000
print(f"Paddle JIT 平均耗时: {paddle_jit_mean:.2f} ms")
print(f"Paddle JIT 标准差:   {paddle_jit_std:.2f} ms")

# 计算加速比
print("\n" + "=" * 60)
print("性能对比结果")
print("=" * 60)
print(f"PyTorch:          {torch_mean:.2f} ms")
print(f"Paddle (无JIT):   {paddle_mean:.2f} ms")
print(f"Paddle (JIT):     {paddle_jit_mean:.2f} ms")

jit_speedup = (torch_mean - paddle_jit_mean) / torch_mean * 100
print(f"\nPaddle JIT 相对 PyTorch 加速比: {jit_speedup:.2f}%")

if jit_speedup >= 30:
    print("✅ 性能测试通过！JIT加速比达到目标（>30%）")
else:
    print("❌ 性能测试未通过！JIT加速比未达到目标")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
