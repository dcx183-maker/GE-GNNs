print("Step 1: Importing modules...")
import numpy as np
import torch
import paddle
import dgl
import pgl

print("Step 2: Setting seeds...")
np.random.seed(42)
torch.manual_seed(42)
paddle.seed(42)
paddle.set_device('cpu')

print("Step 3: Creating mock data...")
num_nodes = 5
edges_u = [0, 1, 2, 3, 0, 4]
edges_v = [1, 2, 3, 4, 2, 0]
feat = np.random.rand(num_nodes, 64).astype(np.float32)

print("Step 4: Creating DGL graph...")
pt_g1 = dgl.graph((edges_u, edges_v), num_nodes=num_nodes)
pt_g1.ndata['h'] = torch.tensor(feat)
print(f"DGL graph created: {pt_g1}")

print("Step 5: Creating PGL graph...")
edges_list = list(zip(edges_u, edges_v))
pd_g1 = pgl.Graph(num_nodes=num_nodes, edges=edges_list, node_feat={"h": paddle.to_tensor(feat)})
pd_g1.graph_ids = paddle.zeros([num_nodes], dtype='int64')
print(f"PGL graph created: {pd_g1}")

print("Step 6: Testing model imports...")
from model.model_GNN import solvgnn_binary as TorchModel
print("PyTorch model imported")
from model_paddle.model_GNN import solvgnn_binary as PaddleModel
print("Paddle model imported")

print("✅ All imports successful!")
