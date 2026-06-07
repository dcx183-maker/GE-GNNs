import numpy as np
import paddle
import time

paddle.seed(42)
paddle.set_device("cpu")

from model_paddle.manual_layers import SolvGNNPaddle

num_nodes = 5
edges_u, edges_v = [0, 1, 2, 3, 0, 4], [1, 2, 3, 4, 2, 0]
sys_u, sys_v = [1, 0, 1, 0], [0, 1, 0, 1]

feat = np.random.rand(num_nodes, 64).astype("float32")
solv1_x_val = np.array([0.4], dtype="float32")

model = SolvGNNPaddle(in_dim=64, hidden_dim=64, n_classes=2)
model.eval()

jit_model = paddle.jit.to_static(model)

warmup_iter = 10
benchmark_iter = 100

print("========== JIT 加速性能对比测试 ==========")
print(f"测试配置: 前向传播 {benchmark_iter} 次，预热 {warmup_iter} 次")
print("=" * 50)

with paddle.no_grad():
    for _ in range(warmup_iter):
        model(feat, solv1_x_val, edges_u, edges_v, sys_u, sys_v, num_nodes)
        jit_model(feat, solv1_x_val, edges_u, edges_v, sys_u, sys_v, num_nodes)

    start_time = time.time()
    for _ in range(benchmark_iter):
        model(feat, solv1_x_val, edges_u, edges_v, sys_u, sys_v, num_nodes)
    end_time = time.time()
    baseline_time = end_time - start_time
    print(f"【基线模式】执行 {benchmark_iter} 次前向传播耗时: {baseline_time:.4f}s")

    start_time = time.time()
    for _ in range(benchmark_iter):
        jit_model(feat, solv1_x_val, edges_u, edges_v, sys_u, sys_v, num_nodes)
    end_time = time.time()
    jit_time = end_time - start_time
    print(f"【JIT模式】执行 {benchmark_iter} 次前向传播耗时: {jit_time:.4f}s")

speedup = (baseline_time - jit_time) / baseline_time * 100
print("=" * 50)
print(f"加速比: {speedup:.2f}%")
print(f"达标状态: {'✅ 通过 (加速 > 30%)' if speedup > 30 else '❌ 未达标 (目标 > 30%)'}")

with open("benchmark_result.txt", "w") as f:
    f.write(f"Benchmark Results for paddle.jit.to_static\n")
    f.write(f"=========================================\n")
    f.write(f"Test Configuration: {benchmark_iter} forward passes\n")
    f.write(f"Baseline Time: {baseline_time:.4f}s\n")
    f.write(f"JIT Time: {jit_time:.4f}s\n")
    f.write(f"Speedup: {speedup:.2f}%\n")
    f.write(f"Status: {'PASSED' if speedup > 30 else 'FAILED'}\n")

print("\n结果已保存到 benchmark_result.txt")

print("\n========== 开启飞桨编译器优化测试 ==========")
import paddle.compiler as compiler

compiler_model = compiler.optimize(model, backend="jit")
compiler_model.eval()

with paddle.no_grad():
    for _ in range(warmup_iter):
        compiler_model(feat, solv1_x_val, edges_u, edges_v, sys_u, sys_v, num_nodes)
    
    start_time = time.time()
    for _ in range(benchmark_iter):
        compiler_model(feat, solv1_x_val, edges_u, edges_v, sys_u, sys_v, num_nodes)
    end_time = time.time()
    compiler_time = end_time - start_time
    print(f"【编译器模式】执行 {benchmark_iter} 次前向传播耗时: {compiler_time:.4f}s")

compiler_speedup = (baseline_time - compiler_time) / baseline_time * 100
print("=" * 50)
print(f"编译器加速比: {compiler_speedup:.2f}%")
print(f"达标状态: {'✅ 通过 (加速 > 30%)' if compiler_speedup > 30 else '❌ 未达标 (目标 > 30%)'}")

with open("benchmark_result.txt", "a") as f:
    f.write(f"\nCompiler Optimization Results:\n")
    f.write(f"=============================\n")
    f.write(f"Compiler Time: {compiler_time:.4f}s\n")
    f.write(f"Compiler Speedup: {compiler_speedup:.2f}%\n")
    f.write(f"Compiler Status: {'PASSED' if compiler_speedup > 30 else 'FAILED'}\n")
