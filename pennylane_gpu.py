import torch
import pennylane as qml
import time

print("========== Quick GPU Benchmark with HEA ==========")
use_cuda = torch.cuda.is_available()
device_type = "cuda" if use_cuda else "cpu"
print(f"torch.cuda.is_available(): {use_cuda}")

n_wires = 5
n_layers = 2  # 设置 HEA 的层数

# 生成符合 StronglyEntanglingLayers 要求的权重形状: (n_layers, n_wires, 3)
weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
# 使用 torch 初始化权重，并确保在正确的设备上
weights = torch.randn(weight_shape, dtype=torch.float64, device=device_type, requires_grad=True)

# --- 测试 lightning.gpu ---
try:
    # 注意：lightning.gpu 在 wires 较少时优势不明显，通常在 20+ wires 时起飞
    dev_gpu = qml.device("lightning.gpu" if use_cuda else "default.qubit", wires=n_wires)

    @qml.qnode(dev_gpu, interface="torch", diff_method="adjoint")
    def circuit(w):
        # 替代之前的手动 RX/RY 和 broadcast
        qml.StronglyEntanglingLayers(weights=w, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    # 预热
    for _ in range(3):
        _ = circuit(weights)

    # GPU benchmark
    if use_cuda: torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = circuit(weights)
    if use_cuda: torch.cuda.synchronize()
    print(f"GPU HEA Forward (50 calls): {(time.time() - start)*1000:.2f} ms")

except Exception as e:
    print("⚠️ lightning.gpu test failed:", e)

# --- 测试 CPU 版本 ---
dev_cpu = qml.device("default.qubit", wires=n_wires)
weights_cpu = weights.detach().cpu() # 移动到 CPU

@qml.qnode(dev_cpu, interface="torch", diff_method="adjoint")
def circuit_cpu(w):
    qml.StronglyEntanglingLayers(weights=w, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))

# 预热
for _ in range(3):
    _ = circuit_cpu(weights_cpu)

start = time.time()
for _ in range(50):
    _ = circuit_cpu(weights_cpu)
print(f"CPU HEA Forward (50 calls): {(time.time() - start)*1000:.2f} ms")
print("========================================\n")