import torch
import pennylane as qml
import time

print("========== Quick GPU Benchmark ==========")
use_cuda = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {use_cuda}")
if use_cuda:
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

n_wires = 5


# --- 辅助函数：替换原本的 broadcast ---
def apply_ring_cnot(wires):
    n = len(wires)
    for i in range(n):
        # 连接 i 和 (i+1)%n 形成环
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


# --- 测试 lightning.gpu ---
try:
    # 注意：确保你安装了 pennylane-lightning[gpu]
    dev_gpu = qml.device("lightning.gpu" if use_cuda else "default.qubit", wires=n_wires)


    @qml.qnode(dev_gpu, interface="torch", diff_method="adjoint")
    def circuit(x):
        for i in range(n_wires):
            qml.RX(x, wires=i)
            qml.RY(x, wires=i)

        # 修复：手动循环实现 Ring CNOT
        apply_ring_cnot(range(n_wires))

        return qml.expval(qml.PauliZ(0))


    x = torch.tensor(0.123, dtype=torch.float64, device="cuda" if use_cuda else "cpu")

    # 预热
    for _ in range(3):
        _ = circuit(x)

    # GPU benchmark
    start = time.time()
    for _ in range(50):
        _ = circuit(x)

    # 确保 CUDA 操作完成以获得准确计时
    if use_cuda:
        torch.cuda.synchronize()

    end = time.time()
    print(f"GPU circuit forward (50 calls): {(end - start) * 1000:.2f} ms")

except Exception as e:
    print("⚠️ lightning.gpu test failed:", e)
    # 如果失败，显式回退以便后续代码能继续运行（尽管下面还有单独的 CPU 测试）
    dev_gpu = qml.device("default.qubit", wires=n_wires)

# --- 测试 CPU 版本 ---
dev_cpu = qml.device("default.qubit", wires=n_wires)


@qml.qnode(dev_cpu, interface="torch", diff_method="adjoint")
def circuit_cpu(x):
    for i in range(n_wires):
        qml.RX(x, wires=i)
        qml.RY(x, wires=i)

    # 修复：手动循环实现 Ring CNOT
    apply_ring_cnot(range(n_wires))

    return qml.expval(qml.PauliZ(0))


x_cpu = torch.tensor(0.123, dtype=torch.float64)
# CPU 预热
for _ in range(3):
    _ = circuit_cpu(x_cpu)

start = time.time()
for _ in range(50):
    _ = circuit_cpu(x_cpu)
end = time.time()
print(f"CPU circuit forward (50 calls): {(end - start) * 1000:.2f} ms")
print("========================================\n")