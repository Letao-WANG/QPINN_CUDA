import torch
import torch.nn as nn
from torch.autograd import grad
import pennylane as qml
import numpy as np
import itertools
import torch.optim as optim
import sys
from functools import reduce


# ===============================
# Device setup
# ===============================
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# Simple PDE parameters
# ===============================
T = 1.0
scale = 10

# ===============================
# Quantum setup (不改)
# ===============================
n_qubits = 2
m = 2
L = 1
blocks = 2

dev = qml.device("lightning.gpu", wires=n_qubits)

# ===============================
# Sampling
# ===============================
def sample_collocation(n):
    t = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    x = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    return t, x

def sample_boundary(n):
    t = torch.linspace(0.99, 0.99, n, device=device).unsqueeze(1)
    x = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    return t, x

# ===============================
# Boundary condition
# ===============================
def boundary_target(x):
    return torch.sin(np.pi * x)

# ===============================
# Observable (不改)
# ===============================
obs_global = reduce(lambda a, b: a @ b,
                    [qml.PauliZ(j) for j in range(n_qubits)])

# ===============================
# Encoding (不改)
# ===============================
def novel_exp_encoding_multi(x_vec, layer):
    for j in range(m):
        coef = 3**layer
        qml.RZ(coef * x_vec[j], wires=j)

# ===============================
# QNode (不改)
# ===============================
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit(u_vec, weights_all):

    qml.templates.StronglyEntanglingLayers(
        weights_all[0], wires=range(n_qubits)
    )

    for l in range(L):
        novel_exp_encoding_multi(u_vec, l)
        qml.templates.StronglyEntanglingLayers(
            weights_all[l+1], wires=range(n_qubits)
        )

    return qml.expval(obs_global)

# ===============================
# QFM (不改)
# ===============================
class QFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights_all = nn.Parameter(
            0.1 * torch.randn(
                (L+1, blocks, n_qubits, 3),
                dtype=torch.float64,
                device=device
            )
        )

    def forward(self, t, x):
        B = t.shape[0]
        inp = torch.cat([t, x], dim=1)
        out = []
        for i in range(B):
            out.append(circuit(inp[i], self.weights_all))
        return torch.stack(out, dim=0).unsqueeze(1) * scale

# ===============================
# QFM_cheb (不改)
# ===============================
class QFM_cheb(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights_all = nn.Parameter(
            0.1 * torch.randn(
                (L+1, blocks, n_qubits, 3),
                dtype=torch.float64,
                device=device
            )
        )

    def forward(self, t, x):
        B = t.shape[0]
        inp = torch.cat([torch.arccos(t), torch.arccos(x)], dim=1)
        out = []
        for i in range(B):
            out.append(circuit(inp[i], self.weights_all))
        return torch.stack(out, dim=0).unsqueeze(1) * scale

# ===============================
# QCM (不改)
# ===============================
class QCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.qfm = QFM()
        self.signs = list(itertools.product([1.0, -1.0], repeat=m))

    def forward(self, t, x):
        B = t.shape[0]
        tx = torch.cat([t, x], dim=1)
        u = torch.arccos(tx)

        vals = []
        for s in self.signs:
            s_vec = torch.tensor(
                s, dtype=u.dtype, device=device
            ).view(1, -1)
            us = u * s_vec
            tmp = []
            for i in range(B):
                tmp.append(circuit(us[i], self.qfm.weights_all))
            vals.append(torch.stack(tmp, dim=0))

        return torch.stack(vals, dim=0).mean(dim=0).unsqueeze(1) * scale

# ===============================
# PDE residual (只改这里)
# ===============================
def pde_residual(model, t, x):
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)

    u = model(t, x)

    u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]

    return u_t + u_x

# ===============================
# Training
# ===============================
qfm_model = QFM()
qfm_cheb_model = QFM_cheb()
qcm_model = QCM()

epochs = 200
mse = nn.MSELoss()

opt_qfm = optim.Adam(qfm_model.parameters(), lr=1e-2)
opt_qfm_cheb = optim.Adam(qfm_cheb_model.parameters(), lr=1e-2)
opt_qcm = optim.Adam(qcm_model.parameters(), lr=1e-2)

tc, xc = sample_collocation(50)
tb, xb = sample_boundary(50)

print("Starting training...")

for epoch in range(epochs):

    # QFM
    loss_f = mse(pde_residual(qfm_model, tc, xc),
                 torch.zeros_like(tc)) \
             + mse(qfm_model(tb, xb),
                   boundary_target(xb))

    opt_qfm.zero_grad()
    loss_f.backward()
    opt_qfm.step()

    # QFM_cheb
    loss_cheb = mse(pde_residual(qfm_cheb_model, tc, xc),
                    torch.zeros_like(tc)) \
                + mse(qfm_cheb_model(tb, xb),
                      boundary_target(xb))

    opt_qfm_cheb.zero_grad()
    loss_cheb.backward()
    opt_qfm_cheb.step()

    # QCM
    loss_qcm = mse(pde_residual(qcm_model, tc, xc),
                   torch.zeros_like(tc)) \
               + mse(qcm_model(tb, xb),
                     boundary_target(xb))

    opt_qcm.zero_grad()
    loss_qcm.backward()
    opt_qcm.step()

    print(f"Epoch {epoch:3d} | "
          f"QFM {loss_f.item():.3e} | "
          f"Cheb {loss_cheb.item():.3e} | "
          f"QCM {loss_qcm.item():.3e}")

print("Training complete.")