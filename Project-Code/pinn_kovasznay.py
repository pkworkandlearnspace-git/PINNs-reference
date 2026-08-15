import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch_xla.core.xla_model as xm

device = xm.xla_device()
print("Using:", device)

Re = 20.0
nu = 1.0 / Re
lambda_val = (Re / 2.0) - np.sqrt((Re**2 / 4.0) + 4.0 * np.pi**2)

# ---------------------------------------------------------------------------
# Run metadata (used for output filenames / tag). Reflects the ACTUAL
# architecture below (unchanged from the original script: 4 hidden layers,
# width 64, activation Tanh). Nothing about the architecture / training loop
# was changed, only saving/plotting/logging features were added.
# ---------------------------------------------------------------------------
ACTIVATION_NAME = "Tanh"
N_HIDDEN_LAYERS = 4
WIDTH = 64
TAG = f"act-{ACTIVATION_NAME}_L{N_HIDDEN_LAYERS}_W{WIDTH}"


def exact_solution(x, y):
    u = 1.0 - np.exp(lambda_val * x) * np.cos(2 * np.pi * y)
    v = (lambda_val / (2 * np.pi)) * np.exp(lambda_val * x) * np.sin(2 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2 * lambda_val * x))
    return u, v, p

# PINNs
class PINN_NavierStokes(nn.Module):
    def __init__(self):
        super(PINN_NavierStokes, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x, y):
        out = self.net(torch.cat([x, y], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

def compute_gradient(output, input_var):
    return torch.autograd.grad(
        output, input_var,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]

# Collocations
N_collocation = 8192
N_boundary = 8192

#PINNs collocations
x_c = torch.empty(N_collocation, 1, device=device).uniform_(-0.5, 1.0).requires_grad_(True)  # <-- FIX: to(device)
y_c = torch.empty(N_collocation, 1, device=device).uniform_(-0.5, 1.5).requires_grad_(True)  # <-- FIX: to(device)

#boundary points
x_b_np = np.concatenate([
    np.full((N_boundary//4, 1), -0.5), np.full((N_boundary//4, 1), 1.0),
    np.random.uniform(-0.5, 1.0, (N_boundary//4, 1)), np.random.uniform(-0.5, 1.0, (N_boundary//4, 1))
])
y_b_np = np.concatenate([
    np.random.uniform(-0.5, 1.5, (N_boundary//4, 1)), np.random.uniform(-0.5, 1.5, (N_boundary//4, 1)),
    np.full((N_boundary//4, 1), -0.5), np.full((N_boundary//4, 1), 1.5)
])

u_b_np, v_b_np, _ = exact_solution(x_b_np, y_b_np)

x_b = torch.tensor(x_b_np, dtype=torch.float32, device=device)  # <-- FIX: to(device)
y_b = torch.tensor(y_b_np, dtype=torch.float32, device=device)  # <-- FIX: to(device)
u_b = torch.tensor(u_b_np, dtype=torch.float32, device=device)  # <-- FIX: to(device)
v_b = torch.tensor(v_b_np, dtype=torch.float32, device=device)  # <-- FIX: to(device)

# train
model = PINN_NavierStokes().to(device)  # <-- FIX: to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5000

loss_history = []  # <-- NEW: collected for the JSON log

print("Training started...")
training_start = time.time()  # <-- NEW

for epoch in range(epochs):
    optimizer.zero_grad()

    u_pred_b, v_pred_b, p_pred_b = model(x_b, y_b)
    loss_u_b = torch.mean((u_pred_b - u_b) ** 2)
    loss_v_b = torch.mean((v_pred_b - v_b) ** 2)
    loss_bc = loss_u_b + loss_v_b

    u_c, v_c, p_c = model(x_c, y_c) #pred

    #AD
    u_x = compute_gradient(u_c, x_c)
    u_y = compute_gradient(u_c, y_c)
    v_x = compute_gradient(v_c, x_c)
    v_y = compute_gradient(v_c, y_c)
    p_x = compute_gradient(p_c, x_c)
    p_y = compute_gradient(p_c, y_c)

    u_xx = compute_gradient(u_x, x_c)
    u_yy = compute_gradient(u_y, y_c)
    v_xx = compute_gradient(v_x, x_c)
    v_yy = compute_gradient(v_y, y_c)

    #Phi res
    f_mass = u_x + v_y
    f_x = u_c * u_x + v_c * u_y + p_x - nu * (u_xx + u_yy)
    f_y = u_c * v_x + v_c * v_y + p_y - nu * (v_xx + v_yy)

    loss_physics = torch.mean(f_mass**2) + torch.mean(f_x**2) + torch.mean(f_y**2) #combine Phi loss

    loss = loss_bc + loss_physics
    loss.backward()
    optimizer.step()
    xm.mark_step()

    # NEW: log every 500 epochs (same cadence as the original print), plus
    # always capture the very last epoch so loss_history has a final entry.
    if epoch % 500 == 0 or epoch == epochs - 1:
        entry = {
            "epoch": epoch,
            "total": loss.item(),
            "bc": loss_bc.item(),
            "physics": loss_physics.item(),
        }
        loss_history.append(entry)
        print(
            f"Epoch: {epoch:4d} | Total Loss: {loss.item():.6f} | "
            f"BC Loss: {loss_bc.item():.6f} | Physics Loss: {loss_physics.item():.6f}"
        )

print("Training finished!")
training_time_seconds = time.time() - training_start  # <-- NEW
print(f"Training time: {training_time_seconds:.2f} s")

# ---------------------------------------------------------------------------
# NEW: save the trained model
# ---------------------------------------------------------------------------
model_path = f"{TAG}.pt"
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)  # <-- FIX: move weights to CPU before saving
print(f"Model saved to {model_path}")

x_test = np.linspace(-0.5, 1.0, 100)
y_test = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x_test, y_test)

x_test_tensor = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)  # <-- FIX: to(device)
y_test_tensor = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device)  # <-- FIX: to(device)

with torch.no_grad():
    u_pred, v_pred, p_pred = model(x_test_tensor, y_test_tensor)

U_pred = u_pred.cpu().numpy().reshape(100, 100)  # <-- FIX: .cpu() before numpy
V_pred = v_pred.cpu().numpy().reshape(100, 100)  # <-- FIX: .cpu() before numpy
P_pred = p_pred.cpu().numpy().reshape(100, 100)  # <-- FIX: .cpu() before numpy
U_exact, V_exact, P_exact = exact_solution(X, Y)  # <-- NEW: also compute v, p exact

def relative_l2_error(pred, exact):  # <-- NEW
    return np.linalg.norm(exact - pred, 2) / np.linalg.norm(exact, 2)

error_u = relative_l2_error(U_pred, U_exact)
error_v = relative_l2_error(V_pred, V_exact)  # <-- NEW
error_p = relative_l2_error(P_pred, P_exact)  # <-- NEW
print(f"Relative L2 Error for u-velocity: {error_u:.4e}")
print(f"Relative L2 Error for v-velocity: {error_v:.4e}")  # <-- NEW
print(f"Relative L2 Error for p:          {error_p:.4e}")  # <-- NEW

# ---------------------------------------------------------------------------
# NEW: 3x3 grid (rows = u, v, p ; cols = Exact, PINN, |Error|)
# ---------------------------------------------------------------------------
fields = [
    ("u", U_exact, U_pred),
    ("v", V_exact, V_pred),
    ("p", P_exact, P_pred),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.suptitle(f"{TAG}  (Kovasznay flow, Re={Re:g})", fontsize=14)

col_titles = ["Exact", "PINN", "|Error|"]
for row, (name, exact, pred) in enumerate(fields):
    error = np.abs(exact - pred)
    data = [exact, pred, error]
    for col in range(3):
        ax = axes[row, col]
        cmap = "jet" if col < 2 else "magma"
        im = ax.contourf(X, Y, data[col], 50, cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        if row == 0:
            ax.set_title(col_titles[col])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = f"{TAG}.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Plot saved to {plot_path}")

# ---------------------------------------------------------------------------
# NEW: save run summary / loss history as JSON
# ---------------------------------------------------------------------------
final_entry = loss_history[-1]

run_summary = {
    "tag": TAG,
    "activation": ACTIVATION_NAME,
    "n_layers": N_HIDDEN_LAYERS,
    "width": WIDTH,
    "loss_history": loss_history,
    "l2_errors": {
        "u": float(error_u),
        "v": float(error_v),
        "p": float(error_p),
    },
    "final_loss": final_entry["total"],
    "training_time_seconds": training_time_seconds,
}

json_path = f"{TAG}.json"
with open(json_path, "w") as f:
    json.dump(run_summary, f, indent=2)

print(f"Run summary saved to {json_path}")
