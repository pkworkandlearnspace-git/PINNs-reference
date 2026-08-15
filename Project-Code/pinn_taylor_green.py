import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch_xla.core.xla_model as xm

device = xm.xla_device()
print("Using:", device)

Re = 100.0
nu = 1.0 / Re

ACTIVATION_NAME = "SiLU"
N_HIDDEN_LAYERS = 6
WIDTH = 256
TAG = f"act-{ACTIVATION_NAME}_L{N_HIDDEN_LAYERS}_W{WIDTH}"


def exact_solution(x, y, t):
    decay_uv = torch.exp(-2.0 * nu * t)
    decay_p = torch.exp(-4.0 * nu * t)
    u = torch.cos(x) * torch.sin(y) * decay_uv
    v = -torch.sin(x) * torch.cos(y) * decay_uv
    p = -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * decay_p
    return u, v, p


class PINN_NavierStokes(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x, y, t):
        out = self.net(torch.cat([x, y, t], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def compute_gradient(output, input_var):
    return torch.autograd.grad(
        output, input_var,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
    )[0]


# Collocations
X_MIN, X_MAX = -math.pi, math.pi
Y_MIN, Y_MAX = -math.pi, math.pi
T_MIN, T_MAX = 0.0, 100.0

N_collocation = 8192
N_initial = 8192
N_boundary = 8192


# PINNs collocations
def sample_collocation(n):
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX).requires_grad_(True)
    y = torch.empty(n, 1, device=device).uniform_(Y_MIN, Y_MAX).requires_grad_(True)
    t = torch.empty(n, 1, device=device).uniform_(T_MIN, T_MAX).requires_grad_(True)
    return x, y, t


# initial collocations
def sample_initial(n):
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX)
    y = torch.empty(n, 1, device=device).uniform_(Y_MIN, Y_MAX)
    t = torch.zeros(n, 1, device=device)
    return x, y, t


# boundary points
def sample_boundary_x(n):
    y = torch.empty(n, 1, device=device).uniform_(Y_MIN, Y_MAX)
    t = torch.empty(n, 1, device=device).uniform_(T_MIN, T_MAX)
    x_left = torch.full((n, 1), X_MIN, device=device)
    x_right = torch.full((n, 1), X_MAX, device=device)
    return x_left, x_right, y, t


def sample_boundary_y(n):
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX)
    t = torch.empty(n, 1, device=device).uniform_(T_MIN, T_MAX)
    y_bottom = torch.full((n, 1), Y_MIN, device=device)
    y_top = torch.full((n, 1), Y_MAX, device=device)
    return x, y_bottom, y_top, t


# train
model = PINN_NavierStokes().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 5000

loss_history = []  # <-- NEW: collected for the JSON log

print("Training started...")
training_start = time.time()  # <-- NEW

for epoch in range(epochs):
    optimizer.zero_grad()

    x_i, y_i, t_i = sample_initial(N_initial)
    u_i, v_i, _ = model(x_i, y_i, t_i)
    u_exact, v_exact, _ = exact_solution(x_i, y_i, t_i)
    loss_ic = torch.mean((u_i - u_exact) ** 2) + torch.mean((v_i - v_exact) ** 2)

    x_c, y_c, t_c = sample_collocation(N_collocation)
    u_c, v_c, p_c = model(x_c, y_c, t_c)  # pred

    # AD
    u_x = compute_gradient(u_c, x_c)
    u_y = compute_gradient(u_c, y_c)
    u_t = compute_gradient(u_c, t_c)

    v_x = compute_gradient(v_c, x_c)
    v_y = compute_gradient(v_c, y_c)
    v_t = compute_gradient(v_c, t_c)

    p_x = compute_gradient(p_c, x_c)
    p_y = compute_gradient(p_c, y_c)

    u_xx = compute_gradient(u_x, x_c)
    u_yy = compute_gradient(u_y, y_c)
    v_xx = compute_gradient(v_x, x_c)
    v_yy = compute_gradient(v_y, y_c)

    # Phi res
    f_mass = u_x + v_y
    f_x = u_t + u_c * u_x + v_c * u_y + p_x - nu * (u_xx + u_yy)
    f_y = v_t + u_c * v_x + v_c * v_y + p_y - nu * (v_xx + v_yy)

    loss_physics = torch.mean(f_mass ** 2) + torch.mean(f_x ** 2) + torch.mean(f_y ** 2)  # combine Phi loss

    # boundaryX
    x_l, x_r, y_b, t_b = sample_boundary_x(N_boundary)
    u_l, v_l, p_l = model(x_l, y_b, t_b)
    u_r, v_r, p_r = model(x_r, y_b, t_b)
    loss_bc_x = (
        torch.mean((u_l - u_r) ** 2)
        + torch.mean((v_l - v_r) ** 2)
        + torch.mean((p_l - p_r) ** 2)
    )

    # boundaryY
    x_by, y_bot, y_top, t_by = sample_boundary_y(N_boundary)
    u_bot, v_bot, p_bot = model(x_by, y_bot, t_by)
    u_top, v_top, p_top = model(x_by, y_top, t_by)
    loss_bc_y = (
        torch.mean((u_bot - u_top) ** 2)
        + torch.mean((v_bot - v_top) ** 2)
        + torch.mean((p_bot - p_top) ** 2)
    )

    loss_bc = loss_bc_x + loss_bc_y

    loss = loss_ic + loss_bc + loss_physics
    loss.backward()
    optimizer.step()
    xm.mark_step()

    # always capture the very last epoch so loss_history has a final entry.
    if epoch % 500 == 0 or epoch == epochs - 1:
        entry = {
            "epoch": epoch,
            "total": loss.item(),
            "bc": loss_bc.item(),
            "physics": loss_physics.item(),
            "ic": loss_ic.item(),
        }
        loss_history.append(entry)
        print(
            f"Epoch: {epoch:5d} | Total: {loss.item():.6e} | "
            f"IC: {loss_ic.item():.6e} | Physics: {loss_physics.item():.6e} "
            f"BC: {loss_bc.item():.6e}"
        )

training_time_seconds = time.time() - training_start  # <-- NEW
print(f"Training finished! ({training_time_seconds:.2f} s)")

# ---------------------------------------------------------------------------
# NEW: save the trained model
# ---------------------------------------------------------------------------
model_path = f"{TAG}.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# evaluate
N_test = 100
x_test = torch.linspace(X_MIN, X_MAX, N_test, device=device)
y_test = torch.linspace(Y_MIN, Y_MAX, N_test, device=device)
X, Y = torch.meshgrid(x_test, y_test, indexing="xy")

model.eval()


def relative_l2_error(pred, exact):
    return (torch.linalg.vector_norm(pred - exact) / (torch.linalg.vector_norm(exact) + 1e-12)).item()


l2_errors_by_time = []
for t_value in [2.0, 10.0, 32.0, 50.0, 100.0]:
    T = torch.full_like(X, t_value)
    with torch.no_grad():
        u_pred, v_pred, p_pred = model(X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1))
        u_exact, v_exact, p_exact = exact_solution(X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1))

    l2_u = relative_l2_error(u_pred, u_exact)
    l2_v = relative_l2_error(v_pred, v_exact)
    l2_p = relative_l2_error(p_pred, p_exact)

    l2_errors_by_time.append({"t": t_value, "u": l2_u, "v": l2_v, "p": l2_p})  # <-- NEW

    print(
        f"t = {t_value:6.1f} | "
        f"L2(u) = {l2_u:.4e} | "
        f"L2(v) = {l2_v:.4e} | "
        f"L2(p) = {l2_p:.4e}"
    )

# Plotting
plot_time = 10.0
T = torch.full_like(X, plot_time)
with torch.no_grad():
    u_pred, v_pred, p_pred = model(X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1))  # <-- NEW: also v, p
    u_exact, v_exact, p_exact = exact_solution(X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1))  # <-- NEW

u_pred = u_pred.cpu().numpy().reshape(N_test, N_test)
u_exact = u_exact.cpu().numpy().reshape(N_test, N_test)
v_pred = v_pred.cpu().numpy().reshape(N_test, N_test)
v_exact = v_exact.cpu().numpy().reshape(N_test, N_test)
p_pred = p_pred.cpu().numpy().reshape(N_test, N_test)
p_exact = p_exact.cpu().numpy().reshape(N_test, N_test)
X_plot, Y_plot = X.cpu().numpy(), Y.cpu().numpy()

# ---------------------------------------------------------------------------
# NEW: 3x3 grid (rows = u, v, p ; cols = Exact, PINN, |Error|)
# ---------------------------------------------------------------------------
fields = [
    ("u", u_exact, u_pred),
    ("v", v_exact, v_pred),
    ("p", p_exact, p_pred),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.suptitle(f"{TAG}  (Taylor-Green Vortex, Re={Re:g}, t={plot_time:g})", fontsize=14)

col_titles = ["Exact", "PINN", "|Error|"]
for row, (name, exact, pred) in enumerate(fields):
    error = np.abs(exact - pred)
    data = [exact, pred, error]
    for col in range(3):
        ax = axes[row, col]
        cmap = "jet" if col < 2 else "magma"
        im = ax.contourf(X_plot, Y_plot, data[col], 50, cmap=cmap)
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
l2_at_plot_time = next(e for e in l2_errors_by_time if e["t"] == plot_time)

run_summary = {
    "tag": TAG,
    "activation": ACTIVATION_NAME,
    "n_layers": N_HIDDEN_LAYERS,
    "width": WIDTH,
    "loss_history": loss_history,
    "l2_errors": {
        "u": l2_at_plot_time["u"],
        "v": l2_at_plot_time["v"],
        "p": l2_at_plot_time["p"],
    },
    "l2_errors_by_time": l2_errors_by_time,
    "final_loss": final_entry["total"],
    "training_time_seconds": training_time_seconds,
}

json_path = f"{TAG}.json"
with open(json_path, "w") as f:
    json.dump(run_summary, f, indent=2)

print(f"Run summary saved to {json_path}")
