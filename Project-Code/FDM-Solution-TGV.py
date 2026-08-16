import json
import math
import time

import numpy as np
import matplotlib.pyplot as plt

Re = 100.0
nu = 1.0 / Re

X_MIN, X_MAX = -math.pi, math.pi
Y_MIN, Y_MAX = -math.pi, math.pi
T_MIN, T_MAX = 0.0, 100.0

N = 100         
dt = 0.02        
n_steps = int(round((T_MAX - T_MIN) / dt))
SAVE_TIMES = [2.0, 10.0, 32.0, 50.0, 100.0]

POISSON_ITERS_PER_STEP = 100
POISSON_ITERS_PRESSURE = 3000

TAG = f"fdm_TaylorGreen_Re{Re:g}_N{N}_classic"


def exact_solution(x, y, t):
    decay_uv = np.exp(-2.0 * nu * t)
    decay_p = np.exp(-4.0 * nu * t)
    u = np.cos(x) * np.sin(y) * decay_uv
    v = -np.sin(x) * np.cos(y) * decay_uv
    p = -0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay_p
    return u, v, p

def exact_vorticity(x, y, t):
    decay_uv = np.exp(-2.0 * nu * t)
    return -2.0 * np.cos(x) * np.cos(y) * decay_uv

x = np.linspace(X_MIN, X_MAX, N, endpoint=False)
y = np.linspace(Y_MIN, Y_MAX, N, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="xy")

def ddx(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)
def ddy(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)
def d2dx2(f):
    return (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / dx ** 2
def d2dy2(f):
    return (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / dy ** 2

def solve_poisson_jacobi(rhs, f_guess, n_iter, h):
    f = f_guess.copy()
    h2 = h * h
    for _ in range(n_iter):
        neighbor_sum = (
            np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1)
            + np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0)
        )
        f = 0.25 * (neighbor_sum - h2 * rhs)
    f -= f.mean()
    return f


def relative_l2_error(pred, exact):
    return float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-12))

omega = exact_vorticity(X, Y, T_MIN).copy()
psi = np.zeros_like(omega)

saved_fields = {}  # t -> (u, v)

print("FDM (classic finite-difference) time-marching started...")
solve_start = time.time()

t_current = 0.0
for step in range(1, n_steps + 1):
    psi = solve_poisson_jacobi(-omega, psi, POISSON_ITERS_PER_STEP, dx)

    u = ddy(psi)
    v = -ddx(psi)

    omega_x = ddx(omega)
    omega_y = ddy(omega)
    omega_xx = d2dx2(omega)
    omega_yy = d2dy2(omega)
    omega_rhs = -(u * omega_x + v * omega_y) + nu * (omega_xx + omega_yy)

    omega = omega + dt * omega_rhs
    t_current = step * dt

    if step % 500 == 0 or step == n_steps:
        elapsed_so_far = time.time() - solve_start
        print(
            f"Step: {step:5d} | t = {t_current:6.2f} | "
            f"max|omega| = {np.max(np.abs(omega)):.4e} | Elapsed: {elapsed_so_far:8.2f}s"
        )

    for t_save in SAVE_TIMES:
        if abs(t_current - t_save) < dt / 2 and t_save not in saved_fields:
            saved_fields[t_save] = (u.copy(), v.copy())

solve_time_seconds = time.time() - solve_start
print("FDM time-marching finished!")
print(f"Total solve time: {solve_time_seconds:.2f} s ({solve_time_seconds / 60.0:.2f} min)")
print(f"Average time per step: {solve_time_seconds / n_steps:.4f} s")

error_history = []
plot_paths = []

for t_val in SAVE_TIMES:
    if t_val not in saved_fields:
        print(f"t = {t_val:6.1f} | ไม่ได้บันทึกผลไว้ระหว่างการรัน (ลองปรับ dt ให้หาร T_MAX ลงตัวพอดี)")
        continue

    u_pred, v_pred = saved_fields[t_val]

    u_x, u_y = ddx(u_pred), ddy(u_pred)
    v_x, v_y = ddx(v_pred), ddy(v_pred)
    p_rhs = -(u_x ** 2 + 2.0 * u_y * v_x + v_y ** 2)
    p_pred = solve_poisson_jacobi(p_rhs, np.zeros_like(p_rhs), POISSON_ITERS_PRESSURE, dx)

    u_exact, v_exact, p_exact = exact_solution(X, Y, t_val)

    err_u = relative_l2_error(u_pred, u_exact)
    err_v = relative_l2_error(v_pred, v_exact)
    err_p = relative_l2_error(p_pred, p_exact)

    entry = {"t": t_val, "l2_u": err_u, "l2_v": err_v, "l2_p": err_p}
    error_history.append(entry)
    print(
        f"t = {t_val:6.1f} | L2(u) = {err_u:.4e} | "
        f"L2(v) = {err_v:.4e} | L2(p) = {err_p:.4e}"
    )

    # -----------------------------------------------------------------
    # graph 3x3 (rows = u, v, p ; cols = Exact, FDM, |Error|)
    # -----------------------------------------------------------------
    fields = [
        ("u", u_exact, u_pred),
        ("v", v_exact, v_pred),
        ("p", p_exact, p_pred),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f"{TAG}  (Taylor-Green decay, t={t_val:g})", fontsize=14)

    col_titles = ["Exact", "FDM (classic)", "|Error|"]
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
    plot_path = f"{TAG}_t{t_val:g}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    plot_paths.append(plot_path)
    print(f"Plot saved to {plot_path}")

# ---------------------------------------------------------------------------
# save to JSON file
# ---------------------------------------------------------------------------
run_summary = {
    "tag": TAG,
    "method": "classic FDM: central differences + explicit Euler + Jacobi Poisson solve",
    "Re": Re,
    "nu": nu,
    "N": N,
    "dt": dt,
    "n_steps": n_steps,
    "poisson_iters_per_step": POISSON_ITERS_PER_STEP,
    "poisson_iters_pressure": POISSON_ITERS_PRESSURE,
    "domain": {"x": [X_MIN, X_MAX], "y": [Y_MIN, Y_MAX], "t": [T_MIN, T_MAX]},
    "save_times": SAVE_TIMES,
    "error_history": error_history,
    "solve_time_seconds": solve_time_seconds,
    "plot_paths": plot_paths,
}

json_path = f"{TAG}.json"
with open(json_path, "w") as f:
    json.dump(run_summary, f, indent=2)

print(f"Run summary saved to {json_path}")
