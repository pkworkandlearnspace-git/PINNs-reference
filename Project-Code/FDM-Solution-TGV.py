import math
import time
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. Analytical Solution (Taylor-Green Vortex) — เหมือนกับที่ใช้ใน PINN
#    (V0 = L = rho = 1 จึงตัดออกจากสูตรเพื่อความกระชับ)
# =====================================================================
Re = 100.0
nu = 1.0 / Re

def exact_solution(x, y, t):`
    decay_uv = np.exp(-2.0 * nu * t)
    decay_p = np.exp(-4.0 * nu * t)
    u = np.cos(x) * np.sin(y) * decay_uv
    v = -np.sin(x) * np.cos(y) * decay_uv
    p = -0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay_p
    return u, v, p

def exact_vorticity(x, y, t):
    # omega = v_x - u_y ; อนุพันธ์ตรงของ exact_solution ข้างบน
    decay_uv = np.exp(-2.0 * nu * t)
    return -2.0 * np.cos(x) * np.cos(y) * decay_uv

# =====================================================================
# 2. Grid & Domain (periodic ทุกด้าน จึงไม่ใส่จุดปลาย endpoint ซ้ำ)
# =====================================================================
X_MIN, X_MAX = -math.pi, math.pi
Y_MIN, Y_MAX = -math.pi, math.pi
T_MIN, T_MAX = 0.0, 100.0

N = 100  # จำนวนจุด grid ต่อแกน (เท่ากับ N_test ที่ใช้ประเมินผลฝั่ง PINN)
x = np.linspace(X_MIN, X_MAX, N, endpoint=False)
y = np.linspace(Y_MIN, Y_MAX, N, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="xy")

dt = 0.02
n_steps = int(round((T_MAX - T_MIN) / dt))  # = 5000, เท่ากับ epochs ของ PINN โดยตั้งใจ

# --- เตรียม wavenumber สำหรับแก้ Poisson equation ด้วย FFT (periodic BC) ---
kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="xy")
K2 = KX ** 2 + KY ** 2
K2[0, 0] = 1.0  # กันหารศูนย์ที่ mode เฉลี่ย (ค่าคงที่จะถูกกำหนดเป็น 0 ทีหลังอยู่แล้ว)

def solve_poisson_fft(rhs):
    """ แก้ ∇²f = rhs แบบ periodic ด้วย FFT, กำหนดให้ mean(f) = 0 (เหมือน exact solution) """
    rhs_hat = np.fft.fft2(rhs)
    f_hat = -rhs_hat / K2
    f_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(f_hat))

def ddx(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)

def ddy(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)

def d2dx2(f):
    return (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / dx ** 2

def d2dy2(f):
    return (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / dy ** 2

# =====================================================================
# 3. Initial Condition (t = 0)
# =====================================================================
omega = exact_vorticity(X, Y, 0.0).copy()

# =====================================================================
# 4. Time-Marching Loop (Explicit Euler + Central FD)
#    เทียบเท่ากับ training loop ของฝั่ง PINN — พิมพ์ความคืบหน้า + เวลาที่ใช้
# =====================================================================
save_times = [2.0, 10.0, 32.0, 50.0, 100.0]
saved_fields = {}  # เก็บ (u, v, p) ที่เวลาต่างๆ ไว้ตรวจสอบ error / plot

print("FDM time-marching started...")
fdm_start_time = time.time()

t_current = 0.0
for step in range(1, n_steps + 1):
    # -- แก้ Poisson หา streamfunction แล้วดึง u, v ออกมา --
    psi = solve_poisson_fft(-omega)          # ∇²psi = -omega
    u = ddy(psi)
    v = -ddx(psi)

    # -- อนุพันธ์ของ vorticity (FDM) --
    omega_x = ddx(omega)
    omega_y = ddy(omega)
    omega_xx = d2dx2(omega)
    omega_yy = d2dy2(omega)

    # -- Vorticity transport equation: omega_t + u*omega_x + v*omega_y = nu*(omega_xx+omega_yy) --
    omega_rhs = -(u * omega_x + v * omega_y) + nu * (omega_xx + omega_yy)
    omega = omega + dt * omega_rhs
    t_current = step * dt

    if step % 500 == 0 or step == n_steps:
        elapsed_so_far = time.time() - fdm_start_time
        print(f"Step: {step:5d} | t = {t_current:6.2f} | max|omega| = {np.max(np.abs(omega)):.4e} | Elapsed: {elapsed_so_far:8.2f}s")

    # -- บันทึก u, v, p ที่ค่า t ที่สนใจ (คำนวณเพิ่มอีกนิดเฉพาะจุดที่บันทึก) --
    for t_save in save_times:
        if abs(t_current - t_save) < dt / 2 and t_save not in saved_fields:
            psi_s = solve_poisson_fft(-omega)
            u_s = ddy(psi_s)
            v_s = -ddx(psi_s)
            u_x = ddx(u_s); u_y = ddy(u_s)
            v_x = ddx(v_s); v_y = ddy(v_s)
            p_rhs = -(u_x ** 2 + 2.0 * u_y * v_x + v_y ** 2)  # pressure-Poisson จาก NS + incompressibility
            p_s = solve_poisson_fft(p_rhs)
            saved_fields[t_save] = (u_s, v_s, p_s)

fdm_end_time = time.time()
total_fdm_time = fdm_end_time - fdm_start_time

print("FDM time-marching finished!")
print(f"Total computation time: {total_fdm_time:.2f} s ({total_fdm_time / 60.0:.2f} min)")
print(f"Average time per step: {total_fdm_time / n_steps:.4f} s")

# =====================================================================
# 5. Evaluation: Relative L2 error ที่หลายค่า t (เทียบรูปแบบเดียวกับฝั่ง PINN)
# =====================================================================
def relative_l2_error(pred, exact):
    return np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-12)

for t_value in save_times:
    if t_value not in saved_fields:
        print(f"t = {t_value:6.1f} | ไม่ได้บันทึกผลไว้ระหว่างการรัน (ลองปรับ dt ให้หาร T_MAX ลงตัวพอดี)")
        continue
    u_pred, v_pred, p_pred = saved_fields[t_value]
    u_exact, v_exact, p_exact = exact_solution(X, Y, t_value)
    print(
        f"t = {t_value:6.1f} | "
        f"L2(u) = {relative_l2_error(u_pred, u_exact):.4e} | "
        f"L2(v) = {relative_l2_error(v_pred, v_exact):.4e} | "
        f"L2(p) = {relative_l2_error(p_pred, p_exact):.4e}"
    )

# =====================================================================
# 6. Plot ผลลัพธ์ที่ t=10 (เทียบรูปแบบเดียวกับฝั่ง PINN)
# =====================================================================
plot_time = 10.0
u_pred, v_pred, p_pred = saved_fields[plot_time]
u_exact, v_exact, p_exact = exact_solution(X, Y, plot_time)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im1 = axes[0].contourf(X, Y, u_exact, 50, cmap="jet")
axes[0].set_title("Exact Taylor-Green u")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].contourf(X, Y, u_pred, 50, cmap="jet")
axes[1].set_title("FDM Taylor-Green u")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
