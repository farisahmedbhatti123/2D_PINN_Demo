# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # 2D PINN Inpainting — with RANS Momentum
#
# **The goal**: reconstruct a 2D CFD velocity field given only K sparse
# sensor observations, using the full RANS momentum equation (with
# Boussinesq closure) as a soft loss term in addition to the continuity
# equation.
#
# This builds on `00_inpainting_complete.py`.  Continuity alone enforces
# kinematics (the velocity field must be divergence-free) but says
# nothing about *what makes the flow look like the flow*.  Adding
# momentum couples velocity to pressure and turbulent stresses,
# constraining the magnitudes and directions in regions of high
# acceleration (turning around baffles, shear layers).
#
# **Outline**:
#
#   1. Load CFD case 834 with all RANS fields (U, p, nu_t, k).
#   2. Pretend we observe (Ux, Uy, p + 2k/3) at K random sensors.
#   3. Pre-train a small NN to represent the CFD's nu_t field — this
#      gives a differentiable nu_t(x, y) for use in the momentum residual.
#   4. Train an MLP on the K sensor observations only — see it fail.
#   5. Train a PINN with continuity + full RANS momentum (Boussinesq closure)
#      — see it close most of the gap.
#   6. Head-to-head: MLP vs PINN reconstruction error.
#
# **What's "Boussinesq closure"**: the RANS momentum equation closes the
# Reynolds-stress tensor by analogy to viscous stress:
#
#     tau_R / rho = nu_t (grad u + grad u^T) - (2/3) k I
#
# So the full RANS momentum becomes:
#
#     (u . grad) u = -grad(p + 2k/3) + div[(nu + nu_t)(grad u + grad u^T)]
#
# We absorb the (2/3)k into the pressure (calling it "modified pressure"
# P_mod = p + 2k/3), and expand the stress divergence to include the
# (grad nu_t) cross-terms that come from the non-uniform turbulent
# viscosity.

# %% [markdown]
# ## 0. Imports

# %%
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

_HERE         = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(".")
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# %% [markdown]
# ## 1. Configuration

# %%
# --- which case, which device ---
CASE_ID    = 834
DATA_DIR   = os.path.join(_PROJECT_ROOT, "data", "tiny")
DEVICE     = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# --- sensors ---
K          = 50
SEED       = 42

# --- main network ---
WIDTH      = 64
DEPTH      = 4

# --- nu_t representation network (small, just fits the CFD nu_t field) ---
NUT_WIDTH  = 32
NUT_DEPTH  = 3
NUT_EPOCHS = 3000

# --- training ---
EPOCHS         = 5000
LR             = 1e-3
VAL_EVERY      = 50
PRINT_EVERY    = 500

# --- physics ---
LAMBDA_CONT    = 1.0     # continuity loss weight
LAMBDA_MOMENT  = 0.001   # momentum loss weight — note: very narrow working window
N_COLLOC       = 4096
WARMUP_FRAC    = 0.2

# --- physical constants ---
NU_MOLECULAR   = 1.0e-6  # kinematic viscosity of water (m^2 / s)

print(f"device: {DEVICE}")
print(f"case: {CASE_ID},  K = {K} sensors,  {EPOCHS} epochs")
print(f"physics weights:  lambda_cont = {LAMBDA_CONT}   lambda_moment = {LAMBDA_MOMENT}")

# %% [markdown]
# ## 2. Load the CFD data
#
# This time we load the **full** field file (extracted earlier from the
# OpenFOAM case via `scripts/extract_openfoam_fields.py`): velocity,
# pressure, turbulent kinetic energy, eddy viscosity.
#
# **Normalization** (same conventions as `00`):
#
#   - Coordinates: `x_norm = x/L`, `y_norm = y/W`
#   - Velocities: `Ux_norm = Ux/U_inlet`, `Uy_norm = Uy/U_inlet`
#   - **Modified pressure**: `P_mod = p + (2/3) k` (folds the isotropic
#     Reynolds-stress contribution into the pressure-gradient term).
#     Normalized as `P_mod_norm = P_mod / U_inlet^2` (dynamic-head units).
#   - `nu_t` kept in physical units (m^2 / s).
#
# Pressure has gauge freedom — the equation only uses ∇p, so the absolute
# level is arbitrary.  We center `P_mod_norm` around zero before
# supervision so the network's output range is symmetric.

# %%
geo_csv = os.path.join(DATA_DIR, "basin_geometries_summary.csv")
vel_csv = os.path.join(DATA_DIR, "results", f"case_{CASE_ID:03d}_full.csv")

geo_row = pd.read_csv(geo_csv).query(f"`Simulation Number` == {CASE_ID}").iloc[0]
L       = float(geo_row['L'])
W       = float(geo_row['W'])
n_baf   = int  (geo_row['n'])
a_baf   = float(geo_row['a'])
U_inlet = float(geo_row['U_inlet'])

df = pd.read_csv(vel_csv).dropna(subset=['Ux', 'Uy', 'p', 'nut', 'k'])
x_phys  = df['x'].values.astype(np.float32)
y_phys  = df['y'].values.astype(np.float32)
x_norm  = (df['x'].values  / L      ).astype(np.float32)
y_norm  = (df['y'].values  / W      ).astype(np.float32)
Ux_norm = (df['Ux'].values / U_inlet).astype(np.float32)
Uy_norm = (df['Uy'].values / U_inlet).astype(np.float32)
nut_phys = df['nut'].values.astype(np.float32)
k_phys   = df['k'].values.astype(np.float32)

# Modified pressure (Boussinesq): P_mod = p + (2/3) k, normalized by U_inlet^2, then centered.
p_mod_phys     = df['p'].values + (2.0/3.0) * df['k'].values
p_mod_norm     = (p_mod_phys / U_inlet**2).astype(np.float32)
p_mod_offset   = float(p_mod_norm.mean())
p_mod_centered = p_mod_norm - p_mod_offset

N = len(x_norm)
print(f"Case {CASE_ID}: L={L:.0f} m, W={W:.0f} m, n={n_baf} baffles, U_inlet={U_inlet:.3f} m/s")
print(f"  {N:,} CFD mesh points loaded")
print(f"  Ux_norm range: [{Ux_norm.min():+.3f}, {Ux_norm.max():+.3f}]")
print(f"  P_mod_norm range (after centering): [{p_mod_centered.min():+.3f}, {p_mod_centered.max():+.3f}]  (offset = {p_mod_offset:+.3f})")
print(f"  nu_t range: [{nut_phys.min():.2e}, {nut_phys.max():.2e}] m^2/s  "
      f"({nut_phys.min()/NU_MOLECULAR:.0f}x to {nut_phys.max()/NU_MOLECULAR:.0f}x molecular viscosity)")

# %% [markdown]
# ## 3. Pick K random sensors
#
# Sensors now measure THREE quantities — Ux, Uy, AND the modified
# pressure.  This corresponds to a real-world setup with co-located
# velocity probes (PIV, ultrasonic) and pressure transducers.

# %%
rng        = np.random.default_rng(SEED)
sensor_idx = rng.choice(N, size=K, replace=False)
eval_idx   = np.setdiff1d(np.arange(N), sensor_idx)

print(f"K = {K} sensors,  {len(eval_idx):,} held-out evaluation points")

def to_col(arr):
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(1)

sens_x   = to_col(x_norm        [sensor_idx])
sens_y   = to_col(y_norm        [sensor_idx])
sens_Ux  = to_col(Ux_norm       [sensor_idx])
sens_Uy  = to_col(Uy_norm       [sensor_idx])
sens_P   = to_col(p_mod_centered[sensor_idx])

eval_x   = to_col(x_norm        [eval_idx])
eval_y   = to_col(y_norm        [eval_idx])
eval_Ux  = to_col(Ux_norm       [eval_idx])
eval_Uy  = to_col(Uy_norm       [eval_idx])
eval_P   = to_col(p_mod_centered[eval_idx])

# %% [markdown]
# ## 4. Plotting helper
#
# Smooth shaded plots over the unstructured CFD mesh.  Same helper as
# `00`.

# %%
def plot_field(ax, x, y, c, *, cmap="viridis", vmin=None, vmax=None,
               title: str = "", edge_cutoff_mult: float = 3.0):
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c)
    tri = mtri.Triangulation(x, y)
    pts        = np.stack([x[tri.triangles], y[tri.triangles]], axis=-1)
    edges      = np.linalg.norm(np.diff(pts[:, [0, 1, 2, 0], :], axis=1), axis=-1)
    max_edge   = edges.max(axis=1)
    tri.set_mask(max_edge > edge_cutoff_mult * np.median(max_edge))
    m = ax.tripcolor(tri, c, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)
    return m

# %% [markdown]
# ## 5. Visualize the ground truth
#
# Four panels: velocity components, magnitude, and the CFD turbulent
# viscosity nu_t.  The nu_t field is what makes this a *RANS* problem
# rather than a laminar one — eddy viscosity is 10^4× larger than
# molecular in the high-shear regions around the baffles.

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
plot_field(axes[0], x_phys, y_phys, Ux_norm * U_inlet, title="Ux (m/s)")
plot_field(axes[1], x_phys, y_phys, Uy_norm * U_inlet, title="Uy (m/s)")
mag = np.sqrt((Ux_norm*U_inlet)**2 + (Uy_norm*U_inlet)**2)
m3 = plot_field(axes[2], x_phys, y_phys, mag, title=f"|U|  +  K={K} sensors")
axes[2].scatter(x_phys[sensor_idx], y_phys[sensor_idx],
                s=35, facecolor='none', edgecolor='red', linewidths=1.4, zorder=5)
plot_field(axes[3], x_phys, y_phys, nut_phys, cmap="magma", title=r"$\nu_t$  (m$^2$/s)")
fig.suptitle(f"Case {CASE_ID} ground truth: velocity + turbulent viscosity", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. The nu_t representation network
#
# To enforce the full Boussinesq momentum equation we need the **gradient
# of nu_t** at collocation points — that's the new term (compared to the
# simplified "Laplacian" form) that captures how non-uniform turbulent
# viscosity contributes to the stress balance.
#
# Two ways to get differentiable nu_t(x, y):
#   - Build a scipy interpolator + numerical gradient — fragile on
#     unstructured meshes.
#   - **Fit a small NN to the CFD's nu_t field** — autograd gives
#     gradients for free.  This is what we do.
#
# The network is tiny (3 layers of 32) — its only job is to represent a
# smooth approximation of CFD's nu_t.  Trained for a few thousand epochs
# against ALL N=4575 CFD mesh values, then frozen.

# %%
class ScalarField(nn.Module):
    """Small NN that maps (x, y) -> scalar.  Used to represent nu_t."""
    def __init__(self, width: int = 32, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# %%
print("Pre-training nu_t representation over all CFD mesh points...")
torch.manual_seed(SEED); np.random.seed(SEED)

x_all = to_col(x_norm)
y_all = to_col(y_norm)
nut_target = to_col(nut_phys)

nut_model = ScalarField(NUT_WIDTH, NUT_DEPTH).to(DEVICE)
opt_nut   = torch.optim.Adam(nut_model.parameters(), lr=1e-3)

for ep in range(NUT_EPOCHS):
    opt_nut.zero_grad()
    loss = F.mse_loss(nut_model(x_all, y_all), nut_target)
    loss.backward(); opt_nut.step()
    if (ep + 1) % 1000 == 0:
        print(f"  ep {ep+1}/{NUT_EPOCHS}:  nu_t MSE = {loss.item():.3e}")

# Freeze: nut_model's parameters won't update during PINN training.
for p in nut_model.parameters():
    p.requires_grad = False

print(f"  final nu_t fit MSE = {loss.item():.3e}  (target std = {nut_target.std().item():.3e})")

# %% [markdown]
# ## 7. The main MLP — 3 outputs now
#
# `(x, y) -> (Ux, Uy, P_mod)`.  Same Tanh-activated architecture as
# `00`, just one more output channel.

# %%
class MLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4, out_dim: int = 3):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

# %% [markdown]
# ## 8. The continuity residual
#
# Same as `00`.  Operates on the first two output channels (Ux, Uy);
# pressure is irrelevant to mass conservation.

# %%
def grad1(out, var):
    """First derivative of `out` w.r.t. `var`, building a graph for further differentiation."""
    return torch.autograd.grad(out, var, torch.ones_like(out),
                               create_graph=True, retain_graph=True)[0]

def continuity_residual(model, x, y):
    out = model(x, y)
    Ux, Uy = out[:, 0:1], out[:, 1:2]
    return grad1(Ux, x) / L + grad1(Uy, y) / W

def continuity_loss(model, x, y):
    r = continuity_residual(model, x, y)
    return torch.mean(r ** 2)

# %% [markdown]
# ## 9. The full Boussinesq momentum residual
#
# The new physics.  Dimensionless form, scaled by `U_inlet^2 / L` so the
# residual is naturally O(1) when the equation is well-satisfied.
#
# Starting from the physical equation:
#
#     (u . grad) u  +  grad(P_mod)  -  div[(nu + nu_t)(grad u + grad u^T)]  =  0
#
# For incompressible flow (`div(u) = 0`), the stress divergence expands as:
#
#     div[(nu + nu_t)(grad u + grad u^T)]
#         = (nu + nu_t) Laplacian(u)        <- the "simplified" Laplacian term
#         + grad(nu_t) . (grad u + grad u^T)  <- the "cross" terms (NEW)
#
# **The cross terms are the new contribution** compared to the simplified
# Laplacian form.  They matter when nu_t varies strongly across the
# domain (as it does here — 500x range).
#
# Network outputs are normalized: Ux/U_inlet, Uy/U_inlet, P/U_inlet^2.
# Inputs are normalized: x/L, y/W.  Chain-rule conversions throughout.

# %%
def momentum_residual(model, nut_model, x, y):
    """
    Returns (R_x, R_y) — dimensionless x- and y-momentum residuals.
    Inputs (x, y) must have requires_grad=True.
    """
    out = model(x, y)
    Ux, Uy, P = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # First derivatives of velocity and modified pressure
    dUx_dx = grad1(Ux, x); dUx_dy = grad1(Ux, y)
    dUy_dx = grad1(Uy, x); dUy_dy = grad1(Uy, y)
    dP_dx  = grad1(P,  x); dP_dy  = grad1(P,  y)

    # Second derivatives of velocity (for the Laplacian term)
    d2Ux_dx2 = grad1(dUx_dx, x); d2Ux_dy2 = grad1(dUx_dy, y)
    d2Uy_dx2 = grad1(dUy_dx, x); d2Uy_dy2 = grad1(dUy_dy, y)

    # nu_t and its gradient from the pre-trained representation
    nu_t    = nut_model(x, y)
    dnut_dx = grad1(nu_t, x)
    dnut_dy = grad1(nu_t, y)

    # Coefficients arising from normalization (chain rule)
    LW         = L / W                                    # aspect ratio (~2.06)
    Re_eff_inv = (NU_MOLECULAR + nu_t) / (U_inlet * L)    # inverse local effective Reynolds
    UL_inv     = 1.0 / (U_inlet * L)                      # scaling factor for cross-terms

    # --- x-momentum ---
    convective_x = Ux * dUx_dx + LW * Uy * dUx_dy
    pressure_x   = dP_dx
    laplace_x    = Re_eff_inv * (d2Ux_dx2 + LW**2 * d2Ux_dy2)
    cross_x      = UL_inv * (2*dnut_dx*dUx_dx
                             + LW**2 * dnut_dy*dUx_dy
                             + LW    * dnut_dy*dUy_dx)
    R_x = convective_x + pressure_x - laplace_x - cross_x

    # --- y-momentum ---
    convective_y = Ux * dUy_dx + LW * Uy * dUy_dy
    pressure_y   = LW * dP_dy
    laplace_y    = Re_eff_inv * (d2Uy_dx2 + LW**2 * d2Uy_dy2)
    cross_y      = UL_inv * (    dnut_dx*dUy_dx
                             + LW    * dnut_dx*dUx_dy
                             + 2 * LW**2 * dnut_dy*dUy_dy)
    R_y = convective_y + pressure_y - laplace_y - cross_y

    return R_x, R_y

def momentum_loss(model, nut_model, x, y):
    R_x, R_y = momentum_residual(model, nut_model, x, y)
    return torch.mean(R_x ** 2) + torch.mean(R_y ** 2)

# %% [markdown]
# ## 10. Train the plain MLP (data only, no physics)
#
# Baseline.  Data loss is now `MSE(Ux) + MSE(Uy) + MSE(P_mod)` — three
# terms because we observe three quantities at each sensor.

# %%
torch.manual_seed(SEED); np.random.seed(SEED)
model_mlp = MLP(WIDTH, DEPTH, out_dim=3).to(DEVICE)
opt       = torch.optim.Adam(model_mlp.parameters(), lr=LR)

print(f"MLP: {model_mlp.num_params():,} parameters,  training on {K} sensors")

train_losses_mlp, val_losses_mlp = [], []
best_val_mlp, best_state_mlp     = float("inf"), None
t0 = time.time()

for ep in range(EPOCHS):
    model_mlp.train(); opt.zero_grad()
    pred = model_mlp(sens_x, sens_y)
    loss = (F.mse_loss(pred[:, 0:1], sens_Ux) +
            F.mse_loss(pred[:, 1:2], sens_Uy) +
            F.mse_loss(pred[:, 2:3], sens_P))
    loss.backward(); opt.step()
    train_losses_mlp.append(loss.item())

    if (ep + 1) % VAL_EVERY == 0:
        model_mlp.eval()
        with torch.no_grad():
            p  = model_mlp(eval_x, eval_y)
            vl = (F.mse_loss(p[:, 0:1], eval_Ux) +
                  F.mse_loss(p[:, 1:2], eval_Uy)).item()    # validate on velocity only
        val_losses_mlp.append(vl)
        if vl < best_val_mlp:
            best_val_mlp   = vl
            best_state_mlp = {k: v.detach().clone() for k, v in model_mlp.state_dict().items()}

    if (ep + 1) % PRINT_EVERY == 0:
        print(f"  ep {ep+1:5d}/{EPOCHS}   train {loss.item():.3e}   val {val_losses_mlp[-1]:.3e}")

model_mlp.load_state_dict(best_state_mlp)
print(f"MLP done in {time.time()-t0:.1f}s  |  best held-out velocity MSE = {best_val_mlp:.3e}")

# %% [markdown]
# ## 11. Visualize the MLP reconstruction

# %%
def predict_field(model, x_n, y_n):
    """Run the model on all mesh points.  Returns (Ux, Uy, P_mod) numpy arrays."""
    model.eval()
    with torch.no_grad():
        p = model(to_col(x_n), to_col(y_n)).cpu().numpy()
    return p[:, 0], p[:, 1], p[:, 2]

mlp_Ux, mlp_Uy, mlp_P = predict_field(model_mlp, x_norm, y_norm)

def show_reconstruction(pred_Ux, pred_Uy, model_name: str, val_mse: float):
    """Truth |U| | reconstruction |U| | pointwise L2 vector error."""
    truth_mag = np.sqrt(Ux_norm**2 + Uy_norm**2)
    pred_mag  = np.sqrt(pred_Ux**2 + pred_Uy**2)
    err       = np.sqrt((pred_Ux - Ux_norm)**2 + (pred_Uy - Uy_norm)**2)

    vmin, vmax = float(min(truth_mag.min(), pred_mag.min())), float(max(truth_mag.max(), pred_mag.max()))
    emax       = float(err.max()) or 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    plot_field(axes[0], x_phys, y_phys, truth_mag, vmin=vmin, vmax=vmax, title="truth |U|")
    plot_field(axes[1], x_phys, y_phys, pred_mag,  vmin=vmin, vmax=vmax, title=f"{model_name} reconstruction")
    mE = plot_field(axes[2], x_phys, y_phys, err,  cmap="magma", vmin=0, vmax=emax,
                    title="L2 vector error  ||vec_pred − vec_truth||")
    for ax in axes:
        ax.scatter(x_phys[sensor_idx], y_phys[sensor_idx],
                   s=18, facecolor='none', edgecolor='red', linewidths=0.8, zorder=5)
    fig.suptitle(f"{model_name}  —  K={K}  —  held-out MSE = {val_mse:.2e}", y=1.02)
    return fig

show_reconstruction(mlp_Ux, mlp_Uy, "MLP (no physics)", best_val_mlp)
plt.show()

# %% [markdown]
# ## 12. How unphysical is the MLP's field?
#
# Measure the divergence ∇·u AND the momentum residuals (R_x and R_y) on
# the full domain.  All three should be small for a physically valid
# flow — but the MLP was never told about any of them.

# %%
def residual_grid(model, n_side: int = 60):
    """Sample model + residuals on a uniform grid."""
    xs = np.linspace(0, 1, n_side, dtype=np.float32)
    ys = np.linspace(0, 1, n_side, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    xt = torch.tensor(X.reshape(-1, 1), device=DEVICE, requires_grad=True)
    yt = torch.tensor(Y.reshape(-1, 1), device=DEVICE, requires_grad=True)
    rc = continuity_residual(model, xt, yt).detach().cpu().numpy().squeeze()
    rx, ry = momentum_residual(model, nut_model, xt, yt)
    rx_np = rx.detach().cpu().numpy().squeeze()
    ry_np = ry.detach().cpu().numpy().squeeze()
    return X.flatten()*L, Y.flatten()*W, rc, rx_np, ry_np

x_g, y_g, rc_mlp, rx_mlp, ry_mlp = residual_grid(model_mlp)
rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
print(f"MLP residuals:")
print(f"  continuity RMS:  {rms(rc_mlp):.3e}   (should be 0)")
print(f"  momentum_x RMS:  {rms(rx_mlp):.3e}   (should be 0)")
print(f"  momentum_y RMS:  {rms(ry_mlp):.3e}   (should be 0)")

# %% [markdown]
# ## 13. Train the PINN — continuity + full Boussinesq momentum
#
# Same network, same sensors, same hyperparameters as the MLP.  The
# additions:
#
#   - 4096 collocation points sampled uniformly in `[0, 1]^2`.
#   - `LAMBDA_CONT * L_continuity` enforces mass conservation.
#   - `LAMBDA_MOMENT * (L_mom_x + L_mom_y)` enforces full Boussinesq momentum.
#
# **Two things to watch in the loss curves**:
#   - The data loss should drop *similarly* to the MLP — if it plateaus
#     well above the MLP's data loss, the momentum weight is too high
#     and physics is fighting the data anchor.
#   - The momentum and continuity residuals should both decrease over
#     training.  If continuity drops to ~1e-4 but momentum stays at ~1e-2,
#     the network found a "kinematically valid but dynamically wrong"
#     solution.
#
# **Why `LAMBDA_MOMENT = 0.001` and not 1.0**: the momentum residual,
# despite being dimensionless, has a much larger gradient magnitude per
# parameter than continuity (it involves second derivatives and three
# coupled terms that must cancel).  Empirical sweep shows the working
# window is 0.0005–0.005.

# %%
# Collocation points (fixed, uniform random).
rng_c = np.random.default_rng(0)
colloc_x = torch.tensor(rng_c.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32),
                        device=DEVICE, requires_grad=True)
colloc_y = torch.tensor(rng_c.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32),
                        device=DEVICE, requires_grad=True)

torch.manual_seed(SEED); np.random.seed(SEED)
model_pinn = MLP(WIDTH, DEPTH, out_dim=3).to(DEVICE)
opt        = torch.optim.Adam(model_pinn.parameters(), lr=LR)

print(f"PINN: {model_pinn.num_params():,} parameters,  K={K} sensors,  "
      f"{N_COLLOC} collocation points,  lambda_c={LAMBDA_CONT}  lambda_m={LAMBDA_MOMENT}")

history = {"data": [], "cont": [], "mom_x": [], "mom_y": [], "total": [], "val": []}
best_val_pinn, best_state_pinn = float("inf"), None
t0 = time.time()

for ep in range(EPOCHS):
    model_pinn.train(); opt.zero_grad()

    # --- Data loss (Ux, Uy, P_mod at sensors) ---
    pred = model_pinn(sens_x, sens_y)
    L_data = (F.mse_loss(pred[:, 0:1], sens_Ux) +
              F.mse_loss(pred[:, 1:2], sens_Uy) +
              F.mse_loss(pred[:, 2:3], sens_P))

    # --- Continuity loss ---
    L_cont = continuity_loss(model_pinn, colloc_x, colloc_y)

    # --- Momentum loss (full Boussinesq) ---
    R_x, R_y = momentum_residual(model_pinn, nut_model, colloc_x, colloc_y)
    L_mx = torch.mean(R_x ** 2)
    L_my = torch.mean(R_y ** 2)

    # --- Combined loss with warmup ramp ---
    ramp    = min(1.0, ep / (EPOCHS * WARMUP_FRAC)) if WARMUP_FRAC > 0 else 1.0
    L_total = L_data + ramp * (LAMBDA_CONT * L_cont +
                               LAMBDA_MOMENT * (L_mx + L_my))

    L_total.backward(); opt.step()

    if (ep + 1) % VAL_EVERY == 0:
        model_pinn.eval()
        with torch.no_grad():
            p  = model_pinn(eval_x, eval_y)
            vl = (F.mse_loss(p[:, 0:1], eval_Ux) +
                  F.mse_loss(p[:, 1:2], eval_Uy)).item()    # velocity-only val
        history["data" ].append(L_data .item())
        history["cont" ].append(L_cont .item())
        history["mom_x"].append(L_mx   .item())
        history["mom_y"].append(L_my   .item())
        history["total"].append(L_total.item())
        history["val"  ].append(vl)
        if vl < best_val_pinn:
            best_val_pinn   = vl
            best_state_pinn = {k: v.detach().clone() for k, v in model_pinn.state_dict().items()}

    if (ep + 1) % PRINT_EVERY == 0:
        print(f"  ep {ep+1:5d}/{EPOCHS}   data {L_data.item():.3e}   cont {L_cont.item():.3e}   "
              f"mom_x {L_mx.item():.3e}   mom_y {L_my.item():.3e}   val {history['val'][-1]:.3e}   ramp {ramp:.2f}")

model_pinn.load_state_dict(best_state_pinn)
print(f"PINN done in {time.time()-t0:.1f}s  |  best held-out velocity MSE = {best_val_pinn:.3e}")

# %% [markdown]
# ## 14. Visualize the PINN reconstruction

# %%
pinn_Ux, pinn_Uy, pinn_P = predict_field(model_pinn, x_norm, y_norm)
show_reconstruction(pinn_Ux, pinn_Uy, f"PINN (cont + Boussinesq momentum, lam_m={LAMBDA_MOMENT})",
                    best_val_pinn)
plt.show()

# %% [markdown]
# ## 15. Compare residuals — MLP vs PINN

# %%
_, _, rc_pinn, rx_pinn, ry_pinn = residual_grid(model_pinn)
print(f"                     MLP            PINN")
print(f"  continuity RMS:  {rms(rc_mlp):.3e}    {rms(rc_pinn):.3e}    reduced {rms(rc_mlp)/rms(rc_pinn):.1f}x")
print(f"  momentum_x RMS:  {rms(rx_mlp):.3e}    {rms(rx_pinn):.3e}    reduced {rms(rx_mlp)/rms(rx_pinn):.1f}x")
print(f"  momentum_y RMS:  {rms(ry_mlp):.3e}    {rms(ry_pinn):.3e}    reduced {rms(ry_mlp)/rms(ry_pinn):.1f}x")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for col, (rc, rx, ry, label) in enumerate([(rc_mlp, rx_mlp, ry_mlp, "MLP"),
                                            (rc_pinn, rx_pinn, ry_pinn, "PINN")]):
    vmax_c = max(np.abs(rc_mlp).max(), np.abs(rc_pinn).max())
    vmax_m = max(np.abs(rx_mlp).max(), np.abs(rx_pinn).max(),
                 np.abs(ry_mlp).max(), np.abs(ry_pinn).max())
    plot_field(axes[col, 0], x_g, y_g, rc, cmap="coolwarm", vmin=-vmax_c, vmax=vmax_c,
               title=f"{label}  continuity  RMS {rms(rc):.2e}")
    plot_field(axes[col, 1], x_g, y_g, rx, cmap="coolwarm", vmin=-vmax_m, vmax=vmax_m,
               title=f"{label}  momentum_x  RMS {rms(rx):.2e}")
    plot_field(axes[col, 2], x_g, y_g, ry, cmap="coolwarm", vmin=-vmax_m, vmax=vmax_m,
               title=f"{label}  momentum_y  RMS {rms(ry):.2e}")
fig.suptitle("Physics residual fields: MLP (top) vs PINN (bottom)", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 16. Head-to-head reconstruction
#
# Six panels: truth, MLP recon, PINN recon, MLP error, PINN error,
# per-point improvement (MLP error − PINN error, blue = PINN wins).

# %%
truth_mag = np.sqrt(Ux_norm**2 + Uy_norm**2)
mlp_mag   = np.sqrt(mlp_Ux**2 + mlp_Uy**2)
pinn_mag  = np.sqrt(pinn_Ux**2 + pinn_Uy**2)

err_mlp   = np.sqrt((mlp_Ux  - Ux_norm)**2 + (mlp_Uy  - Uy_norm)**2)
err_pinn  = np.sqrt((pinn_Ux - Ux_norm)**2 + (pinn_Uy - Uy_norm)**2)
improv    = err_mlp - err_pinn

vmin = float(min(truth_mag.min(), mlp_mag.min(), pinn_mag.min()))
vmax = float(max(truth_mag.max(), mlp_mag.max(), pinn_mag.max()))
emax = float(max(err_mlp.max(), err_pinn.max())) or 1e-12
iabs = float(np.max(np.abs(improv))) or 1e-12

fig, axes = plt.subplots(1, 6, figsize=(24, 3.5))
m_r = plot_field(axes[0], x_phys, y_phys, truth_mag, vmin=vmin, vmax=vmax, title="truth |U|")
plot_field      (axes[1], x_phys, y_phys, mlp_mag,   vmin=vmin, vmax=vmax, title="MLP recon")
plot_field      (axes[2], x_phys, y_phys, pinn_mag,  vmin=vmin, vmax=vmax, title="PINN recon")
m_e = plot_field(axes[3], x_phys, y_phys, err_mlp,   cmap="magma", vmin=0, vmax=emax, title="MLP L2 error")
plot_field      (axes[4], x_phys, y_phys, err_pinn,  cmap="magma", vmin=0, vmax=emax, title="PINN L2 error")
m_i = plot_field(axes[5], x_phys, y_phys, improv,    cmap="RdBu",  vmin=-iabs, vmax=iabs,
                 title="MLP err − PINN err  (blue: PINN wins)")
for ax in axes:
    ax.scatter(x_phys[sensor_idx], y_phys[sensor_idx],
               s=14, facecolor='none', edgecolor='red', linewidths=0.7, zorder=5)
fig.colorbar(m_r, ax=axes[:3].tolist(), fraction=0.02, pad=0.01)
fig.colorbar(m_e, ax=axes[3:5].tolist(), fraction=0.025, pad=0.01)
fig.colorbar(m_i, ax=axes[5],           fraction=0.04, pad=0.01)

rel = 100 * (best_val_mlp - best_val_pinn) / best_val_mlp
fig.suptitle(f"K={K}   MLP MSE {best_val_mlp:.2e}   PINN MSE {best_val_pinn:.2e}   "
             f"({rel:+.0f}% improvement on held-out points)", y=1.04)
plt.show()

# %% [markdown]
# ## 17. Loss curves
#
# Per-term trajectories for the PINN.  Four physics-related losses (data,
# continuity, momentum_x, momentum_y) plus the held-out velocity MSE.

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
epochs_v = np.arange(1, len(history["val"]) + 1) * VAL_EVERY
ax.semilogy(epochs_v, history["data"],  label="data",        color="C0", lw=1.8)
ax.semilogy(epochs_v, history["cont"],  label="continuity",  color="C1", lw=1.2)
ax.semilogy(epochs_v, history["mom_x"], label="momentum_x",  color="C3", lw=1.2)
ax.semilogy(epochs_v, history["mom_y"], label="momentum_y",  color="C4", lw=1.2)
ax.semilogy(epochs_v, history["val"],   label="held-out val", color="C2", lw=2.0, marker="o", ms=3)
ax.set_xlabel("epoch"); ax.set_ylabel("loss  (log)")
ax.set_title("PINN training — per-term loss trajectories")
ax.grid(True, which="both", alpha=0.3); ax.legend()
plt.show()

# %% [markdown]
# ## 18. Try this
#
# Things to experiment with — every knob is in section 1:
#
#   1. **Momentum weight sweep.** Try `LAMBDA_MOMENT` ∈ {0, 0.0005, 0.001,
#      0.005, 0.01, 0.1}.  The working window is narrow — large values
#      catastrophically break training; small values give no improvement
#      over continuity-only.  Plot val MSE vs LAMBDA_MOMENT.
#
#   2. **What does each piece of physics contribute?** Turn off the
#      momentum term (`LAMBDA_MOMENT = 0.0`) and observe the val MSE.
#      Then turn off the continuity term (`LAMBDA_CONT = 0.0`) and keep
#      momentum.  Then both off (MLP).  Compare four configurations.
#
#   3. **Drop the cross-terms.** Inside `momentum_residual`, remove the
#      `cross_x` and `cross_y` contributions (set them to zero).  This
#      reduces full Boussinesq to the simplified Laplacian form.  See
#      what happens to val MSE.
#
#   4. **Drop the (2/3)k contribution.** Use plain `p` instead of `P_mod
#      = p + 2k/3` for the sensor data and pressure-gradient term.
#      Does it matter?
#
#   5. **Sensor sparsity.** Set `K = 20` or `K = 200` and re-run.  Where
#      does momentum stop helping?
#
#   6. **Better nu_t representation.** Increase `NUT_WIDTH` and
#      `NUT_EPOCHS` to fit nu_t more tightly.  Does the PINN result
#      improve in proportion?  This isolates "how much of the residual
#      is due to imperfect nu_t representation".
#
#   7. **Different case.** Switch to `CASE_ID = 267` (n=1).  Re-extract
#      the full CSV first with
#      `python scripts/extract_openfoam_fields.py --case ... --case_id 267`.
#      Does the momentum benefit hold up on a different geometry?
