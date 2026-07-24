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
# # 2D PINN Inpainting — Sparse Velocity Field Reconstruction
#
# **The goal**: reconstruct a 2D CFD velocity field given only K sparse
# "sensor" observations, and show that adding the incompressible
# continuity equation as a soft loss term lets the network fill in the
# gaps between sensors much better than a plain regression.
#
# **Outline**:
#
#   1. Load one CFD case, normalize it.
#   2. Pretend we only see K random "sensor" points.
#   3. Train a plain MLP to fit those K points (data-only).
#   4. Visualize the reconstruction — it's garbage in the gaps.
#   5. Measure the divergence ∇·u of the MLP's predicted field — it's
#      large, i.e., the MLP's field is unphysical.
#   6. Train the same MLP *again*, this time adding the squared continuity
#      residual at random collocation points as a soft loss term.
#   7. Show that the reconstruction now fills the gaps AND has near-zero
#      divergence everywhere.
#
# This is the classic Raissi-style PINN inpainting setup, applied to a
# 2D contact basin geometry.

# %% [markdown]
# ## 0. Imports
#
# We need numpy + pandas for data wrangling, PyTorch for the network and
# autograd, matplotlib for plotting.  Nothing else.

# %%
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Resolve project root from THIS file's location so the script runs no
# matter where it's launched from.
_HERE         = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(".")
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# %% [markdown]
# ## 1. Configuration
#
# Every knob you might want to tweak lives in this one block.  Change
# `K` to 10 / 50 / 500 to see how sensor sparsity affects the result, or
# change `LAMBDA_PHYSICS` to see how strongly the physics term pulls the
# network.

# %%
# --- which case, which device ---
CASE_ID    = 834                                            # 834 = n=4 (default); 267 = n=1 (simpler)
DATA_DIR   = os.path.join(_PROJECT_ROOT, "data", "tiny")
DEVICE     = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# --- sensors ---
K          = 50                                             # number of sensors (the inpainting problem)
SEED       = 42                                             # controls both the sensor split and the network init

# --- network ---
WIDTH      = 64
DEPTH      = 4                                              # number of hidden layers

# --- training ---
EPOCHS     = 5000
LR         = 1e-3
VAL_EVERY  = 50
PRINT_EVERY= 500

# --- physics (PINN only) ---
LAMBDA_PHYSICS = 1.0                                        # weight on the continuity loss
N_COLLOC       = 4096                                       # number of collocation (interior) points
WARMUP_FRAC    = 0.2                                        # linearly ramp the physics term in over this fraction of training

print(f"device: {DEVICE}")
print(f"case: {CASE_ID},  K = {K} sensors,  λ = {LAMBDA_PHYSICS},  {EPOCHS} epochs")

# %% [markdown]
# ## 2. Load the CFD data
#
# Two CSVs:
#
#   - `basin_geometries_summary.csv` has one row per case with the
#     geometry parameters (L, W, n baffles, inlet velocity, ...).
#   - `case_{ID}_velocity.csv` has one row per CFD mesh point with its
#     (x, y, Ux, Uy).
#
# We **normalize** both:
#   - coordinates: `x_norm = x/L`, `y_norm = y/W` → both in [0, 1]
#   - velocities: `Ux_norm = Ux/U_inlet`, `Uy_norm = Uy/U_inlet`
#
# Normalization makes the network's inputs and outputs O(1), which makes
# autograd through the continuity residual numerically stable.

# %%
geo_csv = os.path.join(DATA_DIR, "basin_geometries_summary.csv")
vel_csv = os.path.join(DATA_DIR, "results", f"case_{CASE_ID:03d}_velocity.csv")

geo_row = pd.read_csv(geo_csv).query(f"`Simulation Number` == {CASE_ID}").iloc[0]
L       = float(geo_row['L'])
W       = float(geo_row['W'])
n_baf   = int  (geo_row['n'])
a_baf   = float(geo_row['a'])
U_inlet = float(geo_row['U_inlet'])

vel = pd.read_csv(vel_csv).dropna(subset=['Ux', 'Uy'])
x_phys  = vel['x'].values.astype(np.float32)
y_phys  = vel['y'].values.astype(np.float32)
x_norm  = (vel['x'].values  / L      ).astype(np.float32)
y_norm  = (vel['y'].values  / W      ).astype(np.float32)
Ux_norm = (vel['Ux'].values / U_inlet).astype(np.float32)
Uy_norm = (vel['Uy'].values / U_inlet).astype(np.float32)

N = len(x_norm)
print(f"Case {CASE_ID}: L={L:.0f} m, W={W:.0f} m, n={n_baf} baffles, U_inlet={U_inlet:.3f} m/s")
print(f"  {N:,} CFD mesh points loaded")

# %% [markdown]
# ## 3. Pick K random sensors
#
# We hold out (N − K) points for evaluation: the network never sees them
# during training.  Reconstruction error on this held-out set is the
# metric that matters.

# %%
rng        = np.random.default_rng(SEED)
sensor_idx = rng.choice(N, size=K, replace=False)
eval_idx   = np.setdiff1d(np.arange(N), sensor_idx)

print(f"K = {K} sensors,  {len(eval_idx):,} held-out evaluation points")

# Pre-build tensors once so the training loops don't re-allocate every epoch.
def to_col(arr):
    """Move a 1D numpy array to a (N, 1) torch tensor on DEVICE."""
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(1)

sens_x  = to_col(x_norm [sensor_idx])
sens_y  = to_col(y_norm [sensor_idx])
sens_Ux = to_col(Ux_norm[sensor_idx])
sens_Uy = to_col(Uy_norm[sensor_idx])

eval_x  = to_col(x_norm [eval_idx])
eval_y  = to_col(y_norm [eval_idx])
eval_Ux = to_col(Ux_norm[eval_idx])
eval_Uy = to_col(Uy_norm[eval_idx])

# %% [markdown]
# ## 4. Plotting helper
#
# CFD outputs live on an unstructured mesh.  `tripcolor` with Gouraud
# shading + edge-length masking gives a much nicer rendering than scatter
# — it also avoids drawing long thin triangles that would "bridge" gaps
# across baffles.  This is the one helper we keep around because every
# plot below uses it.

# %%
def plot_field(ax, x, y, c, *, cmap="viridis", vmin=None, vmax=None,
               title: str = "", edge_cutoff_mult: float = 3.0):
    """Smooth-shaded plot of a scalar field on unstructured 2D points."""
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c)
    tri = mtri.Triangulation(x, y)
    # Mask triangles whose longest edge is much longer than typical —
    # these are the ones that bridge across baffles or domain holes.
    pts        = np.stack([x[tri.triangles], y[tri.triangles]], axis=-1)
    edges      = np.linalg.norm(np.diff(pts[:, [0, 1, 2, 0], :], axis=1), axis=-1)
    max_edge   = edges.max(axis=1)
    tri.set_mask(max_edge > edge_cutoff_mult * np.median(max_edge))
    m = ax.tripcolor(tri, c, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)
    return m

# %% [markdown]
# ## 5. Visualize the ground truth + sensor locations
#
# Red rings = sensors.  Everything else is held-out.  Notice how few
# observations the network has to work with in the middle of the basin.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
m1 = plot_field(axes[0], x_phys, y_phys, Ux_norm * U_inlet, title="CFD truth  Ux (m/s)")
m2 = plot_field(axes[1], x_phys, y_phys, Uy_norm * U_inlet, title="CFD truth  Uy (m/s)")
mag = np.sqrt((Ux_norm*U_inlet)**2 + (Uy_norm*U_inlet)**2)
m3 = plot_field(axes[2], x_phys, y_phys, mag, title=f"CFD truth |U|  + K={K} sensors")
axes[2].scatter(x_phys[sensor_idx], y_phys[sensor_idx],
                s=35, facecolor='none', edgecolor='red', linewidths=1.4, zorder=5)
for m, ax in zip([m1, m2, m3], axes):
    fig.colorbar(m, ax=ax, fraction=0.046)
fig.suptitle(f"Case {CASE_ID} ground truth ({N:,} mesh points, K={K} sensors)", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. The neural network
#
# A plain MLP that maps (x, y) → (Ux, Uy).  Just two inputs and two
# outputs since we have a single geometry.
#
# **Why Tanh and not ReLU**: the continuity loss needs the network's
# *derivative*, and that derivative needs to be differentiable too
# (autograd computes ∂Ux/∂x using the chain rule).  ReLU's derivative is
# a step function with zero derivative almost everywhere — useless for
# physics.  Tanh has smooth derivatives at every order.
#
# **Why 4 hidden layers of width 64**: arbitrary.  Big enough to fit the
# flow, small enough to train in seconds.  Try changing `DEPTH` to 2 or 8
# — the PINN should still work, the MLP-only baseline might struggle
# differently.

# %%
class MLP(nn.Module):
    """(x_norm, y_norm) -> (Ux_norm, Uy_norm).  Tanh activations."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        # x, y both shape (N, 1).  Concatenate along dim=1 to feed the net.
        return self.net(torch.cat([x, y], dim=1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

# %% [markdown]
# ## 7. The continuity residual
#
# The physics constraint.  For incompressible 2D flow:
#
# $$ \nabla \cdot \mathbf{u} \;=\; \frac{\partial U_x}{\partial x} + \frac{\partial U_y}{\partial y} \;=\; 0 $$
#
# But our network outputs `Ux_norm`, `Uy_norm` in normalized coordinates
# `x_norm = x/L`, `y_norm = y/W`.  Chain rule:
#
# $$ \frac{\partial U_x}{\partial x} = \frac{1}{L} \frac{\partial U_{x,\text{norm}}}{\partial x_{\text{norm}}} \quad \text{(and similarly for }y\text{)} $$
#
# So the residual we compute is `(1/L)·∂Ux/∂x_norm + (1/W)·∂Uy/∂y_norm`.
# This is the actual `∇·u` in physical units (up to the constant
# `U_inlet` factor we dropped when normalizing the velocities — it just
# scales the residual uniformly and doesn't affect the optimum).
#
# **How autograd works here**: we ask PyTorch to track gradients with
# respect to the input coordinates (`requires_grad=True`), then use
# `torch.autograd.grad` to differentiate the network's outputs w.r.t.
# those inputs.  `create_graph=True` is essential — it builds a *second*
# computation graph through the derivative so that when we call
# `.backward()` on the loss, gradients flow back through the derivatives
# to the network weights.

# %%
def continuity_residual(model, x, y):
    """
    Pointwise ∇·u at the given (x, y) points.  x and y must have
    requires_grad=True so we can differentiate the model's output w.r.t.
    them.  Returns (N, 1) tensor of residual values.
    """
    vel = model(x, y)                      # (N, 2)
    Ux, Uy = vel[:, 0:1], vel[:, 1:2]      # split into (N, 1) each

    dUx_dx = torch.autograd.grad(
        Ux, x, grad_outputs=torch.ones_like(Ux),
        create_graph=True, retain_graph=True,
    )[0]
    dUy_dy = torch.autograd.grad(
        Uy, y, grad_outputs=torch.ones_like(Uy),
        create_graph=True, retain_graph=True,
    )[0]
    return dUx_dx / L + dUy_dy / W


def continuity_loss(model, x, y):
    """Mean squared residual at the collocation points."""
    r = continuity_residual(model, x, y)
    return torch.mean(r ** 2)

# %% [markdown]
# ## 8. Train the plain MLP (no physics)
#
# Standard supervised-regression loop.  The only loss term is MSE between
# the network's prediction at each sensor location and the true (Ux, Uy)
# at that location.  Adam, fixed LR.  Every `VAL_EVERY` epochs, we
# evaluate held-out MSE and keep the best-val checkpoint.

# %%
torch.manual_seed(SEED); np.random.seed(SEED)
model_mlp = MLP(WIDTH, DEPTH).to(DEVICE)
opt       = torch.optim.Adam(model_mlp.parameters(), lr=LR)

print(f"MLP: {model_mlp.num_params():,} parameters,  training on {K} sensors")

train_losses_mlp, val_losses_mlp = [], []
best_val_mlp, best_state_mlp     = float("inf"), None
t0 = time.time()

for ep in range(EPOCHS):
    model_mlp.train(); opt.zero_grad()

    # Forward + data loss
    pred = model_mlp(sens_x, sens_y)
    loss = F.mse_loss(pred[:, 0:1], sens_Ux) + F.mse_loss(pred[:, 1:2], sens_Uy)

    # Backward + step
    loss.backward(); opt.step()
    train_losses_mlp.append(loss.item())

    # Periodic held-out validation
    if (ep + 1) % VAL_EVERY == 0:
        model_mlp.eval()
        with torch.no_grad():
            p  = model_mlp(eval_x, eval_y)
            vl = (F.mse_loss(p[:, 0:1], eval_Ux) +
                  F.mse_loss(p[:, 1:2], eval_Uy)).item()
        val_losses_mlp.append(vl)
        if vl < best_val_mlp:
            best_val_mlp   = vl
            best_state_mlp = {k: v.detach().clone() for k, v in model_mlp.state_dict().items()}

    if (ep + 1) % PRINT_EVERY == 0:
        print(f"  ep {ep+1:5d}/{EPOCHS}   train {loss.item():.3e}   val {val_losses_mlp[-1]:.3e}")

model_mlp.load_state_dict(best_state_mlp)
print(f"MLP done in {time.time()-t0:.1f}s  |  best held-out MSE = {best_val_mlp:.3e}")

# %% [markdown]
# ## 9. Visualize the MLP reconstruction — see the failure
#
# Predict everywhere, plot truth vs reconstruction vs error.  The error
# panel shows the **pointwise L2 vector error** ||vec_pred − vec_truth||
# (non-negative; bright = bad reconstruction).  Expect to see bright
# patches in the regions far from any sensor.

# %%
def predict_field(model, x_n, y_n):
    """Run the model on all mesh points and return numpy (Ux_pred, Uy_pred)."""
    model.eval()
    with torch.no_grad():
        p = model(to_col(x_n), to_col(y_n)).cpu().numpy()
    return p[:, 0], p[:, 1]

pred_Ux_mlp, pred_Uy_mlp = predict_field(model_mlp, x_norm, y_norm)

def show_reconstruction(pred_Ux, pred_Uy, model_name: str, val_mse: float):
    """3-panel comparison: truth |U| | reconstruction |U| | pointwise L2 vector error.

    The error panel uses ||vec_pred − vec_truth|| at each point.  This is
    always ≥ 0 and captures both magnitude and direction errors — unlike
    ||U|_pred − |U|_truth|, which collapses to a scalar and can be small
    even when the predicted vector points the wrong way.
    """
    truth_mag = np.sqrt(Ux_norm**2 + Uy_norm**2)
    pred_mag  = np.sqrt(pred_Ux**2 + pred_Uy**2)
    err       = np.sqrt((pred_Ux - Ux_norm)**2 + (pred_Uy - Uy_norm)**2)

    vmin, vmax = float(min(truth_mag.min(), pred_mag.min())), float(max(truth_mag.max(), pred_mag.max()))
    emax       = float(err.max()) or 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    mT = plot_field(axes[0], x_phys, y_phys, truth_mag, vmin=vmin, vmax=vmax, title="truth |U|")
    mP = plot_field(axes[1], x_phys, y_phys, pred_mag,  vmin=vmin, vmax=vmax, title=f"{model_name} reconstruction")
    mE = plot_field(axes[2], x_phys, y_phys, err,       cmap="magma",  vmin=0, vmax=emax, title="L2 vector error  ||vec_pred − vec_truth||")
    for ax in axes:
        ax.scatter(x_phys[sensor_idx], y_phys[sensor_idx],
                   s=18, facecolor='none', edgecolor='red', linewidths=0.8, zorder=5)
    fig.colorbar(mP, ax=axes[:2].tolist(), fraction=0.025)
    fig.colorbar(mE, ax=axes[2], fraction=0.04)
    fig.suptitle(f"{model_name}  —  K={K}  —  held-out MSE = {val_mse:.2e}", y=1.02)
    return fig

show_reconstruction(pred_Ux_mlp, pred_Uy_mlp, "MLP (no physics)", best_val_mlp)
plt.show()

# %% [markdown]
# ## 10. How unphysical is the MLP's field?
#
# Measure ∇·u on an 80×80 grid using the same autograd machinery we'll
# use as the physics loss in the PINN.  For incompressible flow this
# should be 0 everywhere.  It won't be.

# %%
def divergence_grid(model, n_side: int = 80):
    xs = np.linspace(0, 1, n_side, dtype=np.float32)
    ys = np.linspace(0, 1, n_side, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    xt = torch.tensor(X.reshape(-1, 1), device=DEVICE, requires_grad=True)
    yt = torch.tensor(Y.reshape(-1, 1), device=DEVICE, requires_grad=True)
    r  = continuity_residual(model, xt, yt).detach().cpu().numpy().squeeze()
    return X.flatten() * L, Y.flatten() * W, r

x_g, y_g, r_mlp = divergence_grid(model_mlp)
div_rms_mlp = float(np.sqrt(np.mean(r_mlp ** 2)))
print(f"MLP divergence RMS = {div_rms_mlp:.3e}   (should be 0 for incompressible flow)")

fig, ax = plt.subplots(figsize=(9, 3.5))
vmax = float(np.max(np.abs(r_mlp))) or 1e-12
plot_field(ax, x_g, y_g, r_mlp, cmap="coolwarm", vmin=-vmax, vmax=vmax,
           title=f"MLP ∇·u   (RMS {div_rms_mlp:.2e};  should be 0)")
plt.colorbar(ax.collections[0], ax=ax, fraction=0.04)
plt.show()

# %% [markdown]
# ## 11. Train the same MLP again, but with continuity
#
# Same network class, same hyperparameters, same sensor data.  The only
# things that change:
#
#   1. We generate a batch of `N_COLLOC` random "collocation" points in
#      the interior of the domain.  These are NOT observation points —
#      they're places where we ask the network to satisfy the physics.
#   2. The loss is now `L_data + λ · ramp(ep) · L_continuity`.
#   3. We track per-term losses so we can see which one is doing what.
#
# **Why the warmup**: at the start of training, the network's outputs
# are random.  Computing physics derivatives of random outputs gives
# random gradients that can fight with the data loss and destabilize
# early training.  Linearly ramping the physics weight from 0 to 1 over
# the first `WARMUP_FRAC` of training lets the data term anchor the
# prediction first, then the physics term refines it.

# %%
# Collocation points (fixed throughout training, for simplicity).  In
# research-scale PINNs these are often resampled every N epochs; a fixed
# set works fine on this problem.
rng_c     = np.random.default_rng(0)
colloc_x_np = rng_c.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32)
colloc_y_np = rng_c.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32)
colloc_x  = torch.tensor(colloc_x_np, device=DEVICE, requires_grad=True)
colloc_y  = torch.tensor(colloc_y_np, device=DEVICE, requires_grad=True)

torch.manual_seed(SEED); np.random.seed(SEED)
model_pinn = MLP(WIDTH, DEPTH).to(DEVICE)
opt        = torch.optim.Adam(model_pinn.parameters(), lr=LR)

print(f"PINN: {model_pinn.num_params():,} parameters,  {K} sensors,  {N_COLLOC} collocation points,  λ = {LAMBDA_PHYSICS}")

history = {"data": [], "phys": [], "total": [], "val": []}
best_val_pinn, best_state_pinn = float("inf"), None
t0 = time.time()

for ep in range(EPOCHS):
    model_pinn.train(); opt.zero_grad()

    # --- Data loss (same as MLP) ---
    pred   = model_pinn(sens_x, sens_y)
    L_data = F.mse_loss(pred[:, 0:1], sens_Ux) + F.mse_loss(pred[:, 1:2], sens_Uy)

    # --- Physics loss (NEW) ---
    L_phys = continuity_loss(model_pinn, colloc_x, colloc_y)

    # --- Combine, with linear warmup on the physics weight ---
    ramp    = min(1.0, ep / (EPOCHS * WARMUP_FRAC)) if WARMUP_FRAC > 0 else 1.0
    L_total = L_data + ramp * LAMBDA_PHYSICS * L_phys

    L_total.backward(); opt.step()

    # Periodic held-out validation + per-term history
    if (ep + 1) % VAL_EVERY == 0:
        model_pinn.eval()
        with torch.no_grad():
            p  = model_pinn(eval_x, eval_y)
            vl = (F.mse_loss(p[:, 0:1], eval_Ux) +
                  F.mse_loss(p[:, 1:2], eval_Uy)).item()
        history["data" ].append(L_data .item())
        history["phys" ].append(L_phys .item())
        history["total"].append(L_total.item())
        history["val"  ].append(vl)
        if vl < best_val_pinn:
            best_val_pinn   = vl
            best_state_pinn = {k: v.detach().clone() for k, v in model_pinn.state_dict().items()}

    if (ep + 1) % PRINT_EVERY == 0:
        print(f"  ep {ep+1:5d}/{EPOCHS}   data {L_data.item():.3e}   phys {L_phys.item():.3e}   "
              f"val {history['val'][-1]:.3e}   ramp {ramp:.2f}")

model_pinn.load_state_dict(best_state_pinn)
print(f"PINN done in {time.time()-t0:.1f}s  |  best held-out MSE = {best_val_pinn:.3e}")

# %% [markdown]
# ## 12. Visualize the PINN reconstruction
#
# Same 3-panel layout as the MLP.  Compare side-by-side mentally: the
# PINN error column should be visibly smaller and more concentrated
# near sensor boundaries, not smeared across whole regions.

# %%
pred_Ux_pinn, pred_Uy_pinn = predict_field(model_pinn, x_norm, y_norm)
show_reconstruction(pred_Ux_pinn, pred_Uy_pinn, f"PINN (continuity, λ={LAMBDA_PHYSICS})", best_val_pinn)
plt.show()

# %% [markdown]
# ## 13. Did the divergence actually drop?

# %%
_, _, r_pinn  = divergence_grid(model_pinn)
div_rms_pinn  = float(np.sqrt(np.mean(r_pinn ** 2)))
print(f"MLP   divergence RMS = {div_rms_mlp:.3e}")
print(f"PINN  divergence RMS = {div_rms_pinn:.3e}")
print(f"Reduction: {div_rms_mlp / div_rms_pinn:.1f}×")

fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
vmax = float(max(np.max(np.abs(r_mlp)), np.max(np.abs(r_pinn))))
plot_field(axes[0], x_g, y_g, r_mlp,  cmap="coolwarm", vmin=-vmax, vmax=vmax,
           title=f"MLP   ∇·u   RMS {div_rms_mlp:.2e}")
plot_field(axes[1], x_g, y_g, r_pinn, cmap="coolwarm", vmin=-vmax, vmax=vmax,
           title=f"PINN  ∇·u   RMS {div_rms_pinn:.2e}")
plt.show()

# %% [markdown]
# ## 14. Head-to-head comparison
#
# Six panels in one row:
#
#   1. Truth |U|
#   2. MLP reconstruction |U|
#   3. PINN reconstruction |U|
#   4. MLP **L2 vector error** at each point  =  ||vec_pred − vec_truth||
#   5. PINN L2 vector error at each point
#   6. **Per-point improvement** = MLP_error − PINN_error.  Blue where the
#      PINN does better than the MLP, red where it does worse.
#
# Why L2 vector error and not signed magnitude error?  ||U|_pred − |U|_truth|
# collapses two vector components into one scalar — a prediction pointing
# the wrong direction with the right magnitude shows up as zero error.
# Plotting ||vec_pred − vec_truth|| captures both magnitude AND direction
# errors, and is always non-negative (use a one-sided colormap).
#
# The improvement panel shows *where in the domain* the physics constraint
# helped reconstruction.  Expect to see blue patches in the gaps between
# sensors.

# %%
truth_mag = np.sqrt(Ux_norm**2 + Uy_norm**2)
mlp_mag   = np.sqrt(pred_Ux_mlp**2  + pred_Uy_mlp**2)
pinn_mag  = np.sqrt(pred_Ux_pinn**2 + pred_Uy_pinn**2)

# Pointwise L2 vector error: ||vec_pred - vec_truth||  (always ≥ 0)
err_mlp   = np.sqrt((pred_Ux_mlp  - Ux_norm)**2 + (pred_Uy_mlp  - Uy_norm)**2)
err_pinn  = np.sqrt((pred_Ux_pinn - Ux_norm)**2 + (pred_Uy_pinn - Uy_norm)**2)
improv    = err_mlp - err_pinn        # > 0 means PINN error smaller at this point

vmin = float(min(truth_mag.min(), mlp_mag.min(), pinn_mag.min()))
vmax = float(max(truth_mag.max(), mlp_mag.max(), pinn_mag.max()))
emax = float(max(err_mlp.max(), err_pinn.max())) or 1e-12
iabs = float(np.max(np.abs(improv)))             or 1e-12

fig, axes = plt.subplots(1, 6, figsize=(24, 3.5))
m_r = plot_field(axes[0], x_phys, y_phys, truth_mag, vmin=vmin, vmax=vmax, title="truth |U|")
_   = plot_field(axes[1], x_phys, y_phys, mlp_mag,   vmin=vmin, vmax=vmax, title="MLP recon")
_   = plot_field(axes[2], x_phys, y_phys, pinn_mag,  vmin=vmin, vmax=vmax, title="PINN recon")
m_e = plot_field(axes[3], x_phys, y_phys, err_mlp,   cmap="magma", vmin=0, vmax=emax, title="MLP L2 error")
_   = plot_field(axes[4], x_phys, y_phys, err_pinn,  cmap="magma", vmin=0, vmax=emax, title="PINN L2 error")
m_i = plot_field(axes[5], x_phys, y_phys, improv,    cmap="RdBu",  vmin=-iabs, vmax=iabs,
                 title="MLP err − PINN err   (blue: PINN wins)")
for ax in axes:
    ax.scatter(x_phys[sensor_idx], y_phys[sensor_idx],
               s=14, facecolor='none', edgecolor='red', linewidths=0.7, zorder=5)
fig.colorbar(m_r, ax=axes[:3].tolist(),    fraction=0.02,  pad=0.01)
fig.colorbar(m_e, ax=axes[3:5].tolist(),   fraction=0.025, pad=0.01)
fig.colorbar(m_i, ax=axes[5],              fraction=0.04,  pad=0.01)

imp_pct = 100 * (best_val_mlp - best_val_pinn) / best_val_mlp
fig.suptitle(f"K={K}   MLP MSE {best_val_mlp:.2e}   PINN MSE {best_val_pinn:.2e}   "
             f"({imp_pct:+.0f}% improvement on held-out points)", y=1.04)
plt.show()

# %% [markdown]
# ## 15. Loss curves
#
# For the PINN: data loss, physics loss, total loss, and held-out
# validation loss, all on a log y-axis.

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
epochs_v = np.arange(1, len(history["val"]) + 1) * VAL_EVERY
ax.semilogy(epochs_v, history["data"],  label="data loss",   color="C0", lw=1.8)
ax.semilogy(epochs_v, history["phys"],  label="continuity",  color="C1", lw=1.2)
ax.semilogy(epochs_v, history["total"], label="total",       color="k",  lw=1.5, alpha=0.6)
ax.semilogy(epochs_v, history["val"],   label="held-out val",color="C2", lw=2.0, marker="o", ms=3)
ax.set_xlabel("epoch"); ax.set_ylabel("loss  (log)")
ax.set_title("PINN training — per-term loss trajectories")
ax.grid(True, which="both", alpha=0.3); ax.legend()
plt.show()

# %% [markdown]
# ## 16. Try this
#
# Things to experiment with — every knob is in section 1, every piece of
# logic is in this file:
#
#   1. **Sensor sparsity.** Set `K = 10` or `K = 500` and re-run.  At
#      low K, both models fail (not enough information).  At high K, both
#      succeed (the MLP doesn't need physics).  K ≈ 50 is the regime
#      where physics earns its keep on this case.
#
#   2. **Physics weight.** Sweep `LAMBDA_PHYSICS` ∈ {0.0, 0.01, 0.1, 1.0,
#      10.0}.  Plot val MSE vs λ.  Is bigger always better?
#
#   3. **Warmup.** Set `WARMUP_FRAC = 0.0` (no warmup).  Does the PINN
#      become less stable at high λ?  Look at the early epochs of the
#      loss curves.
#
#   4. **Activation function.** Replace `nn.Tanh()` with `nn.ReLU()` in
#      the MLP definition.  Re-run the PINN section.  What happens?
#      (Hint: the second derivative of ReLU is zero — physics loss
#      becomes useless.)  Try `nn.GELU()` or `nn.SiLU()` — both have
#      smooth second derivatives.
#
#   5. **Different case.** Change `CASE_ID = 267` (n=1, simpler flow).
#      The README claims the sweet spot moves to K ≈ 20 on this case.
#      Confirm or refute.
#
#   6. **Resampled collocation points.** Add a block inside the PINN
#      training loop that regenerates `colloc_x, colloc_y` every 100
#      epochs.  Does the result improve?  (For this geometry, probably
#      not much; for more complex flows, often yes.)
#
#   7. **A different physics term.** Replace the continuity loss with a
#      generic smoothness penalty:
#      `L_phys = mean((dUx/dx)**2 + (dUx/dy)**2 + (dUy/dx)**2 + (dUy/dy)**2)`.
#      Compare the reconstruction.  If it's comparable to the continuity
#      PINN's reconstruction, the lesson is that *any* smoothing term
#      helps in the sparse regime, not continuity specifically.  If the
#      continuity result is clearly better, the physics is doing
#      identifiable work beyond generic smoothing.
