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
# # 03 — Add Continuity Loss: The First PINN
#
# **Goal:** take the same MLP from notebook 02, add *one* extra term to
# the loss, and watch the reconstruction improve where the sensors are
# sparse.
#
# ## What physics is actually being enforced?
#
# Exactly **one** equation: the **incompressible continuity equation**,
#
# $$ \nabla \cdot \mathbf{u} \;=\; \frac{\partial U_x}{\partial x} + \frac{\partial U_y}{\partial y} \;=\; 0. $$
#
# This says "fluid is incompressible" — in 2D, the velocity field must be
# divergence-free everywhere.  That's a **pointwise** constraint on the
# two output components of our MLP.  It is NOT:
#
# - Momentum balance (Navier–Stokes).  That would require pressure and
#   viscosity terms; we're treating the kinematics alone.
# - A boundary condition.  Those come in notebook 04 (inlet, walls, baffles).
# - Anything that knows the geometry has 4 baffles.  The network still
#   has to *learn* the flow shape from sensors — continuity just tells
#   it that whatever shape it learns must be divergence-free.
#
# We impose the equation via a **soft loss**: pick 4096 random points in
# the domain ("collocation points"), evaluate ∂Ux/∂x + ∂Uy/∂y at each
# one with PyTorch autograd, and add the mean-squared residual to the
# total loss:
#
# $$
# \mathcal{L} \;=\; \underbrace{\text{MSE}_\text{sensors}}_{\text{data on K sensors}}
#             \;+\; \lambda \cdot \underbrace{\mathbb{E}_{(x,y)\sim\Omega}\!\left[\!\left(\tfrac{1}{L}\tfrac{\partial U_x}{\partial x_\text{norm}} + \tfrac{1}{W}\tfrac{\partial U_y}{\partial y_\text{norm}}\right)^2\right]}_{\text{continuity at collocation points}}
# $$
#
# (The $1/L$ and $1/W$ come from the chain rule because our coordinates
# are already normalized to $[0,1]$.)
#
# **Why this helps with sparse data:**  A plain MLP trained only on
# sensor values has no reason to produce a divergence-free field
# anywhere.  Between sensors, the two components $U_x$ and $U_y$ are
# effectively independent — the network can put any noisy guess there.
# Adding the continuity term makes $U_x$ and $U_y$ **coupled**
# everywhere, so the sensor information propagates spatially: if the
# sensors pin $U_x$ in some region, continuity drags $U_y$ along with
# it, and vice versa.

# %%
import os, sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from piml.config   import DataPaths, Hyperparams
from piml.data     import load_case, make_sensor_split
from piml.model    import get_device
from piml.train    import train_mlp, train_pinn, reconstruct, PhysicsWeights, PhysicsSampling
from piml.physics  import continuity_residual
from piml.plotting import (
    plot_reconstruction_panel, plot_residual_field,
    plot_loss_components, plot_head_to_head,
)

# %% [markdown]
# ## Setup

# %%
paths  = DataPaths()
device = get_device()
geom, field = load_case(paths)

K     = 50                # sweet spot for the default n=4 case
                          # (for case 267 / n=1, try K=20 instead)
split = make_sensor_split(field, n_sensors=K, seed=42)
hp    = Hyperparams(epochs=3000, width=64, depth=4, print_every=500, val_every=50)

# %% [markdown]
# ## Baseline: data-only MLP (same as notebook 02)
#
# Retrain so we have a side-by-side comparison in memory.

# %%
print("=== MLP baseline ===")
mlp = train_mlp(split, geom, hp, device, verbose=False)
print(f"MLP val MSE: {mlp['final_val_mse']:.3e}")

# %% [markdown]
# ## Add the continuity loss
#
# Same MLP, same sensors, same hyperparameters.  We only turn on the
# continuity term with a modest weight λ = 0.01 and disable the BC terms
# (those come in notebook 04).

# %%
print("\n=== PINN with continuity only ===")
# λ = 1.0 for inpainting — physics is the main signal, not a regularizer.
# The research multi-case surrogate used ~0.01 because data was abundant there.
weights  = PhysicsWeights(continuity=1.0, inlet=0.0, wall=0.0, baffle=0.0)
sampling = PhysicsSampling(n_colloc=4096, resample_every=100)
pinn     = train_pinn(split, geom, hp, device, weights=weights, sampling=sampling, verbose=True)
print(f"PINN val MSE: {pinn['final_val_mse']:.3e}")

# %% [markdown]
# ## Head-to-head: val MSE
#
# This number is the reconstruction error on the held-out CFD points.
# Lower is better.

# %%
imp = (mlp['final_val_mse'] - pinn['final_val_mse']) / mlp['final_val_mse']
print(f"MLP   val MSE: {mlp['final_val_mse']:.3e}")
print(f"PINN  val MSE: {pinn['final_val_mse']:.3e}")
print(f"Relative improvement: {100*imp:+.1f}%")

# %% [markdown]
# ## Measure the divergence residual for both models

# %%
def measure_divergence(model, n_side: int = 80):
    xs = np.linspace(0, 1, n_side, dtype=np.float32)
    ys = np.linspace(0, 1, n_side, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    xt = torch.tensor(X.reshape(-1, 1), device=device, requires_grad=True)
    yt = torch.tensor(Y.reshape(-1, 1), device=device, requires_grad=True)
    r  = continuity_residual(model, xt, yt, geom).detach().cpu().numpy().squeeze()
    return X.flatten() * geom.L, Y.flatten() * geom.W, r

x_g, y_g, r_mlp  = measure_divergence(mlp['model'])
_,   _,   r_pinn = measure_divergence(pinn['model'])

rms = lambda v: np.sqrt(np.mean(v**2))
print(f"MLP  divergence RMS: {rms(r_mlp):.3e}")
print(f"PINN divergence RMS: {rms(r_pinn):.3e}")
print(f"Reduction: {rms(r_mlp) / rms(r_pinn):.1f}×")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
plot_residual_field(x_g, y_g, r_mlp,  ax=axes[0], title=f"MLP  ∇·u   RMS {rms(r_mlp):.2e}")
plot_residual_field(x_g, y_g, r_pinn, ax=axes[1], title=f"PINN ∇·u   RMS {rms(r_pinn):.2e}")
plt.show()

# %% [markdown]
# ## Reconstruction comparison — MLP vs PINN
#
# The full-domain reconstruction panel, same as in notebook 02, but now
# for both models.  Watch what happens in regions far from any sensor.

# %%
sensor_xy = (field['x_phys'][split['sensor_idx']], field['y_phys'][split['sensor_idx']])

pred_mlp  = reconstruct(mlp['model'],  field['x_norm'], field['y_norm'], geom, device)
pred_pinn = reconstruct(pinn['model'], field['x_norm'], field['y_norm'], geom, device)

_ = plot_reconstruction_panel(
    field['x_phys'], field['y_phys'],
    field['Ux_norm'], field['Uy_norm'], pred_mlp[:, 0], pred_mlp[:, 1],
    suptitle=f"MLP (no physics)  — K={K}  val MSE {mlp['final_val_mse']:.2e}",
    sensor_xy=sensor_xy,
)
plt.show()

_ = plot_reconstruction_panel(
    field['x_phys'], field['y_phys'],
    field['Ux_norm'], field['Uy_norm'], pred_pinn[:, 0], pred_pinn[:, 1],
    suptitle=f"PINN (continuity only, λ=1.0)  — K={K}  val MSE {pinn['final_val_mse']:.2e}",
    sensor_xy=sensor_xy,
)
plt.show()

# %% [markdown]
# ## Head-to-head (the "money shot")
#
# Single row: truth  |  MLP reconstruction  |  PINN reconstruction  |
# MLP error  |  PINN error.  Sensor locations are red rings on every panel.
#
# Look at the two error columns: the PINN's error is both smaller in
# magnitude and more *locally* concentrated near sensor boundaries.  The
# MLP's error is smeared over large regions — those are the gaps between
# sensors where it had nothing to interpolate from.

# %%
_ = plot_head_to_head(
    field['x_phys'], field['y_phys'],
    field['Ux_norm'], field['Uy_norm'],
    pred_mlp[:, 0],  pred_mlp[:, 1],
    pred_pinn[:, 0], pred_pinn[:, 1],
    a_label="MLP", b_label="PINN", component="mag",
    suptitle=f"K={K}   MLP MSE {mlp['final_val_mse']:.2e}   "
             f"PINN MSE {pinn['final_val_mse']:.2e}   "
             f"({100*(mlp['final_val_mse']-pinn['final_val_mse'])/mlp['final_val_mse']:+.0f}% improvement)",
    sensor_xy=sensor_xy,
)
plt.show()

# %% [markdown]
# ## Per-term loss trajectories
#
# Multi-term losses need per-term monitoring.  Here you can see the data
# loss settle, the continuity loss ramp in (20% warmup), and the
# validation loss on the held-out field.

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
plot_loss_components(pinn['history'], val_every=hp.val_every, ax=ax,
                     title="PINN training — per-term losses")
plt.show()

# %% [markdown]
# ## Try this
#
# 1. Drop K to 20.  The MLP should collapse (near-zero predictions in
#    large regions); the PINN should still produce something resembling
#    a divergence-free flow.  The physics is doing most of the work.
# 2. Sweep λ ∈ {0, 1e-3, 1e-2, 1e-1, 1}.  For each, record val MSE and
#    divergence RMS.  Is more λ always better?
# 3. Turn off physics warmup (edit `piml/physics.py::warmup_weight` to
#    return 1.0).  Does training become less stable at high λ?
#
# Notebook 04 will add the inlet/wall/baffle BC losses — those pin the
# prediction's values (not just its divergence) in known regions and
# give another jump in reconstruction quality.
