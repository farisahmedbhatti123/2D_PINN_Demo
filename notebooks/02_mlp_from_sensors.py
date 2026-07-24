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
# # 02 — MLP Reconstruction from Sensors (No Physics)
#
# **Goal:** train a plain MLP on just the K sensor observations and see
# how badly it fails in the gaps between sensors.
#
# This is the baseline against which notebook 03's continuity-enforcing
# PINN will be compared.  The failure mode here is the *whole point* — if
# the MLP worked perfectly, physics wouldn't have anything to add.

# %%
import os, sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt

from piml.config   import DataPaths, Hyperparams
from piml.data     import load_case, make_sensor_split
from piml.model    import get_device
from piml.train    import train_mlp, reconstruct
from piml.plotting import (
    plot_loss_curves, plot_reconstruction_panel, plot_velocity_field,
)

# %% [markdown]
# ## Load and split

# %%
paths  = DataPaths()
device = get_device()
print("device:", device)

geom, field = load_case(paths)

K     = 50               # try K ∈ {10, 20, 50, 100, 500}
split = make_sensor_split(field, n_sensors=K, seed=42)
print(f"K = {K}  |  eval set: {len(split['eval_idx']):,} held-out points")

# %% [markdown]
# ## Train the MLP
#
# Data loss only: MSE between the MLP's (Ux, Uy) at each sensor location
# and the sensor observation.  No continuity, no BCs.

# %%
hp = Hyperparams(
    epochs       = 3000,
    width        = 64,
    depth        = 4,
    lr           = 1e-3,
    print_every  = 500,
    val_every    = 50,
)
mlp = train_mlp(split, geom, hp, device, verbose=True)
print(f"\nfinal val MSE: {mlp['final_val_mse']:.3e}")

# %% [markdown]
# ## Training curves
#
# Train loss (sensor MSE) and val loss (held-out eval MSE) on log scale.
# Watch the val curve — if it rises while train falls, the network is
# memorizing the K sensors and extrapolating garbage elsewhere.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
plot_loss_curves(mlp['train_losses'], mlp['val_losses'], val_every=hp.val_every, ax=ax,
                 title=f"MLP from {K} sensors (train vs. held-out val)")
plt.show()

# %% [markdown]
# ## Reconstruction quality on the full domain
#
# We now ask the trained MLP to predict (Ux, Uy) at *every* CFD mesh point
# and compare against the ground truth.  This panel is the "money shot"
# of the notebook.  Look for:
#
# - **Smooth blobs near sensors**: local memorization.
# - **Flat, boring regions away from sensors**: the MLP has no reason to
#   predict anything specific there.
# - **Error panel**: the pattern of the error should roughly *mirror*
#   the real flow structure, since the MLP's prediction is near-constant
#   in the gaps.

# %%
pred = reconstruct(mlp['model'],
                   field['x_norm'], field['y_norm'], geom, device)
fig = plot_reconstruction_panel(
    field['x_phys'], field['y_phys'],
    field['Ux_norm'], field['Uy_norm'],
    pred[:, 0], pred[:, 1],
    suptitle=f"MLP (no physics) — K={K} sensors — val MSE {mlp['final_val_mse']:.2e}",
    sensor_xy=(field['x_phys'][split['sensor_idx']], field['y_phys'][split['sensor_idx']]),
)
plt.show()

# %% [markdown]
# ## Does the reconstruction obey physics?
#
# A real incompressible flow satisfies ∇·u = 0.  The MLP was never told
# this.  Let's measure the divergence of its predicted field on a dense
# grid — if the MLP is physically inconsistent, the divergence will be
# large.

# %%
import torch
from piml.physics import continuity_residual

def measure_divergence(model, geom, n_side: int = 80):
    xs = np.linspace(0, 1, n_side, dtype=np.float32)
    ys = np.linspace(0, 1, n_side, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    xt = torch.tensor(X.reshape(-1, 1), device=device, requires_grad=True)
    yt = torch.tensor(Y.reshape(-1, 1), device=device, requires_grad=True)
    r  = continuity_residual(model, xt, yt, geom).detach().cpu().numpy().squeeze()
    return X.flatten() * geom.L, Y.flatten() * geom.W, r

x_g, y_g, r_mlp = measure_divergence(mlp['model'], geom)
print(f"MLP divergence RMS = {np.sqrt(np.mean(r_mlp**2)):.3e}   "
      f"max|∇·u| = {np.abs(r_mlp).max():.3e}")

# %%
from piml.plotting import plot_residual_field
fig, ax = plt.subplots(figsize=(9, 3.5))
plot_residual_field(x_g, y_g, r_mlp, ax=ax,
                    title=f"MLP  ∇·u  (RMS {np.sqrt(np.mean(r_mlp**2)):.2e};  should be 0)")
plt.show()

# %% [markdown]
# ## Try this
#
# 1. Rerun with K = 500.  How close does the MLP get to the ground truth?
#    At high sensor counts the physics adds little — pure MLP is fine.
# 2. Rerun with K = 20.  The reconstruction should look nearly empty
#    away from the few sensor dots.  This is the regime where physics
#    earns its keep.
# 3. Note down the divergence RMS.  In notebook 03 we'll add a continuity
#    loss and expect this number to drop by 1–2 orders of magnitude.
