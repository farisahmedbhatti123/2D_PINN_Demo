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
# # 01 — Sparse-Data Setup
#
# **Goal:** set up the inpainting problem and see what "sparse" really means.
#
# We have one CFD simulation of 2D flow through a contact basin.  The full
# mesh has thousands of cell centers.  We pretend we only have K "sensor"
# observations at a small random subset of these points (K ≪ N), and the
# rest of the field is unknown.  Our job: reconstruct the full field.
#
# By the end of this notebook you'll have:
#
# 1. A loaded CFD case with its geometry and velocity field.
# 2. A visualization of the full velocity field (what we wish we knew).
# 3. A randomly chosen sensor subset (what we actually observe), overlaid
#    on the ground truth so you can see *where* the data is sparse.
# 4. A sense of how the structure of the flow (inlet, baffle, recirculation)
#    cannot possibly be inferred from sensors alone — which motivates the
#    physics additions in notebooks 02 and 03.

# %%
import os, sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt

from piml.config  import DataPaths
from piml.data    import load_case, make_sensor_split
from piml.plotting import plot_velocity_field, plot_sensors_on_field

# %% [markdown]
# ## Load the demo case
#
# The bundled dataset is a single OpenFOAM simulation (case 267 from a
# larger 1000-case study): L=33 m, W=21 m, one baffle, U_inlet ≈ 0.26 m/s.
# The flow enters at the bottom-left (y=0), wraps around the baffle, and
# exits at the top-right (y=W).

# %%
paths = DataPaths()     # defaults to bundled case 267
geom, field = load_case(paths)
print(geom)

# %% [markdown]
# ## The ground-truth velocity field
#
# This is the full CFD solution — what a direct numerical simulation gives
# you.  In the inpainting problem, we pretend we do *not* have access to
# most of these points.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
Ux_phys = field['Ux_norm'] * geom.U_inlet
Uy_phys = field['Uy_norm'] * geom.U_inlet
plot_velocity_field(field['x_phys'], field['y_phys'], Ux_phys, Uy_phys,
                    ax=axes[0], component="Ux",  title="CFD  Ux  (m/s)")
plot_velocity_field(field['x_phys'], field['y_phys'], Ux_phys, Uy_phys,
                    ax=axes[1], component="Uy",  title="CFD  Uy  (m/s)")
plot_velocity_field(field['x_phys'], field['y_phys'], Ux_phys, Uy_phys,
                    ax=axes[2], component="mag", title="CFD  |U|  (m/s)")
fig.suptitle(f"Ground truth — case {geom.case_id} ({len(field['x_norm']):,} CFD points)", fontsize=12)
fig.tight_layout()

# %% [markdown]
# ## Pick K sensors at random
#
# These K points are the *only* observations the network will see during
# training.  Every other mesh point is held out for evaluation.
#
# Try this: change `K` to 10, 100, 500.  The visual impression of
# "sparsity" changes dramatically.

# %%
K = 50               # <-- tweak me.  50 is the sweet spot on this n=4 case.
split = make_sensor_split(field, n_sensors=K, seed=42)
print(f"Split: {len(split['sensor_idx'])} sensors, {len(split['eval_idx']):,} held-out eval points")

# %% [markdown]
# ## Visualize sensor locations
#
# Red rings show where the sensors sit on top of the CFD magnitude.
# Large empty regions of the domain have *no observations* — and that's
# where the MLP in notebook 02 will fail and the PINN in notebook 03 will
# recover the field using physics.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
plot_sensors_on_field(field, split['sensor_idx'], geom, component="mag", ax=ax,
                      title=f"Case {geom.case_id}: CFD |U| backdrop with K={K} sensor locations")
plt.show()

# %% [markdown]
# ## What does the network actually see?
#
# Plotting only the K sensor observations — no backdrop — makes the
# challenge concrete.  Imagine trying to reconstruct the serpentine flow
# from *just these dots*.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sens = split['sensors']
axes[0].scatter(sens['x_phys'], sens['y_phys'],
                c=sens['Ux_norm'] * geom.U_inlet, s=60, cmap="viridis",
                edgecolor='k', linewidths=0.4)
axes[0].set_aspect("equal")
axes[0].set_xlim(0, geom.L); axes[0].set_ylim(0, geom.W)
axes[0].set_title(f"Sensor Ux observations only (K={K})")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
plt.colorbar(axes[0].collections[0], ax=axes[0], label="Ux (m/s)")

axes[1].scatter(sens['x_phys'], sens['y_phys'],
                c=sens['Uy_norm'] * geom.U_inlet, s=60, cmap="viridis",
                edgecolor='k', linewidths=0.4)
axes[1].set_aspect("equal")
axes[1].set_xlim(0, geom.L); axes[1].set_ylim(0, geom.W)
axes[1].set_title(f"Sensor Uy observations only (K={K})")
axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
plt.colorbar(axes[1].collections[0], ax=axes[1], label="Uy (m/s)")
fig.tight_layout()

# %% [markdown]
# ## Try this
#
# 1. Set `K = 20`.  Replot the sensor overlay.  Would you be able to guess
#    the shape of the flow from those 20 dots alone?
# 2. Set `K = 500`.  Now it starts looking like you *could* reconstruct
#    the field just by interpolating nearest sensors.  A plain MLP will
#    do well in this regime — the physics adds little.
# 3. Change `seed=42` to `seed=1` in `make_sensor_split`.  Different
#    random subsets sample the domain differently; the reconstruction
#    error in later notebooks depends (mildly) on *which* points are seen.
