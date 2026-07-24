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
# # 04 — Sensor Sweep: How Does Physics Help vs. Sensor Count?
#
# **Goal:** quantify, with error bars, how the reconstruction error of the
# plain MLP and the continuity-PINN change as the number of sensors K
# ranges from very-sparse (10) to almost-dense (500).
#
# The README's expected-results table reports a single-seed point estimate
# at each K.  Single-seed numbers can be misleading — at K=50, the answer
# depends a lot on *which* 50 mesh points happened to be picked as sensors.
# Here we run each (K, model) pair across **5 different random sensor
# placements** so we can see the spread, not just the mean.
#
# By the end of this notebook you'll have:
#
# 1. A curve of held-out val MSE vs K for both models, with ±std bands.
# 2. A curve of the *relative* improvement (MLP − PINN)/MLP vs K, with
#    error bars — i.e., where does physics actually help?
# 3. A curve of the divergence RMS vs K — showing that the PINN's field
#    is physically consistent even when its MSE gain is small.

# %%
import os, sys, time

# Resolve the project root from THIS file's location so the script runs
# regardless of where it's launched from (Jupyter CWD, VS Code Run Python
# File from any directory, etc.).  In a Jupyter cell __file__ may be
# undefined; fall back to the current directory in that case.
_HERE         = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(".")
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt

from piml.config   import DataPaths, Hyperparams
from piml.data     import load_case, make_sensor_split
from piml.model    import get_device
from piml.train    import train_mlp, train_pinn, PhysicsWeights, PhysicsSampling
from piml.physics  import continuity_residual

# %% [markdown]
# ## Configuration
#
# - `K_VALUES`: the sensor counts to sweep.
# - `SEEDS`: 5 different random seeds — each selects a different set of K
#   mesh points as sensors.  Network initialization and physics-collocation
#   sampling are held constant (via `hp.seed=42`), so the only thing that
#   varies between seeds is *which observations the model sees*.
# - `EPOCHS`: 1500 is enough to converge on this problem (early-stopping
#   kicks in before 1500 in most cases).
#
# Runtime: 60 training runs total (6 K × 5 seeds × 2 models). On a laptop
# GPU expect ~20 min; on CPU ~45 min.  Flip `RECOMPUTE = False` after a
# successful first run to skip re-training and just re-plot.

# %%
K_VALUES = [10, 20, 50, 100, 200, 500]
SEEDS    = [42, 43, 44, 45, 46]
EPOCHS   = 1500

RESULTS_FILE = os.path.join(_PROJECT_ROOT, "data", "tiny", "sensor_sweep.npz")
RECOMPUTE    = True

# %%
paths  = DataPaths()
device = get_device()
geom, field = load_case(paths)
print(f"device: {device}")
print(f"case {geom.case_id}: n={geom.n}, {len(field['x_norm']):,} CFD points")

# %% [markdown]
# ## Helper: divergence RMS of a trained model
#
# Same measurement used in notebook 03 — sample on an 80×80 grid in
# normalized coords, compute ∂Ux/∂x + ∂Uy/∂y at every point via autograd.

# %%
def divergence_rms(model, n_side: int = 80) -> float:
    xs = np.linspace(0, 1, n_side, dtype=np.float32)
    ys = np.linspace(0, 1, n_side, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    xt = torch.tensor(X.reshape(-1, 1), device=device, requires_grad=True)
    yt = torch.tensor(Y.reshape(-1, 1), device=device, requires_grad=True)
    r  = continuity_residual(model, xt, yt, geom).detach().cpu().numpy().squeeze()
    return float(np.sqrt(np.mean(r ** 2)))

# %% [markdown]
# ## Run the sweep
#
# For each (K, seed) pair: train MLP, train PINN, record val MSE and
# divergence RMS for each.  We use `verbose=False` on the training calls
# and print our own per-run progress.

# %%
def run_sweep():
    results = {
        'K':        [],
        'seed':     [],
        'mlp_mse':  [],
        'pinn_mse': [],
        'mlp_div':  [],
        'pinn_div': [],
        'mlp_sec':  [],
        'pinn_sec': [],
    }

    hp       = Hyperparams(epochs=EPOCHS, print_every=10**9, val_every=50)
    weights  = PhysicsWeights(continuity=1.0, inlet=0.0, wall=0.0, baffle=0.0)
    sampling = PhysicsSampling(n_colloc=4096, resample_every=100)

    total = len(K_VALUES) * len(SEEDS)
    done  = 0
    t_start = time.time()

    for K in K_VALUES:
        for seed in SEEDS:
            split = make_sensor_split(field, n_sensors=K, seed=seed)

            mlp  = train_mlp (split, geom, hp, device, verbose=False)
            pinn = train_pinn(split, geom, hp, device,
                              weights=weights, sampling=sampling, verbose=False)

            results['K'].append(K)
            results['seed'].append(seed)
            results['mlp_mse'].append (mlp ['final_val_mse'])
            results['pinn_mse'].append(pinn['final_val_mse'])
            results['mlp_div'].append (divergence_rms(mlp ['model']))
            results['pinn_div'].append(divergence_rms(pinn['model']))
            results['mlp_sec'].append (mlp ['elapsed_sec'])
            results['pinn_sec'].append(pinn['elapsed_sec'])

            done += 1
            elapsed = time.time() - t_start
            eta     = elapsed / done * (total - done)
            print(f"  [{done:2d}/{total}]  K={K:4d}  seed={seed}  "
                  f"MLP {mlp['final_val_mse']:.3e}  PINN {pinn['final_val_mse']:.3e}  "
                  f"({100*(mlp['final_val_mse']-pinn['final_val_mse'])/mlp['final_val_mse']:+5.1f}%)  "
                  f"elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m")

    return {k: np.array(v) for k, v in results.items()}

# %%
if RECOMPUTE or not os.path.exists(RESULTS_FILE):
    results = run_sweep()
    np.savez(RESULTS_FILE, **results)
    print(f"\nSaved results to {RESULTS_FILE}")
else:
    print(f"Loading cached results from {RESULTS_FILE}")
    results = dict(np.load(RESULTS_FILE))

# %% [markdown]
# ## Aggregate: mean and std at each K
#
# Group by K, compute mean and standard deviation across the 5 seeds.

# %%
def aggregate(results, key):
    """Return (K_values, mean_array, std_array) for a result column."""
    Ks   = np.array(K_VALUES)
    mean = np.array([results[key][results['K'] == K].mean() for K in Ks])
    std  = np.array([results[key][results['K'] == K].std()  for K in Ks])
    return Ks, mean, std

K_arr, mlp_mean,  mlp_std  = aggregate(results, 'mlp_mse')
_,     pinn_mean, pinn_std = aggregate(results, 'pinn_mse')
_,     mdiv_mean, mdiv_std = aggregate(results, 'mlp_div')
_,     pdiv_mean, pdiv_std = aggregate(results, 'pinn_div')

# Relative improvement, computed per-seed then aggregated (more faithful
# than computing means then dividing)
rel_imp_per_run = (results['mlp_mse'] - results['pinn_mse']) / results['mlp_mse']
_, rel_mean, rel_std = aggregate({'K': results['K'], 'r': rel_imp_per_run}, 'r')

# %% [markdown]
# ## Print summary table
#
# Mean ± std across the 5 sensor placements at each K.

# %%
print(f"{'K':>5}  {'MLP MSE':>20}  {'PINN MSE':>20}  {'rel improvement':>20}")
print("-" * 72)
for i, K in enumerate(K_arr):
    print(f"{K:5d}  "
          f"{mlp_mean[i]:.3e} ± {mlp_std[i]:.1e}   "
          f"{pinn_mean[i]:.3e} ± {pinn_std[i]:.1e}   "
          f"{100*rel_mean[i]:+6.1f}% ± {100*rel_std[i]:.1f}%")

# %% [markdown]
# ## Plot 1 — Val MSE vs K
#
# Both axes log-scale.  The shaded band is mean ± 1 std across the 5
# sensor-placement seeds at each K.  Where the bands overlap, the two
# models are statistically indistinguishable at this seed count.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.fill_between(K_arr, mlp_mean  - mlp_std,  mlp_mean  + mlp_std,
                alpha=0.20, color='C0', label="MLP ±1σ")
ax.fill_between(K_arr, pinn_mean - pinn_std, pinn_mean + pinn_std,
                alpha=0.20, color='C1', label="PINN ±1σ")
ax.plot(K_arr, mlp_mean,  'o-', color='C0', lw=2, label="MLP  mean")
ax.plot(K_arr, pinn_mean, 's-', color='C1', lw=2, label="PINN mean")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of sensors K  (log)")
ax.set_ylabel("held-out val MSE  (log)")
ax.set_title("Reconstruction error vs. sensor count")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right")

# Plot 2 — relative improvement
ax = axes[1]
ax.axhline(0, color='gray', lw=0.8, ls=':')
ax.errorbar(K_arr, 100 * rel_mean, yerr=100 * rel_std,
            fmt='o-', color='C2', lw=2, capsize=4,
            label="(MLP − PINN) / MLP  mean ± 1σ")
ax.set_xscale("log")
ax.set_xlabel("number of sensors K  (log)")
ax.set_ylabel("relative improvement  (%)")
ax.set_title("Where does physics actually help?")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right")

fig.suptitle(f"Case {geom.case_id} (n={geom.n})   "
             f"{len(SEEDS)} sensor-placements per K   "
             f"{EPOCHS} epochs", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Plot 2 — Divergence RMS vs K
#
# The PINN's field should be approximately divergence-free at every K (since
# that's exactly the loss term).  The MLP's field has no such constraint and
# its divergence reflects whatever derivatives its tanh interpolation
# happens to produce — usually 1–2 orders of magnitude worse.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.fill_between(K_arr, mdiv_mean - mdiv_std, mdiv_mean + mdiv_std,
                alpha=0.20, color='C0')
ax.fill_between(K_arr, pdiv_mean - pdiv_std, pdiv_mean + pdiv_std,
                alpha=0.20, color='C1')
ax.plot(K_arr, mdiv_mean, 'o-', color='C0', lw=2, label="MLP")
ax.plot(K_arr, pdiv_mean, 's-', color='C1', lw=2, label="PINN")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of sensors K  (log)")
ax.set_ylabel("divergence RMS of predicted field  (log)")
ax.set_title("Physical consistency vs. sensor count")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## Plot 3 — Per-seed scatter at the sweet spot
#
# At K = 50, 5 seeds means 5 (MLP_MSE, PINN_MSE) pairs.  Plotted as scatter,
# we can see whether PINN beats MLP *consistently* or just on average — and
# how big the per-realization spread is compared to the average gap.

# %%
K_focus = 50
mask    = results['K'] == K_focus
mse_m   = results['mlp_mse'][mask]
mse_p   = results['pinn_mse'][mask]
sds     = results['seed'][mask]

fig, ax = plt.subplots(figsize=(5.5, 5.5))
lim_lo  = min(mse_m.min(), mse_p.min()) * 0.85
lim_hi  = max(mse_m.max(), mse_p.max()) * 1.15
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k:', lw=1, label="y = x  (no improvement)")
ax.scatter(mse_m, mse_p, c='C2', s=80, zorder=5)
for x, y, s in zip(mse_m, mse_p, sds):
    ax.annotate(f"seed {s}", (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
ax.set_xlabel("MLP val MSE")
ax.set_ylabel("PINN val MSE")
ax.set_title(f"K = {K_focus}: per-seed PINN vs MLP\n(below the dotted line = PINN wins)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3); ax.legend()
plt.show()

# %% [markdown]
# ## Interpretation
#
# Three things to look for in these plots:
#
# 1. **At very low K** (10–20): both bands sit at the field's intrinsic
#    variance — neither model can reconstruct anything because there's
#    not enough information.  The improvement plot crosses zero.
# 2. **At moderate K** (~50): PINN band sits *below* MLP band, with limited
#    overlap.  The relative-improvement bar is well above zero and its
#    error bar (hopefully) doesn't cross zero.  This is the regime where
#    the physics is buying real data efficiency.
# 3. **At high K** (200–500): both bands converge to nearly the same value
#    and the improvement bar collapses toward zero.  With abundant data,
#    the plain MLP's tanh interpolation already produces something
#    close to a divergence-free field on its own.
#
# The divergence-RMS plot tells a different story: even at high K where
# val MSE is similar, the PINN's field is dramatically more physically
# consistent.  That matters for downstream use of the reconstruction —
# e.g., if you want to compute streamlines or particle trajectories from
# the predicted field, the PINN's prediction is usable and the MLP's is
# not, regardless of which has lower MSE on the validation points.
#
# ## Try this
#
# 1. Increase `SEEDS` to 10 to see whether the K=50 improvement is truly
#    significant or whether the 5-seed error bars were under-counting.
# 2. Switch `paths.case_id = 267` (simpler n=1 flow).  The README claims
#    the sweet spot moves to K=20; does that hold up with error bars?
# 3. Drop EPOCHS to 500.  Does the PINN's relative advantage shrink (less
#    time for the physics term to do its work) or stay the same?
# 4. Run a smoothness-only baseline: replace the continuity loss with
#    `mean((dUx/dx)^2 + (dUx/dy)^2 + (dUy/dx)^2 + (dUy/dy)^2)` and compare
#    its sweep curve.  If the smoothness-only curve matches the PINN
#    curve, then continuity *specifically* isn't what's helping — generic
#    smoothness is.  If the PINN curve sits below the smoothness curve,
#    the physics is doing identifiable work beyond regularization.
