"""
Shared plumbing for the 2D PINN inpainting marimo notebook
(`2D_PINN_Inpainting_marimo.py`).

This is the "slim edition" companion, exactly analogous to `pinn_utils.py`
in the 1D module: everything that is *not* the PINN method itself —
data loading, normalization, and all matplotlib plotting / error helpers —
lives here so the notebook keeps only the network, the autograd residual,
the loss, and the training loops in front of the student.

Coordinates are normalized to [0, 1]^2 (x/L, y/W) and velocities to U_inlet
units, matching `notebooks/00_inpainting_complete.py`.  Nothing here imports
torch: these functions operate on numpy arrays returned by a trained model,
so the file stays light and the autograd stays visible in the notebook.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _read_csv(path: str) -> "pd.DataFrame":
    """
    Read a CSV robustly on OneDrive / cloud-synced folders.

    pandas' C parser opens files with low-level flags that some OneDrive
    "Files On-Demand" placeholders reject with
    `OSError: [Errno 22] Invalid argument`.  Reading the bytes through
    Python's own open() first forces OneDrive to hydrate (download) the file,
    then hands the fully-materialized bytes to pandas — which sidesteps the
    error.  (If you prefer, you can also right-click the `data` folder in
    Explorer and choose "Always keep on this device".)
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    return pd.read_csv(io.BytesIO(raw))


# --------------------------------------------------------------------------
# Paths.  Resolve the data directory from THIS file's location so the module
# works no matter where marimo is launched from.
# --------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "tiny")

DEFAULT_CASE_ID = 834   # n=4 baffles (default); 267 is the simpler n=1 case


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@dataclass
class Case:
    """One CFD case: geometry parameters + normalized mesh fields."""
    case_id: int
    L: float
    W: float
    n_baffles: int
    a_baffle: float
    U_inlet: float
    x_phys: np.ndarray
    y_phys: np.ndarray
    x_norm: np.ndarray
    y_norm: np.ndarray
    Ux_norm: np.ndarray
    Uy_norm: np.ndarray

    @property
    def N(self) -> int:
        return len(self.x_norm)


def load_case(case_id: int = DEFAULT_CASE_ID, data_dir: str | None = None) -> Case:
    """
    Load one CFD case and normalize it, exactly as notebook 00 does:

      - coordinates: x_norm = x/L, y_norm = y/W  -> both in [0, 1]
      - velocities:  Ux_norm = Ux/U_inlet, Uy_norm = Uy/U_inlet

    Normalization makes the network's inputs and outputs O(1), which keeps
    autograd through the continuity residual numerically stable.
    """
    data_dir = data_dir or DATA_DIR
    geo_csv  = os.path.join(data_dir, "basin_geometries_summary.csv")
    vel_csv  = os.path.join(data_dir, "results", f"case_{case_id:03d}_velocity.csv")

    geo_row = _read_csv(geo_csv).query(f"`Simulation Number` == {case_id}").iloc[0]
    L       = float(geo_row["L"])
    W       = float(geo_row["W"])
    n_baf   = int  (geo_row["n"])
    a_baf   = float(geo_row["a"])
    U_inlet = float(geo_row["U_inlet"])

    vel = _read_csv(vel_csv).dropna(subset=["Ux", "Uy"])
    return Case(
        case_id = case_id,
        L = L, W = W, n_baffles = n_baf, a_baffle = a_baf, U_inlet = U_inlet,
        x_phys  = vel["x"].values.astype(np.float32),
        y_phys  = vel["y"].values.astype(np.float32),
        x_norm  = (vel["x"].values  / L      ).astype(np.float32),
        y_norm  = (vel["y"].values  / W      ).astype(np.float32),
        Ux_norm = (vel["Ux"].values / U_inlet).astype(np.float32),
        Uy_norm = (vel["Uy"].values / U_inlet).astype(np.float32),
    )


# --------------------------------------------------------------------------
# Core plotting helper (unstructured CFD mesh)
# --------------------------------------------------------------------------
def plot_field(ax, x, y, c, *, cmap="viridis", vmin=None, vmax=None,
               title: str = "", edge_cutoff_mult: float = 3.0):
    """
    Smooth-shaded plot of a scalar field on unstructured 2D points.

    CFD outputs live on an unstructured mesh, so `tripcolor` with Gouraud
    shading beats scatter.  We mask triangles whose longest edge is far
    longer than typical — those are the ones that would "bridge" across
    baffles or domain holes and paint colour where there is no flow.
    """
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c)
    tri = mtri.Triangulation(x, y)
    pts      = np.stack([x[tri.triangles], y[tri.triangles]], axis=-1)
    edges    = np.linalg.norm(np.diff(pts[:, [0, 1, 2, 0], :], axis=1), axis=-1)
    max_edge = edges.max(axis=1)
    tri.set_mask(max_edge > edge_cutoff_mult * np.median(max_edge))
    m = ax.tripcolor(tri, c, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return m


def _scatter_sensors(ax, case: Case, sensor_idx, *, s=18, lw=0.8):
    ax.scatter(case.x_phys[sensor_idx], case.y_phys[sensor_idx],
               s=s, facecolor="none", edgecolor="red", linewidths=lw, zorder=5)


# --------------------------------------------------------------------------
# "Before" picture: ground truth + sensor locations  (updates live with K/seed)
# --------------------------------------------------------------------------
def plot_ground_truth(case: Case, sensor_idx):
    """3-panel truth: Ux, Uy (m/s), and |U| with the K sensor rings overlaid."""
    U = case.U_inlet
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_field(axes[0], case.x_phys, case.y_phys, case.Ux_norm * U, title="CFD truth  Ux (m/s)")
    plot_field(axes[1], case.x_phys, case.y_phys, case.Uy_norm * U, title="CFD truth  Uy (m/s)")
    mag = np.sqrt((case.Ux_norm * U) ** 2 + (case.Uy_norm * U) ** 2)
    m3 = plot_field(axes[2], case.x_phys, case.y_phys, mag,
                    title=f"CFD truth |U|  + K={len(sensor_idx)} sensors")
    _scatter_sensors(axes[2], case, sensor_idx, s=35, lw=1.4)
    fig.colorbar(m3, ax=axes[2], fraction=0.046)
    fig.suptitle(f"Case {case.case_id} ground truth "
                 f"({case.N:,} mesh points, K={len(sensor_idx)} sensors)", y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Single-model reconstruction: truth | recon | L2 vector error
# --------------------------------------------------------------------------
def plot_reconstruction(case: Case, pred_Ux, pred_Uy, sensor_idx,
                        model_name: str, val_mse: float):
    """
    3-panel comparison: truth |U| | reconstruction |U| | pointwise L2 vector
    error ||vec_pred - vec_truth||.

    The error panel uses the full vector error (always >= 0), which captures
    both magnitude and direction — unlike ||U|_pred - |U|_truth|, which can be
    small even when the predicted vector points the wrong way.
    """
    truth_mag = np.sqrt(case.Ux_norm ** 2 + case.Uy_norm ** 2)
    pred_mag  = np.sqrt(pred_Ux ** 2 + pred_Uy ** 2)
    err       = np.sqrt((pred_Ux - case.Ux_norm) ** 2 + (pred_Uy - case.Uy_norm) ** 2)

    vmin = float(min(truth_mag.min(), pred_mag.min()))
    vmax = float(max(truth_mag.max(), pred_mag.max()))
    emax = float(err.max()) or 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    plot_field(axes[0], case.x_phys, case.y_phys, truth_mag, vmin=vmin, vmax=vmax, title="truth |U|")
    mP = plot_field(axes[1], case.x_phys, case.y_phys, pred_mag, vmin=vmin, vmax=vmax,
                    title=f"{model_name} reconstruction")
    mE = plot_field(axes[2], case.x_phys, case.y_phys, err, cmap="magma", vmin=0, vmax=emax,
                    title="L2 vector error  ||vec_pred − vec_truth||")
    for ax in axes:
        _scatter_sensors(ax, case, sensor_idx)
    fig.colorbar(mP, ax=axes[:2].tolist(), fraction=0.025)
    fig.colorbar(mE, ax=axes[2], fraction=0.04)
    fig.suptitle(f"{model_name}  —  K={len(sensor_idx)}  —  held-out MSE = {val_mse:.2e}", y=1.02)
    return fig


# --------------------------------------------------------------------------
# Divergence fields (the physics diagnostic)
# --------------------------------------------------------------------------
def plot_divergence(x_g, y_g, r, title: str):
    """Single signed divergence field on a coolwarm scale centered at zero."""
    div_rms = float(np.sqrt(np.mean(r ** 2)))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    vmax = float(np.max(np.abs(r))) or 1e-12
    plot_field(ax, x_g, y_g, r, cmap="coolwarm", vmin=-vmax, vmax=vmax, title=title)
    fig.colorbar(ax.collections[0], ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig, div_rms


def plot_divergence_compare(x_g, y_g, r_mlp, r_pinn):
    """Side-by-side MLP vs PINN divergence on a shared symmetric scale."""
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    vmax = float(max(np.max(np.abs(r_mlp)), np.max(np.abs(r_pinn)))) or 1e-12
    plot_field(axes[0], x_g, y_g, r_mlp,  cmap="coolwarm", vmin=-vmax, vmax=vmax,
               title=f"MLP   ∇·u   RMS {rms(r_mlp):.2e}")
    plot_field(axes[1], x_g, y_g, r_pinn, cmap="coolwarm", vmin=-vmax, vmax=vmax,
               title=f"PINN  ∇·u   RMS {rms(r_pinn):.2e}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# The "money plot": 6-panel head-to-head
# --------------------------------------------------------------------------
def plot_headtohead(case: Case, sensor_idx,
                    mlp_Ux, mlp_Uy, pinn_Ux, pinn_Uy,
                    best_val_mlp: float, best_val_pinn: float):
    """
    Six panels: truth, MLP recon, PINN recon, MLP L2 error, PINN L2 error,
    and per-point improvement (MLP error − PINN error; blue = PINN wins).
    """
    truth_mag = np.sqrt(case.Ux_norm ** 2 + case.Uy_norm ** 2)
    mlp_mag   = np.sqrt(mlp_Ux ** 2 + mlp_Uy ** 2)
    pinn_mag  = np.sqrt(pinn_Ux ** 2 + pinn_Uy ** 2)

    err_mlp  = np.sqrt((mlp_Ux  - case.Ux_norm) ** 2 + (mlp_Uy  - case.Uy_norm) ** 2)
    err_pinn = np.sqrt((pinn_Ux - case.Ux_norm) ** 2 + (pinn_Uy - case.Uy_norm) ** 2)
    improv   = err_mlp - err_pinn        # > 0 means PINN error smaller here

    vmin = float(min(truth_mag.min(), mlp_mag.min(), pinn_mag.min()))
    vmax = float(max(truth_mag.max(), mlp_mag.max(), pinn_mag.max()))
    emax = float(max(err_mlp.max(), err_pinn.max())) or 1e-12
    iabs = float(np.max(np.abs(improv))) or 1e-12

    fig, axes = plt.subplots(1, 6, figsize=(24, 3.5))
    m_r = plot_field(axes[0], case.x_phys, case.y_phys, truth_mag, vmin=vmin, vmax=vmax, title="truth |U|")
    plot_field      (axes[1], case.x_phys, case.y_phys, mlp_mag,   vmin=vmin, vmax=vmax, title="MLP recon")
    plot_field      (axes[2], case.x_phys, case.y_phys, pinn_mag,  vmin=vmin, vmax=vmax, title="PINN recon")
    m_e = plot_field(axes[3], case.x_phys, case.y_phys, err_mlp,   cmap="magma", vmin=0, vmax=emax, title="MLP L2 error")
    plot_field      (axes[4], case.x_phys, case.y_phys, err_pinn,  cmap="magma", vmin=0, vmax=emax, title="PINN L2 error")
    m_i = plot_field(axes[5], case.x_phys, case.y_phys, improv,    cmap="RdBu", vmin=-iabs, vmax=iabs,
                     title="MLP err − PINN err  (blue: PINN wins)")
    for ax in axes:
        _scatter_sensors(ax, case, sensor_idx, s=14, lw=0.7)
    fig.colorbar(m_r, ax=axes[:3].tolist(),  fraction=0.02,  pad=0.01)
    fig.colorbar(m_e, ax=axes[3:5].tolist(), fraction=0.025, pad=0.01)
    fig.colorbar(m_i, ax=axes[5],            fraction=0.04,  pad=0.01)

    imp_pct = 100 * (best_val_mlp - best_val_pinn) / best_val_mlp
    win_frac = 100 * float(np.mean(improv > 0))
    fig.suptitle(f"K={len(sensor_idx)}   MLP MSE {best_val_mlp:.2e}   PINN MSE {best_val_pinn:.2e}   "
                 f"({imp_pct:+.0f}% held-out improvement;  PINN wins at {win_frac:.0f}% of points)", y=1.04)
    return fig, imp_pct, win_frac


# --------------------------------------------------------------------------
# Loss curves
# --------------------------------------------------------------------------
def plot_loss_curves(history: dict, val_every: int, warmup_frac: float = 0.2,
                     annotate: bool = True):
    """
    PINN per-term loss trajectories (data, continuity, total, held-out val),
    with three teaching annotations derived from the history itself:

      * the **warmup region** where the physics weight ramps 0 -> 1,
      * the **best held-out checkpoint** (the dip the saved model comes from),
      * the **generalization gap** — final held-out val vs. final data loss.

    Pass `warmup_frac` to match your training ramp (default 0.2), or
    `annotate=False` for the plain curves.
    """
    d   = np.asarray(history["data"]);  ph  = np.asarray(history["phys"])
    tot = np.asarray(history["total"]); val = np.asarray(history["val"])
    epochs_v = np.arange(1, len(val) + 1) * val_every
    total_epochs = len(val) * val_every

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.semilogy(epochs_v, d,   label="data loss",    color="C0", lw=1.8)
    ax.semilogy(epochs_v, ph,  label="continuity",   color="C1", lw=1.2)
    ax.semilogy(epochs_v, tot, label="total",        color="k",  lw=1.5, alpha=0.6)
    ax.semilogy(epochs_v, val, label="held-out val", color="C2", lw=2.0, marker="o", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss  (log)")
    ax.set_title("PINN training — per-term loss trajectories")
    ax.grid(True, which="both", alpha=0.3)

    if annotate and len(val) > 1:
        ymin = min(d.min(), ph.min(), val.min()); ymax = max(tot.max(), val.max())
        ax.set_ylim(ymin * 0.5, ymax * 2.2)
        ax.set_xlim(0, total_epochs * 1.02)

        # 1) warmup region
        warm_end = warmup_frac * total_epochs
        ax.axvspan(0, warm_end, color="0.5", alpha=0.12, zorder=0)
        ax.text(warm_end * 0.5, ymax * 1.4, "physics warmup\n(λ ramps 0→1)",
                ha="center", va="top", fontsize=8, color="0.3")

        # 2) best held-out checkpoint (the dip the saved model comes from)
        bi = int(np.argmin(val)); bx = epochs_v[bi]; bv = val[bi]
        ax.scatter([bx], [bv], s=130, facecolor="none", edgecolor="C2", lw=2.0, zorder=6)
        ax.annotate(f"best held-out MSE\n{bv:.2e} @ epoch {bx}\n(saved checkpoint)",
                    xy=(bx, bv), xytext=(18, 34), textcoords="offset points",
                    fontsize=8, color="C2",
                    arrowprops=dict(arrowstyle="->", color="C2", lw=1.2))

        # 3) generalization gap at the final epoch (val vs data)
        xg = epochs_v[-1]
        ax.annotate("", xy=(xg, val[-1]), xytext=(xg, d[-1]),
                    arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.4))
        ax.text(xg, np.sqrt(d[-1] * val[-1]), f"  generalization\n  gap ≈ {val[-1]/d[-1]:.0f}×",
                ha="right", va="center", fontsize=8, color="0.25")

    ax.legend(loc="center left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Deeper comparison helpers: baseline, error-vs-proximity, angular error
# --------------------------------------------------------------------------
def rbf_interpolate(sx_n, sy_n, vals, qx_n, qy_n, ridge: float = 1e-8):
    """
    Gaussian-RBF interpolation of `vals` from sensor points (sx_n, sy_n) to
    query points (qx_n, qy_n).  Pure numpy, no training and no physics — this
    is the "why not just interpolate the sensors?" baseline.  Coordinates are
    the normalized [0,1] ones so the kernel is isotropic; the shape parameter
    is set from the typical sensor spacing.
    """
    S = np.stack([np.asarray(sx_n), np.asarray(sy_n)], axis=1)
    Q = np.stack([np.asarray(qx_n), np.asarray(qy_n)], axis=1)
    d_ss  = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    shape = np.median(np.sort(d_ss, axis=1)[:, 1]) + 1e-9          # ~ nearest-neighbour spacing
    Phi   = np.exp(-(d_ss / shape) ** 2) + ridge * np.eye(len(S))
    d_qs  = np.sqrt(((Q[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    Psi   = np.exp(-(d_qs / shape) ** 2)
    return (Psi @ np.linalg.solve(Phi, np.asarray(vals))).astype(np.float32)


def heldout_mse(case: Case, eval_idx, pred_Ux, pred_Uy) -> float:
    """Held-out MSE in normalized units: mean sq err of Ux plus that of Uy."""
    e = eval_idx
    return float(np.mean((pred_Ux[e] - case.Ux_norm[e]) ** 2) +
                 np.mean((pred_Uy[e] - case.Uy_norm[e]) ** 2))


def plot_error_vs_distance(case: Case, sensor_idx, eval_idx, preds: dict, n_bins: int = 12):
    """
    Mean held-out L2 vector error binned by distance to the nearest sensor.

    `preds` maps a label -> (pred_Ux, pred_Uy, color), each a full-mesh field
    in normalized units.  The lines track together near sensors and fan apart
    in the interior — that gap is exactly where physics earns its keep.
    """
    sx, sy = case.x_phys[sensor_idx], case.y_phys[sensor_idx]
    ex, ey = case.x_phys[eval_idx],   case.y_phys[eval_idx]
    dist   = np.sqrt((ex[:, None] - sx[None, :]) ** 2 + (ey[:, None] - sy[None, :]) ** 2).min(axis=1)

    edges   = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ibin    = np.clip(np.digitize(dist, edges[1:-1]), 0, len(centers) - 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, (pUx, pUy, color) in preds.items():
        err  = np.sqrt((pUx[eval_idx] - case.Ux_norm[eval_idx]) ** 2 +
                       (pUy[eval_idx] - case.Uy_norm[eval_idx]) ** 2)
        mean = np.array([err[ibin == b].mean() if np.any(ibin == b) else np.nan
                         for b in range(len(centers))])
        sem  = np.array([err[ibin == b].std() / max(np.sqrt(np.sum(ibin == b)), 1)
                         if np.any(ibin == b) else np.nan for b in range(len(centers))])
        ax.plot(centers, mean, marker="o", ms=4, color=color, label=name)
        ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.15)
    ax.set_xlabel("distance to nearest sensor  (m)")
    ax.set_ylabel("mean L2 vector error")
    ax.set_title("Error vs. sensor proximity — physics fills the gaps")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_angular_error_hist(case: Case, mlp_Ux, mlp_Uy, pinn_Ux, pinn_Uy, bins: int = 40):
    """
    Magnitude-weighted distribution of angular (direction) error for MLP vs PINN.

    Each mesh point contributes its velocity angle error, weighted by the true
    local speed -- so near-stagnant points (where the angle is undefined) carry
    almost no weight and cannot distort the picture, without any hard masking.
    A leftward shift of the PINN curve means it points in the right direction
    more often; the dashed lines mark the magnitude-weighted means.
    """
    def _ang(pUx, pUy):
        dot   = pUx * case.Ux_norm + pUy * case.Uy_norm
        npred = np.sqrt(pUx ** 2 + pUy ** 2)
        ntrue = np.sqrt(case.Ux_norm ** 2 + case.Uy_norm ** 2)
        cos   = np.clip(dot / (npred * ntrue + 1e-8), -1.0, 1.0)
        return np.degrees(np.arccos(cos)), ntrue

    a_mlp, w = _ang(mlp_Ux, mlp_Uy)
    a_pinn, _ = _ang(pinn_Ux, pinn_Uy)
    wmean = lambda a: float(np.sum(w * a) / np.sum(w))
    m_mlp, m_pinn = wmean(a_mlp), wmean(a_pinn)
    edges = np.linspace(0, 180, bins + 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(a_mlp,  bins=edges, weights=w, density=True, color="C3", alpha=0.40)
    ax.hist(a_pinn, bins=edges, weights=w, density=True, color="C0", alpha=0.40)
    ax.hist(a_mlp,  bins=edges, weights=w, density=True, histtype="step", color="C3", lw=1.8,
            label=f"MLP   (wtd mean {m_mlp:.1f}\u00b0)")
    ax.hist(a_pinn, bins=edges, weights=w, density=True, histtype="step", color="C0", lw=1.8,
            label=f"PINN  (wtd mean {m_pinn:.1f}\u00b0)")
    ax.axvline(m_mlp,  color="C3", ls="--", lw=1.5)
    ax.axvline(m_pinn, color="C0", ls="--", lw=1.5)
    ax.set_xlabel("angular error  (degrees)")
    ax.set_ylabel("magnitude-weighted density")
    ax.set_title("Direction-error distribution — a leftward shift means the PINN points better")
    ax.set_xlim(0, 180); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# RANS k-epsilon momentum extension: supply the CFD eddy viscosity + pressure
# --------------------------------------------------------------------------
def load_eddy_viscosity(case_id: int = DEFAULT_CASE_ID, data_dir: str | None = None):
    """
    Load the CFD turbulent (eddy) viscosity `nut` and kinematic pressure `p`
    for a case, on the SAME normalized mesh as `load_case`, from
    `results/case_XXX_full.csv`.

    This is how the k-epsilon closure enters the momentum PINN: rather than
    solving the k and epsilon transport equations, we *supply* the eddy
    viscosity nu_t(x, y) that the CFD already computed, and use the effective
    viscosity nu_eff = nu_molecular + nu_t in the RANS momentum residual.
    Returns a dict of float32 arrays aligned with `load_case(case_id)`.
    """
    data_dir = data_dir or DATA_DIR
    geo = _read_csv(os.path.join(data_dir, "basin_geometries_summary.csv"))
    geo = geo.query(f"`Simulation Number` == {case_id}").iloc[0]
    L, W = float(geo["L"]), float(geo["W"])
    full = _read_csv(os.path.join(data_dir, "results", f"case_{case_id:03d}_full.csv"))
    full = full.dropna(subset=["Ux", "Uy", "nut", "p"])
    return {
        "x_norm": (full["x"].values / L).astype(np.float32),
        "y_norm": (full["y"].values / W).astype(np.float32),
        "nut":    full["nut"].values.astype(np.float32),   # eddy viscosity  (m^2/s)
        "p_kin":  full["p"].values.astype(np.float32),     # kinematic pressure p/rho (m^2/s^2)
    }


def nearest_interp(src_x, src_y, src_vals, qx, qy, chunk: int = 1024):
    """Nearest-neighbour interpolation of `src_vals` onto query points (qx, qy).
    Pure numpy, chunked to stay memory-light — used to supply nu_t at the
    random collocation points from the CFD mesh."""
    src = np.stack([np.asarray(src_x), np.asarray(src_y)], axis=1)
    q   = np.stack([np.asarray(qx),    np.asarray(qy)],    axis=1)
    vals = np.asarray(src_vals)
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), chunk):
        d = ((q[i:i + chunk, None, :] - src[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = vals[d.argmin(1)]
    return out


def plot_eddy_viscosity(case: Case, nut, nu_mol: float = 1e-6):
    """Show the supplied turbulent viscosity field (log scale) vs the molecular
    value, to make the point that nu_t dominates."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    m = plot_field(ax, case.x_phys, case.y_phys, np.log10(nut / nu_mol),
                   cmap="magma", title="turbulent viscosity  log10(nu_t / nu_molecular)")
    fig.colorbar(m, ax=ax, fraction=0.04, pad=0.01)
    fig.tight_layout()
    return fig


def plot_pressure_compare(case: Case, p_pred_kin, p_cfd_kin):
    """
    Emergent PINN pressure vs the CFD pressure, both mean-removed.

    Pressure is defined only up to an additive constant, and the network learns
    a *modified* pressure P* = p/rho + (2/3)k (the Boussinesq normal-stress term
    folded in); (2/3)k is tiny here, so after removing the mean the two should
    match in structure.  The striking part: the PINN was never shown pressure.
    """
    pp = np.asarray(p_pred_kin) - float(np.mean(p_pred_kin))
    pc = np.asarray(p_cfd_kin)  - float(np.mean(p_cfd_kin))
    vmax = float(max(np.percentile(np.abs(pp), 99), np.percentile(np.abs(pc), 99))) or 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    plot_field(axes[0], case.x_phys, case.y_phys, pc, cmap="coolwarm", vmin=-vmax, vmax=vmax,
               title="CFD pressure  (mean-removed)")
    mB = plot_field(axes[1], case.x_phys, case.y_phys, pp, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                    title="PINN emergent pressure  (never measured)")
    fig.colorbar(mB, ax=axes, fraction=0.02, pad=0.01)
    return fig


def plot_scalar_field(case: Case, values, title: str, cmap: str = "magma", symmetric: bool = False):
    """Single scalar field on the CFD mesh (e.g. a momentum-residual magnitude)."""
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if symmetric:
        vmax = float(np.percentile(np.abs(values), 99)) or 1e-12
        kw = dict(vmin=-vmax, vmax=vmax)
    else:
        kw = dict(vmin=0.0, vmax=float(np.percentile(values, 99)) or 1e-12)
    m = plot_field(ax, case.x_phys, case.y_phys, values, cmap=cmap, title=title, **kw)
    fig.colorbar(m, ax=ax, fraction=0.04, pad=0.01)
    fig.tight_layout()
    return fig


def plot_loss_curves_rans(history: dict, val_every: int, warmup_frac: float = 0.2, annotate: bool = True):
    """
    Per-term loss trajectories for the RANS-momentum PINN: data, continuity,
    momentum, and held-out val.  Same annotations as `plot_loss_curves`
    (warmup band, best-checkpoint dip, generalization gap = final held-out val
    vs. final data loss) so you can compare the gap against the continuity-only
    run and see whether the extra physics helped generalization.
    """
    d   = np.asarray(history["data"]); c = np.asarray(history["cont"])
    mm  = np.asarray(history["mom"]);  val = np.asarray(history["val"])
    ep = np.arange(1, len(val) + 1) * val_every
    total_epochs = len(val) * val_every

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.semilogy(ep, d,   label="data loss",       color="C0", lw=1.8)
    ax.semilogy(ep, c,   label="continuity",      color="C1", lw=1.2)
    ax.semilogy(ep, mm,  label="momentum (RANS)", color="C4", lw=1.2)
    ax.semilogy(ep, val, label="held-out val",    color="C2", lw=2.0, marker="o", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss  (log)")
    ax.set_title("RANS-momentum PINN — per-term loss trajectories")
    ax.grid(True, which="both", alpha=0.3)

    if annotate and len(val) > 1:
        ymin = min(d.min(), c.min(), mm.min(), val.min()); ymax = max(d.max(), val.max())
        ax.set_ylim(ymin * 0.5, ymax * 2.2); ax.set_xlim(0, total_epochs * 1.02)
        warm_end = warmup_frac * total_epochs
        ax.axvspan(0, warm_end, color="0.5", alpha=0.12, zorder=0)
        ax.text(warm_end * 0.5, ymax * 1.4, "physics warmup\n(ramps 0→1)",
                ha="center", va="top", fontsize=8, color="0.3")
        bi = int(np.argmin(val)); bx = ep[bi]; bv = val[bi]
        ax.scatter([bx], [bv], s=130, facecolor="none", edgecolor="C2", lw=2.0, zorder=6)
        ax.annotate(f"best held-out MSE\n{bv:.2e} @ epoch {bx}", xy=(bx, bv),
                    xytext=(18, 34), textcoords="offset points", fontsize=8, color="C2",
                    arrowprops=dict(arrowstyle="->", color="C2", lw=1.2))
        xg = ep[-1]
        ax.annotate("", xy=(xg, val[-1]), xytext=(xg, d[-1]),
                    arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.4))
        ax.text(xg, np.sqrt(d[-1] * val[-1]), f"  gen. gap ≈ {val[-1]/d[-1]:.0f}×",
                ha="right", va="center", fontsize=8, color="0.25")

    ax.legend(loc="center left"); fig.tight_layout()
    return fig
