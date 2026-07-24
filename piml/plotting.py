"""
Plotting helpers.

Two upgrades over scatter for CFD-like point clouds:
  - `tripcolor` with Gouraud shading for smooth field renderings
  - triangulation masking to suppress long thin triangles that would bridge
    baffle gaps

New in the inpainting edition:
  - `plot_sensors_on_field` — overlay sensor locations on the CFD truth
  - `plot_reconstruction_panel` — ground truth | reconstruction | error,
    with aspect-aware sizing and shared colorbars per row
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# ------------------------------------------------------------------
# Triangulation helper
# ------------------------------------------------------------------

def _triangulate(x, y, edge_cutoff_mult: float = 3.0):
    try:
        tri = mtri.Triangulation(x, y)
    except (RuntimeError, ValueError):
        return None
    pts = np.stack([x[tri.triangles], y[tri.triangles]], axis=-1)
    edges = np.linalg.norm(np.diff(pts[:, [0, 1, 2, 0], :], axis=1), axis=-1)
    max_edge = edges.max(axis=1)
    tri.set_mask(max_edge > edge_cutoff_mult * np.median(max_edge))
    return tri


def _field_plot(ax, x, y, v, *, cmap, vmin, vmax):
    tri = _triangulate(np.asarray(x), np.asarray(y))
    if tri is not None:
        return ax.tripcolor(tri, v, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    return ax.scatter(x, y, c=v, s=2, cmap=cmap, vmin=vmin, vmax=vmax)


# ------------------------------------------------------------------
# Single-field plots
# ------------------------------------------------------------------

def plot_velocity_field(
    x, y, Ux, Uy, *,
    title: str = "", ax: Optional[plt.Axes] = None,
    component: str = "mag", cmap: str = "viridis",
    vmin: Optional[float] = None, vmax: Optional[float] = None,
):
    """Smooth shaded velocity-field plot.  component: 'Ux' | 'Uy' | 'mag'."""
    if component == "Ux":
        c = np.asarray(Ux)
    elif component == "Uy":
        c = np.asarray(Uy)
    elif component == "mag":
        c = np.sqrt(np.asarray(Ux) ** 2 + np.asarray(Uy) ** 2)
    else:
        raise ValueError(f"unknown component {component!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    m = _field_plot(ax, x, y, c, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if title: ax.set_title(title)
    plt.colorbar(m, ax=ax, label=component, fraction=0.046)
    return ax


def plot_residual_field(
    x, y, residual, *, title: str = "continuity residual  ∇·u",
    ax: Optional[plt.Axes] = None, symmetric: bool = True,
):
    """Diverging-colormap plot of a scalar residual field."""
    x = np.asarray(x); y = np.asarray(y); r = np.asarray(residual).squeeze()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    if symmetric:
        vmax = float(np.nanmax(np.abs(r))) or 1e-12
        vmin = -vmax
    else:
        vmin = float(np.nanmin(r)); vmax = float(np.nanmax(r))
    m = _field_plot(ax, x, y, r, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(m, ax=ax, label="∇·u", fraction=0.046)
    return ax


# ------------------------------------------------------------------
# Sensor overlay
# ------------------------------------------------------------------

def plot_sensors_on_field(
    field: dict, sensor_idx, geom, *,
    component: str = "mag", ax: Optional[plt.Axes] = None,
    sensor_color: str = "red", sensor_size: int = 45,
    title: Optional[str] = None,
):
    """
    Overlay sensor locations (red dots) on the full CFD field.

    field:      dict with 'x_phys', 'y_phys', 'Ux_norm', 'Uy_norm'
    sensor_idx: indices into the field arrays that are sensor locations
    geom:       CaseGeometry (used to de-normalize Ux/Uy for a physical-units plot)
    """
    if ax is None:
        W_over_L = (field['y_phys'].max() - field['y_phys'].min()) / \
                   max(field['x_phys'].max() - field['x_phys'].min(), 1e-9)
        _, ax = plt.subplots(figsize=(10, 10 * W_over_L + 0.8))

    Ux = field['Ux_norm'] * geom.U_inlet
    Uy = field['Uy_norm'] * geom.U_inlet

    # Plot the CFD velocity magnitude as the backdrop
    plot_velocity_field(field['x_phys'], field['y_phys'], Ux, Uy,
                        ax=ax, component=component,
                        title=title or f"Case {geom.case_id}: CFD |U| with {len(sensor_idx)} sensors")

    # Overlay sensors
    ax.scatter(field['x_phys'][sensor_idx], field['y_phys'][sensor_idx],
               s=sensor_size, facecolor='none', edgecolor=sensor_color,
               linewidths=1.6, label=f"sensors (K={len(sensor_idx)})", zorder=5)
    ax.legend(loc='upper right', framealpha=0.9)
    return ax


# ------------------------------------------------------------------
# Reconstruction panel (truth | prediction | error)
# ------------------------------------------------------------------

def plot_reconstruction_panel(
    x, y, truth_Ux, truth_Uy, pred_Ux, pred_Uy, *,
    suptitle: str = "", row_height: float = 2.2,
    sensor_xy: Optional[tuple] = None,
):
    """
    3x3 panel: rows (Ux, Uy, |U|), cols (truth, reconstruction, error).

    If `sensor_xy = (x_sens, y_sens)` is provided, sensor locations are
    overlaid on every panel as small red rings — great for seeing where
    the network had data and where it had to extrapolate.
    """
    x = np.asarray(x); y = np.asarray(y)
    truth_Ux  = np.asarray(truth_Ux); truth_Uy = np.asarray(truth_Uy)
    pred_Ux   = np.asarray(pred_Ux);  pred_Uy  = np.asarray(pred_Uy)
    truth_mag = np.sqrt(truth_Ux ** 2 + truth_Uy ** 2)
    pred_mag  = np.sqrt(pred_Ux  ** 2 + pred_Uy  ** 2)

    ar = (x.max() - x.min()) / max(y.max() - y.min(), 1e-9)
    panel_w = row_height * ar
    fig_w = 3 * panel_w + 1.8
    fig_h = 3 * row_height + 0.6
    fig, axes = plt.subplots(3, 3, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.35})

    rows = [
        ("Ux",  truth_Ux,  pred_Ux),
        ("Uy",  truth_Uy,  pred_Uy),
        ("|U|", truth_mag, pred_mag),
    ]

    for i, (label, t, p) in enumerate(rows):
        vmin = float(min(t.min(), p.min()))
        vmax = float(max(t.max(), p.max()))
        err  = p - t
        eabs = float(np.max(np.abs(err))) or 1e-12

        m_t = _field_plot(axes[i, 0], x, y, t,   cmap="viridis", vmin=vmin, vmax=vmax)
        m_p = _field_plot(axes[i, 1], x, y, p,   cmap="viridis", vmin=vmin, vmax=vmax)
        m_e = _field_plot(axes[i, 2], x, y, err, cmap="coolwarm", vmin=-eabs, vmax=eabs)

        axes[i, 0].set_title(f"Truth  {label}")
        axes[i, 1].set_title(f"Reconstr  {label}")
        axes[i, 2].set_title(f"Err  {label}")

        for ax in axes[i]:
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if sensor_xy is not None:
                sx, sy = sensor_xy
                ax.scatter(sx, sy, s=12, facecolor='none',
                           edgecolor='red', linewidths=0.8, zorder=5)

        fig.colorbar(m_p, ax=axes[i, :2].ravel().tolist(), fraction=0.025, pad=0.01)
        fig.colorbar(m_e, ax=axes[i, 2],                   fraction=0.04,  pad=0.01)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.995)
    return fig


# ------------------------------------------------------------------
# Head-to-head MLP vs. PINN comparison (single row)
# ------------------------------------------------------------------

def plot_head_to_head(
    x, y, truth_Ux, truth_Uy, a_Ux, a_Uy, b_Ux, b_Uy, *,
    a_label: str = "MLP", b_label: str = "PINN",
    component: str = "mag",
    suptitle: str = "",
    row_height: float = 2.4,
    sensor_xy: Optional[tuple] = None,
):
    """
    One-row head-to-head: [ truth | A reconstr | B reconstr | A err | B err ]

    Component of the velocity to display: 'Ux' | 'Uy' | 'mag' (default).
    Useful as the "money shot" single figure comparing models at a glance.
    """
    x = np.asarray(x); y = np.asarray(y)
    if component == "Ux":
        t, a, b = truth_Ux, a_Ux, b_Ux
    elif component == "Uy":
        t, a, b = truth_Uy, a_Uy, b_Uy
    else:
        t = np.sqrt(np.asarray(truth_Ux)**2 + np.asarray(truth_Uy)**2)
        a = np.sqrt(np.asarray(a_Ux)**2 + np.asarray(a_Uy)**2)
        b = np.sqrt(np.asarray(b_Ux)**2 + np.asarray(b_Uy)**2)
    t, a, b = map(np.asarray, (t, a, b))
    err_a, err_b = a - t, b - t

    ar = (x.max() - x.min()) / max(y.max() - y.min(), 1e-9)
    panel_w = row_height * ar
    fig, axes = plt.subplots(1, 5, figsize=(5 * panel_w + 2.0, row_height + 1.2),
                             gridspec_kw={"wspace": 0.12})

    vmin = float(min(t.min(), a.min(), b.min()))
    vmax = float(max(t.max(), a.max(), b.max()))
    eabs = float(max(np.max(np.abs(err_a)), np.max(np.abs(err_b)))) or 1e-12

    panels = [
        (f"Truth  {component}",          t,     vmin,  vmax, "viridis"),
        (f"{a_label}  recon",            a,     vmin,  vmax, "viridis"),
        (f"{b_label}  recon",            b,     vmin,  vmax, "viridis"),
        (f"{a_label}  err",              err_a, -eabs, eabs, "coolwarm"),
        (f"{b_label}  err",              err_b, -eabs, eabs, "coolwarm"),
    ]
    mappables = []
    for ax, (ttl, v, vm0, vm1, cmap) in zip(axes, panels):
        m = _field_plot(ax, x, y, v, cmap=cmap, vmin=vm0, vmax=vm1)
        mappables.append(m)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=10)
        if sensor_xy is not None:
            sx, sy = sensor_xy
            ax.scatter(sx, sy, s=10, facecolor='none',
                       edgecolor='red', linewidths=0.7, zorder=5)

    fig.colorbar(mappables[2], ax=axes[:3].tolist(), fraction=0.02, pad=0.01)
    fig.colorbar(mappables[4], ax=axes[3:].tolist(), fraction=0.03, pad=0.01)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    return fig


# ------------------------------------------------------------------
# Loss curves
# ------------------------------------------------------------------

def plot_loss_curves(train_losses, val_losses, *, val_every: int = 50,
                     ax: Optional[plt.Axes] = None, title: str = "Training curves"):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    et = np.arange(1, len(train_losses) + 1)
    ev = np.arange(1, len(val_losses) + 1) * val_every
    ax.semilogy(et, train_losses, label="train", alpha=0.7)
    if len(val_losses) > 0:
        ax.semilogy(ev, val_losses, label="val", marker='o', ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.3); ax.legend()
    return ax


def plot_loss_components(history: dict, *, val_every: int = 50,
                         ax: Optional[plt.Axes] = None,
                         title: str = "Loss components (log)"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    styles = {
        'data':       dict(color='C0', lw=1.8),
        'continuity': dict(color='C1', lw=1.2),
        'inlet':      dict(color='C2', lw=1.0, ls='--'),
        'wall':       dict(color='C3', lw=1.0, ls='--'),
        'baffle':     dict(color='C4', lw=1.0, ls='--'),
        'total':      dict(color='k',  lw=1.8, alpha=0.6),
        'val':        dict(color='C5', lw=2.0, marker='o', ms=3),
    }
    for k, style in styles.items():
        v = np.asarray(history.get(k, []), dtype=float)
        if v.size == 0 or np.all(v == 0):
            continue
        epochs = np.arange(1, v.size + 1) * val_every
        safe = np.where(v > 0, v, np.nan)
        ax.semilogy(epochs, safe, label=k, **style)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=9, loc='best')
    return ax
