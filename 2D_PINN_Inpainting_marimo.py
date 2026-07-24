import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2D PINN Inpainting — A Guided Tutorial (marimo, slim edition)
    ### Reconstructing a CFD velocity field from a handful of sensors

    This is the **slim** marimo version of `notebooks/00_inpainting_complete.py`. All the
    plotting and data-loading plumbing lives in the companion file **`pinn_utils_2d.py`**
    (same folder). Only the code that *is* the PINN method — the network, the autograd
    continuity residual, the loss, and the two training loops — stays visible here.

    #### The problem

    We have a steady CFD simulation of flow through a **baffled contact basin** (water winding
    around a set of baffles). Pretend we only measured the velocity at **K scattered "sensor"
    points**, far fewer than the thousands of mesh cells. Can we reconstruct the whole field?

    1. **Baseline — a plain MLP.** Fit the K sensors by regression. It nails the sensors and
       produces garbage in the gaps.
    2. **The PINN.** Same network, same sensors, but we *also* penalize violations of the
       **incompressible continuity equation** $\nabla\!\cdot\mathbf{u}=0$ at thousands of
       interior *collocation* points. That one physical law couples $U_x$ and $U_y$ everywhere,
       so a measurement in one place constrains the field elsewhere — and the gaps fill in.

    This is the classic Raissi-style PINN *inpainting* setup, applied to a 2D turbulent flow.

    ---

    > **marimo is reactive:** move a slider and every dependent cell re-runs — no "Run All".
    > Because training is slow, each model sits behind a **▶ Train** button, so the notebook
    > opens instantly. Cheap things (picking sensors, the "before" picture) update live as you
    > drag sliders.
    >
    > **Two files travel together:** this notebook does `from pinn_utils_2d import ...`, so keep
    > `pinn_utils_2d.py` in the same folder.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A quick glossary

    | Symbol / term | Plain meaning |
    |---|---|
    | $(x, y)$ | position in the basin |
    | $\mathbf{u}=(U_x, U_y)$ | the **velocity** we want to reconstruct everywhere |
    | **sensor** | one of the $K$ mesh points whose velocity we pretend to have measured |
    | **held-out points** | every *other* mesh point — the network never sees these; we grade on them |
    | **collocation points** | scattered interior $(x,y)$ where we *check the physics*, not the data |
    | **continuity** | $\partial U_x/\partial x + \partial U_y/\partial y = 0$ — mass conservation for incompressible flow |
    | **residual** | how badly a guess violates the equation (zero = perfect) |
    | **loss** | one number measuring "how wrong we are"; training pushes it toward 0 |
    | **autograd** | PyTorch's automatic differentiation — exact derivatives of the network |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 0 — Setup

    We load the libraries and pull the shared plumbing from **`pinn_utils_2d.py`**: the CFD
    data loader and every plotting helper. We set the matplotlib backend to `Agg` (static
    images) **before** importing the helpers, so figures render cleanly inside marimo.
    """)
    return


@app.cell
def _():
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib

    matplotlib.use("Agg")  # static images for marimo (set before importing pinn_utils_2d)

    from pinn_utils_2d import (
        load_case,
        plot_field,                 # core mesh plotter
        plot_ground_truth,          # "before" picture
        plot_reconstruction,        # single-model recon
        plot_divergence,            # single divergence field
        plot_divergence_compare,    # MLP vs PINN divergence
        plot_headtohead,            # the 6-panel money plot
        plot_loss_curves,           # PINN loss trajectories
        plot_loss_curves_rans,      # RANS-momentum loss trajectories
        rbf_interpolate,            # interpolation baseline (no training/physics)
        heldout_mse,                # held-out MSE helper
        plot_error_vs_distance,     # error vs. sensor proximity
        plot_angular_error_hist,    # direction-error distribution (MLP vs PINN)
        load_eddy_viscosity,        # CFD nu_t + pressure (k-epsilon momentum extension)
        nearest_interp,             # supply nu_t at collocation points
        plot_eddy_viscosity,        # show the supplied nu_t field
        plot_scalar_field,          # generic mesh scalar (momentum residual)
        plot_pressure_compare,      # emergent PINN pressure vs CFD
    )

    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return (
        DEVICE,
        F,
        heldout_mse,
        load_case,
        load_eddy_viscosity,
        nearest_interp,
        nn,
        np,
        plot_angular_error_hist,
        plot_divergence,
        plot_divergence_compare,
        plot_eddy_viscosity,
        plot_error_vs_distance,
        plot_field,
        plot_ground_truth,
        plot_headtohead,
        plot_loss_curves,
        plot_loss_curves_rans,
        plot_pressure_compare,
        plot_reconstruction,
        plot_scalar_field,
        rbf_interpolate,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 — Load the CFD case

    One CSV per case: each row is a mesh point with its $(x, y, U_x, U_y)$. We **normalize**
    coordinates to $[0,1]^2$ (via $x/L$, $y/W$) and velocities by the inlet speed, so the
    network's inputs and outputs are all $O(1)$ — which keeps the autograd through the
    continuity residual numerically stable.

    `CASE_ID = 834` is the 4-baffle case (the default). `267` is a simpler 1-baffle geometry.
    """)
    return


@app.cell
def _(load_case):
    CASE_ID = 834                       # 834 = n=4 (default);  267 = n=1 (simpler)
    case = load_case(CASE_ID)
    return (case,)


@app.cell
def _(case, mo):
    mo.md(f"""
    **Running case {case.case_id}:** &nbsp; $L$={case.L:.0f} m &nbsp;·&nbsp; $W$={case.W:.0f} m
    &nbsp;·&nbsp; **{case.n_baffles} baffles** &nbsp;·&nbsp; $U_\\text{{inlet}}$={case.U_inlet:.3f} m/s
    &nbsp;·&nbsp; **{case.N:,} mesh points** loaded.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — Pick the sensors (live)

    Choose how many sensors $K$ we get, and which random draw (`seed`) places them. Everything
    else — the whole basin — is **held out**: the network never sees it during training, and
    reconstruction error on those held-out points is the only score that matters.

    Drag the sliders and the picture below redraws instantly. Notice how little the network has
    to work with in the middle of the basin.
    """)
    return


@app.cell
def _(mo):
    K    = mo.ui.slider(10, 500, value=50, step=10, label="K  (number of sensors)")
    seed = mo.ui.slider(40, 60, value=42, step=1, label="seed  (which points get sensed)")
    mo.hstack([K, seed], justify="start", gap=2)
    return K, seed


@app.cell
def _(K, case, np, seed):
    # Split the mesh into K sensors + everything-else (held out).  This stays visible: it *is*
    # the inpainting setup.  Only `sensor_idx` / `eval_idx` change when you move the sliders.
    _rng       = np.random.default_rng(int(seed.value))
    sensor_idx = _rng.choice(case.N, size=int(K.value), replace=False)
    eval_idx   = np.setdiff1d(np.arange(case.N), sensor_idx)
    return eval_idx, sensor_idx


@app.cell
def _(case, mo, plot_ground_truth, sensor_idx):
    mo.as_html(plot_ground_truth(case, sensor_idx))  # helper from pinn_utils_2d — returns a Figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — The network and the continuity residual

    The pieces that *are* the PINN method stay visible below.

    - **The network** is a plain MLP mapping $(x, y) \to (U_x, U_y)$: two inputs, `tanh`
      hidden layers, two outputs.
    - **Why `tanh` and not `ReLU`?** The physics loss needs the network's *derivative*, and that
      derivative must itself be differentiable (autograd differentiates the output w.r.t. the
      inputs). `ReLU`'s second derivative is zero almost everywhere — useless for physics.
      `tanh` is smooth at every order.
    - **The continuity residual.** The network outputs live in normalized units, so the chain
      rule puts the $1/L$ and $1/W$ factors back: $\nabla \cdot \mathbf{u} = \tfrac{1}{L}\,\partial U_x/\partial x_\text{norm} + \tfrac{1}{W}\,\partial U_y/\partial y_\text{norm}$
      `create_graph=True` is what lets gradients later flow back through this derivative to the
      weights.
    """)
    return


@app.cell
def _(case, nn, torch):
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
            return self.net(torch.cat([x, y], dim=1))

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    def continuity_residual(model, x, y):
        """Pointwise divergence of u.  x, y must have requires_grad=True."""
        vel = model(x, y)
        Ux, Uy = vel[:, 0:1], vel[:, 1:2]
        dUx_dx = torch.autograd.grad(Ux, x, torch.ones_like(Ux),
                                     create_graph=True, retain_graph=True)[0]
        dUy_dy = torch.autograd.grad(Uy, y, torch.ones_like(Uy),
                                     create_graph=True, retain_graph=True)[0]
        return dUx_dx / case.L + dUy_dy / case.W

    def continuity_loss(model, x, y):
        r = continuity_residual(model, x, y)
        return torch.mean(r ** 2)

    return MLP, continuity_loss, continuity_residual


@app.cell
def _(DEVICE, case, eval_idx, sensor_idx, torch):
    # Pre-build the sensor / held-out tensors once (rebuilt only when the sliders change).
    def _col(arr):
        return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    sens_x  = _col(case.x_norm [sensor_idx]); sens_y  = _col(case.y_norm [sensor_idx])
    sens_Ux = _col(case.Ux_norm[sensor_idx]); sens_Uy = _col(case.Uy_norm[sensor_idx])
    eval_x  = _col(case.x_norm [eval_idx]);   eval_y  = _col(case.y_norm [eval_idx])
    eval_Ux = _col(case.Ux_norm[eval_idx]);   eval_Uy = _col(case.Uy_norm[eval_idx])
    return eval_Ux, eval_Uy, eval_x, eval_y, sens_Ux, sens_Uy, sens_x, sens_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — Train the plain MLP (data only, no physics)

    Standard supervised regression: the only loss is the mismatch between the network and the
    true velocity at the $K$ sensor locations. Every `VAL_EVERY` epochs we score the held-out
    points and keep the best checkpoint. Set the epoch budget and click **Train**.

    The reconstruction panels below grade each model with the **pointwise L2 vector error**
    $\;e(\mathbf{x}) = \lVert \mathbf{u}_\text{pred}(\mathbf{x})-\mathbf{u}_\text{true}(\mathbf{x})\rVert_2
    = \sqrt{\left(U_x^\text{pred}-U_x^\text{true}\right)^2 + \left(U_y^\text{pred}-U_y^\text{true}\right)^2}$.
    It penalizes a wrong *direction* as well as a wrong *speed* — unlike
    $\bigl|\,\lVert\mathbf{u}_\text{pred}\rVert-\lVert\mathbf{u}_\text{true}\rVert\,\bigr|$, which can look
    small even when the predicted vector points the wrong way.
    """)
    return


@app.cell
def _(mo):
    epochs_mlp = mo.ui.slider(1000, 8000, value=5000, step=500, label="MLP training epochs")
    run_mlp    = mo.ui.run_button(label="▶ Train MLP")
    mo.hstack([epochs_mlp, run_mlp], justify="start", gap=2)
    return epochs_mlp, run_mlp


@app.cell
def _(
    DEVICE,
    F,
    MLP,
    epochs_mlp,
    eval_Ux,
    eval_Uy,
    eval_x,
    eval_y,
    mo,
    run_mlp,
    sens_Ux,
    sens_Uy,
    sens_x,
    sens_y,
    time,
    torch,
):
    mo.stop(not run_mlp.value, mo.md("▶ *Click **Train MLP** above to start.*"))

    VAL_EVERY = 50
    torch.manual_seed(42)
    model_mlp = MLP(width=64, depth=4).to(DEVICE)
    _opt = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)

    _n = int(epochs_mlp.value)
    best_val_mlp, _best_state = float("inf"), None
    _t0 = time.time()
    with mo.status.progress_bar(total=_n, title="Training MLP (Adam)") as _bar:
        for _ep in range(_n):
            model_mlp.train(); _opt.zero_grad()
            _pred = model_mlp(sens_x, sens_y)
            _loss = F.mse_loss(_pred[:, 0:1], sens_Ux) + F.mse_loss(_pred[:, 1:2], sens_Uy)
            _loss.backward(); _opt.step()

            if (_ep + 1) % VAL_EVERY == 0:
                model_mlp.eval()
                with torch.no_grad():
                    _p  = model_mlp(eval_x, eval_y)
                    _vl = (F.mse_loss(_p[:, 0:1], eval_Ux) +
                           F.mse_loss(_p[:, 1:2], eval_Uy)).item()
                if _vl < best_val_mlp:
                    best_val_mlp = _vl
                    _best_state  = {k: v.detach().clone() for k, v in model_mlp.state_dict().items()}
                _bar.update(increment=VAL_EVERY)

    model_mlp.load_state_dict(_best_state)
    mo.md(f"**MLP done** in {time.time()-_t0:.1f}s &nbsp;·&nbsp; "
          f"best held-out MSE = **{best_val_mlp:.3e}** &nbsp;·&nbsp; "
          f"{model_mlp.num_params():,} params on {sens_x.shape[0]} sensors")
    return VAL_EVERY, best_val_mlp, model_mlp


@app.cell
def _(DEVICE, case, continuity_residual, np, torch):
    # A reusable diagnostic: sample divergence on an 80x80 grid using the SAME autograd machinery
    # we use as the physics loss.  Returns physical-space grid coords + the residual field.
    def divergence_grid(model, n_side: int = 80):
        _xs = np.linspace(0, 1, n_side, dtype=np.float32)
        _ys = np.linspace(0, 1, n_side, dtype=np.float32)
        _X, _Y = np.meshgrid(_xs, _ys)
        _xt = torch.tensor(_X.reshape(-1, 1), device=DEVICE, requires_grad=True)
        _yt = torch.tensor(_Y.reshape(-1, 1), device=DEVICE, requires_grad=True)
        _r  = continuity_residual(model, _xt, _yt).detach().cpu().numpy().squeeze()
        return _X.flatten() * case.L, _Y.flatten() * case.W, _r

    def predict_field(model, x_n, y_n):
        """Run the model on all mesh points -> numpy (Ux, Uy)."""
        model.eval()
        with torch.no_grad():
            _p = model(torch.as_tensor(x_n, device=DEVICE).unsqueeze(1),
                       torch.as_tensor(y_n, device=DEVICE).unsqueeze(1)).cpu().numpy()
        return _p[:, 0], _p[:, 1]

    return divergence_grid, predict_field


@app.cell
def _(
    best_val_mlp,
    case,
    divergence_grid,
    mo,
    model_mlp,
    plot_divergence,
    plot_reconstruction,
    predict_field,
    sensor_idx,
):
    # MLP reconstruction + how unphysical its field is (divergence should be 0, but won't be).
    _mlp_Ux, _mlp_Uy = predict_field(model_mlp, case.x_norm, case.y_norm)
    _x_g, _y_g, _r_mlp = divergence_grid(model_mlp)
    _div_rms = float((_r_mlp ** 2).mean() ** 0.5)
    _fig_div, _ = plot_divergence(
        _x_g, _y_g, _r_mlp, title=f"MLP div(u)   (RMS {_div_rms:.2e};  should be 0)"
    )
    mo.vstack([
        mo.as_html(plot_reconstruction(case, _mlp_Ux, _mlp_Uy, sensor_idx, "MLP (no physics)", best_val_mlp)),
        mo.md(f"**MLP divergence RMS = {_div_rms:.3e}** — the plain fit is not a physical flow. "
              f"Bright error patches sit far from any sensor."),
        mo.as_html(_fig_div),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 — Train the same network *with* continuity

    Identical network, identical sensor data. The only additions:

    1. **Collocation points** — thousands of random interior $(x, y)$ where we ask the network
       to satisfy $\nabla\!\cdot\mathbf{u}=0$. These are *not* observations. (Note: for
       simplicity they're drawn uniformly over the whole $[0,1]^2$ bounding box — including
       inside the solid baffles, where there is no flow. Both the collocation loss and the
       divergence diagnostic below do this, so the MLP-vs-PINN comparison stays fair; masking
       the baffle regions is a natural refinement.)
    2. The loss becomes $\;L_\text{data} + \lambda\cdot\text{ramp}(ep)\cdot L_\text{continuity}$.
    3. A **warmup ramp** eases the physics weight in over the first fraction of training, so the
       data term can anchor the prediction before the physics starts pulling.

    $\lambda$ sets how hard the physics pulls. Try the default (1.0) first, then sweep it.
    """)
    return


@app.cell
def _(mo):
    lam         = mo.ui.slider(0.0, 5.0, value=1.0, step=0.25, label="λ  (continuity weight)")
    epochs_pinn = mo.ui.slider(1000, 8000, value=5000, step=500, label="PINN training epochs")
    run_pinn    = mo.ui.run_button(label="▶ Train PINN")
    mo.vstack([
        mo.md("### Controls"),
        mo.hstack([lam, epochs_pinn], justify="start", gap=2),
        run_pinn,
    ])
    return epochs_pinn, lam, run_pinn


@app.cell
def _(
    DEVICE,
    F,
    MLP,
    VAL_EVERY,
    continuity_loss,
    epochs_pinn,
    eval_Ux,
    eval_Uy,
    eval_x,
    eval_y,
    lam,
    mo,
    np,
    run_pinn,
    sens_Ux,
    sens_Uy,
    sens_x,
    sens_y,
    time,
    torch,
):
    mo.stop(not run_pinn.value, mo.md("▶ *Click **Train PINN** above to start.*"))

    N_COLLOC, WARMUP_FRAC = 4096, 0.2
    _rng = np.random.default_rng(0)   # collocation set fixed throughout training
    colloc_x = torch.tensor(_rng.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32),
                            device=DEVICE, requires_grad=True)
    colloc_y = torch.tensor(_rng.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32),
                            device=DEVICE, requires_grad=True)

    torch.manual_seed(42)
    model_pinn = MLP(width=64, depth=4).to(DEVICE)
    _opt = torch.optim.Adam(model_pinn.parameters(), lr=1e-3)

    _n = int(epochs_pinn.value)
    _lambda = float(lam.value)
    history = {"data": [], "phys": [], "total": [], "val": []}
    best_val_pinn, _best_state = float("inf"), None
    _t0 = time.time()
    with mo.status.progress_bar(total=_n, title="Training PINN (data + continuity)") as _bar:
        for _ep in range(_n):
            model_pinn.train(); _opt.zero_grad()
            _pred   = model_pinn(sens_x, sens_y)
            _L_data = F.mse_loss(_pred[:, 0:1], sens_Ux) + F.mse_loss(_pred[:, 1:2], sens_Uy)
            _L_phys = continuity_loss(model_pinn, colloc_x, colloc_y)
            _ramp   = min(1.0, _ep / (_n * WARMUP_FRAC)) if WARMUP_FRAC > 0 else 1.0
            _L_tot  = _L_data + _ramp * _lambda * _L_phys
            _L_tot.backward(); _opt.step()

            if (_ep + 1) % VAL_EVERY == 0:
                model_pinn.eval()
                with torch.no_grad():
                    _p  = model_pinn(eval_x, eval_y)
                    _vl = (F.mse_loss(_p[:, 0:1], eval_Ux) +
                           F.mse_loss(_p[:, 1:2], eval_Uy)).item()
                history["data" ].append(_L_data.item())
                history["phys" ].append(_L_phys.item())
                history["total"].append(_L_tot.item())
                history["val"  ].append(_vl)
                if _vl < best_val_pinn:
                    best_val_pinn = _vl
                    _best_state   = {k: v.detach().clone() for k, v in model_pinn.state_dict().items()}
                _bar.update(increment=VAL_EVERY)

    model_pinn.load_state_dict(_best_state)
    mo.md(f"**PINN done** in {time.time()-_t0:.1f}s &nbsp;·&nbsp; "
          f"best held-out MSE = **{best_val_pinn:.3e}** &nbsp;·&nbsp; λ = {_lambda}")
    return best_val_pinn, history, model_pinn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6 — Head to head

    Same network, same sensors — the only difference is that one obeyed one physical law. The
    six panels below show truth, both reconstructions, both error fields, and a per-point
    **improvement** map (blue where the PINN beats the MLP). The divergence panels confirm the
    PINN's field is genuinely physical, and the loss curves show the data and physics terms
    dropping together.

    Watch the title: at the **sweet-spot** $K\approx 50$ the PINN typically cuts held-out error
    by roughly a quarter to a third. Push $K$ very low (both fail) or very high (the MLP no
    longer needs help) and the gap shrinks — retrain after moving the sliders to see it.
    """)
    return


@app.cell
def _(
    best_val_mlp,
    best_val_pinn,
    case,
    divergence_grid,
    history,
    mo,
    model_mlp,
    model_pinn,
    plot_divergence_compare,
    plot_headtohead,
    plot_loss_curves,
    predict_field,
    sensor_idx,
):
    _mlp_Ux,  _mlp_Uy  = predict_field(model_mlp,  case.x_norm, case.y_norm)
    _pinn_Ux, _pinn_Uy = predict_field(model_pinn, case.x_norm, case.y_norm)

    _fig_hh, _imp_pct, _win = plot_headtohead(
        case, sensor_idx, _mlp_Ux, _mlp_Uy, _pinn_Ux, _pinn_Uy, best_val_mlp, best_val_pinn
    )

    _xg, _yg, _r_mlp  = divergence_grid(model_mlp)
    _,   _,   _r_pinn = divergence_grid(model_pinn)
    _rms = lambda v: float((v ** 2).mean() ** 0.5)

    mo.vstack([
        mo.as_html(_fig_hh),
        mo.md(f"**Held-out MSE:** MLP {best_val_mlp:.2e} -> PINN {best_val_pinn:.2e} "
              f"(**{_imp_pct:+.0f}%**). PINN wins at **{_win:.0f}%** of held-out points."),
        mo.md(f"**Divergence RMS:** MLP {_rms(_r_mlp):.2e} -> PINN {_rms(_r_pinn):.2e} "
              f"(**{_rms(_r_mlp)/max(_rms(_r_pinn),1e-12):.0f}x smaller**) — the PINN field is physical."),
        mo.as_html(plot_divergence_compare(_xg, _yg, _r_mlp, _r_pinn)),
        mo.as_html(plot_loss_curves(history, val_every=50)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7 — Deeper comparison: where, how, and versus what baseline

    A single held-out MSE hides *where* and *what kind* of error each model makes. Three sharper
    views (retrain both models above first, then these update):

    1. **An RBF (radial-basis-function) interpolation baseline** — a smooth surface built to pass
       through the $K$ sensor values by placing one radial "bump" per sensor and summing them:
       $\;f(\mathbf{x}) = \sum_{j=1}^{K} w_j\,\varphi(\lVert \mathbf{x}-\mathbf{x}_j\rVert)$, with a
       Gaussian kernel $\varphi(r)=e^{-(r/\varepsilon)^2}$. The weights $w_j$ are found by solving a
       small linear system so the surface matches every sensor exactly, and the width $\varepsilon$
       is set from the typical sensor spacing. It is **mesh-free and needs no network training and no
       physics** — we simply interpolate $U_x$ and $U_y$ separately. This is the honest "why not just
       interpolate the sensors?" baseline the network has to beat.
    2. **Error vs. distance to the nearest sensor** — turns "the PINN wins in the gaps" into a curve.
    3. **Angular-error distribution** — how far each field points in the *wrong direction* (magnitude
       alone can look fine while the vector is wrong), as a magnitude-weighted histogram of MLP vs
       PINN. A leftward shift of the PINN curve means it points the right way more often.
    """)
    return


@app.cell
def _(
    case,
    eval_idx,
    heldout_mse,
    mo,
    model_mlp,
    model_pinn,
    predict_field,
    rbf_interpolate,
    sensor_idx,
):
    # Full-mesh predictions for all three methods (normalized units).
    mlp_Ux,  mlp_Uy  = predict_field(model_mlp,  case.x_norm, case.y_norm)
    pinn_Ux, pinn_Uy = predict_field(model_pinn, case.x_norm, case.y_norm)
    interp_Ux = rbf_interpolate(case.x_norm[sensor_idx], case.y_norm[sensor_idx],
                                case.Ux_norm[sensor_idx], case.x_norm, case.y_norm)
    interp_Uy = rbf_interpolate(case.x_norm[sensor_idx], case.y_norm[sensor_idx],
                                case.Uy_norm[sensor_idx], case.x_norm, case.y_norm)

    _mi = heldout_mse(case, eval_idx, interp_Ux, interp_Uy)
    _mm = heldout_mse(case, eval_idx, mlp_Ux,   mlp_Uy)
    _mp = heldout_mse(case, eval_idx, pinn_Ux,  pinn_Uy)
    mo.md(f"""
    **Held-out MSE** &nbsp; RBF interp **{_mi:.2e}** &nbsp;·&nbsp; MLP **{_mm:.2e}** &nbsp;·&nbsp; PINN **{_mp:.2e}**

    PINN vs interp **{100*(_mi-_mp)/_mi:+.0f}%** &nbsp;·&nbsp; PINN vs MLP **{100*(_mm-_mp)/_mm:+.0f}%**
    &nbsp; — if plain interpolation already beats the MLP, that is the honest baseline the physics has to clear.
    """)
    return interp_Ux, interp_Uy, mlp_Ux, mlp_Uy, pinn_Ux, pinn_Uy


@app.cell
def _(
    case,
    eval_idx,
    interp_Ux,
    interp_Uy,
    mlp_Ux,
    mlp_Uy,
    mo,
    pinn_Ux,
    pinn_Uy,
    plot_error_vs_distance,
    sensor_idx,
):
    _preds = {
        "RBF interp": (interp_Ux, interp_Uy, "C7"),
        "MLP":        (mlp_Ux,    mlp_Uy,    "C3"),
        "PINN":       (pinn_Ux,   pinn_Uy,   "C0"),
    }
    mo.as_html(plot_error_vs_distance(case, sensor_idx, eval_idx, _preds))
    return


@app.cell
def _(
    case,
    mlp_Ux,
    mlp_Uy,
    mo,
    pinn_Ux,
    pinn_Uy,
    plot_angular_error_hist,
):
    mo.as_html(plot_angular_error_hist(case, mlp_Ux, mlp_Uy, pinn_Ux, pinn_Uy))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 8 — Adding turbulent momentum (RANS k-ε closure)

    Continuity couples $U_x$ and $U_y$, but it is only *one* equation. The next rung on the
    completeness ladder is the **momentum balance**. For this turbulent basin the relevant form is
    the steady, incompressible **RANS** momentum equation with a Boussinesq eddy-viscosity closure:

    $$(\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla P^{*} + \nabla\!\cdot\!\big[(\nu+\nu_t)(\nabla\mathbf{u}+\nabla\mathbf{u}^{\top})\big].$$

    Three things this needs that continuity did not:

    1. **Pressure.** Momentum contains $\nabla P$, and we have no pressure sensors — so we add a
       **third network output** $p$ and let it be *inferred* from momentum consistency (it is fixed
       only up to a constant). That output is the *modified* pressure $P^{*}=p/\rho+\tfrac{2}{3}k$;
       the $\tfrac{2}{3}k$ piece is negligible here.
    2. **Second derivatives.** The viscous term is a Laplacian, so the residual needs
       $\partial^2\mathbf{u}$ — a second autograd pass (another reason for the smooth `tanh`).
    3. **The turbulence, supplied — not solved.** We do **not** solve the $k$ and $\varepsilon$
       transport equations. Instead we *supply* the eddy viscosity $\nu_t(x,y)$ the CFD already
       computed and use $\nu_{\text{eff}}=\nu+\nu_t$. That is the honest move for this dataset: the
       median $\nu_t$ is **~16,000×** the molecular $\nu$, so a molecular-only ("laminar") momentum
       law would be badly wrong — turbulence carries essentially all the momentum transport. (We
       also freeze $\nu_t$ locally, i.e. neglect $\nabla\nu_t$, so the viscous term stays a clean
       $\nu_{\text{eff}}\nabla^2\mathbf{u}$.)

    The loss becomes $\;L_\text{data} + \lambda_c\,L_\text{continuity} + \lambda_m\,L_\text{momentum}$,
    each physics term eased in with the same warmup ramp. Momentum is a *partial* extra constraint,
    so expect a modest further gain over continuity — and a striking bonus: a pressure field the
    network reconstructs without ever seeing one.
    """)
    return


@app.cell
def _(case, load_eddy_viscosity, mo, nearest_interp, np, plot_eddy_viscosity):
    NU_MOL = 1e-6                                   # molecular kinematic viscosity (transportProperties: nu = 1e-6)
    ev = load_eddy_viscosity(case.case_id)          # CFD nu_t + kinematic pressure, same mesh as `case`
    nut_mesh  = nearest_interp(ev["x_norm"], ev["y_norm"], ev["nut"],   case.x_norm, case.y_norm)
    p_cfd_kin = nearest_interp(ev["x_norm"], ev["y_norm"], ev["p_kin"], case.x_norm, case.y_norm)
    mo.vstack([
        mo.md(f"Supplied eddy viscosity: median $\\nu_t/\\nu \\approx$ **{np.median(nut_mesh)/NU_MOL:,.0f}×** — "
              f"turbulence, not molecular viscosity, carries the momentum here."),
        mo.as_html(plot_eddy_viscosity(case, nut_mesh, NU_MOL)),
    ])
    return NU_MOL, ev, nut_mesh, p_cfd_kin


@app.cell
def _(case, nn, torch):
    class MLP3(nn.Module):
        """(x_norm, y_norm) -> (Ux_norm, Uy_norm, p_norm).  Tanh; adds a pressure output."""

        def __init__(self, width: int = 64, depth: int = 4):
            super().__init__()
            layers = [nn.Linear(2, width), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.Tanh()]
            layers += [nn.Linear(width, 3)]
            self.net = nn.Sequential(*layers)

        def forward(self, x, y):
            return self.net(torch.cat([x, y], dim=1))

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    def rans_momentum_residual(model, x, y, nu_eff):
        """
        Steady incompressible RANS momentum with a supplied eddy viscosity
        nu_eff = nu + nu_t (Boussinesq).  The pressure output is the modified
        pressure P* = p/rho + 2/3 k.  nu_eff is treated as locally frozen
        (grad nu_t neglected), so the viscous term is nu_eff * laplacian(u).
        The 1/L, 1/W (and 1/L^2, 1/W^2) factors convert normalized derivatives to
        physical ones; the residual is nondimensionalized by U^2/L.
        """
        U, L, W = case.U_inlet, case.L, case.W
        out = model(x, y)
        un, vn, pn = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        g = lambda f, z: torch.autograd.grad(f, z, torch.ones_like(f),
                                             create_graph=True, retain_graph=True)[0]
        un_x = g(un, x); un_y = g(un, y); vn_x = g(vn, x); vn_y = g(vn, y)
        pn_x = g(pn, x); pn_y = g(pn, y)
        un_xx = g(un_x, x); un_yy = g(un_y, y); vn_xx = g(vn_x, x); vn_yy = g(vn_y, y)

        u = U * un; v = U * vn
        dudx = (U / L) * un_x; dudy = (U / W) * un_y
        dvdx = (U / L) * vn_x; dvdy = (U / W) * vn_y
        d2udx2 = (U / L**2) * un_xx; d2udy2 = (U / W**2) * un_yy
        d2vdx2 = (U / L**2) * vn_xx; d2vdy2 = (U / W**2) * vn_yy
        dPdx = (U**2 / L) * pn_x; dPdy = (U**2 / W) * pn_y

        Rx = u * dudx + v * dudy + dPdx - nu_eff * (d2udx2 + d2udy2)
        Ry = u * dvdx + v * dvdy + dPdy - nu_eff * (d2vdx2 + d2vdy2)
        scale = U**2 / L
        return Rx / scale, Ry / scale

    def momentum_loss(model, x, y, nu_eff):
        Rx, Ry = rans_momentum_residual(model, x, y, nu_eff)
        return torch.mean(Rx ** 2 + Ry ** 2)

    return MLP3, momentum_loss, rans_momentum_residual


@app.cell
def _(mo):
    lam_cont    = mo.ui.slider(0.0, 5.0, value=1.0, step=0.25, label="λ_c  (continuity)")
    lam_mom     = mo.ui.slider(0.0, 2.0, value=0.2, step=0.1,  label="λ_m  (RANS momentum)")
    epochs_rans = mo.ui.slider(1000, 8000, value=5000, step=500, label="RANS PINN epochs")
    run_rans    = mo.ui.run_button(label="▶ Train RANS-momentum PINN")
    mo.vstack([
        mo.md("### Controls"),
        mo.hstack([lam_cont, lam_mom, epochs_rans], justify="start", gap=1),
        run_rans,
    ])
    return epochs_rans, lam_cont, lam_mom, run_rans


@app.cell
def _(
    DEVICE,
    F,
    MLP3,
    NU_MOL,
    continuity_loss,
    epochs_rans,
    ev,
    eval_Ux,
    eval_Uy,
    eval_x,
    eval_y,
    lam_cont,
    lam_mom,
    mo,
    momentum_loss,
    nearest_interp,
    np,
    run_rans,
    sens_Ux,
    sens_Uy,
    sens_x,
    sens_y,
    time,
    torch,
):
    mo.stop(not run_rans.value, mo.md("▶ *Click **Train RANS-momentum PINN** above to start.*"))

    _N_COLLOC, _WARMUP_FRAC = 4096, 0.2
    _rng = np.random.default_rng(0)
    _cxn = _rng.uniform(0, 1, _N_COLLOC).astype(np.float32)
    _cyn = _rng.uniform(0, 1, _N_COLLOC).astype(np.float32)
    _colloc_x = torch.tensor(_cxn.reshape(-1, 1), device=DEVICE, requires_grad=True)
    _colloc_y = torch.tensor(_cyn.reshape(-1, 1), device=DEVICE, requires_grad=True)
    # supply nu_eff = nu + nu_t at the collocation points from the CFD field
    _nueff = NU_MOL + nearest_interp(ev["x_norm"], ev["y_norm"], ev["nut"], _cxn, _cyn)
    _nu_eff_col = torch.tensor(_nueff.reshape(-1, 1), device=DEVICE)

    torch.manual_seed(42)
    model_rans = MLP3(width=64, depth=4).to(DEVICE)
    _opt = torch.optim.Adam(model_rans.parameters(), lr=1e-3)
    _n = int(epochs_rans.value); _lc = float(lam_cont.value); _lm = float(lam_mom.value)
    history_rans = {"data": [], "cont": [], "mom": [], "val": []}
    best_val_rans, _best = float("inf"), None
    _t0 = time.time()
    with mo.status.progress_bar(total=_n, title="Training RANS-momentum PINN") as _bar:
        for _ep in range(_n):
            model_rans.train(); _opt.zero_grad()
            _pred = model_rans(sens_x, sens_y)
            _Ld = F.mse_loss(_pred[:, 0:1], sens_Ux) + F.mse_loss(_pred[:, 1:2], sens_Uy)
            _Lc = continuity_loss(model_rans, _colloc_x, _colloc_y)
            _Lm = momentum_loss(model_rans, _colloc_x, _colloc_y, _nu_eff_col)
            _ramp = min(1.0, _ep / (_n * _WARMUP_FRAC)) if _WARMUP_FRAC > 0 else 1.0
            _L = _Ld + _ramp * _lc * _Lc + _ramp * _lm * _Lm
            _L.backward(); _opt.step()
            if (_ep + 1) % 50 == 0:
                model_rans.eval()
                with torch.no_grad():
                    _p = model_rans(eval_x, eval_y)
                    _vl = (F.mse_loss(_p[:, 0:1], eval_Ux) + F.mse_loss(_p[:, 1:2], eval_Uy)).item()
                history_rans["data"].append(_Ld.item()); history_rans["cont"].append(_Lc.item())
                history_rans["mom"].append(_Lm.item()); history_rans["val"].append(_vl)
                if _vl < best_val_rans:
                    best_val_rans = _vl
                    _best = {k: v.detach().clone() for k, v in model_rans.state_dict().items()}
                _bar.update(increment=50)

    model_rans.load_state_dict(_best)
    mo.md(f"**RANS-momentum PINN done** in {time.time()-_t0:.1f}s &nbsp;·&nbsp; "
          f"best held-out MSE = **{best_val_rans:.3e}** &nbsp;·&nbsp; "
          f"λ_c={_lc}, λ_m={_lm}, {model_rans.num_params():,} params")
    return best_val_rans, history_rans, model_rans


@app.cell
def _(
    DEVICE,
    NU_MOL,
    best_val_rans,
    case,
    mo,
    model_rans,
    np,
    nut_mesh,
    p_cfd_kin,
    plot_pressure_compare,
    plot_scalar_field,
    rans_momentum_residual,
    torch,
):
    # momentum-residual magnitude on the mesh (smaller where the law is satisfied)
    _xt = torch.tensor(case.x_norm.reshape(-1, 1), device=DEVICE, requires_grad=True)
    _yt = torch.tensor(case.y_norm.reshape(-1, 1), device=DEVICE, requires_grad=True)
    _nueff = torch.tensor((NU_MOL + nut_mesh).reshape(-1, 1), device=DEVICE)
    _Rx, _Ry = rans_momentum_residual(model_rans, _xt, _yt, _nueff)
    _rmag = torch.sqrt(_Rx ** 2 + _Ry ** 2).detach().cpu().numpy().ravel()

    # emergent pressure (3rd output), in physical kinematic units
    model_rans.eval()
    with torch.no_grad():
        _out = model_rans(torch.as_tensor(case.x_norm, device=DEVICE).unsqueeze(1),
                          torch.as_tensor(case.y_norm, device=DEVICE).unsqueeze(1)).cpu().numpy()
    _p_pred_kin = (case.U_inlet ** 2) * _out[:, 2]

    mo.vstack([
        mo.md(f"**RANS-momentum PINN held-out MSE = {best_val_rans:.3e}** — compare with the "
              f"continuity-only PINN above; momentum adds a *partial* further constraint."),
        mo.as_html(plot_scalar_field(case, _rmag, "RANS momentum residual magnitude (nondimensional)", cmap="magma")),
        mo.md("The network's **emergent pressure** — inferred purely from momentum consistency, never "
              "measured — beside the CFD pressure (both mean-removed):"),
        mo.as_html(plot_pressure_compare(case, _p_pred_kin, p_cfd_kin)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per-term loss trajectories (RANS-momentum PINN)

    Same read as the continuity-only curves, now with the momentum term. Watch the **held-out val**
    (green) and its distance to the **data loss** (blue) — the *generalization gap*. If the extra
    physics is helping the network generalize, this gap should be no worse — ideally tighter — than
    the continuity-only run in Step 6.
    """)
    return


@app.cell
def _(history_rans, mo, plot_loss_curves_rans):
    mo.as_html(plot_loss_curves_rans(history_rans, val_every=50))
    return


@app.cell
def _(history, history_rans, mo):
    _gc = history["val"][-1] / history["data"][-1]
    _gr = history_rans["val"][-1] / history_rans["data"][-1]
    mo.md(f"""
    **Does momentum shrink the generalization gap?** &nbsp; final held-out ÷ data ratio:
    continuity-only **{_gc:.0f}×** → + momentum **{_gr:.0f}×** &nbsp;·&nbsp; best held-out MSE
    **{min(history['val']):.2e} → {min(history_rans['val']):.2e}**.

    *Caveat:* a physics term also raises the data loss (it competes with the pure fit), so the ratio
    can shrink even when held-out error does not improve — judge generalization by the **held-out val
    itself**, not the ratio alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrapping up — what to take away

    - **The loss is the physics.** The PINN and the MLP differ by exactly one term — a squared
      continuity residual at collocation points — yet the PINN reconstructs the gaps the MLP
      cannot, because that one law couples $U_x$ and $U_y$ across the whole domain.
    - **Derivatives come for free** from autograd — that is what makes "obey the equation"
      measurable, and why we need a smooth activation like `tanh`.
    - **Physics buys data efficiency in the sparse regime.** The win is largest around
      $K\approx 50$. It is *not* magic at every scale — see below.
    - **More physics is a ladder, not a switch.** Step 8 adds the RANS $k\text{-}\varepsilon$
      momentum balance (with the eddy viscosity *supplied* from the CFD) on top of continuity. It is
      a *partial* extra constraint, so it buys a further, smaller gain — and, as a bonus, the network
      reconstructs a **pressure field it was never shown**. Completeness (continuity → momentum →
      closure) is what steadily tightens the generalization gap.

    **Things to try (move sliders, then retrain):**

    1. **Sensor sparsity.** Set $K=20$ then $K=300$ and retrain both models. At low $K$ both
       fail; at high $K$ the MLP interpolates fine on its own and the gap collapses — sometimes
       continuity even *slightly hurts*. That is incomplete physics meeting sufficient data, and
       it is a feature to teach.
    2. **The physics weight $\lambda$.** Sweep it $\{0, 0.25, 1, 3, 5\}$. Is bigger always
       better? (Watch the data loss in the curves — too much $\lambda$ and physics fights the
       data anchor.)
    3. **Seed variance.** Hold $K=50$ and step the seed 42 to 46, retraining each time. The
       improvement swings ~2x across *which* points you happen to sense — always report a range,
       never one lucky number.
    4. **A different case.** Set `CASE_ID = 267` in Step 1 (simpler 1-baffle flow). The sweet
       spot moves to a lower $K$.
    5. **The momentum weight $\lambda_m$ (Step 8).** Sweep it and watch the RANS loss trajectories:
       does adding momentum lower the held-out val and tighten the generalization gap, or just
       trade data fit for physics? Compare its gap with the continuity-only run.

    **Where the plumbing lives:** all data-loading and plotting code is in `pinn_utils_2d.py`,
    documented with docstrings. The `plot_*` helpers return their Figure, which is why the cells
    above wrap them in `mo.as_html(...)`.

    **Reference.** Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed
    neural networks.* Journal of Computational Physics, 378, 686-707.
    doi:10.1016/j.jcp.2018.10.045
    """)
    return


if __name__ == "__main__":
    app.run()
