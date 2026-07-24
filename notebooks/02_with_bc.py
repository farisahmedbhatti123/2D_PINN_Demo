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
# # 02 — Inpainting with Boundary Conditions
#
# Builds on `01_with_momentum.py` by adding **explicit boundary-condition
# losses**: inlet, outlet, wall no-slip, and baffle no-slip.  Trains six
# configurations from the same K=50 sensor split so we can isolate what
# each piece of physics — and BC information — actually buys:
#
# | config | data | continuity | momentum | BC |
# |---|---|---|---|---|
# | MLP                     | ✓ |    |    |    |
# | MLP + BC                | ✓ |    |    | ✓ |
# | Continuity              | ✓ | ✓ |    |    |
# | Continuity + BC         | ✓ | ✓ |    | ✓ |
# | Momentum                | ✓ | ✓ | ✓ |    |
# | Momentum + BC           | ✓ | ✓ | ✓ | ✓ |
#
# **BC formulations**:
#   - Inlet (y=0, x ∈ [0, w/L]): Ux=0, Uy=1 (in U_inlet units)
#   - Outlet (top wall for even n, bottom for odd): zero-gradient ∂U/∂n=0
#   - Outer walls (excluding openings): no-slip u=0
#   - Baffles (vertical line segments): no-slip u=0

# %% [markdown]
# ## 0. Imports

# %%
import os, sys, time, pathlib, json
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

# --- nu_t representation network (pre-trained once, reused across all configs) ---
NUT_WIDTH  = 32
NUT_DEPTH  = 3
NUT_EPOCHS = 3000

# --- training ---
EPOCHS         = 5000
LR             = 1e-3
VAL_EVERY      = 50
PRINT_EVERY    = 1000

# --- physics weights ---
LAMBDA_CONT    = 1.0       # continuity
LAMBDA_MOMENT  = 0.001     # momentum (narrow working window 0.0005-0.005)
LAMBDA_BC      = 3e-4      # BC losses.  Sweep on MLP+BC (see phase_a_lambda_bc_sweep.log)
                           # showed monotonic deterioration above ~1e-3; sweet spot at 3e-4
                           # gives -30% vs MLP-only.  Above 0.1 the BC overwhelms the
                           # 50-sensor data signal and val MSE actually gets worse than no BC.
N_COLLOC       = 4096
WARMUP_FRAC    = 0.2

# --- BC sampling ---
N_INLET   = 256
N_OUTLET  = 256
N_WALL    = 512
N_BAFFLE  = 64

# --- physical constants ---
NU_MOLECULAR  = 1.0e-6
INLET_WIDTH_M = 1.5

print(f"device: {DEVICE}")
print(f"case: {CASE_ID},  K = {K} sensors,  {EPOCHS} epochs per config")
print(f"physics weights: lam_cont={LAMBDA_CONT}  lam_moment={LAMBDA_MOMENT}  lam_BC={LAMBDA_BC}")

# %% [markdown]
# ## 2. Load CFD data + sensor split
#
# Same case_834_full.csv as `01_with_momentum`.  Modified pressure
# `P_mod = p + 2k/3`, centered.

# %%
geo_csv = os.path.join(DATA_DIR, "basin_geometries_summary.csv")
vel_csv = os.path.join(DATA_DIR, "results", f"case_{CASE_ID:03d}_full.csv")

geo_row = pd.read_csv(geo_csv).query(f"`Simulation Number` == {CASE_ID}").iloc[0]
L       = float(geo_row['L']);   W = float(geo_row['W'])
n_baf   = int  (geo_row['n']);   a_baf = float(geo_row['a'])
U_inlet = float(geo_row['U_inlet'])

df = pd.read_csv(vel_csv).dropna(subset=['Ux', 'Uy', 'p', 'nut', 'k'])
x_phys  = df['x'].values.astype(np.float32)
y_phys  = df['y'].values.astype(np.float32)
x_norm  = (df['x'].values / L).astype(np.float32)
y_norm  = (df['y'].values / W).astype(np.float32)
Ux_norm = (df['Ux'].values / U_inlet).astype(np.float32)
Uy_norm = (df['Uy'].values / U_inlet).astype(np.float32)
nut_phys = df['nut'].values.astype(np.float32)
k_phys   = df['k'].values.astype(np.float32)

p_mod_phys     = df['p'].values + (2.0/3.0) * df['k'].values
p_mod_norm     = (p_mod_phys / U_inlet**2).astype(np.float32)
p_mod_offset   = float(p_mod_norm.mean())
p_mod_centered = p_mod_norm - p_mod_offset

N = len(x_norm)
print(f"Case {CASE_ID}: L={L:.0f} m, W={W:.0f} m, n={n_baf} baffles, U_inlet={U_inlet:.3f} m/s")
print(f"  {N:,} CFD mesh points loaded")

# Sensor split (same seed 42 as prior notebooks)
rng        = np.random.default_rng(SEED)
sensor_idx = rng.choice(N, size=K, replace=False)
eval_idx   = np.setdiff1d(np.arange(N), sensor_idx)

def to_col(arr):
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(1)

sens_x  = to_col(x_norm        [sensor_idx])
sens_y  = to_col(y_norm        [sensor_idx])
sens_Ux = to_col(Ux_norm       [sensor_idx])
sens_Uy = to_col(Uy_norm       [sensor_idx])
sens_P  = to_col(p_mod_centered[sensor_idx])

eval_x  = to_col(x_norm        [eval_idx])
eval_y  = to_col(y_norm        [eval_idx])
eval_Ux = to_col(Ux_norm       [eval_idx])
eval_Uy = to_col(Uy_norm       [eval_idx])

print(f"K = {K} sensors,  {len(eval_idx):,} held-out evaluation points")

# %% [markdown]
# ## 3. BC point samplers
#
# Four samplers in normalized [0,1]² coordinates.  See A0 verification
# script in the project root for geometry-correctness visualization.

# %%
def sample_inlet(n=N_INLET, seed=1):
    rng = np.random.default_rng(seed)
    x_max = min(INLET_WIDTH_M / L, 1.0)
    return (rng.uniform(0, x_max, (n, 1)).astype(np.float32),
            np.zeros((n, 1), dtype=np.float32))

def sample_outlet(n=N_OUTLET, seed=4):
    rng = np.random.default_rng(seed)
    x_min = max(1.0 - INLET_WIDTH_M / L, 0.0)
    x = rng.uniform(x_min, 1.0, (n, 1)).astype(np.float32)
    y = np.ones((n, 1), dtype=np.float32) if n_baf % 2 == 0 else np.zeros((n, 1), dtype=np.float32)
    return x, y

def sample_walls(n=N_WALL, seed=2):
    rng = np.random.default_rng(seed)
    n_per = n // 4; n_left = n_right = n_bot = n_per; n_top = n - 3*n_per
    inlet_x_max  = min(INLET_WIDTH_M / L, 1.0)
    outlet_x_min = max(1.0 - INLET_WIDTH_M / L, 0.0)
    xs, ys = [], []
    xs.append(np.zeros((n_left, 1), dtype=np.float32));  ys.append(rng.uniform(0, 1, (n_left, 1)).astype(np.float32))
    xs.append(np.ones ((n_right, 1), dtype=np.float32)); ys.append(rng.uniform(0, 1, (n_right, 1)).astype(np.float32))
    bx = rng.uniform(inlet_x_max, outlet_x_min if n_baf%2==1 else 1.0, (n_bot, 1))
    xs.append(bx.astype(np.float32)); ys.append(np.zeros((n_bot, 1), dtype=np.float32))
    tx = rng.uniform(0.0, outlet_x_min if n_baf%2==0 else 1.0, (n_top, 1))
    xs.append(tx.astype(np.float32)); ys.append(np.ones((n_top, 1), dtype=np.float32))
    return np.vstack(xs), np.vstack(ys)

def sample_baffles(n_per_baffle=N_BAFFLE, seed=3):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for k in range(1, n_baf + 1):
        bx = k / (n_baf + 1)
        if k % 2 == 1: y0, y1, tip = 0.0, a_baf, a_baf
        else:          y0, y1, tip = 1.0 - a_baf, 1.0, 1.0 - a_baf
        n_uni = int(n_per_baffle * 0.7); n_tip = n_per_baffle - n_uni
        y_u = rng.uniform(y0, y1, (n_uni, 1)).astype(np.float32)
        y_t = np.clip(tip + rng.normal(0, 0.02, (n_tip, 1)), y0, y1).astype(np.float32)
        xs.append(np.full((n_uni, 1), bx, dtype=np.float32)); ys.append(y_u)
        xs.append(np.full((n_tip, 1), bx, dtype=np.float32)); ys.append(y_t)
    return np.vstack(xs), np.vstack(ys)

# Build BC tensors once (fixed point sets, reused across all configs and all epochs)
def _bc_to_tensor(x_np, y_np, grad=False):
    xt = torch.as_tensor(x_np, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y_np, dtype=torch.float32, device=DEVICE)
    if grad:
        xt = xt.clone().requires_grad_(True)
        yt = yt.clone().requires_grad_(True)
    return xt, yt

_in_np  = sample_inlet ()
_out_np = sample_outlet()
_wa_np  = sample_walls ()
_ba_np  = sample_baffles()

in_x,   in_y   = _bc_to_tensor(*_in_np)
out_x,  out_y  = _bc_to_tensor(*_out_np, grad=True)   # outlet needs autograd for ∂U/∂y
wall_x, wall_y = _bc_to_tensor(*_wa_np)
baf_x,  baf_y  = _bc_to_tensor(*_ba_np)

print(f"BC samples: inlet={in_x.shape[0]}  outlet={out_x.shape[0]}  "
      f"wall={wall_x.shape[0]}  baffle={baf_x.shape[0]}")

# %% [markdown]
# ## 4. Architecture, residuals, BC losses

# %%
class ScalarField(nn.Module):
    """Small NN that maps (x, y) -> scalar.  Used for the frozen nu_t representation."""
    def __init__(self, width=NUT_WIDTH, depth=NUT_DEPTH):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

class MLP(nn.Module):
    """Main network: (x, y) -> (Ux, Uy, P_mod).  Tanh activations for autograd-friendliness."""
    def __init__(self, width=WIDTH, depth=DEPTH, out_dim=3):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

def grad1(out, var):
    """First derivative; builds a graph for further differentiation."""
    return torch.autograd.grad(out, var, torch.ones_like(out),
                               create_graph=True, retain_graph=True)[0]

def continuity_residual(model, x, y):
    out = model(x, y)
    Ux, Uy = out[:, 0:1], out[:, 1:2]
    return grad1(Ux, x) / L + grad1(Uy, y) / W

def continuity_loss(model, x, y):
    return torch.mean(continuity_residual(model, x, y) ** 2)

def momentum_residual(model, nut_model, x, y):
    """Dimensionless Boussinesq momentum residual.  Returns (R_x, R_y)."""
    out = model(x, y)
    Ux, Uy, P = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    dUx_dx = grad1(Ux, x); dUx_dy = grad1(Ux, y)
    dUy_dx = grad1(Uy, x); dUy_dy = grad1(Uy, y)
    dP_dx  = grad1(P,  x); dP_dy  = grad1(P,  y)
    d2Ux_dx2 = grad1(dUx_dx, x); d2Ux_dy2 = grad1(dUx_dy, y)
    d2Uy_dx2 = grad1(dUy_dx, x); d2Uy_dy2 = grad1(dUy_dy, y)
    nu_t    = nut_model(x, y)
    dnut_dx = grad1(nu_t, x); dnut_dy = grad1(nu_t, y)
    LW         = L / W
    Re_eff_inv = (NU_MOLECULAR + nu_t) / (U_inlet * L)
    UL_inv     = 1.0 / (U_inlet * L)
    conv_x = Ux * dUx_dx + LW * Uy * dUx_dy
    pres_x = dP_dx
    lap_x  = Re_eff_inv * (d2Ux_dx2 + LW**2 * d2Ux_dy2)
    cr_x   = UL_inv * (2*dnut_dx*dUx_dx + LW**2 * dnut_dy*dUx_dy + LW * dnut_dy*dUy_dx)
    R_x = conv_x + pres_x - lap_x - cr_x
    conv_y = Ux * dUy_dx + LW * Uy * dUy_dy
    pres_y = LW * dP_dy
    lap_y  = Re_eff_inv * (d2Uy_dx2 + LW**2 * d2Uy_dy2)
    cr_y   = UL_inv * (dnut_dx*dUy_dx + LW * dnut_dx*dUx_dy + 2 * LW**2 * dnut_dy*dUy_dy)
    R_y = conv_y + pres_y - lap_y - cr_y
    return R_x, R_y

# --- BC losses ---
def bc_loss_inlet(model, x, y, target_Ux=0.0, target_Uy=1.0):
    out = model(x, y)
    return torch.mean((out[:, 0:1] - target_Ux) ** 2 + (out[:, 1:2] - target_Uy) ** 2)

def bc_loss_wall(model, x, y):
    out = model(x, y)
    return torch.mean(out[:, 0:1] ** 2 + out[:, 1:2] ** 2)

def bc_loss_baffle(model, x, y):
    return bc_loss_wall(model, x, y)

def bc_loss_outlet(model, x, y):
    """Zero-gradient: ∂Ux/∂y = ∂Uy/∂y = 0.  x, y must require_grad."""
    out = model(x, y)
    Ux, Uy = out[:, 0:1], out[:, 1:2]
    dUx_dy = grad1(Ux, y)
    dUy_dy = grad1(Uy, y)
    return torch.mean(dUx_dy ** 2 + dUy_dy ** 2)

# %% [markdown]
# ## 5. Pre-train the nu_t representation network (once, reused across all configs)
#
# Same approach as `01_with_momentum`: fit a tiny Tanh net against the
# CFD's nu_t field, then freeze.  This gives a differentiable nu_t(x, y)
# for the momentum residual's cross-terms.

# %%
torch.manual_seed(SEED); np.random.seed(SEED)
print("Pre-training nu_t representation...")
x_all = to_col(x_norm); y_all = to_col(y_norm); nut_target = to_col(nut_phys)

nut_model = ScalarField(NUT_WIDTH, NUT_DEPTH).to(DEVICE)
opt_nut   = torch.optim.Adam(nut_model.parameters(), lr=1e-3)
for ep in range(NUT_EPOCHS):
    opt_nut.zero_grad()
    nut_loss = F.mse_loss(nut_model(x_all, y_all), nut_target)
    nut_loss.backward(); opt_nut.step()
    if (ep + 1) % 1000 == 0:
        print(f"  nu_t ep {ep+1}/{NUT_EPOCHS}:  MSE = {nut_loss.item():.3e}")
for p in nut_model.parameters():
    p.requires_grad = False
print(f"  nu_t pre-train done.  final MSE = {nut_loss.item():.3e}  (target std^2 = {nut_target.std().item()**2:.3e})")

# %% [markdown]
# ## 6. Training helper
#
# `run_config(name, bc, cont, mom)` trains one configuration and returns
# a results dict (history, best val MSE, trained predictions, residual
# RMS on a uniform grid).

# %%
# Fixed collocation points for all PINN configs.  Resampling each step would
# give marginally smoother training but adds run-to-run variance we don't want
# while ablating physics terms.
rng_c = np.random.default_rng(0)
colloc_x = torch.tensor(rng_c.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32),
                        device=DEVICE, requires_grad=True)
colloc_y = torch.tensor(rng_c.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32),
                        device=DEVICE, requires_grad=True)

def predict_field(model, x_n, y_n):
    model.eval()
    with torch.no_grad():
        p = model(to_col(x_n), to_col(y_n)).cpu().numpy()
    return p[:, 0], p[:, 1], p[:, 2]

def residual_rms_grid(model, n_side=60):
    """Continuity + momentum residual RMS on a uniform 60x60 grid (interior)."""
    xs = np.linspace(0, 1, n_side, dtype=np.float32)
    ys = np.linspace(0, 1, n_side, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    xt = torch.tensor(X.reshape(-1, 1), device=DEVICE, requires_grad=True)
    yt = torch.tensor(Y.reshape(-1, 1), device=DEVICE, requires_grad=True)
    rc = continuity_residual(model, xt, yt).detach().cpu().numpy().squeeze()
    rx, ry = momentum_residual(model, nut_model, xt, yt)
    rx = rx.detach().cpu().numpy().squeeze()
    ry = ry.detach().cpu().numpy().squeeze()
    return {
        "cont":   float(np.sqrt(np.mean(rc ** 2))),
        "mom_x":  float(np.sqrt(np.mean(rx ** 2))),
        "mom_y":  float(np.sqrt(np.mean(ry ** 2))),
    }

def run_config(name, bc, cont, mom):
    """Train one configuration.  Same MLP architecture, same sensor split,
       same EPOCHS for all.  Physics + BC turned on/off via flags."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = MLP(WIDTH, DEPTH, out_dim=3).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = model.num_params()
    print(f"\n[{name}]  bc={bc}  cont={cont}  mom={mom}   ({n_params:,} params)")

    history = {k: [] for k in
               ['data', 'cont', 'mom_x', 'mom_y',
                'bc_in', 'bc_wall', 'bc_baf', 'bc_out',
                'total', 'val']}
    best_val, best_state = float("inf"), None
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()

        # Data loss (always on)
        pred = model(sens_x, sens_y)
        L_data = (F.mse_loss(pred[:, 0:1], sens_Ux) +
                  F.mse_loss(pred[:, 1:2], sens_Uy) +
                  F.mse_loss(pred[:, 2:3], sens_P))

        # Physics losses (conditional)
        L_cont = continuity_loss(model, colloc_x, colloc_y) if cont else torch.zeros((), device=DEVICE)
        if mom:
            R_x, R_y = momentum_residual(model, nut_model, colloc_x, colloc_y)
            L_mx = torch.mean(R_x ** 2); L_my = torch.mean(R_y ** 2)
        else:
            L_mx = torch.zeros((), device=DEVICE); L_my = torch.zeros((), device=DEVICE)

        # BC losses (conditional)
        if bc:
            L_in  = bc_loss_inlet (model, in_x,   in_y)
            L_wa  = bc_loss_wall  (model, wall_x, wall_y)
            L_ba  = bc_loss_baffle(model, baf_x,  baf_y)
            L_out = bc_loss_outlet(model, out_x,  out_y)
        else:
            L_in = L_wa = L_ba = L_out = torch.zeros((), device=DEVICE)

        ramp     = min(1.0, ep / (EPOCHS * WARMUP_FRAC)) if WARMUP_FRAC > 0 else 1.0
        L_phys   = ramp * (LAMBDA_CONT * L_cont + LAMBDA_MOMENT * (L_mx + L_my))
        L_bound  = ramp * LAMBDA_BC * (L_in + L_wa + L_ba + L_out)
        L_total  = L_data + L_phys + L_bound

        L_total.backward(); opt.step()

        if (ep + 1) % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                p = model(eval_x, eval_y)
                vl = (F.mse_loss(p[:, 0:1], eval_Ux) +
                      F.mse_loss(p[:, 1:2], eval_Uy)).item()
            history['data' ].append(L_data .item())
            history['cont' ].append(L_cont .item())
            history['mom_x'].append(L_mx   .item())
            history['mom_y'].append(L_my   .item())
            history['bc_in'].append(L_in   .item())
            history['bc_wall'].append(L_wa .item())
            history['bc_baf'].append(L_ba  .item())
            history['bc_out'].append(L_out .item())
            history['total'].append(L_total.item())
            history['val'  ].append(vl)
            if vl < best_val:
                best_val   = vl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % PRINT_EVERY == 0:
            extras = []
            if cont: extras.append(f"cont {L_cont.item():.2e}")
            if mom:  extras.append(f"mom_x {L_mx.item():.2e}  mom_y {L_my.item():.2e}")
            if bc:   extras.append(f"bc_in {L_in.item():.2e}  bc_wa {L_wa.item():.2e}  bc_ba {L_ba.item():.2e}  bc_out {L_out.item():.2e}")
            print(f"  ep {ep+1:5d}  data {L_data.item():.2e}  "
                  + "  ".join(extras) + f"  val {history['val'][-1]:.2e}")

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    # Final residual RMS on grid
    res_rms = residual_rms_grid(model)
    # Field prediction on full mesh
    pred_Ux, pred_Uy, pred_P = predict_field(model, x_norm, y_norm)
    print(f"[{name}] DONE in {elapsed:.1f}s  |  best val MSE = {best_val:.3e}  |  "
          f"cont RMS = {res_rms['cont']:.2e}  mom_x RMS = {res_rms['mom_x']:.2e}  mom_y RMS = {res_rms['mom_y']:.2e}")

    return {
        "name":         name,
        "bc":           bc,  "cont": cont,  "mom": mom,
        "best_val":     best_val,
        "elapsed":      elapsed,
        "history":      {k: np.array(v, dtype=np.float32) for k, v in history.items()},
        "residual_rms": res_rms,
        "pred_Ux":      pred_Ux.astype(np.float32),
        "pred_Uy":      pred_Uy.astype(np.float32),
        "pred_P":       pred_P.astype(np.float32),
        "n_params":     n_params,
    }

# %% [markdown]
# ## 7. Train all 6 configurations

# %%
configs = [
    {"name": "MLP",         "bc": False, "cont": False, "mom": False},
    {"name": "MLP + BC",    "bc": True,  "cont": False, "mom": False},
    {"name": "Cont",        "bc": False, "cont": True,  "mom": False},
    {"name": "Cont + BC",   "bc": True,  "cont": True,  "mom": False},
    {"name": "Mom",         "bc": False, "cont": True,  "mom": True},
    {"name": "Mom + BC",    "bc": True,  "cont": True,  "mom": True},
]
results = []
for cfg in configs:
    print(f"\n{'='*70}\n  CONFIG: {cfg['name']}\n{'='*70}")
    r = run_config(**cfg)
    results.append(r)

# Save lightweight summary so the comparison can be re-run without re-training
summary = []
for r in results:
    summary.append({
        "name":     r["name"],
        "bc":       r["bc"],   "cont": r["cont"],  "mom":     r["mom"],
        "best_val": r["best_val"],
        "elapsed":  r["elapsed"],
        "residual_rms": r["residual_rms"],
    })
_out_dir = pathlib.Path(_PROJECT_ROOT) / "figures_run"
_out_dir.mkdir(exist_ok=True)
with open(_out_dir / "phase_a_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
np.savez(_out_dir / "phase_a_predictions.npz",
         **{f"{r['name'].replace(' + ', '_').replace(' ', '_')}_Ux": r["pred_Ux"] for r in results},
         **{f"{r['name'].replace(' + ', '_').replace(' ', '_')}_Uy": r["pred_Uy"] for r in results},
         x_phys=x_phys, y_phys=y_phys,
         Ux_norm=Ux_norm, Uy_norm=Uy_norm,
         sensor_idx=sensor_idx)

print(f"\n{'='*70}")
print(f"All {len(results)} configurations trained.  Summary saved to {_out_dir.name}/phase_a_summary.json")
print(f"{'='*70}")
for r in results:
    print(f"  {r['name']:15s}  val MSE = {r['best_val']:.3e}  ({r['elapsed']:.1f}s)")

# %% [markdown]
# ## 8. Phase A comparison report
#
# Bar chart of val MSE for all six configurations, residual table, and a
# 7-panel head-to-head with the truth.

# %%
def plot_field(ax, x, y, c, *, cmap="viridis", vmin=None, vmax=None, title=""):
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c)
    tri = mtri.Triangulation(x, y)
    pts = np.stack([x[tri.triangles], y[tri.triangles]], axis=-1)
    edges = np.linalg.norm(np.diff(pts[:, [0, 1, 2, 0], :], axis=1), axis=-1)
    tri.set_mask(edges.max(axis=1) > 3.0 * np.median(edges.max(axis=1)))
    m = ax.tripcolor(tri, c, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)
    return m

# --- (a) Val MSE bar chart ---
fig, ax = plt.subplots(figsize=(10, 4.5))
names    = [r["name"] for r in results]
vals     = [r["best_val"] for r in results]
colors   = ['gray' if not (r["bc"] or r["cont"] or r["mom"]) else
            ('C0' if r["bc"] and not r["cont"] and not r["mom"] else
             'C1' if r["cont"] and not r["mom"] and not r["bc"] else
             'C2' if r["cont"] and not r["mom"] and r["bc"]  else
             'C3' if r["mom"] and not r["bc"] else 'C4')
            for r in results]
bars = ax.bar(names, vals, color=colors, edgecolor='black', linewidth=0.6)
ax.set_yscale('log')
ax.set_ylabel('held-out velocity MSE  (log)')
ax.set_title(f'Phase A: K={K} sensor inpainting across 6 configurations')
for bar, v in zip(bars, vals):
    ax.annotate(f'{v:.2e}', xy=(bar.get_x() + bar.get_width()/2, v),
                xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
ax.grid(True, axis='y', alpha=0.3, which='both')
plt.xticks(rotation=15)
fig.tight_layout()
plt.show()

# --- (b) Residual RMS table ---
print(f"\n{'config':15s}  {'val MSE':>10s}   {'cont RMS':>10s}  {'mom_x RMS':>10s}  {'mom_y RMS':>10s}")
print('-' * 72)
for r in results:
    rr = r["residual_rms"]
    print(f"{r['name']:15s}   {r['best_val']:.3e}    {rr['cont']:.2e}    {rr['mom_x']:.2e}    {rr['mom_y']:.2e}")

# --- (c) 7-panel reconstruction: truth + all 6 reconstructions ---
truth_mag = np.sqrt(Ux_norm**2 + Uy_norm**2)
panels = [("Truth |U|", truth_mag)] + [
    (r["name"], np.sqrt(r["pred_Ux"]**2 + r["pred_Uy"]**2)) for r in results
]
vmin = float(min(p[1].min() for p in panels))
vmax = float(max(p[1].max() for p in panels))

fig, axes = plt.subplots(1, 7, figsize=(26, 3.5), gridspec_kw={"wspace": 0.06})
for ax, (lbl, c) in zip(axes, panels):
    m = plot_field(ax, x_phys, y_phys, c, vmin=vmin, vmax=vmax, title=lbl)
    ax.scatter(x_phys[sensor_idx], y_phys[sensor_idx],
               s=10, facecolor='none', edgecolor='red', linewidths=0.6, zorder=5)
fig.colorbar(m, ax=axes.tolist(), fraction=0.012, pad=0.005, label='|U_norm|')
fig.suptitle(f'Phase A reconstruction across 6 configs (red rings = sensor locations, K={K})', y=1.04)
plt.show()

# --- (d) L2 vector-error fields, same 7 panels ---
err_panels = [("", None)] + [
    (r["name"], np.sqrt((r["pred_Ux"] - Ux_norm)**2 + (r["pred_Uy"] - Uy_norm)**2))
    for r in results
]
emax = float(max(p[1].max() for p in err_panels if p[1] is not None))

fig, axes = plt.subplots(1, 7, figsize=(26, 3.5), gridspec_kw={"wspace": 0.06})
axes[0].set_visible(False)
for ax, (lbl, c) in zip(axes[1:], err_panels[1:]):
    m = plot_field(ax, x_phys, y_phys, c, cmap='magma', vmin=0, vmax=emax, title=lbl + " err")
    ax.scatter(x_phys[sensor_idx], y_phys[sensor_idx],
               s=10, facecolor='none', edgecolor='cyan', linewidths=0.6, zorder=5)
fig.colorbar(m, ax=axes.tolist(), fraction=0.012, pad=0.005, label='L2 vec error')
fig.suptitle('Phase A pointwise L2 vector errors across 6 configs', y=1.04)
plt.show()

# --- (e) Loss-curves comparison: val MSE trajectories ---
fig, ax = plt.subplots(figsize=(10, 5))
epochs_v = np.arange(1, len(results[0]["history"]["val"]) + 1) * VAL_EVERY
for r in results:
    style = '-' if r["bc"] else '--'
    lw    = 2.0 if r["mom"] else (1.5 if r["cont"] else 1.0)
    ax.semilogy(epochs_v, r["history"]["val"], style, lw=lw, label=r["name"])
ax.set_xlabel('epoch'); ax.set_ylabel('val MSE  (log)')
ax.set_title('Validation trajectory across configurations')
ax.grid(True, which='both', alpha=0.3); ax.legend(ncol=2, fontsize=9, loc='best')
plt.show()

# %% [markdown]
# ## 9. Try this
#
# 1. **Tune LAMBDA_BC.**  Try 0.1, 1.0, 10.0.  Watch how the relative
#    weight of BC vs physics losses affects convergence.
# 2. **Drop one BC at a time.**  Modify `run_config` to allow per-BC
#    flags.  Does dropping the outlet zero-gradient hurt more than
#    dropping the wall no-slip?
# 3. **Vary K.**  Re-run all six configs at K=20 and K=200.  Does the
#    BC contribution scale with sparsity?
