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
# # 03 - Inpainting with Full k-epsilon Closure
#
# Builds on `02_with_bc.py` by replacing the pre-trained scalar nu_t
# representation with a two-network architecture:
#
#   main_net : (x, y) -> (Ux, Uy, P_mod)
#   turb_net : (x, y) -> (z_k, z_eps)   decoded via 10**z to physical k, eps
#
# nu_t = C_mu k^2 / eps is computed from turb_net outputs by the closure
# formula (verified in B0 to machine precision against CFD's nut field).
#
# Two new configurations:
#
#   - "Mom + BC + turb"   turb_net frozen at warm-start values; momentum
#                         uses the closure-derived nu_t.  Isolates "does
#                         a better nu_t representation alone help?"
#   - "k-eps + BC"        turb_net trainable; full k and eps transport
#                         residuals enforced; wall-k BC added.  This is
#                         the headline full-closure config.

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
CASE_ID    = 834
DATA_DIR   = os.path.join(_PROJECT_ROOT, "data", "tiny")
DEVICE     = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

K          = 50
SEED       = 42

# Main MLP
WIDTH = 64; DEPTH = 4
# TurbNet
TURB_WIDTH = 32; TURB_DEPTH = 3
TURB_WARMUP_EPOCHS = 5000   # warm-start the turbulence net before main training

EPOCHS         = 5000
LR             = 1e-3
VAL_EVERY      = 50
PRINT_EVERY    = 1000

# Physics weights
LAMBDA_CONT      = 1.0
LAMBDA_MOMENT    = 0.001     # narrow working window 0.0005-0.005 (from 01_with_momentum)
LAMBDA_BC        = 3e-4      # velocity BC (inlet/outlet/wall/baffle) - from Phase A sweep
LAMBDA_K_EQ      = 1.0       # k-equation transport (normalized residual, see scaling below)
LAMBDA_EPS_EQ    = 1.0       # eps-equation transport (normalized)
LAMBDA_BC_WALLK  = 1e-3      # wall-k BC weight - similar order to velocity-BC weight

# Sampling
N_COLLOC  = 4096
N_INLET   = 256
N_OUTLET  = 256
N_WALL    = 512
N_BAFFLE  = 64
WARMUP_FRAC = 0.2

# Physical constants
NU_MOLECULAR  = 1.0e-6
INLET_WIDTH_M = 1.5

# k-eps closure constants (OpenFOAM defaults)
C_MU      = 0.09
C_1EPS    = 1.44
C_2EPS    = 1.92
SIGMA_K   = 1.0
SIGMA_EPS = 1.3

# Wall-k target: log10(k_wall) = -5 -> k = 1e-5 m^2/s^2
LOG10_K_WALL_TARGET = -5.0

# Residual normalization scales (from B0/B2 typical magnitudes).  Used so the
# loss weights stay close to O(1) instead of needing absurd values like 1e8.
K_RES_SCALE   = 1.07e-5   # m^2/s^3  ~ k_scale * U_inlet / L
EPS_RES_SCALE = 1.30e-7   # m^2/s^4  ~ eps_scale * U_inlet / L

print(f"device: {DEVICE}")
print(f"case: {CASE_ID},  K = {K} sensors,  {EPOCHS} epochs per config")
print(f"weights:  cont={LAMBDA_CONT}  mom={LAMBDA_MOMENT}  BC_vel={LAMBDA_BC}  "
      f"k_eq={LAMBDA_K_EQ}  eps_eq={LAMBDA_EPS_EQ}  BC_wallk={LAMBDA_BC_WALLK}")

# %% [markdown]
# ## 2. Load CFD data + sensor split + log-normalization for k, eps

# %%
geo_csv = os.path.join(DATA_DIR, "basin_geometries_summary.csv")
vel_csv = os.path.join(DATA_DIR, "results", f"case_{CASE_ID:03d}_full.csv")

geo_row = pd.read_csv(geo_csv).query(f"`Simulation Number` == {CASE_ID}").iloc[0]
L, W      = float(geo_row['L']),   float(geo_row['W'])
n_baf     = int  (geo_row['n']);   a_baf = float(geo_row['a'])
U_inlet   = float(geo_row['U_inlet'])

df = pd.read_csv(vel_csv).dropna(subset=['Ux','Uy','p','nut','k','epsilon'])
x_phys  = df['x'].values.astype(np.float32)
y_phys  = df['y'].values.astype(np.float32)
x_norm  = (df['x'].values / L).astype(np.float32)
y_norm  = (df['y'].values / W).astype(np.float32)
Ux_norm = (df['Ux'].values / U_inlet).astype(np.float32)
Uy_norm = (df['Uy'].values / U_inlet).astype(np.float32)
nut_phys = df['nut'].values.astype(np.float32)
k_phys   = df['k'].values.astype(np.float32)
eps_phys = df['epsilon'].values.astype(np.float32)

p_mod_phys   = df['p'].values + (2.0/3.0) * df['k'].values
p_mod_norm   = (p_mod_phys / U_inlet**2).astype(np.float32)
p_mod_offset = float(p_mod_norm.mean())
p_mod_cent   = p_mod_norm - p_mod_offset

N = len(x_norm)
print(f"Case {CASE_ID}: L={L:.0f}m, W={W:.0f}m, n={n_baf} baffles, U_inlet={U_inlet:.3f}m/s, N={N:,}")

# Log-normalization constants
LOG10_K_MEAN   = float(np.log10(k_phys).mean())
LOG10_K_STD    = float(np.log10(k_phys).std())
LOG10_EPS_MEAN = float(np.log10(eps_phys).mean())
LOG10_EPS_STD  = float(np.log10(eps_phys).std())
print(f"log10(k):       mean={LOG10_K_MEAN:.4f}  std={LOG10_K_STD:.4f}")
print(f"log10(epsilon): mean={LOG10_EPS_MEAN:.4f}  std={LOG10_EPS_STD:.4f}")

Z_K_WALL_TARGET = (LOG10_K_WALL_TARGET - LOG10_K_MEAN) / LOG10_K_STD
print(f"Z_K_WALL_TARGET = {Z_K_WALL_TARGET:.3f}  (corresponds to k_wall = {10**LOG10_K_WALL_TARGET:.1e})")

# Sensor split
rng        = np.random.default_rng(SEED)
sensor_idx = rng.choice(N, size=K, replace=False)
eval_idx   = np.setdiff1d(np.arange(N), sensor_idx)
def to_col(arr):
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(1)

sens_x  = to_col(x_norm        [sensor_idx]); sens_y  = to_col(y_norm[sensor_idx])
sens_Ux = to_col(Ux_norm       [sensor_idx]); sens_Uy = to_col(Uy_norm[sensor_idx])
sens_P  = to_col(p_mod_cent    [sensor_idx])
eval_x  = to_col(x_norm        [eval_idx]);   eval_y  = to_col(y_norm[eval_idx])
eval_Ux = to_col(Ux_norm       [eval_idx]);   eval_Uy = to_col(Uy_norm[eval_idx])

# Full-mesh tensors for warm-start of TurbNet
x_all = to_col(x_norm); y_all = to_col(y_norm)
log_k_norm_target   = to_col(((np.log10(k_phys)   - LOG10_K_MEAN)   / LOG10_K_STD).astype(np.float32)).squeeze(1).unsqueeze(1)
log_eps_norm_target = to_col(((np.log10(eps_phys) - LOG10_EPS_MEAN) / LOG10_EPS_STD).astype(np.float32)).squeeze(1).unsqueeze(1)

print(f"K = {K} sensors, {len(eval_idx):,} held-out evaluation points")

# %% [markdown]
# ## 3. BC samplers (same as 02_with_bc.py)

# %%
def sample_inlet(n=N_INLET, seed=1):
    rng = np.random.default_rng(seed); x_max = min(INLET_WIDTH_M/L, 1.0)
    return (rng.uniform(0, x_max, (n, 1)).astype(np.float32),
            np.zeros((n, 1), dtype=np.float32))
def sample_outlet(n=N_OUTLET, seed=4):
    rng = np.random.default_rng(seed); x_min = max(1.0-INLET_WIDTH_M/L, 0.0)
    x = rng.uniform(x_min, 1.0, (n, 1)).astype(np.float32)
    y = np.ones((n, 1), dtype=np.float32) if n_baf%2==0 else np.zeros((n, 1), dtype=np.float32)
    return x, y
def sample_walls(n=N_WALL, seed=2):
    rng = np.random.default_rng(seed); n_per = n//4; n_top = n - 3*n_per
    ix_max = min(INLET_WIDTH_M/L, 1.0); ox_min = max(1.0-INLET_WIDTH_M/L, 0.0)
    xs, ys = [], []
    xs.append(np.zeros((n_per, 1), dtype=np.float32));  ys.append(rng.uniform(0, 1, (n_per, 1)).astype(np.float32))
    xs.append(np.ones ((n_per, 1), dtype=np.float32)); ys.append(rng.uniform(0, 1, (n_per, 1)).astype(np.float32))
    bx = rng.uniform(ix_max, ox_min if n_baf%2==1 else 1.0, (n_per, 1))
    xs.append(bx.astype(np.float32)); ys.append(np.zeros((n_per, 1), dtype=np.float32))
    tx = rng.uniform(0.0, ox_min if n_baf%2==0 else 1.0, (n_top, 1))
    xs.append(tx.astype(np.float32)); ys.append(np.ones((n_top, 1), dtype=np.float32))
    return np.vstack(xs), np.vstack(ys)
def sample_baffles(n_per_baffle=N_BAFFLE, seed=3):
    rng = np.random.default_rng(seed); xs, ys = [], []
    for k in range(1, n_baf+1):
        bx = k/(n_baf+1)
        if k%2==1: y0,y1,tip = 0.0, a_baf, a_baf
        else:      y0,y1,tip = 1.0-a_baf, 1.0, 1.0-a_baf
        n_uni = int(n_per_baffle*0.7); n_tip = n_per_baffle - n_uni
        y_u = rng.uniform(y0, y1, (n_uni, 1)).astype(np.float32)
        y_t = np.clip(tip + rng.normal(0, 0.02, (n_tip, 1)), y0, y1).astype(np.float32)
        xs.append(np.full((n_uni, 1), bx, dtype=np.float32)); ys.append(y_u)
        xs.append(np.full((n_tip, 1), bx, dtype=np.float32)); ys.append(y_t)
    return np.vstack(xs), np.vstack(ys)

def _bc_to_t(x_np, y_np, grad=False):
    xt = torch.as_tensor(x_np, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y_np, dtype=torch.float32, device=DEVICE)
    if grad:
        xt = xt.clone().requires_grad_(True); yt = yt.clone().requires_grad_(True)
    return xt, yt

in_x,   in_y   = _bc_to_t(*sample_inlet ())
out_x,  out_y  = _bc_to_t(*sample_outlet(), grad=True)
wall_x, wall_y = _bc_to_t(*sample_walls ())
baf_x,  baf_y  = _bc_to_t(*sample_baffles())

# %% [markdown]
# ## 4. Architectures: MainNet (U, P) and TurbNet (k, eps)

# %%
class MainNet(nn.Module):
    """(x, y) -> (Ux, Uy, P_mod)."""
    def __init__(self, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

class TurbNet(nn.Module):
    """(x, y) -> normalized log-space (z_k, z_eps).
       Decode via 10**(z * sigma + mu) to get physical k, eps > 0."""
    def __init__(self, width=TURB_WIDTH, depth=TURB_DEPTH):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("log10_k_mean",   torch.tensor(LOG10_K_MEAN,   dtype=torch.float32))
        self.register_buffer("log10_k_std",    torch.tensor(LOG10_K_STD,    dtype=torch.float32))
        self.register_buffer("log10_eps_mean", torch.tensor(LOG10_EPS_MEAN, dtype=torch.float32))
        self.register_buffer("log10_eps_std",  torch.tensor(LOG10_EPS_STD,  dtype=torch.float32))
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))
    def k_eps(self, x, y):
        z = self.forward(x, y)
        log10_k   = z[:, 0:1] * self.log10_k_std   + self.log10_k_mean
        log10_eps = z[:, 1:2] * self.log10_eps_std + self.log10_eps_mean
        return torch.pow(10.0, log10_k), torch.pow(10.0, log10_eps)
    def nu_t(self, x, y):
        k, eps = self.k_eps(x, y)
        return C_MU * k * k / eps
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# %% [markdown]
# ## 5. Residuals (continuity, momentum, k-eq, eps-eq)

# %%
def grad1(out, var):
    if not out.requires_grad:
        return 0.0 * var
    g = torch.autograd.grad(out, var, torch.ones_like(out),
                            create_graph=True, retain_graph=True, allow_unused=True)[0]
    if g is None:
        return 0.0 * var
    return g

def continuity_residual(main_net, x, y):
    out = main_net(x, y)
    Ux, Uy = out[:, 0:1], out[:, 1:2]
    return grad1(Ux, x) / L + grad1(Uy, y) / W

def continuity_loss(main_net, x, y):
    return torch.mean(continuity_residual(main_net, x, y) ** 2)

def momentum_residual(main_net, turb_net, x, y):
    """Dimensionless Boussinesq momentum residual (R_x, R_y), using nu_t from
       turb_net via the closure C_mu k^2 / eps."""
    out = main_net(x, y)
    Ux, Uy, P = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    dUx_dx = grad1(Ux, x); dUx_dy = grad1(Ux, y)
    dUy_dx = grad1(Uy, x); dUy_dy = grad1(Uy, y)
    dP_dx  = grad1(P,  x); dP_dy  = grad1(P,  y)
    d2Ux_dx2 = grad1(dUx_dx, x); d2Ux_dy2 = grad1(dUx_dy, y)
    d2Uy_dx2 = grad1(dUy_dx, x); d2Uy_dy2 = grad1(dUy_dy, y)
    nu_t    = turb_net.nu_t(x, y)
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

def k_eq_residual(main_net, turb_net, x, y):
    """k-equation residual in physical units (m^2/s^3).  Returns the full residual;
       caller normalizes by K_RES_SCALE for the loss."""
    out_v = main_net(x, y)
    Ux_n, Uy_n = out_v[:, 0:1], out_v[:, 1:2]
    Ux = Ux_n * U_inlet; Uy = Uy_n * U_inlet
    k, eps = turb_net.k_eps(x, y)
    nu_t   = C_MU * k * k / eps
    # Velocity derivatives (chain rule from x_norm)
    dUx_dx = grad1(Ux, x) / L; dUx_dy = grad1(Ux, y) / W
    dUy_dx = grad1(Uy, x) / L; dUy_dy = grad1(Uy, y) / W
    # k derivatives
    dk_dx = grad1(k, x) / L;  dk_dy = grad1(k, y) / W
    d2k_dx2 = grad1(dk_dx, x) / L
    d2k_dy2 = grad1(dk_dy, y) / W
    # nu_t derivatives (for the diffusion-coefficient-gradient term)
    dnut_dx = grad1(nu_t, x) / L; dnut_dy = grad1(nu_t, y) / W
    # Convection
    conv = Ux * dk_dx + Uy * dk_dy
    # Diffusion: div[(nu + nu_t/sigma_k) grad k]
    D = NU_MOLECULAR + nu_t / SIGMA_K
    dD_dx = dnut_dx / SIGMA_K; dD_dy = dnut_dy / SIGMA_K
    diff = D * (d2k_dx2 + d2k_dy2) + dD_dx * dk_dx + dD_dy * dk_dy
    # Production
    P_k = nu_t * (2 * dUx_dx**2 + 2 * dUy_dy**2 + (dUx_dy + dUy_dx)**2)
    # Residual: LHS - RHS = conv - diff - P_k + eps
    return conv - diff - P_k + eps

def eps_eq_residual(main_net, turb_net, x, y):
    """eps-equation residual in physical units (m^2/s^4)."""
    out_v = main_net(x, y)
    Ux_n, Uy_n = out_v[:, 0:1], out_v[:, 1:2]
    Ux = Ux_n * U_inlet; Uy = Uy_n * U_inlet
    k, eps = turb_net.k_eps(x, y)
    nu_t   = C_MU * k * k / eps
    dUx_dx = grad1(Ux, x) / L; dUx_dy = grad1(Ux, y) / W
    dUy_dx = grad1(Uy, x) / L; dUy_dy = grad1(Uy, y) / W
    deps_dx = grad1(eps, x) / L; deps_dy = grad1(eps, y) / W
    d2eps_dx2 = grad1(deps_dx, x) / L
    d2eps_dy2 = grad1(deps_dy, y) / W
    dnut_dx = grad1(nu_t, x) / L; dnut_dy = grad1(nu_t, y) / W
    conv = Ux * deps_dx + Uy * deps_dy
    D = NU_MOLECULAR + nu_t / SIGMA_EPS
    dD_dx = dnut_dx / SIGMA_EPS; dD_dy = dnut_dy / SIGMA_EPS
    diff = D * (d2eps_dx2 + d2eps_dy2) + dD_dx * deps_dx + dD_dy * deps_dy
    P_k = nu_t * (2 * dUx_dx**2 + 2 * dUy_dy**2 + (dUx_dy + dUy_dx)**2)
    P_eps = C_1EPS * (eps / k) * P_k
    D_eps = C_2EPS * (eps * eps / k)
    return conv - diff - P_eps + D_eps

# %% [markdown]
# ## 6. BC losses

# %%
def bc_loss_inlet(main_net, x, y, target_Ux=0.0, target_Uy=1.0):
    out = main_net(x, y)
    return torch.mean((out[:, 0:1] - target_Ux) ** 2 + (out[:, 1:2] - target_Uy) ** 2)

def bc_loss_wall(main_net, x, y):
    out = main_net(x, y)
    return torch.mean(out[:, 0:1] ** 2 + out[:, 1:2] ** 2)

def bc_loss_baffle(main_net, x, y):
    return bc_loss_wall(main_net, x, y)

def bc_loss_outlet(main_net, x, y):
    out = main_net(x, y); Ux, Uy = out[:, 0:1], out[:, 1:2]
    return torch.mean(grad1(Ux, y) ** 2 + grad1(Uy, y) ** 2)

def bc_loss_wallk(turb_net, x, y, target_z_k=Z_K_WALL_TARGET):
    """Wall-k BC: pull z_k toward Z_K_WALL_TARGET (corresponds to k = 1e-5 m^2/s^2)."""
    z = turb_net(x, y)
    return torch.mean((z[:, 0:1] - target_z_k) ** 2)

# %% [markdown]
# ## 7. Warm-start the TurbNet (reused across all configs)
#
# We pre-train the TurbNet against log-normalized CFD k and eps targets, same as B1.

# %%
print("Warm-starting TurbNet against CFD k, eps targets...")
torch.manual_seed(SEED); np.random.seed(SEED)
turb_warm = TurbNet().to(DEVICE)
opt_w = torch.optim.Adam(turb_warm.parameters(), lr=1e-3)
t0 = time.time()
for ep in range(TURB_WARMUP_EPOCHS):
    opt_w.zero_grad()
    z = turb_warm(x_all, y_all)
    l_k   = F.mse_loss(z[:, 0:1], log_k_norm_target)
    l_eps = F.mse_loss(z[:, 1:2], log_eps_norm_target)
    loss  = l_k + l_eps
    loss.backward(); opt_w.step()
    if (ep + 1) % 1000 == 0:
        print(f"  warm-start ep {ep+1}: MSE_k={l_k.item():.3e}  MSE_eps={l_eps.item():.3e}")
print(f"  TurbNet warm-start done in {time.time()-t0:.1f}s.  Final loss = {loss.item():.3e}")
WARM_STATE = {k: v.detach().clone() for k, v in turb_warm.state_dict().items()}

# %% [markdown]
# ## 8. Training helper

# %%
rng_c = np.random.default_rng(0)
colloc_x = torch.tensor(rng_c.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32),
                        device=DEVICE, requires_grad=True)
colloc_y = torch.tensor(rng_c.uniform(0, 1, (N_COLLOC, 1)).astype(np.float32),
                        device=DEVICE, requires_grad=True)

def predict_field(main_net, x_n, y_n):
    main_net.eval()
    with torch.no_grad():
        p = main_net(to_col(x_n), to_col(y_n)).cpu().numpy()
    return p[:, 0], p[:, 1], p[:, 2]

def predict_turb(turb_net, x_n, y_n):
    """Return k_pred, eps_pred, nut_pred at the given (normalized) points."""
    turb_net.eval()
    with torch.no_grad():
        k, eps = turb_net.k_eps(to_col(x_n), to_col(y_n))
        nut    = C_MU * k * k / eps
    return (k.cpu().numpy().squeeze().astype(np.float64),
            eps.cpu().numpy().squeeze().astype(np.float64),
            nut.cpu().numpy().squeeze().astype(np.float64))

def run_config(name, *, bc, cont, mom, keps, wallk, freeze_turb):
    """Train one configuration.
       bc:           enable velocity BC losses (inlet+outlet+wall+baffle)
       cont:         enable continuity loss
       mom:          enable momentum loss (uses turb_net for nu_t)
       keps:         enable k-eq and eps-eq transport losses
       wallk:        enable wall-k BC loss
       freeze_turb:  if True, turb_net is loaded from WARM_STATE and frozen
                     (used for Mom+BC+turb).  If False, turb_net is loaded
                     from WARM_STATE and trained jointly with main_net."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    main_net = MainNet().to(DEVICE)
    turb_net = TurbNet().to(DEVICE)
    turb_net.load_state_dict(WARM_STATE)
    if freeze_turb:
        for p in turb_net.parameters(): p.requires_grad = False
        params = list(main_net.parameters())
    else:
        params = list(main_net.parameters()) + list(turb_net.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    n_params = main_net.num_params() + (0 if freeze_turb else turb_net.num_params())

    print(f"\n[{name}]  bc={bc}  cont={cont}  mom={mom}  keps={keps}  wallk={wallk}  "
          f"freeze_turb={freeze_turb}   ({n_params:,} trainable params)")

    history = {k: [] for k in
               ['data', 'cont', 'mom_x', 'mom_y', 'k_eq', 'eps_eq',
                'bc_in', 'bc_wall', 'bc_baf', 'bc_out', 'bc_wallk',
                'total', 'val']}
    best_val, best_state_main, best_state_turb = float("inf"), None, None
    t0 = time.time()

    for ep in range(EPOCHS):
        main_net.train()
        if not freeze_turb: turb_net.train()
        opt.zero_grad()

        # Data loss (always on; same 3-channel data as 01 and 02)
        pred = main_net(sens_x, sens_y)
        L_data = (F.mse_loss(pred[:, 0:1], sens_Ux) +
                  F.mse_loss(pred[:, 1:2], sens_Uy) +
                  F.mse_loss(pred[:, 2:3], sens_P))

        # Continuity
        L_cont = continuity_loss(main_net, colloc_x, colloc_y) if cont else torch.zeros((), device=DEVICE)

        # Momentum
        if mom:
            R_x, R_y = momentum_residual(main_net, turb_net, colloc_x, colloc_y)
            L_mx = torch.mean(R_x ** 2); L_my = torch.mean(R_y ** 2)
        else:
            L_mx = L_my = torch.zeros((), device=DEVICE)

        # k-eps transport
        if keps:
            R_k_phys   = k_eq_residual  (main_net, turb_net, colloc_x, colloc_y)
            R_eps_phys = eps_eq_residual(main_net, turb_net, colloc_x, colloc_y)
            # Normalize by physical scales so the loss weights stay O(1)
            L_k_eq   = torch.mean((R_k_phys   / K_RES_SCALE  ) ** 2)
            L_eps_eq = torch.mean((R_eps_phys / EPS_RES_SCALE) ** 2)
        else:
            L_k_eq = L_eps_eq = torch.zeros((), device=DEVICE)

        # Velocity BCs
        if bc:
            L_in  = bc_loss_inlet (main_net, in_x,   in_y)
            L_wa  = bc_loss_wall  (main_net, wall_x, wall_y)
            L_ba  = bc_loss_baffle(main_net, baf_x,  baf_y)
            L_out = bc_loss_outlet(main_net, out_x,  out_y)
        else:
            L_in = L_wa = L_ba = L_out = torch.zeros((), device=DEVICE)

        # k-wall BC
        if wallk and not freeze_turb:
            L_wallk = bc_loss_wallk(turb_net, wall_x, wall_y) + \
                      bc_loss_wallk(turb_net, baf_x,  baf_y)
        else:
            L_wallk = torch.zeros((), device=DEVICE)

        ramp = min(1.0, ep / (EPOCHS * WARMUP_FRAC)) if WARMUP_FRAC > 0 else 1.0
        L_phys     = ramp * (LAMBDA_CONT * L_cont +
                             LAMBDA_MOMENT * (L_mx + L_my) +
                             LAMBDA_K_EQ   * L_k_eq +
                             LAMBDA_EPS_EQ * L_eps_eq)
        L_boundary = ramp * (LAMBDA_BC       * (L_in + L_wa + L_ba + L_out) +
                             LAMBDA_BC_WALLK * L_wallk)
        L_total    = L_data + L_phys + L_boundary

        L_total.backward(); opt.step()

        if (ep + 1) % VAL_EVERY == 0:
            main_net.eval()
            with torch.no_grad():
                p = main_net(eval_x, eval_y)
                vl = (F.mse_loss(p[:, 0:1], eval_Ux) + F.mse_loss(p[:, 1:2], eval_Uy)).item()
            for key, val in [('data', L_data), ('cont', L_cont),
                             ('mom_x', L_mx), ('mom_y', L_my),
                             ('k_eq', L_k_eq), ('eps_eq', L_eps_eq),
                             ('bc_in', L_in), ('bc_wall', L_wa),
                             ('bc_baf', L_ba), ('bc_out', L_out),
                             ('bc_wallk', L_wallk), ('total', L_total)]:
                history[key].append(val.item())
            history['val'].append(vl)
            if vl < best_val:
                best_val = vl
                best_state_main = {k: v.detach().clone() for k, v in main_net.state_dict().items()}
                if not freeze_turb:
                    best_state_turb = {k: v.detach().clone() for k, v in turb_net.state_dict().items()}

        if (ep + 1) % PRINT_EVERY == 0:
            extras = []
            if cont:  extras.append(f"cont {L_cont.item():.2e}")
            if mom:   extras.append(f"mom_x {L_mx.item():.2e}  mom_y {L_my.item():.2e}")
            if keps:  extras.append(f"k_eq {L_k_eq.item():.2e}  eps_eq {L_eps_eq.item():.2e}")
            if bc:    extras.append(f"bc {L_in.item():.1e}/{L_wa.item():.1e}/{L_ba.item():.1e}/{L_out.item():.1e}")
            if wallk: extras.append(f"bc_wallk {L_wallk.item():.2e}")
            print(f"  ep {ep+1:5d}  data {L_data.item():.2e}  " + "  ".join(extras) + f"  val {history['val'][-1]:.2e}")

    main_net.load_state_dict(best_state_main)
    if not freeze_turb and best_state_turb is not None:
        turb_net.load_state_dict(best_state_turb)
    elapsed = time.time() - t0

    pred_Ux, pred_Uy, pred_P = predict_field(main_net, x_norm, y_norm)
    k_pred, eps_pred, nut_pred = predict_turb(turb_net, x_norm, y_norm)
    print(f"[{name}] DONE in {elapsed:.1f}s  |  best val MSE = {best_val:.3e}")
    # Quality of turb_net's nu_t at convergence
    nu_t_rel_err = float(np.median(np.abs(nut_pred - nut_phys) / nut_phys))
    print(f"  nu_t median rel err vs CFD = {nu_t_rel_err:.3e}")

    return {
        "name": name, "bc": bc, "cont": cont, "mom": mom, "keps": keps,
        "wallk": wallk, "freeze_turb": freeze_turb,
        "best_val": best_val, "elapsed": elapsed,
        "history": {k: np.array(v, dtype=np.float32) for k, v in history.items()},
        "pred_Ux": pred_Ux.astype(np.float32),
        "pred_Uy": pred_Uy.astype(np.float32),
        "pred_P":  pred_P.astype(np.float32),
        "k_pred":  k_pred,    "eps_pred": eps_pred,    "nut_pred": nut_pred,
        "nu_t_rel_err": nu_t_rel_err,
    }

# %% [markdown]
# ## 9. Train the two new configurations

# %%
results = []

# Config 1: Mom + BC + warm-started turb_net (turb frozen, no transport eqs)
print(f"\n{'='*70}\n  CONFIG: Mom + BC + turb\n{'='*70}")
r1 = run_config("Mom + BC + turb",
                bc=True, cont=True, mom=True, keps=False, wallk=False, freeze_turb=True)
results.append(r1)

# Config 2: Full k-eps + BC (turb trainable, transport eqs enforced, wall-k BC on)
print(f"\n{'='*70}\n  CONFIG: k-eps + BC\n{'='*70}")
r2 = run_config("k-eps + BC",
                bc=True, cont=True, mom=True, keps=True, wallk=True, freeze_turb=False)
results.append(r2)

# Save lightweight summary
_out_dir = pathlib.Path(_PROJECT_ROOT) / "figures_run"
_out_dir.mkdir(exist_ok=True)
summary = []
for r in results:
    summary.append({"name": r["name"],
                    "bc": r["bc"], "cont": r["cont"], "mom": r["mom"],
                    "keps": r["keps"], "wallk": r["wallk"], "freeze_turb": r["freeze_turb"],
                    "best_val": r["best_val"], "elapsed": r["elapsed"],
                    "nu_t_rel_err": r["nu_t_rel_err"]})
with open(_out_dir / "phase_b_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
np.savez(_out_dir / "phase_b_predictions.npz",
         x_phys=x_phys, y_phys=y_phys,
         Ux_norm=Ux_norm, Uy_norm=Uy_norm, k_phys=k_phys, eps_phys=eps_phys, nut_phys=nut_phys,
         sensor_idx=sensor_idx,
         **{f"{r['name'].replace(' + ', '_').replace(' ', '_').replace('-', 'm')}_Ux":  r["pred_Ux"]  for r in results},
         **{f"{r['name'].replace(' + ', '_').replace(' ', '_').replace('-', 'm')}_Uy":  r["pred_Uy"]  for r in results},
         **{f"{r['name'].replace(' + ', '_').replace(' ', '_').replace('-', 'm')}_k":   r["k_pred"]   for r in results},
         **{f"{r['name'].replace(' + ', '_').replace(' ', '_').replace('-', 'm')}_eps": r["eps_pred"] for r in results},
         **{f"{r['name'].replace(' + ', '_').replace(' ', '_').replace('-', 'm')}_nut": r["nut_pred"] for r in results})

print(f"\nPhase B summary saved.")
for r in results:
    print(f"  {r['name']:20s}  val MSE = {r['best_val']:.3e}  ({r['elapsed']:.1f}s)  "
          f"nu_t rel err = {r['nu_t_rel_err']:.2e}")

# %% [markdown]
# ## 10. Compare against Phase A (loaded from saved summary)

# %%
def plot_field(ax, x, y, c, *, cmap="viridis", vmin=None, vmax=None, title="", log=False):
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c)
    tri = mtri.Triangulation(x, y)
    pts = np.stack([x[tri.triangles], y[tri.triangles]], axis=-1)
    edges = np.linalg.norm(np.diff(pts[:, [0, 1, 2, 0], :], axis=1), axis=-1)
    tri.set_mask(edges.max(axis=1) > 3.0 * np.median(edges.max(axis=1)))
    c_plot = np.log10(np.maximum(c, 1e-12)) if log else c
    m = ax.tripcolor(tri, c_plot, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)
    return m

# Load Phase A summary for cross-phase comparison
phase_a_path = _out_dir / "phase_a_summary.json"
if phase_a_path.exists():
    with open(phase_a_path) as f:
        phase_a = json.load(f)
else:
    phase_a = []

all_names = [c["name"] for c in phase_a] + [r["name"] for r in results]
all_vals  = [c["best_val"] for c in phase_a] + [r["best_val"] for r in results]
colors    = (['lightgray'] * len(phase_a)) + ['C5', 'C8']
edges     = ['black'] * len(all_names)

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(all_names, all_vals, color=colors, edgecolor=edges, linewidth=0.6)
ax.set_yscale('log')
ax.set_ylabel('held-out velocity MSE (log)')
ax.set_title(f'Phase B headline: k-eps closure  vs  Phase A configs  (K={K}, single seed)')
for bar, v in zip(bars, all_vals):
    ax.annotate(f'{v:.2e}', xy=(bar.get_x() + bar.get_width()/2, v),
                xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
ax.grid(True, axis='y', alpha=0.3, which='both')
plt.xticks(rotation=15)
fig.tight_layout()
plt.show()

# Residual table
print(f"\n{'config':22s}  {'val MSE':>10s}   {'nu_t rel err':>12s}")
print('-' * 60)
for r in results:
    print(f"{r['name']:22s}   {r['best_val']:.3e}    {r['nu_t_rel_err']:.2e}")

# %% [markdown]
# ## 11. nu_t quality: CFD vs Phase A 01-style direct fit vs warm-started TurbNet vs trained k-eps

# %%
# Compute Phase A's direct-fit nu_t implicitly via the warm-start in 01_with_momentum
# style.  We already have the warm-started TurbNet that matches Mom+BC+turb's nu_t.
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), gridspec_kw={"wspace": 0.15})
nut_warm = results[0]["nut_pred"]   # Mom+BC+turb uses frozen warm-started turb
nut_keps = results[1]["nut_pred"]   # k-eps+BC trained turb_net

vmin = float(min(np.log10(nut_phys).min(), np.log10(nut_warm).min(), np.log10(nut_keps).min()))
vmax = float(max(np.log10(nut_phys).max(), np.log10(nut_warm).max(), np.log10(nut_keps).max()))

plot_field(axes[0], x_phys, y_phys, nut_phys, log=True, cmap='magma', vmin=vmin, vmax=vmax,
           title='CFD log10(nu_t)')
plot_field(axes[1], x_phys, y_phys, nut_warm, log=True, cmap='magma', vmin=vmin, vmax=vmax,
           title=f'Mom+BC+turb (warm-frozen TurbNet)\nrel err = {results[0]["nu_t_rel_err"]:.2e}')
plot_field(axes[2], x_phys, y_phys, nut_keps, log=True, cmap='magma', vmin=vmin, vmax=vmax,
           title=f'k-eps+BC (trained TurbNet)\nrel err = {results[1]["nu_t_rel_err"]:.2e}')

# Rel error field of the k-eps trained TurbNet
rel_err_field = np.abs(nut_keps - nut_phys) / np.maximum(nut_phys, 1e-12)
plot_field(axes[3], x_phys, y_phys, rel_err_field, cmap='viridis', vmin=0, vmax=1.0,
           title='k-eps+BC: |nu_t_pred - nu_t_CFD| / nu_t_CFD')
fig.suptitle('nu_t reconstruction quality across configurations', y=1.02)
plt.show()

# %% [markdown]
# ## 12. Velocity reconstruction — head-to-head with the two new configs and the Phase A winner

# %%
truth_mag = np.sqrt(Ux_norm**2 + Uy_norm**2)
# Load Phase A predictions for Mom+BC (best Phase A config)
pa_pred_path = _out_dir / "phase_a_predictions.npz"
if pa_pred_path.exists():
    pa = np.load(pa_pred_path)
    mom_bc_Ux = pa["Mom_BC_Ux"];  mom_bc_Uy = pa["Mom_BC_Uy"]
    mom_bc_mag = np.sqrt(mom_bc_Ux**2 + mom_bc_Uy**2)
    err_mom_bc = np.sqrt((mom_bc_Ux - Ux_norm)**2 + (mom_bc_Uy - Uy_norm)**2)
else:
    mom_bc_mag = err_mom_bc = None

panels = [("Truth |U|", truth_mag, None)]
if mom_bc_mag is not None:
    panels.append((f"Mom+BC (Phase A)\nval={2.7e-2 if mom_bc_mag is None else '':}", mom_bc_mag, err_mom_bc))
for r in results:
    mag = np.sqrt(r["pred_Ux"]**2 + r["pred_Uy"]**2)
    err = np.sqrt((r["pred_Ux"] - Ux_norm)**2 + (r["pred_Uy"] - Uy_norm)**2)
    panels.append((f"{r['name']}\nval={r['best_val']:.2e}", mag, err))

ncol = len(panels)
fig, axes = plt.subplots(2, ncol, figsize=(4*ncol, 6),
                         gridspec_kw={"wspace": 0.05, "hspace": 0.25})
vmin = float(min(p[1].min() for p in panels))
vmax = float(max(p[1].max() for p in panels))
emax = float(max((p[2].max() for p in panels if p[2] is not None), default=1))
for j, (lbl, mag, err) in enumerate(panels):
    m_r = plot_field(axes[0, j], x_phys, y_phys, mag, vmin=vmin, vmax=vmax, title=lbl)
    if err is None:
        axes[1, j].set_visible(False)
    else:
        m_e = plot_field(axes[1, j], x_phys, y_phys, err, cmap='magma', vmin=0, vmax=emax,
                         title=f"L2 vec err (RMS {np.sqrt((err**2).mean()):.2e})")
    axes[0, j].scatter(x_phys[sensor_idx], y_phys[sensor_idx], s=10, facecolor='none',
                       edgecolor='red', linewidths=0.6, zorder=5)
fig.suptitle(f'Velocity reconstruction comparison (K={K})', y=1.02)
plt.show()

# %% [markdown]
# ## 13. Per-term loss trajectories — k-eps+BC

# %%
fig, ax = plt.subplots(figsize=(11, 5))
h = results[1]["history"]
epochs_v = np.arange(1, len(h["val"]) + 1) * VAL_EVERY
for key, color, style in [
    ('data',     'C0', '-'), ('cont',     'C1', '-'),
    ('mom_x',    'C3', '-'), ('mom_y',    'C4', '-'),
    ('k_eq',     'C2', '-'), ('eps_eq',   'C6', '-'),
    ('bc_wallk', 'C7', '--'),
    ('val',      'k',  '-'),
]:
    v = np.asarray(h.get(key, []), dtype=float)
    if v.size == 0 or np.all(v == 0): continue
    safe = np.where(v > 0, v, np.nan)
    ax.semilogy(epochs_v, safe, style, lw=(2.2 if key=='val' else 1.4), color=color,
                label=key, marker='o' if key=='val' else None, ms=3)
ax.set_xlabel('epoch'); ax.set_ylabel('loss (log)')
ax.set_title('k-eps+BC training: per-term loss trajectories')
ax.grid(True, which='both', alpha=0.3); ax.legend(ncol=2, fontsize=9)
plt.show()
