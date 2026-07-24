"""
Training loops for the inpainting demo.

Two functions:
  train_mlp(...)  — data loss only (sensor MSE)
  train_pinn(...) — data + physics losses with per-term tracking

Both return a dict that includes the trained model, per-epoch loss history,
and a reconstructed-field prediction on a user-supplied evaluation grid.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from .config   import CaseGeometry, Hyperparams
from .model    import MLP
from .physics  import continuity_loss, inlet_bc_loss, wall_bc_loss, baffle_bc_loss, warmup_weight
from .sampling import sample_collocation, sample_inlet_bc, sample_wall_bc, sample_baffle_bc, to_tensors


@dataclass
class PhysicsWeights:
    """
    Per-term weights on the physics loss.  Set to 0 to disable a term.

    Defaults are tuned for the **inpainting regime** (K ~ 10–50 sensors),
    where physics is the primary information source and needs high weight.
    Research-scale multi-case surrogate training uses values ~100× smaller.
    """
    continuity: float = 1.0
    inlet:      float = 1.0
    wall:       float = 1.0
    baffle:     float = 1.0


@dataclass
class PhysicsSampling:
    n_colloc:        int = 4096
    n_inlet:         int = 256
    n_wall:          int = 512
    n_baffle_per:    int = 64
    resample_every:  int = 100


def _to_t(arr, device, grad=False):
    t = torch.as_tensor(arr, dtype=torch.float32, device=device)
    if arr.ndim == 1:
        t = t.unsqueeze(1)
    if grad:
        t = t.clone().requires_grad_(True)
    return t


def _build_sensor_tensors(sensors: dict, device: str):
    return {
        'x':  _to_t(sensors['x_norm'], device),
        'y':  _to_t(sensors['y_norm'], device),
        'Ux': _to_t(sensors['Ux_norm'], device),
        'Uy': _to_t(sensors['Uy_norm'], device),
    }


def _build_eval_tensors(evalp: dict, device: str):
    return {
        'x':  _to_t(evalp['x_norm'], device),
        'y':  _to_t(evalp['y_norm'], device),
        'Ux': _to_t(evalp['Ux_norm'], device),
        'Uy': _to_t(evalp['Uy_norm'], device),
    }


# ======================================================================
# Data-only MLP training
# ======================================================================

def train_mlp(
    split:  dict,
    geom:   CaseGeometry,
    hp:     Hyperparams,
    device: str,
    apply_mask: bool = False,
    verbose: bool = True,
) -> dict:
    """Train on sensor observations only.  No physics."""
    torch.manual_seed(hp.seed); np.random.seed(hp.seed)

    sens = _build_sensor_tensors(split['sensors'], device)
    evl  = _build_eval_tensors(split['eval'],    device)

    model = MLP(width=hp.width, depth=hp.depth).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=hp.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=hp.lr_patience)

    if verbose:
        print(f"MLP: {split['n_sensors']} sensors, {evl['x'].shape[0]:,} eval pts  "
              f"({model.num_params():,} params)")

    train_losses, val_losses = [], []
    best_val, best_state, patience = float('inf'), None, 0
    t0 = time.time()

    for ep in range(hp.epochs):
        model.train(); opt.zero_grad()
        pred = model(sens['x'], sens['y'], n=geom.n, a=geom.a, apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)
        loss = nn.functional.mse_loss(pred[:, 0:1], sens['Ux']) + \
               nn.functional.mse_loss(pred[:, 1:2], sens['Uy'])
        loss.backward(); opt.step(); sched.step(loss.item())
        train_losses.append(loss.item())

        if (ep + 1) % hp.val_every == 0:
            model.eval()
            with torch.no_grad():
                p = model(evl['x'], evl['y'], n=geom.n, a=geom.a, apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)
                vl = (nn.functional.mse_loss(p[:, 0:1], evl['Ux']) +
                      nn.functional.mse_loss(p[:, 1:2], evl['Uy'])).item()
            val_losses.append(vl)
            if vl < best_val:
                best_val, best_state, patience = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += hp.val_every
            if patience >= hp.early_stop_patience:
                if verbose: print(f"  Early stop at epoch {ep+1}")
                break

        if verbose and (ep + 1) % hp.print_every == 0:
            vl_now = val_losses[-1] if val_losses else float('nan')
            print(f"  ep {ep+1:5d}/{hp.epochs}  train {loss.item():.3e}  val {vl_now:.3e}  "
                  f"lr {opt.param_groups[0]['lr']:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    # Final eval
    model.eval()
    with torch.no_grad():
        p = model(evl['x'], evl['y'], n=geom.n, a=geom.a, apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)
        final_val = (nn.functional.mse_loss(p[:, 0:1], evl['Ux']) +
                     nn.functional.mse_loss(p[:, 1:2], evl['Uy'])).item()

    if verbose:
        print(f"  Done in {elapsed:.1f}s  |  val MSE {final_val:.3e}")

    return {
        'model':         model,
        'train_losses':  np.array(train_losses, dtype=np.float32),
        'val_losses':    np.array(val_losses,   dtype=np.float32),
        'final_val_mse': float(final_val),
        'elapsed_sec':   float(elapsed),
    }


# ======================================================================
# PINN training (data + physics)
# ======================================================================

def train_pinn(
    split:    dict,
    geom:     CaseGeometry,
    hp:       Hyperparams,
    device:   str,
    weights:  PhysicsWeights  = PhysicsWeights(),
    sampling: PhysicsSampling = PhysicsSampling(),
    apply_mask: bool = False,
    verbose: bool = True,
) -> dict:
    """Train with data loss + weighted physics losses.  Returns per-term history."""
    torch.manual_seed(hp.seed); np.random.seed(hp.seed)

    sens = _build_sensor_tensors(split['sensors'], device)
    evl  = _build_eval_tensors(split['eval'],      device)

    model = MLP(width=hp.width, depth=hp.depth).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=hp.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=hp.lr_patience)

    def _resample():
        return {
            'colloc': to_tensors(sample_collocation(geom, sampling.n_colloc,  seed=np.random.randint(1 << 30)),
                                 device, requires_grad=True),
            'inlet':  to_tensors(sample_inlet_bc  (geom, sampling.n_inlet,   seed=np.random.randint(1 << 30)), device),
            'wall':   to_tensors(sample_wall_bc   (geom, sampling.n_wall,    seed=np.random.randint(1 << 30)), device),
            'baffle': to_tensors(sample_baffle_bc (geom, sampling.n_baffle_per, seed=np.random.randint(1 << 30)), device),
        }

    batches = _resample()

    if verbose:
        print(f"PINN: {split['n_sensors']} sensors, {evl['x'].shape[0]:,} eval pts  ({model.num_params():,} params)")
        print(f"      colloc {batches['colloc']['x'].shape[0]:,}  wall {batches['wall']['x'].shape[0]:,}  "
              f"baffle {batches['baffle']['x'].shape[0]:,}  inlet {batches['inlet']['x'].shape[0]:,}")
        print(f"      weights: {weights}")

    history = {k: [] for k in ['data', 'continuity', 'inlet', 'wall', 'baffle', 'total', 'val']}
    best_val, best_state, patience = float('inf'), None, 0
    t0 = time.time()

    for ep in range(hp.epochs):
        model.train(); opt.zero_grad()

        # data loss
        pred = model(sens['x'], sens['y'], n=geom.n, a=geom.a,
                     apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)
        L_data = nn.functional.mse_loss(pred[:, 0:1], sens['Ux']) + \
                 nn.functional.mse_loss(pred[:, 1:2], sens['Uy'])

        # physics losses
        ramp = warmup_weight(ep, hp.epochs, frac=0.2)

        L_cont = torch.tensor(0.0, device=device)
        if weights.continuity > 0:
            L_cont = continuity_loss(model, batches['colloc']['x'], batches['colloc']['y'],
                                     geom, apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)

        L_in = torch.tensor(0.0, device=device)
        if weights.inlet > 0:
            L_in = inlet_bc_loss(model, batches['inlet']['x'], batches['inlet']['y'],
                                 geom, apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)

        L_wa = torch.tensor(0.0, device=device)
        if weights.wall > 0:
            L_wa = wall_bc_loss(model, batches['wall']['x'], batches['wall']['y'],
                                geom, apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)

        L_ba = torch.tensor(0.0, device=device)
        if weights.baffle > 0 and batches['baffle']['x'].numel() > 0:
            L_ba = baffle_bc_loss(model, batches['baffle']['x'], batches['baffle']['y'],
                                  geom, apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)

        L_phys = ramp * (weights.continuity * L_cont + weights.inlet * L_in +
                         weights.wall * L_wa + weights.baffle * L_ba)
        L_total = L_data + L_phys
        L_total.backward(); opt.step(); sched.step(L_total.item())

        if (ep + 1) % hp.val_every == 0:
            model.eval()
            with torch.no_grad():
                p = model(evl['x'], evl['y'], n=geom.n, a=geom.a,
                          apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)
                vl = (nn.functional.mse_loss(p[:, 0:1], evl['Ux']) +
                      nn.functional.mse_loss(p[:, 1:2], evl['Uy'])).item()
            history['data'].append(L_data.item())
            history['continuity'].append(L_cont.item())
            history['inlet'].append(L_in.item())
            history['wall'].append(L_wa.item())
            history['baffle'].append(L_ba.item())
            history['total'].append(L_total.item())
            history['val'].append(vl)

            if vl < best_val:
                best_val, best_state, patience = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += hp.val_every
            if patience >= hp.early_stop_patience:
                if verbose: print(f"  Early stop at epoch {ep+1}")
                break

        if verbose and (ep + 1) % hp.print_every == 0:
            print(f"  ep {ep+1:5d}  data {L_data.item():.3e}  cont {L_cont.item():.3e}  "
                  f"in {L_in.item():.3e}  wall {L_wa.item():.3e}  baf {L_ba.item():.3e}  "
                  f"val {history['val'][-1]:.3e}  ramp {ramp:.2f}")

        if sampling.resample_every > 0 and (ep + 1) % sampling.resample_every == 0:
            batches = _resample()

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        p = model(evl['x'], evl['y'], n=geom.n, a=geom.a,
                  apply_mask=apply_mask, sdf_alpha=hp.sdf_alpha)
        final_val = (nn.functional.mse_loss(p[:, 0:1], evl['Ux']) +
                     nn.functional.mse_loss(p[:, 1:2], evl['Uy'])).item()

    if verbose:
        print(f"  Done in {elapsed:.1f}s  |  val MSE {final_val:.3e}")

    return {
        'model':         model,
        'history':       {k: np.array(v, dtype=np.float32) for k, v in history.items()},
        'final_val_mse': float(final_val),
        'elapsed_sec':   float(elapsed),
    }


# ======================================================================
# Convenience: reconstruct field on arbitrary points
# ======================================================================

def reconstruct(model, x_norm, y_norm, geom: CaseGeometry, device: str,
                apply_mask: bool = False, sdf_alpha: float = 800.0) -> np.ndarray:
    """Run the model on arbitrary (x_norm, y_norm) points.  Returns (N, 2) numpy array."""
    x = _to_t(np.asarray(x_norm), device)
    y = _to_t(np.asarray(y_norm), device)
    model.eval()
    with torch.no_grad():
        p = model(x, y, n=geom.n, a=geom.a, apply_mask=apply_mask, sdf_alpha=sdf_alpha)
    return p.cpu().numpy()
