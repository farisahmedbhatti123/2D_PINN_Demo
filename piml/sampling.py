"""
Collocation + BC point samplers (single-case).

For inpainting we generate these anew from the fixed case geometry — no
per-case parameterization.  Returned dicts carry the tensors the training
loop needs; `to_tensors` moves them to GPU and sets requires_grad on xy.
"""

from typing import Dict
import numpy as np
import torch

from .config import CaseGeometry, INLET_WIDTH_M


def sample_collocation(geom: CaseGeometry, n_points: int = 4096, seed: int = 0) -> dict:
    """Uniform interior collocation points in normalized coords."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n_points, 1)).astype(np.float32)
    y = rng.uniform(0, 1, (n_points, 1)).astype(np.float32)
    return {'x': x, 'y': y}


def sample_inlet_bc(geom: CaseGeometry, n_points: int = 256, seed: int = 1) -> dict:
    """Points on the inlet: y=0, x ∈ [0, INLET_WIDTH/L]."""
    rng = np.random.default_rng(seed)
    x_max = min(INLET_WIDTH_M / geom.L, 1.0)
    x = rng.uniform(0, x_max, (n_points, 1)).astype(np.float32)
    y = np.zeros((n_points, 1), dtype=np.float32)
    return {'x': x, 'y': y}


def sample_wall_bc(geom: CaseGeometry, n_points: int = 512, seed: int = 2) -> dict:
    """
    Points on the four outer walls, *excluding* inlet and outlet openings.

    Outlet location convention:
      - n odd:  outlet on the bottom wall at the x=1 side (serpentine ends down)
      - n even: outlet on the top wall at the x=1 side    (serpentine ends up)
    """
    rng = np.random.default_rng(seed)
    n_per = n_points // 4
    n_left, n_right, n_bottom = n_per, n_per, n_per
    n_top = n_points - n_left - n_right - n_bottom

    inlet_x_max  = min(INLET_WIDTH_M / geom.L, 1.0)
    outlet_x_min = max(1.0 - INLET_WIDTH_M / geom.L, 0.0)

    x_parts, y_parts = [], []
    # Left wall
    x_parts.append(np.zeros((n_left, 1), dtype=np.float32))
    y_parts.append(rng.uniform(0, 1, (n_left, 1)).astype(np.float32))
    # Right wall
    x_parts.append(np.ones((n_right, 1), dtype=np.float32))
    y_parts.append(rng.uniform(0, 1, (n_right, 1)).astype(np.float32))
    # Bottom wall
    if geom.n % 2 == 1:
        bot_x = rng.uniform(inlet_x_max, outlet_x_min, (n_bottom, 1))
    else:
        bot_x = rng.uniform(inlet_x_max, 1.0, (n_bottom, 1))
    x_parts.append(bot_x.astype(np.float32))
    y_parts.append(np.zeros((n_bottom, 1), dtype=np.float32))
    # Top wall
    if geom.n % 2 == 0:
        top_x = rng.uniform(0.0, outlet_x_min, (n_top, 1))
    else:
        top_x = rng.uniform(0.0, 1.0, (n_top, 1))
    x_parts.append(top_x.astype(np.float32))
    y_parts.append(np.ones((n_top, 1), dtype=np.float32))

    return {'x': np.vstack(x_parts), 'y': np.vstack(y_parts)}


def sample_baffle_bc(geom: CaseGeometry, n_per_baffle: int = 64, seed: int = 3) -> dict:
    """
    Points directly on each baffle segment (zero-thickness).

    70% uniformly along the segment, 30% clustered near the tip (high
    gradient region).  Returns empty arrays when n_baffles = 0.
    """
    if geom.n <= 0:
        return {'x': np.zeros((0, 1), dtype=np.float32),
                'y': np.zeros((0, 1), dtype=np.float32)}

    rng = np.random.default_rng(seed)
    x_parts, y_parts = [], []
    for k in range(1, geom.n + 1):
        bx = k / (geom.n + 1)
        if k % 2 == 1:
            y_start, y_end, tip = 0.0, geom.a, geom.a
        else:
            y_start, y_end, tip = 1.0 - geom.a, 1.0, 1.0 - geom.a

        n_uni = int(n_per_baffle * 0.7)
        n_tip = n_per_baffle - n_uni

        y_uni = rng.uniform(y_start, y_end, (n_uni, 1)).astype(np.float32)
        y_tip = np.clip(tip + rng.normal(0, 0.02, (n_tip, 1)),
                        y_start, y_end).astype(np.float32)
        x_parts.append(np.full((n_uni, 1), bx, dtype=np.float32)); y_parts.append(y_uni)
        x_parts.append(np.full((n_tip, 1), bx, dtype=np.float32)); y_parts.append(y_tip)

    return {'x': np.vstack(x_parts), 'y': np.vstack(y_parts)}


def to_tensors(batch: dict, device: str, requires_grad: bool = False) -> Dict[str, torch.Tensor]:
    """Move numpy batch to device.  Set requires_grad on x,y for collocation batches."""
    out = {}
    for k, v in batch.items():
        t = torch.as_tensor(v, dtype=torch.float32, device=device)
        if requires_grad and k in ('x', 'y'):
            t = t.clone().requires_grad_(True)
        out[k] = t
    return out
