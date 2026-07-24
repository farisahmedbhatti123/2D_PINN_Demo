"""
Physics loss terms for the inpainting demo.

All operate on a single-case MLP.  Coordinates are in normalized [0,1]² and
velocities in U_inlet units.  The continuity residual rescales via the
chain rule:  (1/L) ∂Ux/∂x_norm + (1/W) ∂Uy/∂y_norm = ∇·u (physical).
"""

import torch
from .config import CaseGeometry


def _forward(model, x, y, geom: CaseGeometry, apply_mask: bool, sdf_alpha: float):
    return model(x, y, n=geom.n, a=geom.a, apply_mask=apply_mask, sdf_alpha=sdf_alpha)


def continuity_residual(
    model, x, y, geom: CaseGeometry,
    apply_mask: bool = False, sdf_alpha: float = 800.0,
):
    """
    Pointwise continuity residual in physical units.  x and y must have
    requires_grad=True so autograd can differentiate the network output.

    Returns (N, 1) residual.  Mean(residual²) is the continuity loss.
    """
    vel = _forward(model, x, y, geom, apply_mask, sdf_alpha)
    Ux, Uy = vel[:, 0:1], vel[:, 1:2]

    dUx_dx = torch.autograd.grad(
        Ux, x, grad_outputs=torch.ones_like(Ux),
        create_graph=True, retain_graph=True,
    )[0]
    dUy_dy = torch.autograd.grad(
        Uy, y, grad_outputs=torch.ones_like(Uy),
        create_graph=True, retain_graph=True,
    )[0]
    return dUx_dx / geom.L + dUy_dy / geom.W


def continuity_loss(model, x, y, geom, apply_mask=False, sdf_alpha=800.0):
    r = continuity_residual(model, x, y, geom, apply_mask, sdf_alpha)
    return torch.mean(r ** 2)


def inlet_bc_loss(
    model, x, y, geom,
    target_Ux: float = 0.0, target_Uy: float = 1.0,
    apply_mask: bool = False, sdf_alpha: float = 800.0,
):
    """
    Inlet BC: at y_norm=0 and x_norm ∈ [0, INLET_WIDTH/L],
        Ux_norm = 0,  Uy_norm = 1  (flow enters in +y at inlet velocity).
    """
    vel = _forward(model, x, y, geom, apply_mask, sdf_alpha)
    Ux, Uy = vel[:, 0:1], vel[:, 1:2]
    return torch.mean((Ux - target_Ux) ** 2 + (Uy - target_Uy) ** 2)


def wall_bc_loss(model, x, y, geom, apply_mask=False, sdf_alpha=800.0):
    """No-slip on outer walls: u = 0 at sampled wall points."""
    vel = _forward(model, x, y, geom, apply_mask, sdf_alpha)
    return torch.mean(vel ** 2)


def baffle_bc_loss(model, x, y, geom, apply_mask=False, sdf_alpha=800.0):
    """No-slip on baffle surfaces (soft; hard version = apply_mask=True)."""
    vel = _forward(model, x, y, geom, apply_mask, sdf_alpha)
    return torch.mean(vel ** 2)


def warmup_weight(epoch: int, total_epochs: int, frac: float = 0.2) -> float:
    """Linear ramp 0→1 over the first `frac` of training; then 1.0."""
    if total_epochs <= 0 or frac <= 0:
        return 1.0
    return float(min(1.0, epoch / (total_epochs * frac)))
