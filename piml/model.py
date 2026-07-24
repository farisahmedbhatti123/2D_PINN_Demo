"""
Network + SDF baffle mask.

For single-case inpainting the network is as small as it gets:
    (x_norm, y_norm) ∈ [0,1]² → (Ux_norm, Uy_norm)

No geometry conditioning on the inputs — there's only one geometry.  The
baffle mask, when used, lives outside the network as a post-multiplication
(notebook 05).  This keeps the MLP module itself trivially readable.
"""

import torch
import torch.nn as nn


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:
            pass
    return "cpu"


def sdf_mask(
    x, y, n_baffles: int, a: float,
    alpha: float = 800.0, gamma: float = 20.0, n_max: int = 8,
):
    """
    Differentiable mask D(x, y) ∈ [0, 1] based on signed distance to baffle
    line segments.  Exactly 0 on the baffle surface, ~1 away from baffles.

    Baffle convention (in normalized coords, both in [0, 1]):
        baffle k sits at x = k / (n_baffles + 1)
        odd k: extends from y=0 to y=a   (rises from the bottom)
        even k: extends from y=1-a to y=1 (hangs from the top)

    Args:
        x, y: (N, 1) normalized coordinates, each in [0, 1]
        n_baffles, a: scalar geometry parameters for this case
        alpha: mask sharpness.  Large = tight wall.  Default 800.
        gamma: unused in single-case (kept for signature symmetry).
        n_max: upper bound on baffle count (default 8).

    Returns:
        D: (N, 1) mask value, differentiable w.r.t. x, y.
    """
    if n_baffles <= 0:
        return torch.ones_like(x)

    device, dtype = x.device, x.dtype
    per_baffle = []
    for k in range(1, n_baffles + 1):
        bx = k / (n_baffles + 1)
        if k % 2 == 1:         # odd: rises from y=0
            y_start, y_end = 0.0, a
        else:                  # even: hangs from y=1
            y_start, y_end = 1.0 - a, 1.0

        dx = x - bx
        y_clamped = torch.clamp(y, min=y_start, max=y_end)
        dy = y - y_clamped
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-14)
        per_baffle.append(torch.tanh(alpha * dist))

    D = torch.ones_like(x)
    for m in per_baffle:
        D = D * m
    return D


class MLP(nn.Module):
    """
    Plain (x, y) → (Ux, Uy) MLP with Tanh activations.

    Tanh is chosen over ReLU because the continuity residual needs a smooth
    second derivative (autograd twice), and ReLU's is zero a.e.

    Optional baffle mask: pass `apply_mask=True` to multiply the output by
    the SDF mask — enforces u = 0 on baffles *by construction* (notebook 05).
    """

    def __init__(self, width: int = 64, depth: int = 4, out_dim: int = 2):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, *, n: int = 0, a: float = 0.0,
                apply_mask: bool = False, sdf_alpha: float = 800.0):
        xy = torch.cat([x, y], dim=1)        # (N, 2)
        vel = self.net(xy)                   # (N, 2)
        if apply_mask and n > 0:
            m = sdf_mask(x, y, n, a, alpha=sdf_alpha)
            vel = vel * m
        return vel

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
