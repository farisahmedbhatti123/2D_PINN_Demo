"""
Configuration for the inpainting PIML teaching demo.
"""

import os
from dataclasses import dataclass, field

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Physical constants ---
INLET_WIDTH_M  = 1.5   # inlet/outlet opening width (m)
NU_MOLECULAR   = 1.0e-6


@dataclass
class DataPaths:
    geo_csv:     str = os.path.join(_PROJECT_ROOT, "data", "tiny", "basin_geometries_summary.csv")
    results_dir: str = os.path.join(_PROJECT_ROOT, "data", "tiny", "results")
    case_id:     int = 834     # n=4, L=37, W=18 — default demo case
    # Alternative bundled case: 267 (n=1, simpler flow)


@dataclass
class CaseGeometry:
    """
    Fixed geometric + flow parameters for a single case.  Populated by
    `data.load_case`.  Keeps downstream code clean — no more passing
    L, W, n, a, U_inlet as seven separate arguments.
    """
    case_id:  int
    L:        float       # basin length (m)
    W:        float       # basin width (m)
    n:        int         # baffle count
    a:        float       # baffle projection fraction in [0, 1]
    U_inlet:  float       # inlet velocity (m/s)


@dataclass
class Hyperparams:
    """
    Training knobs.  Defaults are tuned for ~1 minute laptop-GPU runs on the
    bundled case (~4500 CFD points, we train on a tiny fraction of them).
    """
    # network
    width:  int = 64
    depth:  int = 4
    out_dim: int = 2       # (Ux_norm, Uy_norm)

    # optimization
    epochs:      int   = 3000
    lr:          float = 1e-3
    lr_patience: int   = 300
    seed:        int   = 42

    # validation + early stopping
    val_every:           int = 50
    print_every:         int = 500
    early_stop_patience: int = 1000

    # baffle SDF mask (only used when notebook 05 is run)
    sdf_alpha: float = 800.0
    sdf_gamma: float = 20.0
