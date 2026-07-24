"""
Data loading for the sparse-reconstruction demo.

One case, read once.  Then pick K "sensors" at random positions (train
subset) and hold out the rest as the ground-truth evaluation grid.

Normalization policy:
    x_norm = x/L,  y_norm = y/W    → both in [0, 1]
    Ux_norm = Ux/U_inlet, Uy_norm = Uy/U_inlet
This keeps the network inputs/outputs O(1), which stabilizes autograd
through the continuity residual.
"""

import os
from dataclasses import asdict
from typing import Tuple
import numpy as np
import pandas as pd

from .config import DataPaths, CaseGeometry


def load_case(paths: DataPaths = DataPaths()) -> Tuple[CaseGeometry, dict]:
    """
    Load one case.  Returns (CaseGeometry, field_dict).

    `field_dict` contains the full CFD-mesh data in normalized coords:
        {
          'x_norm', 'y_norm':   (N,) point positions
          'Ux_norm', 'Uy_norm': (N,) velocity components
          'x_phys',  'y_phys':  (N,) physical coords (for plotting only)
        }
    """
    geo = pd.read_csv(paths.geo_csv)
    row = geo[geo['Simulation Number'] == paths.case_id]
    if len(row) == 0:
        raise ValueError(f"case_id {paths.case_id} not in {paths.geo_csv}")
    row = row.iloc[0]

    geom = CaseGeometry(
        case_id = int(paths.case_id),
        L       = float(row['L']),
        W       = float(row['W']),
        n       = int(row['n']),
        a       = float(row['a']),
        U_inlet = float(row['U_inlet']),
    )

    csv = os.path.join(paths.results_dir, f'case_{paths.case_id:03d}_velocity.csv')
    vel = pd.read_csv(csv)

    valid = ~(np.isnan(vel['Ux']) | np.isnan(vel['Uy']))
    vel = vel[valid]

    field = {
        'x_phys':  vel['x'].values.astype(np.float32),
        'y_phys':  vel['y'].values.astype(np.float32),
        'x_norm':  (vel['x'].values / geom.L).astype(np.float32),
        'y_norm':  (vel['y'].values / geom.W).astype(np.float32),
        'Ux_norm': (vel['Ux'].values / geom.U_inlet).astype(np.float32),
        'Uy_norm': (vel['Uy'].values / geom.U_inlet).astype(np.float32),
    }
    print(f"Loaded case {geom.case_id}: n={geom.n}, L={geom.L:.0f}, W={geom.W:.0f}, "
          f"U_inlet={geom.U_inlet:.3f} m/s, {len(field['x_norm']):,} CFD points.")
    return geom, field


def make_sensor_split(field: dict, n_sensors: int, seed: int = 42) -> dict:
    """
    Split the full CFD field into:
      - sensors: K randomly chosen points — these are the *only* observations
                 the network sees during training.
      - eval:    all OTHER points — held out for reconstruction error assessment.

    Returns a dict with:
        sensor_idx: (K,) indices into field arrays
        eval_idx:   (N-K,) complementary indices
        sensors:    {'x_norm', 'y_norm', 'Ux_norm', 'Uy_norm'}
        eval:       same keys, over held-out points
    """
    N = len(field['x_norm'])
    if n_sensors > N:
        raise ValueError(f"asked for {n_sensors} sensors but only {N} CFD points available")

    rng        = np.random.default_rng(seed)
    sensor_idx = rng.choice(N, size=n_sensors, replace=False)
    eval_idx   = np.setdiff1d(np.arange(N), sensor_idx, assume_unique=False)

    def sub(idx):
        return {k: field[k][idx] for k in ('x_norm', 'y_norm', 'Ux_norm', 'Uy_norm',
                                            'x_phys', 'y_phys')}

    return {
        'sensor_idx': sensor_idx,
        'eval_idx':   eval_idx,
        'sensors':    sub(sensor_idx),
        'eval':       sub(eval_idx),
        'n_sensors':  int(n_sensors),
        'seed':       int(seed),
    }
