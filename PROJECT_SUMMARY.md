# 2D PINN Inpainting — Project Summary

A self-contained example for the Applied AI Summer Workshop. Walks through
sparse-sensor velocity-field reconstruction using physics-informed neural
networks (PINNs) on a 2D CFD case.

## The problem

Given a CFD simulation of a contact basin (4,575 mesh cells, serpentine
flow around 4 baffles), pretend you only have K=50 sparse "sensor"
observations of velocity and pressure. Reconstruct the full field
everywhere else. With enough sensors a plain regression network does
fine — the regime where PINNs matter is the moderate-sparse zone where
50 observations under-determine a 4,575-point field and physics has to
fill the gaps.

This is the classic Raissi-style inpainting setup, applied to a
turbulent-flow case (Re ≈ 4.5×10⁶) with RANS k-ε closure.

## What's in here

```
2D_PINN_Training/
  PROJECT_SUMMARY.md                              <- this file
  README.md                                        <- original project README
  setup_check.py                                   <- environment smoke test
  requirements.txt
  piml/                                            <- modular library (used by older notebooks)
    config.py, data.py, model.py, physics.py, sampling.py, train.py, plotting.py
  data/tiny/
    basin_geometries_summary.csv                   <- case parameters (L, W, n, U_inlet, ...)
    results/case_267_velocity.csv                  <- 1-baffle case
    results/case_834_velocity.csv                  <- 4-baffle case (default)
    results/case_834_full.csv                      <- with p, nut, k, epsilon
  case_834/                                        <- raw OpenFOAM case directory
    velocity_backup/808/                           <- the steady-state RANS solution
    constant/, system/, 0/                         <- standard OpenFOAM directories
  scripts/extract_openfoam_fields.py               <- parses OpenFOAM ASCII files -> CSV
  notebooks/
    00_inpainting_complete.{py,ipynb}              <- the main self-contained walkthrough
    01_with_momentum.{py,ipynb}                    <- the same problem + RANS momentum
    01_sparse_data_setup.{py,ipynb}                <- (legacy) staged version part 1
    02_mlp_from_sensors.{py,ipynb}                 <- (legacy) staged version part 2
    03_add_continuity.{py,ipynb}                   <- (legacy) staged version part 3
    04_sensor_sweep.{py,ipynb}                     <- K-vs-improvement sweep with error bars
```

## Pedagogical progression

The example builds in three steps. Each step uses the *same* sensor data
and the *same* network architecture — only the loss function changes.

| step | physics enforced | result on K=50 case |
|---|---|---|
| Data-only MLP | none | baseline held-out MSE |
| + continuity loss | ∇·u = 0 | ~28% lower held-out error (range +16% to +42% across seeds) |
| + RANS momentum | full Boussinesq closure | further reduction (single-seed; softens under multi-seed — see caveat below) |

The progression shows that each piece of additional physics buys a further
reduction in reconstruction error, with no single piece dominating. Treat the
continuity figure as "about a quarter to a third," not a fixed number — it
varies ~2.5× across which 50 points you happen to sense (see `README.md` and
`ADVANCED_PHYSICS_GUIDE.md` for the honest multi-seed reading).

## Two ways to use the code

**Self-contained notebooks (recommended for the workshop):**

- `notebooks/00_inpainting_complete.{py,ipynb}` — full walkthrough,
  data-only MLP through continuity-PINN, ~500 lines, zero imports from
  `piml/`. Every layer of the network and every loss term is in front of
  the student.
- `notebooks/01_with_momentum.{py,ipynb}` — adds full RANS momentum
  with Boussinesq closure. ~600 lines, also self-contained. Builds on
  `00`'s structure with a new pre-trained `ν_t` representation and a
  more elaborate residual function.

**Modular library version (legacy, more compact):**

- `notebooks/01_sparse_data_setup.py` → `notebooks/03_add_continuity.py`
  use the `piml/` library and are shorter per file (~150 lines each),
  but require students to navigate multiple files to understand the full
  picture.

For students, the self-contained `00` and `01` are the canonical entry
points. The library version is still useful if you want a shorter
classroom demo or are extending the code for research.

## Key results

The continuity figure below is the re-confirmed multi-seed result (case 834,
K=50, 5000 epochs, averaged over 3 sensor seeds). The momentum row is a single
illustrative seed and is expected to soften under the same multi-seed treatment
— treat it as illustrative, not authoritative. There's a multi-seed sweep in
`04_sensor_sweep.py` if you want error bars. The MSE magnitudes below are
representative single-seed values; the headline percentages are what to quote.

### Reconstruction quality (held-out velocity MSE)

```
MLP (data only)                  baseline
PINN  + continuity               ~28% lower vs MLP   (range +16% to +42% across seeds)
PINN  + continuity + momentum    further reduction   (single-seed; softens multi-seed)
```

### Where the PINN wins (head-to-head)

The "money plot" in section 14 of `00` (and the equivalent in `01`) shows
six panels: truth, MLP recon, PINN recon, MLP error, PINN error, and a
per-point "improvement" panel (blue where PINN wins). At K=50, PINN beats
MLP at ~61% of held-out points. Wins are concentrated in the sensor-sparse
interior; some losses near sensors where the MLP can match exactly.

### How the bright error bands evolve

The MLP's worst errors live in the high-velocity serpentine channels (not
in the sensor gaps as one might naively expect). Continuity alone does
little for those bands because it's a kinematic constraint. Adding
momentum specifically darkens them — the network can no longer set
velocity magnitudes arbitrarily, because momentum couples them to the
pressure gradient.

## Important caveats & known issues

1. **λ_momentum is in a narrow working window.** Set `LAMBDA_MOMENT =
   0.001`. An order of magnitude up (0.01) gives basically no benefit,
   two orders of magnitude up (0.1) hurts, and 1.0 catastrophically
   breaks training. This is the single most fragile knob in the
   momentum notebook.

2. **The `nut_model` representation is imperfect.** The small NN that
   fits ν_t to the CFD data has RMSE ~85% of the field's standard
   deviation — adequate for the gradient signal but not great. A bigger
   net or more epochs would tighten this and might improve the momentum
   PINN's headline number further. Quick improvement worth trying.

3. **Pressure has a gauge.** The CFD has a specific pressure reference
   (set by the outlet boundary condition in OpenFOAM). The PINN's
   predicted pressure can be off by a constant. We deal with this by
   subtracting the mean of `P_mod_norm` before training, but this
   means the network's `P_mod` output is centered around zero, not
   physical.

4. **RANS closure is taken as a given.** We use CFD's `ν_t` field as
   input — we do NOT enforce the k-equation or ε-equation. So we're
   doing "RANS momentum" but bypassing the actual closure equations.
   Adding closure transport equations would be the natural next step
   for a research project.

5. **Reynolds-stress simplification.** We use the Boussinesq closure
   (eddy-viscosity model), which assumes Reynolds stresses are
   proportional to mean strain rate. For some flows this is wrong;
   for the serpentine basin it's the same assumption the CFD solver
   already made, so consistency is fine.

6. **Single-case demo.** All notebooks operate on one geometry. The
   sensor sweep in `04_sensor_sweep.py` varies K but not geometry. A
   multi-case version would test generalization across topologies.

## Suggested classroom exercises

In rough order of complexity:

1. **K sweep** (already in `04`): see where physics stops helping.
2. **λ sweep** for both physics terms — find the sweet spot
   empirically, plot val MSE vs λ on log-log.
3. **Activation function swap** — replace Tanh with ReLU (and observe
   the physics loss going to noise because ReLU's second derivative is
   zero) or SiLU/GELU (which should still work).
4. **Drop the cross-terms** in `momentum_residual` to reduce full
   Boussinesq back to the simplified Laplacian form — show the result
   degrades.
5. **Different case** — switch `CASE_ID` to 267 (simpler n=1 case);
   re-extract the full CSV; verify the story holds.
6. **Smoothness baseline** — replace continuity with a generic
   smoothness penalty (sum of squared first-derivatives of u). If it
   gives similar results, then "continuity" specifically isn't doing
   identifiable work beyond regularization. If continuity beats
   smoothness, then physics is doing real work.

## Things explicitly not done

- No boundary condition losses (inlet/wall/baffle). The original
  `piml/` library has these in `piml/physics.py`, but the simplified
  walkthrough notebooks don't use them.
- No hard SDF mask for baffles. Baffles are implicit in the data; the
  network learns their effect.
- No Fourier features or sinusoidal activations (SIREN). The Tanh MLP
  has a known spectral bias against high frequencies that probably
  caps the achievable accuracy in shear layers.
- No transient (time-dependent) version. Everything is steady-state.

## How the data was generated

The CFD data comes from a steady-state incompressible RANS simulation
in OpenFOAM (`simpleFoam` solver, k-ε turbulence model). The full
OpenFOAM case is in `case_834/`. The Python extractor
`scripts/extract_openfoam_fields.py` parses OpenFOAM's ASCII field
files and writes `data/tiny/results/case_834_full.csv` with columns
`x, y, Ux, Uy, p, nut, k, epsilon`. To extract another case from a
different OpenFOAM directory, run:

```
python scripts/extract_openfoam_fields.py \
  --case <openfoam_case_dir> \
  --time <time_subdir, e.g. velocity_backup/808> \
  --case_id <numeric_id>
```

The extractor cross-checks against the existing velocity CSV when
asked, confirming cell ordering matches.

## Quick start

```bash
pip install -r requirements.txt
python setup_check.py                                   # smoke test
jupyter notebook notebooks/00_inpainting_complete.ipynb # start here
```

Expected runtime per notebook on a laptop GPU (CUDA): ~1 minute for
`00`, ~3 minutes for `01` (more loss terms, second derivatives, plus
the `nut_model` pre-training). On CPU: 2-3× slower.
