# 2D PINN Training — Sparse Field Reconstruction

A classroom-ready PIML (Physics-Informed Machine Learning) teaching module for
the Applied AI Summer Workshop.

**Worked example:** reconstruct a 2D CFD velocity field from K sparse "sensor"
observations, where K is much smaller than the number of mesh points — then
show that adding the incompressible continuity equation as a soft loss term
lets the network fill the gaps between sensors far better than plain
regression. This is the classic Raissi-style inpainting setup, applied to a
turbulent contact-basin flow.

> Given (Ux, Uy) at K randomly chosen points in a 2D baffled basin, reconstruct
> the velocity field everywhere else — once with a plain MLP, once with the same
> MLP plus a continuity-equation loss.

---

## Start here

The canonical path is the **self-contained notebooks** — everything (network,
loss, plots) is in one file, with nothing to import:

| Notebook | Role | Physics |
|---|---|---|
| `notebooks/00_inpainting_complete.{py,ipynb}` | **Workshop module — start here** | continuity only |
| `notebooks/01_with_momentum.{py,ipynb}` | Advanced extension | + RANS momentum |

For the deeper story — why partial physics helps in the sparse regime and
stops helping in the data-rich one, how sensor placement changes everything,
and the completeness ladder from continuity to full closure — see
[`ADVANCED_PHYSICS_GUIDE.md`](ADVANCED_PHYSICS_GUIDE.md).

Workshop framing materials (deck + one-page handout) live alongside this repo:
`PINN_Workshop_Module.pptx` and `PINN_Workshop_Handout.pdf`.

## Quick start

```bash
pip install -r requirements.txt
python setup_check.py                                   # environment smoke test
jupyter notebook notebooks/00_inpainting_complete.ipynb # start here
```

Expected runtime: ~30–60 s for `00` on a laptop (CPU is fine), a few minutes
for `01` (more loss terms, second derivatives, plus νₜ pre-training).

## Directory layout

```
2D_PINN_Training/
  README.md                              <- this file
  PROJECT_SUMMARY.md                     <- fuller project write-up
  ADVANCED_PHYSICS_GUIDE.md              <- the "going deeper" track
  setup_check.py                         <- environment smoke test
  requirements.txt
  notebooks/
    00_inpainting_complete.{py,ipynb}    <- WORKSHOP MODULE (continuity)  ← start
    01_with_momentum.{py,ipynb}          <- advanced: + RANS momentum
    01_sparse_data_setup.{py,ipynb}      <- (legacy, library-based staged version)
    02_mlp_from_sensors.{py,ipynb}       <- (legacy)
    03_add_continuity.{py,ipynb}         <- (legacy)
    04_sensor_sweep.{py,ipynb}           <- K-vs-improvement sweep with error bars
  piml/                                  <- modular library used by the legacy notebooks
    config.py  data.py  model.py  physics.py  sampling.py  train.py  plotting.py
  data/tiny/
    basin_geometries_summary.csv         <- case parameters (L, W, n, U_inlet, ...)
    results/case_834_velocity.csv        <- 4-baffle case (default)
    results/case_834_full.csv            <- with p, nut, k, epsilon
    results/case_267_velocity.csv        <- 1-baffle case (simpler alternative)
  case_834/                              <- raw OpenFOAM case directory
  scripts/extract_openfoam_fields.py     <- parses OpenFOAM ASCII fields -> CSV
  phase_a_*, phase_b_*, phase_c_*        <- research-phase experiment scripts + logs
```

The self-contained notebooks (`00`, `01`) are the teaching entry points. The
`piml/` library plus the legacy staged notebooks (`01_sparse_data_setup` →
`03_add_continuity`) do the same thing split across modules — shorter per file,
but you have to navigate several files to see the whole picture.

## What physics is enforced?

**Notebook `00` imposes exactly one equation** — the incompressible continuity
equation,

∂Ux/∂x + ∂Uy/∂y = 0,

as a squared residual (via PyTorch autograd) at thousands of random
collocation points per step, added as a soft loss term. Nothing else: **not**
momentum, **not** boundary conditions, **not** the baffle no-slip. The
continuity term couples Ux and Uy everywhere, so sensor information propagates
spatially through the physics rather than by proximity interpolation alone —
that coupling is what buys the data efficiency.

Notebook `01` adds the RANS momentum balance (Boussinesq closure). See the
advanced guide for the completeness ladder and what each level buys.

## Expected results (case 834, n = 4 baffles)

Numbers below are **re-confirmed** for the continuity demo: case 834, K = 50,
5000 epochs, averaged over 3 sensor seeds.

```
Plain MLP (data only)         baseline
+ continuity                  ~28% lower held-out error   (range +16% to +42%)
```

Honest reading: **the direction is robust — continuity reliably helps and never
hurts at K = 50 — but the size varies** roughly 2.5× across which 50 points you
happen to sense, and it depends on how long you train. Quote it as "about a
quarter to a third," not a fixed figure. (An earlier single-seed write-up
quoted larger numbers, e.g. ~43%; that was the high end of the seed range.)

**Where physics earns its keep — the K regimes:**

| K | Continuity benefit | Why |
|---|---|---|
| ≲ 30 (too sparse) | small / unreliable | not enough information anywhere |
| ≈ 50 (sweet spot) | largest (~+28% mean) | enough to anchor, sparse enough that physics must fill gaps |
| ≳ 150 (data-rich) | small, can go slightly negative | dense data already implies the physics; a fixed-weight partial law can over-constrain |

PIML's value is maximized in the **moderate-sparse regime**. Below it, neither
model reconstructs the field; above it, the MLP interpolates fine on its own
and one partial law adds little. The fact that continuity can *slightly hurt*
at high K is a feature to teach, not a bug — it shows partial physics meeting
sufficient data (see the advanced guide).

## Swapping in a different case

```python
CASE_ID = 267    # simpler n=1 case; the sweet spot moves to a lower K
```

...and ensure `data/tiny/results/case_267_velocity.csv` exists (it does), or
copy another case from the full dataset into `data/tiny/`.

## How the data was generated

Steady-state incompressible RANS in OpenFOAM (`simpleFoam`, k-ε turbulence).
The full case is in `case_834/`; `scripts/extract_openfoam_fields.py` parses
the OpenFOAM ASCII fields into `data/tiny/results/case_834_full.csv` with
columns `x, y, Ux, Uy, p, nut, k, epsilon`.
