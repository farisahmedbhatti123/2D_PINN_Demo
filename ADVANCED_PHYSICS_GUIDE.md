# Going Deeper: From One Law to Full Physics

*An advanced track for the 2D PINN inpainting module. The workshop session
teaches the core idea — sparse data plus one physics law reconstructs a flow
field. This guide is for participants who want to push further: add more
complete physics, understand where and why physics stops helping, and explore
how sensor placement changes the whole picture.*

---

## The organizing idea: physics completeness × data sparsity

The workshop demo enforces exactly one equation — incompressible continuity,
∇·u = 0. That is a *kinematic* constraint: it says mass is conserved, nothing
more. It is not the full physics of the flow. The data we reconstruct comes
from a complete RANS simulation (`simpleFoam`, k-ε turbulence) that also obeys
momentum balance and a turbulence closure. Continuity is *correct* — the true
field satisfies it exactly — but it is *partial*.

That single observation organizes everything in this guide:

> **When data is scarce, even partial physics helps a lot** — any correct
> constraint shrinks the space of possible fields. **As data grows, partial
> physics becomes redundant** — dense sensors already encode the full physics,
> including the parts continuity does not capture — and a fixed-weight
> continuity penalty can even *over-constrain* and mildly hurt.

So the value of a physics term lives on a 2-D surface: how *complete* the
physics is, against how *sparse* the data is. The three notebooks trace one
axis of that surface (increasing physics completeness); the `K` sweep traces
the other (increasing data).

---

## The completeness ladder

### Level 1 — Continuity (the workshop module, notebook `00`)

∂u/∂x + ∂v/∂y = 0, added as a squared residual at interior collocation points.
Couples the two velocity components so a measurement in one place constrains
the field elsewhere.

What we measured (case 834, K = 50, 5000 epochs, averaged over 3 sensor seeds):
held-out error drops about **28%** versus a data-only network, ranging
**+16% to +42%** across draws — always positive, never harmful at this K. This
is the honest headline number; treat it as "about a quarter to a third," not a
fixed figure, because it depends on which points are sensed and how long you
train.

What continuity does **not** do: it is blind to velocity *magnitude*. The plain
network's worst errors live in the high-speed serpentine channels, and
continuity barely touches those — it is a kinematic constraint, not a dynamic
one. Fixing the bright error bands needs momentum.

### Level 2 — RANS momentum (notebook `01_with_momentum`)

Adds the steady incompressible RANS momentum balance with the Boussinesq
(eddy-viscosity) closure, using the CFD's νₜ field as input. This couples
velocity to the pressure gradient, so the network can no longer set speeds
arbitrarily — exactly the constraint continuity was missing.

In the repository's single-seed runs this buys a further improvement on top of
continuity (the documented figures put continuity + momentum around a
two-thirds reduction versus the plain network at K = 50). **Caveat for
honesty:** those momentum numbers are single-seed and were not re-confirmed in
the multi-seed sweep behind this guide — continuity was. Given that the
re-run continuity number (~28% mean) came in below the originally documented
single-seed figure (~43%), expect the momentum headline to soften under
multi-seed averaging too. Confirming it is itself a good advanced exercise.

**The fragile knob.** `LAMBDA_MOMENT` works only in a narrow window around
0.001. An order of magnitude up gives little benefit; two orders up hurts; 1.0
breaks training. This is the single most delicate hyperparameter in the module
and worth demonstrating live — it shows that physics-informed training is not
free, and that loss weighting is a real engineering problem.

### Level 3 — Turbulence closure (not implemented)

Level 2 *uses* the CFD's νₜ as a given; it does not enforce the k-equation or
ε-equation. Adding those transport equations — so the network predicts the
turbulence field instead of being handed it — is the natural research-scale
next step and is left as an open extension.

---

## When physics stops helping (and starts hurting)

Sweeping the number of sensors K at fixed physics weight shows three regimes:

| Regime | What happens | Continuity benefit |
|---|---|---|
| Too sparse (K ≲ 30) | No information anywhere; absolute error is high for both models | small / unreliable |
| Sweet spot (K ≈ 50) | Enough to anchor, sparse enough that physics must fill the rest | largest (~+28% mean) |
| Data-rich (K ≳ 150) | Dense data already implies the full physics | small, and can go negative |

In our grid, continuity at K = 100 came in **slightly worse** than the
data-only network (about −6% on one draw). That is not a bug — it is the
signature of *incomplete physics meeting sufficient data*. Once the sensors are
dense enough to pin the field (including the momentum and turbulence structure
that continuity does not express), a fixed-weight continuity penalty is mostly
redundant and can pull the fit slightly off. The cure is either to lower the
physics weight as data grows, or to make the physics *more complete* (Level 2),
which is the prediction that motivates the momentum notebook: fuller physics
should keep helping further into the data-rich regime.

---

## Advanced topic: how you place the sensors

The workshop draws the K sensors at random. *Where* they go matters as much as
how many there are — and the obvious strategy is not the winner.

A natural hypothesis is to put sensors where the field changes fastest (shear
layers, channel edges) — "sample where it's hard." The repository's own
research scripts tested exactly this, and the result is counterintuitive:

| Sensor strategy (K = 50, k-ε + BC model) | held-out error |
|---|---|
| Random | 1.14×10⁻² |
| Error-greedy / active (importance-sampled by baseline error) | 1.74×10⁻² |
| Boundary-biased | 2.08×10⁻² |

Both "smart" strategies did **worse** than random (`phase_c_active_sensors.log`,
`phase_c_boundary_biased_sensors.log`). The reason is the lesson: high-gradient
regions are intrinsically the hardest to fit, and concentrating sensors there
*starves the rest of the domain* of coverage. You trade global reconstruction
for local detail you cannot even nail. Random wins because space-filling
coverage controls global error.

That opens a genuinely interesting design question rather than closing it:

- **Geometry-aware, space-filling placement** (farthest-point sampling or
  k-means over the actual mesh nodes). Hypothesis: smooths the noisy
  improvement-vs-K curve and lowers the sensing threshold by guaranteeing
  coverage. Note that naive Latin-hypercube sampling in the bounding box does
  *not* do this on a baffled (non-convex) domain — points snap to voids and
  the result is erratic; the sampler must respect the geometry.
- **Physics/curvature-aware placement** done *well* — not pure error-greedy,
  but balancing coverage against information. This is active learning, and
  whether it can beat random here is open.

---

## Suggested advanced exercises

1. **Confirm the momentum gain under multiple seeds.** Re-run notebook `01` at
   3–5 sensor seeds and report mean ± range, the way the continuity number was
   re-confirmed. Does momentum still clearly beat continuity?

2. **Map the over-constraint regime.** Sweep K from 20 to 300 for both
   continuity and momentum. Find where each stops helping. Does momentum's
   useful regime extend to higher K, as the completeness idea predicts?

3. **Tune λ against K.** Instead of a fixed physics weight, lower λ as K grows.
   Can you make continuity never hurt in the data-rich regime?

4. **Build a geometry-aware sampler** (farthest-point over mesh nodes) and put
   it head-to-head with random and the error-greedy strategy from `phase_c`.

5. **Continuity vs. generic smoothness.** Replace the continuity loss with a
   plain first-derivative smoothness penalty (tune its weight fairly — it needs
   a much smaller λ because it lacks the 1/L, 1/W scaling). In our runs,
   tuned smoothness reached ~0.024 versus continuity's ~0.019 at K = 50 —
   continuity does identifiable work *beyond* generic smoothing. Confirm it.

6. **Activation function.** Swap Tanh for ReLU and watch the physics loss go to
   noise (ReLU's second derivative is zero). Then try GELU/SiLU, which keep
   smooth higher derivatives.

---

## Honest caveats to teach alongside the wins

- **Seed variance is large.** At K = 50 the improvement ranges roughly 2.5×
  across sensor draws. Always report a range, never a single lucky number.
- **The win is epoch-dependent.** The PINN needs enough training to exploit the
  physics; short runs understate the benefit. State the epoch budget whenever
  you quote a number.
- **λ is a real knob, not a detail.** Continuity tolerates a broad range
  (sweet spot ~1–3 in our sweep); momentum is fragile (~0.001). Loss weighting
  is where physics-informed training earns or loses its keep.
- **Pressure has a gauge.** The network's predicted pressure can be off by a
  constant; the momentum notebook centers it, so `P_mod` is not physical.
