"""
piml — Physics-Informed Machine Learning teaching library (inpainting edition)

Demo problem: **sparse velocity-field reconstruction**.  Given a small set of
"sensor" observations of a 2D flow field inside a contact basin, reconstruct
the full field.  With enough sensors, a plain MLP works fine.  With few
sensors — the regime where PIML matters — the MLP guesses garbage in the
gaps, and the continuity equation becomes essential information.

The library is deliberately narrow-scoped: one geometry, one flow field,
one governing equation.  Every module should be readable top-to-bottom in
under 5 minutes.
"""

from .config  import CaseGeometry, Hyperparams, DataPaths
from .data    import load_case, make_sensor_split
from .model   import MLP, sdf_mask, get_device
from .physics import (
    continuity_residual, continuity_loss,
    inlet_bc_loss, wall_bc_loss, baffle_bc_loss,
    warmup_weight,
)
from .sampling import (
    sample_collocation, sample_inlet_bc, sample_wall_bc, sample_baffle_bc,
    to_tensors,
)
from .train    import train_mlp, train_pinn, PhysicsWeights, PhysicsSampling
from .plotting import (
    plot_velocity_field, plot_residual_field,
    plot_sensors_on_field, plot_reconstruction_panel,
    plot_head_to_head, plot_loss_curves, plot_loss_components,
)

__all__ = [
    "CaseGeometry", "Hyperparams", "DataPaths",
    "load_case", "make_sensor_split",
    "MLP", "sdf_mask", "get_device",
    "continuity_residual", "continuity_loss",
    "inlet_bc_loss", "wall_bc_loss", "baffle_bc_loss", "warmup_weight",
    "sample_collocation", "sample_inlet_bc", "sample_wall_bc", "sample_baffle_bc",
    "to_tensors",
    "train_mlp", "train_pinn", "PhysicsWeights", "PhysicsSampling",
    "plot_velocity_field", "plot_residual_field",
    "plot_sensors_on_field", "plot_reconstruction_panel",
    "plot_head_to_head", "plot_loss_curves", "plot_loss_components",
]
