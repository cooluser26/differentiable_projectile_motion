"""Differentiable projectile-motion utilities built with JAX."""

from .physics import analytic_trajectory, decompose_velocity, dynamics
from .simulate import simulate_projectile
from .optimize import optimize_launch, target_loss

__all__ = [
    "analytic_trajectory",
    "decompose_velocity",
    "dynamics",
    "simulate_projectile",
    "target_loss",
    "optimize_launch",
]
