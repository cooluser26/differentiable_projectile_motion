"""JAX projectile simulator built around ``jax.lax.scan``."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .integrators import euler_step, rk2_step
from .physics import decompose_velocity, dynamics


@partial(jax.jit, static_argnames=("n_steps", "method"))
def simulate_projectile(
    x0: float,
    y0: float,
    v0: float,
    theta_deg: float,
    dt: float,
    n_steps: int,
    drag_coefficient: float = 0.0,
    method: str = "rk2",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate projectile motion and return final state and positions.

    Parameters
    ----------
    x0, y0:
        Initial position in metres.
    v0:
        Initial speed in m/s.
    theta_deg:
        Launch angle in degrees.
    dt:
        Time step in seconds.
    n_steps:
        Number of integration steps. This is static under JIT compilation.
    drag_coefficient:
        Lumped quadratic-drag coefficient. Set to zero for vacuum motion.
    method:
        Either ``"euler"`` or ``"rk2"``.
    """
    vx0, vy0 = decompose_velocity(v0, theta_deg)
    state0 = jnp.array([x0, y0, vx0, vy0], dtype=jnp.result_type(v0, dt, float))

    if method == "euler":
        stepper = euler_step
    elif method == "rk2":
        stepper = rk2_step
    else:
        raise ValueError(f"Unknown integration method: {method}")

    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        return dynamics(state, drag_coefficient)

    def scan_fn(state: jnp.ndarray, _: None):
        new_state = stepper(state, dt, rhs)
        return new_state, new_state[:2]

    final_state, positions = lax.scan(scan_fn, state0, xs=None, length=n_steps)
    return final_state, positions


def time_grid(dt: float, n_steps: int) -> jnp.ndarray:
    """Return times matching the positions emitted by ``simulate_projectile``."""
    return jnp.arange(1, n_steps + 1) * dt
