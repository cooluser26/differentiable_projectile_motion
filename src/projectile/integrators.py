"""Time-integration methods used by the simulator."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

State = jnp.ndarray
Dynamics = Callable[[State], State]


def euler_step(state: State, dt: float, rhs: Dynamics) -> State:
    """Advance one step with forward Euler (first order)."""
    return state + dt * rhs(state)


def rk2_step(state: State, dt: float, rhs: Dynamics) -> State:
    """Advance one step with the midpoint RK2 method (second order)."""
    k1 = rhs(state)
    midpoint = state + 0.5 * dt * k1
    k2 = rhs(midpoint)
    return state + dt * k2
