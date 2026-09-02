"""Gradient-based launch-parameter optimization."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .simulate import simulate_projectile


def target_loss(
    params: jnp.ndarray,
    target_xy: jnp.ndarray,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 100,
    drag_coefficient: float = 0.0,
    method: str = "rk2",
    speed_penalty: float = 0.0,
) -> jnp.ndarray:
    """Squared terminal-position error for params = [v0, theta_deg]."""
    v0, theta_deg = params
    final_state, _ = simulate_projectile(
        x0=x0,
        y0=y0,
        v0=v0,
        theta_deg=theta_deg,
        dt=dt,
        n_steps=n_steps,
        drag_coefficient=drag_coefficient,
        method=method,
    )
    terminal_xy = final_state[:2]
    miss = terminal_xy - target_xy
    return jnp.dot(miss, miss) + speed_penalty * v0**2


@partial(jax.jit, static_argnames=("n_steps", "method", "optimization_steps"))
def optimize_launch(
    initial_params: jnp.ndarray,
    target_xy: jnp.ndarray,
    *,
    learning_rate: float = 1e-2,
    optimization_steps: int = 500,
    x0: float = 0.0,
    y0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 100,
    drag_coefficient: float = 0.0,
    method: str = "rk2",
    speed_penalty: float = 0.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Optimize speed and launch angle with gradient descent.

    Returns the optimized parameter vector and histories of parameters and loss.
    """

    def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
        return target_loss(
            params,
            target_xy,
            x0=x0,
            y0=y0,
            dt=dt,
            n_steps=n_steps,
            drag_coefficient=drag_coefficient,
            method=method,
            speed_penalty=speed_penalty,
        )

    value_and_grad = jax.value_and_grad(loss_fn)

    def step(params: jnp.ndarray, _: None):
        loss, grads = value_and_grad(params)
        new_params = params - learning_rate * grads
        return new_params, (new_params, loss)

    final_params, history = lax.scan(
        step,
        initial_params,
        xs=None,
        length=optimization_steps,
    )
    return final_params, history
