"""Physics helpers for projectile motion."""

from __future__ import annotations

import jax.numpy as jnp

G = 9.81


def decompose_velocity(v0: float, theta_deg: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Decompose launch speed into horizontal and vertical components."""
    theta = jnp.deg2rad(theta_deg)
    return v0 * jnp.cos(theta), v0 * jnp.sin(theta)


def dynamics(state: jnp.ndarray, drag_coefficient: float = 0.0) -> jnp.ndarray:
    """Return d(state)/dt for state = [x, y, vx, vy].

    The drag model is quadratic in speed and uses a lumped coefficient k:
        a_drag = -k * |v| * v
    so k=0 recovers vacuum projectile motion.
    """
    _, _, vx, vy = state
    speed = jnp.sqrt(vx**2 + vy**2)

    ax = -drag_coefficient * speed * vx
    ay = -G - drag_coefficient * speed * vy

    return jnp.array([vx, vy, ax, ay])


def analytic_trajectory(
    x0: float,
    y0: float,
    v0: float,
    theta_deg: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """Exact gravity-only trajectory evaluated at ``times``."""
    vx0, vy0 = decompose_velocity(v0, theta_deg)
    x = x0 + vx0 * times
    y = y0 + vy0 * times - 0.5 * G * times**2
    return jnp.stack((x, y), axis=-1)
