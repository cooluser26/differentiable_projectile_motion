import jax
import jax.numpy as jnp

from projectile.physics import analytic_trajectory
from projectile.simulate import simulate_projectile, time_grid


def test_rk2_matches_constant_gravity_solution():
    dt = 0.01
    n_steps = 200
    final_state, _ = simulate_projectile(
        x0=0.0,
        y0=0.0,
        v0=20.0,
        theta_deg=35.0,
        dt=dt,
        n_steps=n_steps,
        method="rk2",
    )
    exact = analytic_trajectory(
        0.0,
        0.0,
        20.0,
        35.0,
        time_grid(dt, n_steps),
    )[-1]
    assert jnp.allclose(final_state[:2], exact, rtol=1e-5, atol=1e-5)


def test_euler_error_decreases_when_dt_is_halved():
    def error(dt):
        end_time = 2.0
        n_steps = int(round(end_time / dt))
        final_state, _ = simulate_projectile(
            x0=0.0,
            y0=0.0,
            v0=20.0,
            theta_deg=35.0,
            dt=dt,
            n_steps=n_steps,
            method="euler",
        )
        exact = analytic_trajectory(
            0.0,
            0.0,
            20.0,
            35.0,
            time_grid(dt, n_steps),
        )[-1]
        return jnp.linalg.norm(final_state[:2] - exact)

    coarse = error(0.04)
    fine = error(0.02)
    assert fine < coarse
    assert jnp.isclose(coarse / fine, 2.0, rtol=0.08)


def test_simulator_is_differentiable_with_respect_to_angle():
    def terminal_x(theta_deg):
        final_state, _ = simulate_projectile(
            x0=0.0,
            y0=0.0,
            v0=20.0,
            theta_deg=theta_deg,
            dt=0.01,
            n_steps=100,
            method="rk2",
        )
        return final_state[0]

    derivative = jax.grad(terminal_x)(35.0)
    assert jnp.isfinite(derivative)
    assert derivative != 0.0


def test_drag_reduces_horizontal_travel():
    kwargs = dict(
        x0=0.0,
        y0=0.0,
        v0=20.0,
        theta_deg=35.0,
        dt=0.01,
        n_steps=100,
        method="rk2",
    )
    vacuum, _ = simulate_projectile(**kwargs, drag_coefficient=0.0)
    drag, _ = simulate_projectile(**kwargs, drag_coefficient=0.02)
    assert drag[0] < vacuum[0]
