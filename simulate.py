import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

g = 9.81 # m/s^2

def decompose_vel(v0, theta):
    '''Decompose initial vel into x-y components'''
    theta = jnp.deg2rad(theta)
    vx = v0 * jnp.cos(theta)
    vy = v0 * jnp.sin(theta)

    return vx, vy

def simulate_projectile(x0, y0, v0, theta, dt, n_steps):
    vx, vy = decompose_vel(v0, theta)
    state0 = (x0, y0, vx, vy)

    def scan_fn(state, _):

        x, y, vx, vy = state
        x_new = x + vx * dt
        y_new = y + vy * dt
        vx_new = vx
        vy_new = vy - g * dt

        new_state = (x_new, y_new, vx_new, vy_new)
        output = (x_new, y_new)
    
        return (new_state, output)

    return lax.scan(scan_fn, state0, xs=None, length=n_steps)

def analytic_projectile(x0, y0, v0, theta, dt, n_steps):
    '''Computes projectile with only gravity using SUVAT'''
    vx, vy = decompose_vel(v0, theta)
    time = jnp.arange(1, n_steps + 1) * dt

    x_analytic = x0 + vx * time
    y_analytic = y0 + vy * time - 0.5 * g * time**2

    return x_analytic, y_analytic