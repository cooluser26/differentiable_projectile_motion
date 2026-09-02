"""Use ``jax.vmap`` to sweep many launch angles at fixed trajectory shape."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from projectile.simulate import simulate_projectile


def main() -> None:
    angles = jnp.linspace(15.0, 75.0, 25)

    def run_angle(theta_deg):
        final_state, _ = simulate_projectile(
            x0=0.0,
            y0=0.0,
            v0=20.0,
            theta_deg=theta_deg,
            dt=0.01,
            n_steps=200,
            method="rk2",
        )
        return final_state[:2]

    terminal_xy = jax.vmap(run_angle)(angles)

    plt.figure(figsize=(5, 4))
    plt.plot(angles, terminal_xy[:, 0], label="terminal x")
    plt.plot(angles, terminal_xy[:, 1], label="terminal y")
    plt.xlabel("launch angle [deg]")
    plt.ylabel("terminal position [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vmap_angle_sweep.png", dpi=200)


if __name__ == "__main__":
    main()
