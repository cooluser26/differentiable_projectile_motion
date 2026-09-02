"""Optimize launch speed and angle to hit a terminal target."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from projectile.optimize import optimize_launch
from projectile.simulate import simulate_projectile


def main() -> None:
    target = jnp.array([20.0, 5.0])
    initial_params = jnp.array([15.0, 30.0])  # [speed, angle_deg]

    final_params, (param_history, loss_history) = optimize_launch(
        initial_params,
        target,
        learning_rate=2e-3,
        optimization_steps=1000,
        dt=0.01,
        n_steps=150,
        method="rk2",
    )

    speed, angle = final_params
    print(f"optimized speed: {float(speed):.3f} m/s")
    print(f"optimized angle: {float(angle):.3f} deg")
    print(f"final loss: {float(loss_history[-1]):.6e}")

    _, trajectory = simulate_projectile(
        x0=0.0,
        y0=0.0,
        v0=speed,
        theta_deg=angle,
        dt=0.01,
        n_steps=150,
        method="rk2",
    )

    plt.figure(figsize=(6, 4))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="optimized trajectory")
    plt.scatter([target[0]], [target[1]], marker="x", s=80, label="target")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimized_target.png", dpi=200)

    plt.figure(figsize=(5, 4))
    plt.semilogy(loss_history)
    plt.xlabel("optimization step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig("optimization_history.png", dpi=200)


if __name__ == "__main__":
    main()
