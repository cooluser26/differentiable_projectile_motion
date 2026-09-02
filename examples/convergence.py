"""Compare Euler and RK2 against the analytic gravity-only solution."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from projectile.physics import analytic_trajectory
from projectile.simulate import simulate_projectile, time_grid


def terminal_error(dt: float, method: str, end_time: float = 2.0) -> float:
    n_steps = int(round(end_time / dt))
    final_state, _ = simulate_projectile(
        x0=0.0,
        y0=0.0,
        v0=20.0,
        theta_deg=35.0,
        dt=dt,
        n_steps=n_steps,
        method=method,
    )
    times = time_grid(dt, n_steps)
    exact = analytic_trajectory(0.0, 0.0, 20.0, 35.0, times)[-1]
    return float(jnp.linalg.norm(final_state[:2] - exact))


def main() -> None:
    dt_values = [0.16, 0.08, 0.04, 0.02, 0.01]

    euler_errors = [terminal_error(dt, "euler") for dt in dt_values]
    rk2_errors = [terminal_error(dt, "rk2") for dt in dt_values]

    print("dt       Euler error       RK2 error")
    for dt, euler_error, rk2_error in zip(dt_values, euler_errors, rk2_errors):
        print(f"{dt:<8.3f} {euler_error:<17.8e} {rk2_error:.8e}")

    plt.figure(figsize=(5, 4))
    plt.loglog(dt_values, euler_errors, "o-", label="Euler")
    plt.loglog(dt_values, rk2_errors, "o-", label="RK2")
    plt.xlabel("dt [s]")
    plt.ylabel("terminal position error [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=200)


if __name__ == "__main__":
    main()
