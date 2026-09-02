"""Compare trajectories with and without quadratic drag."""

import matplotlib.pyplot as plt

from projectile.simulate import simulate_projectile


def main() -> None:
    common = dict(
        x0=0.0,
        y0=0.0,
        v0=25.0,
        theta_deg=45.0,
        dt=0.01,
        n_steps=300,
        method="rk2",
    )

    _, vacuum = simulate_projectile(**common, drag_coefficient=0.0)
    _, drag = simulate_projectile(**common, drag_coefficient=0.02)

    plt.figure(figsize=(6, 4))
    plt.plot(vacuum[:, 0], vacuum[:, 1], label="vacuum")
    plt.plot(drag[:, 0], drag[:, 1], label="quadratic drag")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("drag_comparison.png", dpi=200)


if __name__ == "__main__":
    main()
