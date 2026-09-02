# Differentiable Projectile Motion

A compact JAX project for learning numerical integration, automatic differentiation, vectorization, JIT compilation, and gradient-based optimization through a simple physics problem.

## What this project demonstrates

- Projectile dynamics written as a first-order ODE system
- Time integration with `jax.lax.scan`
- Forward Euler and midpoint RK2 integrators
- Comparison against the analytic constant-gravity solution
- First-order convergence of Euler
- Quadratic air drag
- Automatic differentiation with `jax.grad`
- Batched parameter sweeps with `jax.vmap`
- JIT compilation with static simulation length
- Gradient-based optimization of launch speed and angle
- Unit tests and GitHub Actions CI

## Physics

The state is

```text
state = [x, y, vx, vy]
```

with dynamics

```text
dx/dt  = vx
dy/dt  = vy
dvx/dt = ax
dvy/dt = ay
```

For vacuum motion,

```text
ax = 0
ay = -g
```

and with the optional lumped quadratic-drag model,

```text
a_drag = -k |v| v
```

where `k=0` recovers vacuum projectile motion.

## Installation

```bash
git clone https://github.com/cooluser26/differentiable_projectile_motion.git
cd differentiable_projectile_motion
python -m pip install -e ".[dev]"
```

## Basic simulation

```python
from projectile.simulate import simulate_projectile

final_state, trajectory = simulate_projectile(
    x0=0.0,
    y0=0.0,
    v0=20.0,
    theta_deg=35.0,
    dt=0.01,
    n_steps=200,
    method="rk2",
)

x = trajectory[:, 0]
y = trajectory[:, 1]
```

`simulate_projectile` uses `jax.lax.scan`, so the evolving state is the scan carry while the saved `(x, y)` positions form the output trajectory.

## Numerical convergence

Run:

```bash
python examples/convergence.py
```

This compares the numerical solution against the analytic constant-gravity trajectory for several time steps. Forward Euler should show approximately first-order global error:

```text
error ~ dt
```

For constant acceleration, midpoint RK2 reproduces the position solution to floating-point precision because the acceleration is exactly represented over each step. The RK2 implementation becomes meaningfully second-order once the acceleration depends on the state, such as when drag is enabled.

## Air drag

```bash
python examples/drag_comparison.py
```

The example compares vacuum motion with a quadratic-drag trajectory using the same RK2 integrator.

## Vectorization with `vmap`

```bash
python examples/vmap_sweep.py
```

The example evaluates many launch angles in parallel while keeping `dt` and `n_steps` fixed. This is the shape-compatible use case that `jax.vmap` is designed for.

## Differentiable optimization

The terminal position is differentiable with respect to launch parameters. The optimization example treats

```text
params = [v0, theta_deg]
```

as trainable parameters and minimizes squared distance to a target position after a fixed simulation time.

```bash
python examples/optimize_target.py
```

Under the hood, the project uses `jax.value_and_grad` through the complete `lax.scan` simulation and performs gradient descent using another `lax.scan`.

## Why optimize at a fixed terminal time?

A hard ground-impact event requires choosing the first index at which `y <= 0`. That discrete index selection is not smoothly differentiable. This project therefore uses a fixed simulation horizon for the differentiable optimization example and keeps event-based landing calculations separate from the autodiff path.

## Repository layout

```text
differentiable_projectile_motion/
├── src/projectile/
│   ├── __init__.py
│   ├── physics.py
│   ├── integrators.py
│   ├── simulate.py
│   └── optimize.py
├── examples/
│   ├── convergence.py
│   ├── drag_comparison.py
│   ├── optimize_target.py
│   └── vmap_sweep.py
├── tests/
│   └── test_simulation.py
├── .github/workflows/tests.yml
└── pyproject.toml
```

## JAX concepts used

### `lax.scan`
Carries the projectile state from one integration step to the next without a Python time loop.

### `grad` / `value_and_grad`
Differentiates terminal-position loss with respect to launch speed and angle.

### `vmap`
Runs the same fixed-shape simulation over many launch angles without manually writing a Python loop.

### `jit`
Compiles the simulator and optimizer. `n_steps`, the integration method, and the optimization length are static arguments because they determine compiled control flow or output shape.

## Tests

```bash
pytest -q
```

The tests check:

- RK2 against the exact constant-gravity solution
- Euler convergence when `dt` is halved
- finite nonzero gradients through the simulator
- reduced horizontal travel when drag is enabled

## Suggested extensions

Good next exercises are:

1. Add RK4 and compare convergence rates.
2. Add wind as an external velocity field.
3. Add a differentiable soft approximation to ground impact.
4. Optimize launch parameters under a speed or energy constraint.
5. Compare NumPy, eager JAX, and JIT-compiled JAX runtime.
6. Batch thousands of trajectories with `vmap`.
7. Replace gradient descent with Optax.

## Purpose

This repository is intentionally small enough to understand end-to-end. The goal is not to build a production ballistics package, but to use a familiar mechanics problem to expose the core ideas behind differentiable scientific computing in JAX.
