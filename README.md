# Differentiable Projectile Motion

A small JAX project for numerical integration, autodiff, vectorization, JIT compilation, and gradient-based optimization using projectile motion.

## Features

- Forward Euler and midpoint RK2 integrators
- Time integration with `jax.lax.scan`
- Analytic constant-gravity benchmark
- Quadratic air drag
- `jax.grad` / `jax.value_and_grad`
- Batched parameter sweeps with `jax.vmap`
- JIT-compiled simulation and optimization
- Pytest + GitHub Actions CI

## Installation

```bash
git clone https://github.com/cooluser26/differentiable_projectile_motion.git
cd differentiable_projectile_motion
python -m pip install -e ".[dev]"
```

## Example

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

## Examples

```bash
python examples/convergence.py
python examples/drag_comparison.py
python examples/vmap_sweep.py
python examples/optimize_target.py
```

## Structure

```text
src/projectile/
├── physics.py
├── integrators.py
├── simulate.py
└── optimize.py

examples/
tests/
```

## Tests

```bash
pytest -q
```
