"""Shared fixtures for GPU-versus-oracle kernel tests."""

import numpy as np

from marrow.core.lattice import build_lattice

CUBE = build_lattice(np.zeros(3), 1.0, np.ones((1, 1, 1), dtype=bool))
BLOCK = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))


def oracle_predict(state, params, h):
    """The oracle's predict step, lifted out of step() verbatim."""
    gravity = np.asarray(params.gravity, dtype=np.float64)
    movable = state.inv_mass > 0.0
    predicted = state.nodes.copy()
    predicted[movable] += state.velocities[movable] * h + gravity * (h * h)
    return predicted


def assert_close(gpu_out, cpu_out, tol, what):
    """Fail with the worst offender named, not just 'arrays differ'."""
    gpu_out = np.asarray(gpu_out, dtype=np.float64)
    cpu_out = np.asarray(cpu_out, dtype=np.float64)
    assert gpu_out.shape == cpu_out.shape, (
        f"{what}: shape {gpu_out.shape} vs {cpu_out.shape}"
    )
    diff = np.abs(gpu_out - cpu_out)
    worst = int(np.argmax(diff.max(axis=1)))
    assert diff.max() < tol, (
        f"{what}: max |GPU - oracle| = {diff.max():.3e} > {tol:.1e} "
        f"at element {worst}: GPU {gpu_out[worst]} oracle {cpu_out[worst]}"
    )
