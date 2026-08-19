# Tetrahedral Fiber Constraints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an anisotropic per-tet constraint driven by a travelling activation wave, so a Marrow body can contract along baked fiber directions and crawl.

**Architecture:** One new XPBD constraint, `C = |F a| - s`, projected through the existing `project()` helper between the deviatoric and hydrostatic passes. Fiber directions and wave phase are baked once from a Curve object at Tetrahedralize and ride a per-tet RGBA32F texture; activation is computed on the GPU from push constants and a per-substep clock. The numpy oracle in `solver_ref.py` gets the same term first, and the GPU is diffed against it.

**Tech Stack:** Python 3.11+ (Blender bundled), numpy, GLSL compute via Blender's `gpu` module, pytest for `tests/core`, an assert-based runner for `tests/blender`.

**Spec:** `docs/superpowers/specs/2026-08-19-marrow-fiber-constraints-design.md`

## Global Constraints

- **`marrow/core/` must never import `bpy`.** Guarded by `tests/core/test_no_bpy.py`.
- **The addon must use relative imports only.** Installed its package is `bl_ext.user_default.marrow`, not `marrow`. Guarded by `tests/core/test_packaging.py`.
- **Never `pip install` into Blender's bundled Python.**
- **Core suite:** `.venv/Scripts/python -m pytest tests/core`
- **Blender suite:** `blender -b --factory-startup --python tests/blender/run_tests.py` — **run this on Blender 5.2.** 4.5 fails every GPU test for environmental reasons, not code reasons.
- **Blender tests are plain module-level `test_*()` functions with bare `assert`.** No pytest, no fixtures, no classes. The runner auto-discovers `tests/blender/test_*.py`.
- **GPU-versus-oracle tolerance is `2e-5`**, float32 across a full constraint projection on a unit-scale cage.
- **Every GPU kernel change must be mirrored in `marrow/core/solver_ref.py` and diffed against it.** A sign error in a compute shader is otherwise indistinguishable from a sign error in the constraint algebra.
- **Fiber defaults are inert:** `fiber_k=0.0`, `wave_amp=0.0`, `wave_len=1.0`, `wave_speed=0.0`, `waveform=0`. Every existing test must keep passing untouched.

---

### Task 1: Fiber baking from a polyline

**Files:**
- Create: `marrow/core/fiber.py`
- Test: `tests/core/test_fiber.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `tet_centroids(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray` — `(T, 3)` float64.
  - `fiber_from_polyline(points: np.ndarray, centroids: np.ndarray) -> np.ndarray` — `(T, 4)` float64, `xyz` = unit tangent, `w` = arclength. All-zero row means "no fiber, skip".

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_fiber.py`:

```python
import numpy as np

from marrow.core.fiber import fiber_from_polyline, tet_centroids


def _line(n=5, length=4.0):
    """A straight polyline along +X from the origin."""
    xs = np.linspace(0.0, length, n)
    return np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)


def test_centroids_are_the_mean_of_the_four_nodes():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    assert np.allclose(tet_centroids(nodes, tets), [[0.25, 0.25, 0.25]])


def test_straight_line_gives_axis_direction_and_arclength_phase():
    centroids = np.array([[0.5, 0.2, 0.0], [2.5, -0.3, 0.1], [3.9, 0.0, 0.0]])
    out = fiber_from_polyline(_line(), centroids)

    assert out.shape == (3, 4)
    assert np.allclose(out[:, :3], np.array([[1.0, 0.0, 0.0]] * 3), atol=1e-9)
    # Phase is the arclength of the nearest point, which for a line along X
    # is the centroid's own x.
    assert np.allclose(out[:, 3], [0.5, 2.5, 3.9], atol=1e-9)


def test_directions_are_unit_length():
    centroids = np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 1.0]])
    out = fiber_from_polyline(_line(), centroids)
    assert np.allclose(np.linalg.norm(out[:, :3], axis=1), 1.0, atol=1e-9)


def test_tangent_follows_a_corner():
    """An L: along +X to (1,0,0), then along +Y to (1,2,0)."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    out = fiber_from_polyline(points, np.array([[0.5, 0.0, 0.0], [1.0, 1.5, 0.0]]))
    assert np.allclose(out[0, :3], [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(out[1, :3], [0.0, 1.0, 0.0], atol=1e-9)
    # Arclength keeps accumulating around the corner: 1.0 along X plus 1.5.
    assert abs(out[1, 3] - 2.5) < 1e-9


def test_duplicate_points_do_not_poison_the_result():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    out = fiber_from_polyline(points, np.array([[1.0, 0.5, 0.0]]))
    assert np.all(np.isfinite(out)), "a zero-length segment must not produce NaN"
    assert np.allclose(out[0, :3], [1.0, 0.0, 0.0], atol=1e-9)


def test_a_degenerate_polyline_yields_no_fiber():
    """One point, or every point identical, means there is no direction to
    give. Zero rows are the solver's 'skip this tet' signal."""
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.allclose(fiber_from_polyline(np.zeros((1, 3)), centroids), 0.0)
    assert np.allclose(fiber_from_polyline(np.zeros((4, 3)), centroids), 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_fiber.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.core.fiber'`

- [ ] **Step 3: Write the implementation**

Create `marrow/core/fiber.py`:

```python
"""Fiber directions and wave phase, sampled from a polyline. Pure numpy.

The fiber constraint needs two things per tet: a direction to contract
along, and a scalar that says where along the creature this tet sits so the
wave can reach it at the right moment. A curve gives both from one sample -
the tangent at the nearest point, and the arclength at that point - which is
why the direction is not simply painted.

Directions are rest-space, because the constraint measures F a and F maps
rest to world. Callers sample against the cage's rest nodes, once.
"""

import numpy as np

# Below this a segment carries no direction: normalizing it would divide by
# roughly zero and hand the solver a NaN that spreads through the whole cage.
_MIN_SEGMENT = 1e-12


def tet_centroids(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """The mean of each tet's four nodes, (T, 3)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    if tets.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return nodes[tets].mean(axis=1)


def fiber_from_polyline(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Per-tet (direction, arclength) from the nearest point on a polyline.

    Returns (T, 4): xyz is the unit tangent of the nearest segment, w is the
    arclength from the start of the polyline to the nearest point. A row of
    zeros means no fiber could be assigned, which every consumer reads as
    "skip this tet" rather than as a direction.
    """
    points = np.asarray(points, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    out = np.zeros((centroids.shape[0], 4), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or centroids.shape[0] == 0:
        return out

    starts = points[:-1]
    deltas = points[1:] - starts
    lengths = np.linalg.norm(deltas, axis=1)

    # Zero-length segments are dropped rather than repaired. A curve
    # evaluated with duplicate control points is common and harmless; what
    # is not harmless is letting one become the nearest "segment" and
    # normalizing it.
    keep = lengths > _MIN_SEGMENT
    if not np.any(keep):
        return out

    # Arclength is measured along the WHOLE polyline, including the segments
    # dropped above, so phase stays continuous across a duplicate point.
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])[:-1]

    starts, deltas = starts[keep], deltas[keep]
    lengths, cumulative = lengths[keep], cumulative[keep]
    tangents = deltas / lengths[:, None]

    # (T, S) closest approach of every centroid to every segment.
    rel = centroids[:, None, :] - starts[None, :, :]
    t = np.einsum("tsc,sc->ts", rel, deltas) / (lengths * lengths)[None, :]
    t = np.clip(t, 0.0, 1.0)
    nearest = starts[None, :, :] + t[:, :, None] * deltas[None, :, :]
    pick = np.argmin(np.linalg.norm(centroids[:, None, :] - nearest, axis=2), axis=1)

    rows = np.arange(centroids.shape[0])
    out[:, :3] = tangents[pick]
    out[:, 3] = cumulative[pick] + t[rows, pick] * lengths[pick]
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_fiber.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Run the whole core suite for regressions**

Run: `.venv/Scripts/python -m pytest tests/core`
Expected: PASS, including `test_no_bpy.py` and `test_packaging.py`.

- [ ] **Step 6: Commit**

```bash
git add marrow/core/fiber.py tests/core/test_fiber.py
git commit -m "feat: bake fiber directions and wave phase from a polyline"
```

---

### Task 2: The fiber term in the numpy oracle

**Files:**
- Modify: `marrow/core/solver_ref.py` (`SolverParams`, `step`, `solve_constraints`)
- Test: `tests/core/test_solver_fiber.py`

**Interfaces:**
- Consumes: `fiber_from_polyline` output shape `(T, 4)` from Task 1.
- Produces:
  - `SolverParams` fields `fiber_k`, `wave_amp`, `wave_len`, `wave_speed`, `waveform`.
  - `fiber_activation(phase: float, t: float, params: SolverParams) -> float`
  - `solve_constraints(state, tets, dm_inv, rest_vol, params, h, fiber=None, t=0.0)`
  - `step(state, tets, dm_inv, rest_vol, params, targets=None, fiber=None)`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_solver_fiber.py`:

```python
import numpy as np

from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import (
    SolverParams,
    fiber_activation,
    make_state,
    precompute,
    solve_constraints,
    step,
)

CUBE = build_lattice(np.zeros(3), 1.0, np.ones((1, 1, 1), dtype=bool))


def _fibers_along_x(n_tets, phase=0.75):
    """Every tet pulls along +X, all at the same phase so they fire together."""
    fiber = np.zeros((n_tets, 4), dtype=np.float64)
    fiber[:, 0] = 1.0
    fiber[:, 3] = phase
    return fiber


def _extent(nodes, axis):
    return float(nodes[:, axis].max() - nodes[:, axis].min())


def test_activation_is_one_when_amplitude_is_zero():
    params = SolverParams(wave_amp=0.0)
    assert fiber_activation(0.5, 0.0, params) == 1.0


def test_smooth_waveform_peaks_at_half_a_cycle():
    params = SolverParams(wave_amp=0.4, wave_len=1.0, wave_speed=0.0, waveform=0)
    assert abs(fiber_activation(0.0, 0.0, params) - 1.0) < 1e-12
    assert abs(fiber_activation(0.5, 0.0, params) - 0.6) < 1e-12


def test_square_waveform_is_on_or_off():
    params = SolverParams(wave_amp=0.4, wave_len=1.0, wave_speed=0.0, waveform=1)
    assert fiber_activation(0.25, 0.0, params) == 1.0
    assert abs(fiber_activation(0.75, 0.0, params) - 0.6) < 1e-12


def test_phase_wraps_the_same_way_for_negative_time():
    """wave_time * wave_speed drives the phase negative almost immediately,
    and GLSL fract and numpy % must agree there or the GPU diverges."""
    params = SolverParams(wave_amp=1.0, wave_len=1.0, wave_speed=1.0, waveform=0)
    assert abs(fiber_activation(0.0, 0.25, params) - fiber_activation(0.75, 0.0, params)) < 1e-12


def test_the_wave_reaches_two_tets_at_different_times():
    early = SolverParams(wave_amp=1.0, wave_len=1.0, wave_speed=1.0, waveform=1)
    at_t0 = (fiber_activation(0.1, 0.0, early), fiber_activation(0.6, 0.0, early))
    assert at_t0[0] != at_t0[1], "tets at different arclength must not fire together"


def test_contraction_shortens_along_the_fiber_and_bulges_across():
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = _fibers_along_x(CUBE.n_tets)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0),
        fiber_k=1.0e4,
        wave_amp=0.4,
        wave_len=1.0,
        wave_speed=0.0,
        waveform=1,          # square, phase 0.75 -> fully on, held
    )
    state = make_state(CUBE.nodes)
    x0, y0 = _extent(CUBE.nodes, 0), _extent(CUBE.nodes, 1)
    for _ in range(20):
        step(state, CUBE.tets, dm_inv, rest_vol, params, fiber=fiber)

    assert _extent(state.nodes, 0) < x0 - 1e-3, "fiber must shorten along +X"
    assert _extent(state.nodes, 1) > y0 + 1e-4, "volume must go sideways"


def test_zero_fiber_stiffness_reproduces_the_current_solve():
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = _fibers_along_x(CUBE.n_tets)
    params = SolverParams(gravity=(0.0, 0.0, 0.0), fiber_k=0.0, wave_amp=0.9)
    h = params.dt / params.substeps

    with_fiber = make_state(CUBE.nodes)
    without = make_state(CUBE.nodes)
    for st in (with_fiber, without):
        st.predicted[:] = CUBE.nodes * 1.2

    solve_constraints(with_fiber, CUBE.tets, dm_inv, rest_vol, params, h, fiber=fiber)
    solve_constraints(without, CUBE.tets, dm_inv, rest_vol, params, h)
    assert np.array_equal(with_fiber.predicted, without.predicted)


def test_a_zero_direction_row_is_skipped():
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = np.zeros((CUBE.n_tets, 4), dtype=np.float64)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=1.0e4, wave_amp=0.9, wave_len=1.0, waveform=1,
    )
    h = params.dt / params.substeps
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes
    solve_constraints(state, CUBE.tets, dm_inv, rest_vol, params, h, fiber=fiber)
    assert np.array_equal(state.predicted, CUBE.nodes)


def test_fiber_alone_still_solves_with_both_stiffnesses_off():
    """The early bail in solve_constraints must know about fiber_k, or the
    one test that isolates this feature passes for the wrong reason."""
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = _fibers_along_x(CUBE.n_tets)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=1.0e4, wave_amp=0.5, wave_len=1.0, waveform=1,
    )
    h = params.dt / params.substeps
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes
    solve_constraints(state, CUBE.tets, dm_inv, rest_vol, params, h, fiber=fiber)
    assert not np.array_equal(state.predicted, CUBE.nodes)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_solver_fiber.py -v`
Expected: FAIL, `ImportError: cannot import name 'fiber_activation'`

- [ ] **Step 3: Add the parameters and the activation function**

In `marrow/core/solver_ref.py`, add to `SolverParams` after `pin_kinematic`:

```python
    # --- fiber ---
    # Anisotropic muscle term. Inert at these defaults, so every body that
    # does not ask for fibers solves exactly as it did before.
    fiber_k: float = 0.0    # fiber stiffness; 0 disables the term
    wave_amp: float = 0.0   # peak contraction, 0.3 shortens to 70%
    wave_len: float = 1.0   # wave period in arclength units
    wave_speed: float = 0.0  # cycles per second; negative reverses travel
    waveform: int = 0       # 0 smooth cosine, 1 square
```

Add, next to `attach_compliance`:

```python
def fiber_activation(phase: float, t: float, params: SolverParams) -> float:
    """Target stretch along the fiber for one tet at time ``t``.

    1.0 is rest. Below 1.0 the tet is being told to shorten. The phase
    argument is the tet's baked arclength, so two tets at different points
    along the body reach their peak at different times - which is the whole
    difference between a travelling wave and a body that pulses in unison.

    Mirrored in kernels.SOLVE_SRC. GLSL fract and numpy % agree on negative
    inputs, which matters because t * wave_speed drives this negative within
    the first second.
    """
    cycle = (phase / params.wave_len - t * params.wave_speed) % 1.0
    if params.waveform == 0:
        pulse = 0.5 * (1.0 - np.cos(2.0 * np.pi * cycle))
    else:
        pulse = 1.0 if cycle >= 0.5 else 0.0
    return 1.0 - params.wave_amp * pulse
```

- [ ] **Step 4: Add the constraint and the clock**

In `solve_constraints`, change the signature and the early bail:

```python
def solve_constraints(state, tets, dm_inv, rest_vol, params, h,
                      fiber=None, t=0.0) -> None:
    """Stable neo-Hookean, plus an optional anisotropic fiber term."""
    fiber_on = params.fiber_k > 0.0 and fiber is not None
    if params.mu <= 0.0 and params.lam <= 0.0 and not fiber_on:
        return
```

Insert this block inside the per-tet loop, **between** the deviatoric block and the hydrostatic block:

```python
        # Fiber: resist stretch along a, and drive it below rest length when
        # the wave says so. Sits between the two isotropic terms because the
        # hydrostatic pass below rebuilds F from the positions this moved,
        # and that is what turns shortening into a sideways bulge.
        if fiber_on:
            a = fiber[t_i, :3]
            if float(a @ a) > 0.5:
                s = fiber_activation(float(fiber[t_i, 3]), t, params)
                fa = f @ a
                fiber_len = float(np.linalg.norm(fa))
                if fiber_len > 1e-12:
                    grads = _grads_from_dcdf(np.outer(fa / fiber_len, a), dm_inv[t_i])
                    _apply(
                        state, idx, grads, fiber_len - s,
                        1.0 / (params.fiber_k * abs(rest_vol[t_i])), h, 0.0,
                    )
```

The loop variable in `solve_constraints` is currently named `t`. Rename it to `t_i` throughout the function — every `tets[t]`, `dm_inv[t]`, `rest_vol[t]`, `lam_dev[t]`, `lam_hyd[t]` — so `t` can mean simulation time here and in the kernel alike. Do the rename in the same edit; leaving both meanings of `t` in one scope is how this gets silently wrong later.

The fiber multiplier is passed as `0.0` and its return discarded: unlike the deviatoric and hydrostatic terms this constraint is projected once per substep, so there is no multiplier to accumulate across iterations.

In `step`, add the parameter and the clock:

```python
def step(state: SolverState, tets, dm_inv, rest_vol, params: SolverParams,
         targets=None, fiber=None) -> None:
```

Inside, before the substep loop:

```python
    # Simulation clock for the fiber wave. Advances per substep, not per
    # frame: a per-frame clock steps the wave in visible stairs at low
    # substep counts. Local, so a fresh step() sequence always starts at the
    # same phase - the GPU mirrors this with GPUSolver.sim_time.
    t = 0.0
```

Change the `solve_constraints` call to pass them, and advance the clock at the end of each substep iteration:

```python
        solve_constraints(state, tets, dm_inv, rest_vol, params, h, fiber=fiber, t=t)
```

```python
        t += h
```

Place `t += h` as the last statement of the substep loop body, after the integrate block.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_solver_fiber.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 6: Run the whole core suite for regressions**

Run: `.venv/Scripts/python -m pytest tests/core`
Expected: PASS. `test_solver_constraints.py` and `test_solver_integration.py` must be untouched — they call `step` and `solve_constraints` positionally and the new arguments are keyword-with-default.

- [ ] **Step 7: Commit**

```bash
git add marrow/core/solver_ref.py tests/core/test_solver_fiber.py
git commit -m "feat: anisotropic fiber constraint in the numpy oracle"
```

---

### Task 3: Pack fibers to a texture, and the GLSL term

**Files:**
- Modify: `marrow/core/layout.py`
- Modify: `marrow/gpu/kernels.py` (`SOLVE_SRC`)
- Test: `tests/core/test_layout.py` (add one test)
- Test: `tests/blender/test_fiber_vs_oracle.py` (create)

**Interfaces:**
- Consumes: `SolverParams` fiber fields and `fiber_activation` from Task 2; `(T, 4)` fiber arrays from Task 1.
- Produces:
  - `pack_fiber(fiber: np.ndarray) -> np.ndarray` — `(H, W, 4)` float32 image, one texel per tet.
  - `SOLVE_SRC` gains image `fiber` and push constants `fiber_k`, `wave_amp`, `wave_len`, `wave_speed`, `wave_time`, `waveform`.

- [ ] **Step 1: Write the failing packing test**

Append to `tests/core/test_layout.py`:

```python
def test_pack_fiber_is_one_texel_per_tet():
    from marrow.core.layout import pack_fiber

    fiber = np.array([[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 1.75]])
    image = pack_fiber(fiber)
    flat = image.reshape(-1, 4)
    assert image.dtype == np.float32
    assert np.allclose(flat[0], [1.0, 0.0, 0.0, 0.25])
    assert np.allclose(flat[1], [0.0, 1.0, 0.0, 1.75])
    assert np.allclose(flat[2], 0.0), "unused texels must be zero, which reads as no fiber"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_layout.py::test_pack_fiber_is_one_texel_per_tet -v`
Expected: FAIL, `ImportError: cannot import name 'pack_fiber'`

- [ ] **Step 3: Add `pack_fiber`**

In `marrow/core/layout.py`, after `pack_rest`:

```python
def pack_fiber(fiber: np.ndarray) -> np.ndarray:
    """One texel per tet: rest-space direction in rgb, arclength in a.

    A zero row is not padding to be trimmed - it is the signal that a tet
    was never assigned a fiber, and the kernel skips it. Which means the
    blank tail of the image is already correct for a partly-fibered cage.
    """
    fiber = np.asarray(fiber, dtype=np.float64)
    image = _blank(fiber.shape[0])
    _write(image, fiber.astype(np.float32))
    return image
```

Add `pack_fiber` to the module's exports if `__all__` is present; at time of writing `layout.py` has none, so nothing else is needed.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Add the GLSL term**

In `marrow/gpu/kernels.py`, inside `SOLVE_SRC`'s `main()`, between the `// --- deviatoric ---` block and the `// --- hydrostatic ---` block:

```glsl
  // --- fiber ---
  // Transversely isotropic term: C = |F a| - s, with a the rest-space fiber
  // direction and s the activation. s < 1 shortens the tet along a; the
  // hydrostatic pass below then has to put that volume somewhere, which is
  // the bulge. A zero direction means this tet was never assigned a fiber.
  //
  // Transcribed from solver_ref.solve_constraints and fiber_activation. The
  // two must stay in step - test_fiber_vs_oracle is what notices if they do
  // not.
  //
  // f here is the deviatoric pass's F, stale by one projection. That is the
  // same staleness the deviatoric-then-hydrostatic split already accepts,
  // and the oracle does it too, so parity holds either way.
  if (fiber_k > 0.0 && !is_torn) {
    vec4 fb = imageLoad(fiber, texel(t));
    vec3 a = fb.xyz;
    if (dot(a, a) > 0.5) {
      float cycle = fract(fb.w / wave_len - wave_time * wave_speed);
      // Smooth is muscle; square is the literal (@Frame%10)/10 blink the
      // technique came from.
      float pulse = (waveform == 0)
        ? 0.5 * (1.0 - cos(6.2831853 * cycle))
        : step(0.5, cycle);
      float s = 1.0 - wave_amp * pulse;

      vec3 fa = f * a;
      float fiber_len = length(fa);
      if (fiber_len > 1e-12) {
        mat3 dcdf = outerProduct(fa / fiber_len, a);
        mat3 g = dcdf * dm_inv_t;
        vec3 g1v = g[0];
        vec3 g2v = g[1];
        vec3 g3v = g[2];
        vec3 g0v = -(g1v + g2v + g3v);
        project(idx, g0v, g1v, g2v, g3v, fiber_len - s,
                1.0 / (fiber_k * rest_vol), h);
      }
    }
  }
```

- [ ] **Step 6: Write the parity test**

Create `tests/blender/test_fiber_vs_oracle.py`:

```python
import gpu
import numpy as np

from _oracle_harness import BLOCK, CUBE, assert_close
from marrow.core.coloring import color_tets
from marrow.core.layout import (
    color_order,
    color_ordered,
    pack_fiber,
    pack_nodes,
    pack_rest,
    pack_tets,
    unpack_vec3,
)
from marrow.core.solver_ref import SolverParams, make_state, precompute, solve_constraints
from marrow.gpu.kernels import SOLVE_SRC, build
from marrow.gpu.textures import blank, download, flush, make_flush_shader, upload

gpu.init()

TOL = 2e-5

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "tets", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "rest", {"READ"}),
    ("R32F", "FLOAT_2D", "torn", {"READ", "WRITE"}),
    ("R32F", "FLOAT_2D", "live", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "fiber", {"READ"}),
]
PUSH = [
    ("FLOAT", "h"),
    ("FLOAT", "mu"),
    ("FLOAT", "lam"),
    ("FLOAT", "tear_threshold"),
    ("INT", "color_begin"),
    ("INT", "color_end"),
    ("FLOAT", "fiber_k"),
    ("FLOAT", "wave_amp"),
    ("FLOAT", "wave_len"),
    ("FLOAT", "wave_speed"),
    ("FLOAT", "wave_time"),
    ("INT", "waveform"),
]


def _fibers(n_tets, phase=0.75):
    """Every tet along +X at one phase, so the whole cage fires together."""
    fiber = np.zeros((n_tets, 4), dtype=np.float64)
    fiber[:, 0] = 1.0
    fiber[:, 3] = phase
    return fiber


def _run_solve(mesh, state, params, h, fiber, wave_time=0.0, torn=None):
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, offsets = color_ordered(mesh.tets, colors)
    order = color_order(colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)

    shader = build("solve", SOLVE_SRC, IMAGES, PUSH)
    tex_p = upload(pack_nodes(state.predicted, state.inv_mass))
    tex_t = upload(pack_tets(ordered))
    tex_r = upload(pack_rest(dm_inv, rest_vol))
    # Fiber rows are per-tet, so they must ride the same colour permutation
    # the tets did or every tet contracts along its neighbour's direction.
    tex_f = upload(pack_fiber(fiber[order]))
    tex_torn = blank(mesh.n_tets, fmt="R32F") if torn is None else upload(torn, fmt="R32F")
    tex_live = blank(mesh.n_nodes, fmt="R32F")

    for c in range(len(offsets) - 1):
        begin, end = int(offsets[c]), int(offsets[c + 1])
        if end <= begin:
            continue
        shader.bind()
        shader.image("p", tex_p)
        shader.image("tets", tex_t)
        shader.image("rest", tex_r)
        shader.image("torn", tex_torn)
        shader.image("live", tex_live)
        shader.image("fiber", tex_f)
        shader.uniform_float("h", h)
        shader.uniform_float("tear_threshold", 0.0)
        shader.uniform_float("mu", params.mu)
        shader.uniform_float("lam", params.lam)
        shader.uniform_int("color_begin", begin)
        shader.uniform_int("color_end", end)
        shader.uniform_float("fiber_k", params.fiber_k)
        shader.uniform_float("wave_amp", params.wave_amp)
        shader.uniform_float("wave_len", params.wave_len)
        shader.uniform_float("wave_speed", params.wave_speed)
        shader.uniform_float("wave_time", wave_time)
        shader.uniform_int("waveform", params.waveform)
        gpu.compute.dispatch(shader, (end - begin + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), mesh.n_nodes)


def _run_oracle(mesh, state, params, h, fiber, wave_time=0.0):
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, _ = color_ordered(mesh.tets, colors)
    order = color_order(colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)
    solve_constraints(
        state, ordered, dm_inv, rest_vol, params, h,
        fiber=fiber[order], t=wave_time,
    )
    return state.predicted.copy()


def _paired_states(mesh, deform):
    a, b = make_state(mesh.nodes), make_state(mesh.nodes)
    for st in (a, b):
        st.predicted[:] = deform(mesh.nodes.copy())
    return a, b


def test_fiber_matches_oracle_with_a_square_wave():
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4,
        wave_amp=0.4, wave_len=1.0, wave_speed=0.0, waveform=1,
    )
    h = params.dt / params.substeps
    fiber = _fibers(CUBE.n_tets)
    gpu_state, cpu_state = _paired_states(CUBE, lambda n: n)
    assert_close(
        _run_solve(CUBE, gpu_state, params, h, fiber),
        _run_oracle(CUBE, cpu_state, params, h, fiber),
        TOL,
        "fiber square wave on a cube",
    )


def test_fiber_matches_oracle_with_a_smooth_wave_in_motion():
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4,
        wave_amp=0.35, wave_len=0.4, wave_speed=1.5, waveform=0,
    )
    h = params.dt / params.substeps
    fiber = np.zeros((BLOCK.n_tets, 4), dtype=np.float64)
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(BLOCK.n_tets, 3))
    fiber[:, :3] = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    fiber[:, 3] = rng.uniform(0.0, 2.0, size=BLOCK.n_tets)

    def squash(nodes):
        nodes[:, 2] *= 0.85
        return nodes

    gpu_state, cpu_state = _paired_states(BLOCK, squash)
    # A non-zero time is the point: it is what drives the phase negative,
    # where fract and % have to agree.
    assert_close(
        _run_solve(BLOCK, gpu_state, params, h, fiber, wave_time=0.37),
        _run_oracle(BLOCK, cpu_state, params, h, fiber, wave_time=0.37),
        TOL,
        "fiber smooth wave mid-travel",
    )


def test_zero_direction_rows_match_the_oracle():
    """Half the cage has no fiber. Both sides must skip exactly those."""
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4,
        wave_amp=0.5, wave_len=1.0, wave_speed=0.0, waveform=1,
    )
    h = params.dt / params.substeps
    fiber = _fibers(BLOCK.n_tets)
    fiber[::2, :3] = 0.0
    gpu_state, cpu_state = _paired_states(BLOCK, lambda n: n)
    assert_close(
        _run_solve(BLOCK, gpu_state, params, h, fiber),
        _run_oracle(BLOCK, cpu_state, params, h, fiber),
        TOL,
        "fiber with unassigned tets",
    )


def test_zero_fiber_stiffness_is_a_noop():
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=0.0, wave_amp=0.9, wave_len=1.0, waveform=1,
    )
    h = params.dt / params.substeps
    fiber = _fibers(CUBE.n_tets)
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes + 0.1
    out = _run_solve(CUBE, state, params, h, fiber)
    assert np.allclose(out, CUBE.nodes + 0.1, atol=TOL)


def test_a_torn_tet_ignores_its_fiber():
    """Tearing means the material goes slack. Torn muscle must not pull.

    The oracle has no tearing, so this is GPU-only by construction: mark
    every tet torn, turn the isotropic terms off, and nothing may move.
    """
    from marrow.core.layout import pack_scalar

    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=1.0e4, wave_amp=0.5, wave_len=1.0, waveform=1,
    )
    h = params.dt / params.substeps
    fiber = _fibers(CUBE.n_tets)
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes
    torn = pack_scalar(np.ones(CUBE.n_tets))
    out = _run_solve(CUBE, state, params, h, fiber, torn=torn)
    assert np.allclose(out, CUBE.nodes, atol=TOL), (
        "a torn tet still contracted"
    )
```

- [ ] **Step 7: Run the Blender suite**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: PASS for `test_fiber_vs_oracle` (5 tests) and `test_kernels_compile`, and no regression in `test_solve_vs_oracle`. If the whole run shows a handful of unrelated GPU failures, re-run before reading them as a regression — see the README's note on GPU reliability.

- [ ] **Step 8: Commit**

```bash
git add marrow/core/layout.py marrow/gpu/kernels.py tests/core/test_layout.py tests/blender/test_fiber_vs_oracle.py
git commit -m "feat: fiber constraint in the solve kernel, diffed against the oracle"
```

---

### Task 4: Wire fibers into GPUSolver

**Files:**
- Modify: `marrow/gpu/solver.py` (`__init__`, the solve build, `substep_constraints`)
- Test: `tests/blender/test_fiber_solver.py` (create)

**Interfaces:**
- Consumes: `pack_fiber`, the `SOLVE_SRC` images and push constants from Task 3.
- Produces:
  - `GPUSolver(..., fiber=None)` — accepts a `(T, 4)` array **in mesh tet order**; the solver applies the colour permutation itself.
  - `GPUSolver.sim_time: float` — seconds since the solver was built, advanced per substep.

- [ ] **Step 1: Write the failing test**

Create `tests/blender/test_fiber_solver.py`:

```python
import gpu
import numpy as np

from _oracle_harness import CUBE
from marrow.core.solver_ref import SolverParams
from marrow.gpu.solver import GPUSolver

gpu.init()


def _fibers_along_x(n_tets, phase=0.75):
    fiber = np.zeros((n_tets, 4), dtype=np.float64)
    fiber[:, 0] = 1.0
    fiber[:, 3] = phase
    return fiber


def _params(**kw):
    base = dict(gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4, wave_amp=0.4,
                wave_len=1.0, wave_speed=0.0, waveform=1)
    base.update(kw)
    return SolverParams(**base)


def _extent(nodes, axis):
    return float(nodes[:, axis].max() - nodes[:, axis].min())


def test_a_solver_built_without_fibers_still_runs():
    """Every existing body takes this path. The image must still bind."""
    solver = GPUSolver(CUBE, np.ones(CUBE.n_nodes), SolverParams())
    solver.step()
    assert np.all(np.isfinite(solver.positions()))


def test_fibers_contract_the_body_along_their_direction():
    inv_mass = np.ones(CUBE.n_nodes)
    solver = GPUSolver(CUBE, inv_mass, _params(), fiber=_fibers_along_x(CUBE.n_tets))
    before = _extent(CUBE.nodes, 0)
    for _ in range(20):
        solver.step()
    after = _extent(solver.positions(), 0)
    assert after < before - 1e-3, f"fiber did not contract: {before} -> {after}"


def test_sim_time_advances_one_substep_at_a_time():
    params = _params()
    solver = GPUSolver(CUBE, np.ones(CUBE.n_nodes), params,
                       fiber=_fibers_along_x(CUBE.n_tets))
    assert solver.sim_time == 0.0
    solver.step()
    expected = params.dt
    assert abs(solver.sim_time - expected) < 1e-9, (
        f"one frame must advance the clock by dt, got {solver.sim_time}"
    )


def test_zero_fiber_stiffness_matches_a_solver_with_no_fibers():
    params = _params(fiber_k=0.0)
    with_fiber = GPUSolver(CUBE, np.ones(CUBE.n_nodes), params,
                           fiber=_fibers_along_x(CUBE.n_tets))
    without = GPUSolver(CUBE, np.ones(CUBE.n_nodes), params)
    for _ in range(5):
        with_fiber.step()
        without.step()
    assert np.allclose(with_fiber.positions(), without.positions(), atol=1e-6)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: FAIL on `test_fiber_solver` with `TypeError: __init__() got an unexpected keyword argument 'fiber'`.

- [ ] **Step 3: Accept and permute the fiber array**

In `marrow/gpu/solver.py`, add `fiber=None` to the end of the `__init__` signature, after `pin_kinematic=False`.

Import `pack_fiber` alongside the other packers at the top of the file.

After the line that sets `self._tet_order = color_order(colors)` (around line 158), and next to the other texture uploads:

```python
        # Fiber rows arrive in mesh tet order and must ride the same colour
        # permutation the tets did, or every tet contracts along its
        # neighbour's direction. Always allocated, even with no fibers: the
        # kernel needs every image bound, and a blank one reads as "no tet
        # has a fiber" at the cost of one texel per tet.
        if fiber is None:
            fiber_rows = np.zeros((self.mesh.n_tets, 4), dtype=np.float64)
        else:
            fiber_rows = np.asarray(fiber, dtype=np.float64)[self._tet_order]
        self.tex_fiber = upload_verified(pack_fiber(fiber_rows))

        # Seconds of simulated time, for the fiber wave. Per substep, not
        # per frame - a per-frame clock steps the wave in visible stairs at
        # low substep counts. Reset by a live restart, which rebuilds this
        # object outright.
        self.sim_time = 0.0
```

- [ ] **Step 4: Declare the image and push constants**

In the `self.sh_solve = kernels.build(...)` call, add to the images list:

```python
             ("RGBA32F", "FLOAT_2D", "fiber", {"READ"}),
```

and to the push constants list:

```python
             ("FLOAT", "fiber_k"), ("FLOAT", "wave_amp"),
             ("FLOAT", "wave_len"), ("FLOAT", "wave_speed"),
             ("FLOAT", "wave_time"), ("INT", "waveform"),
```

- [ ] **Step 5: Bind them and advance the clock**

In `substep_constraints`, inside the colour loop, alongside the existing `self.sh_solve.*` calls:

```python
            self.sh_solve.image("fiber", self.tex_fiber)
            self.sh_solve.uniform_float("fiber_k", self.params.fiber_k)
            self.sh_solve.uniform_float("wave_amp", self.params.wave_amp)
            self.sh_solve.uniform_float("wave_len", self.params.wave_len)
            self.sh_solve.uniform_float("wave_speed", self.params.wave_speed)
            self.sh_solve.uniform_float("wave_time", self.sim_time)
            self.sh_solve.uniform_int("waveform", int(self.params.waveform))
```

As the **last** statement of `substep_constraints`, after the collider dispatch:

```python
        # After every dispatch in this substep, so each colour saw one value.
        self.sim_time += h
```

- [ ] **Step 6: Run the Blender suite**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: PASS for `test_fiber_solver` (4 tests), no regression elsewhere.

- [ ] **Step 7: Commit**

```bash
git add marrow/gpu/solver.py tests/blender/test_fiber_solver.py
git commit -m "feat: bind fibers and a simulation clock in GPUSolver"
```

---

### Task 5: Persist fibers on the cage mesh

**Files:**
- Modify: `marrow/blender/storage.py`
- Test: `tests/blender/test_fiber_storage.py` (create)

**Interfaces:**
- Consumes: `(T, 4)` fiber arrays from Task 1.
- Produces:
  - `storage.FIBER_KEY = "marrow_fiber"`
  - `write_fiber(mesh, fiber: np.ndarray) -> None`
  - `read_fiber(mesh) -> np.ndarray | None` — `(T, 4)` float64, or `None` when the cage has no fibers.

- [ ] **Step 1: Write the failing test**

Create `tests/blender/test_fiber_storage.py`:

```python
import bpy
import numpy as np

from marrow.blender.storage import clear_marrow_data, read_fiber, write_fiber


def _mesh(name):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
    mesh.update()
    return mesh


def test_fiber_round_trips_through_the_mesh():
    mesh = _mesh("fiber_round_trip")
    fiber = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.25]])
    write_fiber(mesh, fiber)
    out = read_fiber(mesh)
    assert out is not None
    assert out.shape == (2, 4)
    assert np.allclose(out, fiber, atol=1e-6), f"{out} != {fiber}"


def test_a_cage_without_fibers_reads_as_none():
    assert read_fiber(_mesh("fiber_absent")) is None


def test_clear_marrow_data_strips_the_fiber_key():
    mesh = _mesh("fiber_cleared")
    write_fiber(mesh, np.array([[1.0, 0.0, 0.0, 0.0]]))
    clear_marrow_data(mesh)
    assert read_fiber(mesh) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: FAIL on import, `ImportError: cannot import name 'read_fiber'`.

- [ ] **Step 3: Add the key, the accessors and the strip**

In `marrow/blender/storage.py`, next to `BLEND_KEY`:

```python
# Per-tet fiber rows, stored on the CAGE mesh like the tets: T*4 floats,
# (dir x, dir y, dir z, arclength). An ID property rather than an attribute
# because the cage mesh has one vertex per NODE, and there is no per-tet
# domain to hang this on. Absent on a body with no fiber curve, which the
# solver takes as "no fiber pass".
FIBER_KEY = "marrow_fiber"
```

After `read_blend`:

```python
def write_fiber(mesh, fiber: np.ndarray) -> None:
    mesh[FIBER_KEY] = np.asarray(fiber, dtype=np.float32).ravel().tolist()


def read_fiber(mesh):
    """Stored fiber rows as (T, 4), or None if this cage has none."""
    if FIBER_KEY not in mesh.keys():
        return None
    flat = np.array(mesh[FIBER_KEY], dtype=np.float32).astype(np.float64)
    if flat.size == 0:
        return None
    return flat.reshape(-1, 4)
```

In `clear_marrow_data`, extend the ID-property loop:

```python
    for key in (BLEND_KEY, BLEND_W_KEY, FIBER_KEY):
```

- [ ] **Step 4: Run the Blender suite**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: PASS for `test_fiber_storage` (3 tests), and `test_detetrahedralize` and `test_storage` unchanged.

- [ ] **Step 5: Commit**

```bash
git add marrow/blender/storage.py tests/blender/test_fiber_storage.py
git commit -m "feat: persist fiber rows on the cage mesh"
```

---

### Task 6: Evaluate a curve and bake at Tetrahedralize

**Files:**
- Create: `marrow/blender/curve.py`
- Modify: `marrow/blender/ops.py` (`_tetrahedralize_iter`)
- Test: `tests/blender/test_fiber_bake.py` (create)

**Interfaces:**
- Consumes: `fiber_from_polyline`, `tet_centroids` (Task 1); `write_fiber` (Task 5).
- Produces: `polyline_from_curve(context, curve_obj) -> np.ndarray` — `(S, 3)` float64 **world-space** points in path order, or `(0, 3)` when the object is not a usable curve.

The spec put curve evaluation in `ops.py`. It goes in its own module instead: `ops.py` is already 690 lines, chain-walking an evaluated mesh is self-contained, and it is far easier to test on its own.

Fiber directions are sampled in **world space**, because that is the space `tetmesh.nodes` are in and therefore the space `dm_inv` is built from. No extra transform is needed anywhere; do not add one.

- [ ] **Step 1: Write the failing test**

Create `tests/blender/test_fiber_bake.py`:

```python
import bpy
import numpy as np

from marrow.blender.curve import polyline_from_curve


def _straight_curve(name="spine", length=4.0):
    """A two-point poly curve along +X, at the origin."""
    data = bpy.data.curves.new(name, type="CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("POLY")
    spline.points.add(1)                      # a new spline starts with one
    spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (length, 0.0, 0.0, 1.0)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def test_polyline_runs_along_the_curve_in_order():
    obj = _straight_curve()
    points = polyline_from_curve(bpy.context, obj)
    assert points.shape[1] == 3
    assert points.shape[0] >= 2
    xs = points[:, 0]
    assert np.all(np.diff(xs) > 0.0), f"points are not in path order: {xs}"
    assert abs(xs[0]) < 1e-6 and abs(xs[-1] - 4.0) < 1e-6


def test_polyline_is_world_space():
    obj = _straight_curve("moved_spine")
    obj.location = (10.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    points = polyline_from_curve(bpy.context, obj)
    assert abs(points[:, 0].min() - 10.0) < 1e-6, "curve was not taken to world space"


def test_a_non_curve_object_yields_nothing():
    mesh_obj = bpy.data.objects.new("not_a_curve", bpy.data.meshes.new("m"))
    bpy.context.collection.objects.link(mesh_obj)
    assert polyline_from_curve(bpy.context, mesh_obj).shape == (0, 3)


def test_tetrahedralize_bakes_fibers_when_a_curve_is_set():
    from marrow.blender.session import find_cage
    from marrow.blender.storage import read_fiber

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    body = bpy.context.active_object
    body.marrow.resolution = 1.0
    body.marrow.fiber_curve = _straight_curve("body_spine")
    bpy.ops.marrow.tetrahedralize()

    cage = find_cage(body)
    assert cage is not None
    fiber = read_fiber(cage.data)
    assert fiber is not None, "a curve was set but no fibers were baked"
    assert fiber.shape[1] == 4
    lengths = np.linalg.norm(fiber[:, :3], axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5), "directions must be unit length"
    assert np.allclose(fiber[:, 1:3], 0.0, atol=1e-5), "a +X curve gives +X fibers"


def test_tetrahedralize_without_a_curve_bakes_nothing():
    from marrow.blender.session import find_cage
    from marrow.blender.storage import read_fiber

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(20.0, 0.0, 0.0))
    body = bpy.context.active_object
    body.marrow.resolution = 1.0
    bpy.ops.marrow.tetrahedralize()
    assert read_fiber(find_cage(body).data) is None
```

The last two tests need the `fiber_curve` property from Task 7. Implement Task 6 and Task 7 in either order, but the suite is only green once both have landed — if Task 6 is run alone, those two fail with `AttributeError: 'MarrowSettings' object has no attribute 'fiber_curve'`, which is expected and is not a defect in this task.

- [ ] **Step 2: Run it to verify it fails**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: FAIL on import, `ModuleNotFoundError: No module named 'marrow.blender.curve'`.

- [ ] **Step 3: Write the curve evaluation**

Create `marrow/blender/curve.py`:

```python
"""A Curve object as an ordered world-space polyline.

Fibers are sampled against the curve after Blender has evaluated it, so
resolution, modifiers and shape keys are all accounted for. Evaluating to a
mesh is what makes that possible - but a mesh has no notion of "along", so
the vertices have to be walked back into path order through the edges.
"""

import numpy as np


def polyline_from_curve(context, curve_obj) -> np.ndarray:
    """Ordered world-space points along ``curve_obj``, (S, 3).

    Returns an empty (0, 3) array for anything that is not a curve, or for a
    curve that does not evaluate to a single open path - a bevelled or
    extruded curve becomes a tube, and a tube has no unambiguous direction
    to hand a fiber. Callers treat empty as "no fibers", which is the same
    thing they do when no curve was set at all.
    """
    if curve_obj is None or curve_obj.type != "CURVE":
        return np.zeros((0, 3), dtype=np.float64)

    depsgraph = context.evaluated_depsgraph_get()
    evaluated = curve_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        count = len(mesh.vertices)
        if count < 2:
            return np.zeros((0, 3), dtype=np.float64)

        coords = np.empty(count * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)

        neighbours = [[] for _ in range(count)]
        for edge in mesh.edges:
            a, b = edge.vertices[0], edge.vertices[1]
            neighbours[a].append(b)
            neighbours[b].append(a)

        order = _walk_chain(neighbours, count)
        if order is None:
            return np.zeros((0, 3), dtype=np.float64)
        coords = coords[order]
    finally:
        # to_mesh() allocates; the matching free is not optional.
        evaluated.to_mesh_clear()

    world = np.array(curve_obj.matrix_world.to_4x4())
    return coords @ world[:3, :3].T + world[:3, 3]


def _walk_chain(neighbours, count):
    """Vertex indices from one end of an open chain to the other, or None.

    None means the evaluated geometry is not a simple path: a branch, a ring
    or several disconnected pieces. Guessing a direction through any of
    those would put fibers somewhere the user did not ask for, so the caller
    declines instead.
    """
    degrees = [len(n) for n in neighbours]
    if any(d == 0 or d > 2 for d in degrees):
        return None
    ends = [i for i, d in enumerate(degrees) if d == 1]
    if len(ends) != 2:
        return None

    order = [ends[0]]
    previous = -1
    current = ends[0]
    while True:
        nexts = [n for n in neighbours[current] if n != previous]
        if not nexts:
            break
        previous, current = current, nexts[0]
        order.append(current)
        if len(order) > count:
            return None
    return order if len(order) == count else None
```

- [ ] **Step 4: Bake during Tetrahedralize**

In `marrow/blender/ops.py`, add to the storage import block: `write_fiber`. Add near the other core imports:

```python
from ..blender.curve import polyline_from_curve
from ..core.fiber import fiber_from_polyline, tet_centroids
```

Use relative imports matching the file's existing style — `from .curve import polyline_from_curve` if `ops.py` imports its siblings that way; check the top of the file and match it.

In `_tetrahedralize_iter`, immediately after `write_tetmesh(cage_mesh, tetmesh, colors)` and the `write_blend` call:

```python
    # Fibers are baked here and frozen. The direction is rest-space, because
    # the constraint measures F a and F maps rest to world, so an animated
    # curve would have no meaning as a source. Changing the curve means
    # tetrahedralizing again, and the panel says so.
    #
    # Sampled in world space: that is the space tetmesh.nodes are in, and
    # therefore the space dm_inv is built from.
    spine = polyline_from_curve(context, obj.marrow.fiber_curve)
    if spine.shape[0] >= 2:
        write_fiber(
            cage_mesh,
            fiber_from_polyline(spine, tet_centroids(tetmesh.nodes, tetmesh.tets)),
        )
```

- [ ] **Step 5: Run the Blender suite**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: PASS for the first three `test_fiber_bake` tests. The last two pass once Task 7 has landed.

- [ ] **Step 6: Commit**

```bash
git add marrow/blender/curve.py marrow/blender/ops.py tests/blender/test_fiber_bake.py
git commit -m "feat: bake fibers from a curve during Tetrahedralize"
```

---

### Task 7: Properties, panel and session wiring

**Files:**
- Modify: `marrow/blender/ui.py` (`MarrowSettings`, `MARROW_PT_panel.draw`)
- Modify: `marrow/blender/session.py` (`_build_solver`, the settings read)
- Test: `tests/blender/test_fiber_ui.py` (create)

**Interfaces:**
- Consumes: `read_fiber` (Task 5), `GPUSolver(fiber=...)` (Task 4), the `SolverParams` fiber fields (Task 2).
- Produces: `MarrowSettings.fiber_enabled`, `.fiber_curve`, `.fiber_stiffness`, `.wave_amplitude`, `.wave_length`, `.wave_speed`, `.waveform`.

- [ ] **Step 1: Write the failing test**

Create `tests/blender/test_fiber_ui.py`:

```python
import bpy
import numpy as np


def _body_with_cage(location, with_curve=True):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    body = bpy.context.active_object
    body.marrow.resolution = 1.0
    if with_curve:
        data = bpy.data.curves.new(f"spine{location[0]}", type="CURVE")
        data.dimensions = "3D"
        spline = data.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (location[0] - 2.0, location[1], location[2], 1.0)
        spline.points[1].co = (location[0] + 2.0, location[1], location[2], 1.0)
        curve = bpy.data.objects.new(f"spine{location[0]}", data)
        bpy.context.collection.objects.link(curve)
        body.marrow.fiber_curve = curve
    bpy.ops.marrow.tetrahedralize()
    return body


def test_the_defaults_are_inert():
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(40.0, 0.0, 0.0))
    settings = bpy.context.active_object.marrow
    assert settings.fiber_enabled is False
    assert settings.fiber_curve is None
    assert settings.waveform == "SMOOTH"


def test_the_curve_slot_rejects_a_mesh():
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(50.0, 0.0, 0.0))
    body = bpy.context.active_object
    mesh_obj = bpy.data.objects.new("decoy", bpy.data.meshes.new("decoy"))
    bpy.context.collection.objects.link(mesh_obj)
    try:
        body.marrow.fiber_curve = mesh_obj
    except (TypeError, RuntimeError):
        return  # rejected outright, which is the stricter of the two outcomes
    assert body.marrow.fiber_curve is None, "poll must reject a non-curve"


def test_session_passes_fiber_stiffness_through_when_enabled():
    from marrow.blender.session import MarrowSession

    body = _body_with_cage((60.0, 0.0, 0.0))
    body.marrow.fiber_enabled = True
    body.marrow.fiber_stiffness = 1234.0
    session = MarrowSession(body)
    assert abs(session.params.fiber_k - 1234.0) < 1e-6
    assert session.solver is not None


def test_fiber_disabled_zeroes_the_stiffness():
    from marrow.blender.session import MarrowSession

    body = _body_with_cage((70.0, 0.0, 0.0))
    body.marrow.fiber_enabled = False
    body.marrow.fiber_stiffness = 1234.0
    session = MarrowSession(body)
    assert session.params.fiber_k == 0.0


def test_a_cage_with_no_fibers_still_builds_a_session():
    from marrow.blender.session import MarrowSession

    body = _body_with_cage((80.0, 0.0, 0.0), with_curve=False)
    body.marrow.fiber_enabled = True
    session = MarrowSession(body)
    assert session.solver is not None, "fiber on without baked data must not crash"


def _drawn_for(obj):
    """Panel controls actually offered, via the recorder test_panel_gating
    already uses. Blender hands out a real UILayout only inside a draw
    callback, so the panel is driven against a stand-in instead."""
    from test_panel_gating import _Layout, _Panel
    from marrow.blender.ui import MARROW_PT_panel

    bpy.context.view_layer.objects.active = obj
    layout = _Layout()
    MARROW_PT_panel.draw(_Panel(layout), bpy.context)
    return layout.drawn


def test_the_panel_offers_the_wave_controls_once_fibers_are_baked():
    body = _body_with_cage((90.0, 0.0, 0.0))
    body.marrow.fiber_enabled = True
    drawn = _drawn_for(body)
    for name in ("fiber_enabled", "fiber_curve", "wave_amplitude",
                 "wave_length", "wave_speed", "waveform"):
        assert ("prop", name) in drawn, f"{name} was not offered"


def test_the_panel_explains_itself_when_no_fibers_are_baked():
    """The curve is baked at Tetrahedralize, so setting it in the panel does
    nothing on its own. That must be said, not left to look broken."""
    body = _body_with_cage((100.0, 0.0, 0.0), with_curve=False)
    drawn = _drawn_for(body)
    labels = [text for kind, text in drawn if kind == "label"]
    assert any("Tetrahedralize" in text for text in labels), (
        f"no explanation offered, got {labels}"
    )
    assert ("prop", "wave_amplitude") not in drawn, (
        "wave controls drive nothing without baked fibers"
    )
```

`MarrowSession`'s constructor signature is whatever `session.py` already exposes — check it before writing these tests and match it; the assertions above are about `session.params` and `session.solver`, both of which exist today.

- [ ] **Step 2: Run it to verify it fails**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: FAIL, `AttributeError: 'MarrowSettings' object has no attribute 'fiber_enabled'`.

- [ ] **Step 3: Add the properties**

In `marrow/blender/ui.py`, inside `MarrowSettings`:

```python
    fiber_enabled: bpy.props.BoolProperty(
        name="Fiber",
        description=(
            "Contract along fiber directions baked from a curve. Needs a "
            "cage tetrahedralized with a Curve set below"
        ),
        default=False,
    )
    fiber_curve: bpy.props.PointerProperty(
        name="Curve",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "CURVE",
        description=(
            "Curve running along the body. Its tangent is the fiber "
            "direction and its arclength is the wave phase. Sampled once at "
            "Tetrahedralize, so changing it means tetrahedralizing again"
        ),
    )
    fiber_stiffness: bpy.props.FloatProperty(
        name="Fiber Stiffness",
        description="Resistance to stretch along the fiber, and how hard it pulls",
        default=1.0e4,
        min=0.0,
        soft_max=1.0e6,
    )
    wave_amplitude: bpy.props.FloatProperty(
        name="Amplitude",
        description=(
            "Peak contraction. 0.3 shortens to 70% of rest length at the "
            "crest of the wave"
        ),
        default=0.3,
        min=0.0,
        max=0.9,
    )
    wave_length: bpy.props.FloatProperty(
        name="Wavelength",
        description="Distance between crests, measured along the curve",
        default=1.0,
        min=1.0e-4,
        soft_max=10.0,
        unit="LENGTH",
    )
    wave_speed: bpy.props.FloatProperty(
        name="Speed",
        description=(
            "Cycles per second. The wave travels at Wavelength x Speed in "
            "world units per second; negative reverses it"
        ),
        default=1.0,
        soft_min=-10.0,
        soft_max=10.0,
    )
    waveform: bpy.props.EnumProperty(
        name="Waveform",
        description="Shape of the contraction pulse",
        items=[
            ("SMOOTH", "Smooth", "Cosine. Organic muscle"),
            ("SQUARE", "Square", "Hard on and off"),
        ],
        default="SMOOTH",
    )
```

- [ ] **Step 4: Draw the box**

In `MARROW_PT_panel.draw`, after the Pin box and before whatever follows it, add:

```python
        # Below the elastic settings because fiber is a material term like
        # them, and above the contact boxes because it is not contact.
        fiber = sim.box()
        fiber.prop(settings, "fiber_enabled")
        column = fiber.column()
        column.enabled = settings.fiber_enabled
        column.prop(settings, "fiber_curve")

        from .storage import read_fiber

        cage_obj = find_cage(obj)
        if read_fiber(cage_obj.data) is None:
            # The curve is baked at Tetrahedralize, so setting it here does
            # nothing on its own. Say that rather than let it look broken.
            column.label(text="Tetrahedralize to bake fibers", icon="INFO")
        else:
            column.prop(settings, "fiber_stiffness")
            column.prop(settings, "wave_amplitude")
            column.prop(settings, "wave_length")
            column.prop(settings, "wave_speed")
            column.prop(settings, "waveform")
```

`find_cage` is already imported in `draw` above; reuse that binding rather than importing it twice.

- [ ] **Step 5: Wire the session**

In `marrow/blender/session.py`, add `read_fiber` to the storage import.

In the settings read where `SolverParams(...)` is built (around line 280):

```python
        self.params = SolverParams(
            substeps=int(settings.substeps),
            mu=float(settings.stiffness),
            lam=float(settings.volume_preservation),
            damping=float(settings.damping),
            # Zero when the toggle is off, so the kernel branch goes dead
            # rather than the pass being conditionally dispatched.
            fiber_k=(
                float(settings.fiber_stiffness) if settings.fiber_enabled else 0.0
            ),
            wave_amp=float(settings.wave_amplitude),
            wave_len=float(settings.wave_length),
            wave_speed=float(settings.wave_speed),
            waveform=0 if settings.waveform == "SMOOTH" else 1,
        )
```

Where `read_blend` is called (around line 107):

```python
        self.fiber = read_fiber(cage_obj.data)
```

And in `_build_solver`, pass it to the constructor:

```python
            fiber=self.fiber,
```

- [ ] **Step 6: Run the Blender suite**

Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: PASS for `test_fiber_ui`, and the two `test_fiber_bake` tests that were waiting on `fiber_curve` now pass too.

- [ ] **Step 7: Commit**

```bash
git add marrow/blender/ui.py marrow/blender/session.py tests/blender/test_fiber_ui.py
git commit -m "feat: fiber panel, properties and session wiring"
```

---

### Task 8: Demo scene and documentation

**Files:**
- Create: `tools/fiber_demo.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above, through the public operators and properties.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the demo script**

Create `tools/fiber_demo.py`:

```python
"""Build a fiber locomotion scene. Run from Blender's Scripting tab.

Not a test: a snake crawling forward is the acceptance criterion for this
feature but makes a slow, flaky assertion. This puts the scene on screen so
it can be judged by eye, which is the honest way to judge it.
"""

import bpy

LENGTH = 6.0
RADIUS = 0.35


def build():
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=RADIUS, depth=LENGTH, location=(0.0, 0.0, RADIUS),
        rotation=(0.0, 1.5708, 0.0), vertices=24,
    )
    body = bpy.context.active_object
    body.name = "Snake"

    data = bpy.data.curves.new("Spine", type="CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (-LENGTH / 2.0, 0.0, RADIUS, 1.0)
    spline.points[1].co = (LENGTH / 2.0, 0.0, RADIUS, 1.0)
    spine = bpy.data.objects.new("Spine", data)
    bpy.context.collection.objects.link(spine)

    settings = body.marrow
    settings.resolution = 0.18
    settings.fiber_curve = spine
    bpy.context.view_layer.objects.active = body
    bpy.ops.marrow.tetrahedralize()

    settings.fiber_enabled = True
    settings.fiber_stiffness = 3.0e4
    settings.wave_amplitude = 0.35
    settings.wave_length = 1.5
    settings.wave_speed = 1.2
    settings.waveform = "SMOOTH"
    # Locomotion is contraction plus grip. With friction at zero the wave
    # travels and the body goes nowhere.
    settings.friction = 0.8
    settings.ground_enabled = True
    settings.ground_z = 0.0
    settings.substeps = 20

    print(f"Snake built: {len(bpy.data.objects['Snake_cage'].data.vertices)} nodes. Press play.")


build()
```

The cage object's name follows `CAGE_SUFFIX` in `ops.py`; check its value and use it in that final print rather than the literal `_cage` if they differ.

- [ ] **Step 2: Run the demo and watch it**

Run: open Blender 5.2, Scripting tab, run `tools/fiber_demo.py`, press play.
Expected: the tube shows a travelling contraction wave and creeps along +X or -X. If it flails without translating, friction is too low; if it barely deforms, `fiber_stiffness` is too low relative to `stiffness`.

- [ ] **Step 3: Update the README**

Change the limitations line:

```
- **No plasticity, anisotropy or per-region materials.**
```

to:

```
- **No plasticity or per-region materials.** Fiber adds anisotropy along one baked direction; `mu` and `lam` are still global.
```

Add to the Settings table, after **Damping**:

| Setting | What it does |
|---|---|
| **Fiber** | Contract along baked fiber directions. See [Fiber](#fiber). |
| **Curve** | Curve running along the body. Its tangent is the fiber direction, its arclength the wave phase. |
| **Fiber Stiffness** | Resistance to stretch along the fiber, and how hard it pulls. |
| **Amplitude** | Peak contraction. 0.3 shortens to 70% of rest length. |
| **Wavelength** | Distance between crests, along the curve. |
| **Speed** | Cycles per second. Travel velocity is Wavelength x Speed. |
| **Waveform** | Smooth cosine, or hard on/off square. |

Add a `### Fiber` section after `### Tearing`:

```markdown
### Fiber

Every tet in Marrow is otherwise isotropic: it resists distortion equally in
all directions. **Fiber** adds one direction that is different, and then
drives it - the tet is told to shorten along its fiber, and the volume
constraint puts the material it displaces out sideways. That is a muscle.

Directions come from a **Curve** you point at the body. Each tet takes the
tangent at the nearest point on that curve, and the arclength at that point
as its place in the wave, so the contraction travels from one end to the
other instead of the whole body pulsing at once. Both are sampled at
Tetrahedralize and frozen, because a fiber direction is a property of the
rest shape - change the curve and tetrahedralize again.

The wave itself is procedural, not keyframed. **Wavelength** and **Speed**
set its shape and how fast it travels: the crest moves at Wavelength x Speed
in world units per second, and a negative Speed sends it the other way.
**Amplitude** is how hard it squeezes. **Waveform** picks a smooth cosine or
a hard on/off square.

Contraction alone does not move a body anywhere. Locomotion is contraction
plus grip, so a crawling creature needs [Friction](#friction) above zero and
something to push against. `tools/fiber_demo.py` builds a working scene to
start from.

A torn tet loses its fiber along with its stiffness. Torn muscle does not
pull.
```

- [ ] **Step 4: Run both suites one last time**

Run: `.venv/Scripts/python -m pytest tests/core`
Run: `blender -b --factory-startup --python tests/blender/run_tests.py`
Expected: both green. The Blender suite was 254 tests before this work and gains roughly 20.

- [ ] **Step 5: Commit**

```bash
git add tools/fiber_demo.py README.md
git commit -m "docs: fiber constraints, settings table and demo scene"
```

---

## Notes for the implementer

**The oracle is not optional.** Task 2 lands before Task 3 on purpose. If the GPU term is written first and the oracle second, the oracle gets written to agree with whatever the shader already does, and the one mechanism in this repo for catching a sign error is quietly disabled.

**Fiber rows are per-tet and tets get permuted.** `GPUSolver` colour-orders its tets at build time (`self._tet_order`). Any array indexed by tet - fiber included - must ride that same permutation. This is the single most likely bug in this plan, and it does not crash: it produces a body that contracts in plausible-looking wrong directions.

**`t` means two things.** In `solve_constraints` today it is the tet loop index. Task 2 renames it to `t_i` so `t` can be simulation time, matching the kernel. Do the rename completely in one edit.

**The GPU suite is occasionally flaky.** A full run sometimes reports a few failures that pass when their module runs alone, and the failing set changes between runs. Re-run before treating red as a regression, and check you invoked Blender 5.2.
