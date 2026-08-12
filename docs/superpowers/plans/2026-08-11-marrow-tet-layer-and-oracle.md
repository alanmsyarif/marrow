# Marrow Tet Layer and CPU Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Marrow's tetrahedral cage layer (generate, store, bind, colour) plus a numpy CPU reference solver that will serve as the test oracle for the GPU kernels in the next plan.

**Architecture:** All simulation and geometry math lives in `marrow/core/`, which never imports `bpy` and is tested with ordinary pytest. A thin `marrow/blender/` layer adapts it to Blender: BVH inside-tests, attribute storage, operators, UI. This split is what makes the math testable at all, since Blender's bundled Python cannot run pytest.

**Tech Stack:** Python 3.12+ (tests) / 3.13 (Blender runtime), numpy, Blender 5.2 `bpy` and `mathutils`, pytest.

## Global Constraints

- Blender 5.2.0 LTS is the only supported target. Addon manifest declares `blender_version_min = "5.2.0"`.
- `marrow/core/` MUST NOT import `bpy` or `mathutils`. Enforced by a test.
- No dependency outside numpy and Blender's bundled modules. No scipy, no torch, no compiled extension.
- Never `pip install` into Blender's bundled Python. Test dependencies go in a venv built on system Python 3.12.
- Arrays: positions `float64` in core, `int32` for indices. Conversion to `float32` happens only at the GPU boundary, which is out of scope for this plan.
- Every tet must have positive signed volume. Negative-volume tets are a defect, not a tolerance.
- Product name is "Marrow" everywhere in user-facing strings.

---

## File Structure

| Path | Responsibility |
|---|---|
| `marrow/__init__.py` | Addon entry, `register()` / `unregister()` |
| `marrow/blender_manifest.toml` | Extension manifest |
| `marrow/core/__init__.py` | Namespace only |
| `marrow/core/tetmesh.py` | `TetMesh` container, volumes, orientation repair |
| `marrow/core/lattice.py` | Cube-split tet lattice over a voxel grid |
| `marrow/core/bind.py` | Barycentric bind of render verts to tets, and deform |
| `marrow/core/coloring.py` | Greedy graph colouring of tets by shared nodes |
| `marrow/core/solver_ref.py` | numpy XPBD stable neo-Hookean reference solver |
| `marrow/blender/__init__.py` | Namespace only |
| `marrow/blender/inside_bvh.py` | BVHTree inside/outside test producing a cell mask |
| `marrow/blender/storage.py` | TetMesh and bind data to/from mesh attributes |
| `marrow/blender/ops.py` | `MARROW_OT_tetrahedralize` operator |
| `marrow/blender/ui.py` | Properties and N-panel |
| `tests/core/*` | pytest, pure numpy |
| `tests/blender/run_tests.py` | Assert-based runner executed by `blender -b` |
| `tools/spike_00_gpu_context.py` | Done, build step 0 |

---

### Task 1: Test harness and the bpy-free boundary

**Files:**
- Create: `marrow/__init__.py`, `marrow/core/__init__.py`, `marrow/blender/__init__.py`
- Create: `pytest.ini`, `requirements-dev.txt`, `.gitignore`
- Test: `tests/core/test_no_bpy.py`

**Interfaces:**
- Consumes: nothing
- Produces: the `marrow.core` package import path, and the venv + pytest workflow every later task uses

- [ ] **Step 1: Create the venv and install pytest**

```bash
cd /c/Users/user/Documents/marrow
"/c/Users/user/AppData/Local/Programs/Python/Python312/python.exe" -m venv .venv
./.venv/Scripts/python.exe -m pip install -q --upgrade pip
./.venv/Scripts/python.exe -m pip install -q pytest numpy
./.venv/Scripts/python.exe -m pytest --version
```

Expected: prints a pytest version. If `pip` cannot reach the network, stop and report; there is no offline fallback for pytest.

- [ ] **Step 2: Write the boundary test**

`tests/core/test_no_bpy.py`:

```python
"""marrow.core must never depend on Blender. This is what makes it testable."""
import ast
import pathlib

CORE = pathlib.Path(__file__).resolve().parents[2] / "marrow" / "core"
FORBIDDEN = {"bpy", "mathutils", "gpu", "bmesh", "gpu_extras", "bpy_extras", "blf", "aud", "bl_math"}


def _imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


def test_core_has_no_blender_imports():
    offenders = {}
    for path in CORE.rglob("*.py"):
        bad = _imported_roots(path) & FORBIDDEN
        if bad:
            offenders[path.name] = sorted(bad)
    assert offenders == {}, f"marrow.core must not import Blender modules: {offenders}"


def test_core_package_imports_standalone():
    import marrow.core  # noqa: F401
```

- [ ] **Step 3: Run it to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_no_bpy.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow'`

- [ ] **Step 4: Create the packages and config**

`marrow/__init__.py`:

```python
"""Marrow: GPU tetrahedral soft body for Blender."""

__version__ = "0.1.0"
```

`marrow/core/__init__.py`:

```python
"""Blender-free geometry and simulation math. Never import bpy here."""
```

`marrow/blender/__init__.py`:

```python
"""Blender adapters for marrow.core."""
```

`pytest.ini`:

```ini
[pytest]
testpaths = tests/core
pythonpath = .
```

`requirements-dev.txt`:

```
pytest
numpy
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
*.blend1
.superpowers/
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/core -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add marrow pytest.ini requirements-dev.txt .gitignore tests
git commit -m "test: add core test harness and bpy-free boundary check"
```

---

### Task 2: TetMesh container, volumes, orientation repair

**Files:**
- Create: `marrow/core/tetmesh.py`
- Test: `tests/core/test_tetmesh.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `TetMesh(nodes: np.ndarray, tets: np.ndarray)` frozen dataclass, `nodes` shape `(N,3)` float64, `tets` shape `(T,4)` int32
  - `signed_volumes(nodes, tets) -> np.ndarray` shape `(T,)` float64
  - `repair_orientation(tets, nodes) -> np.ndarray` shape `(T,4)` int32, swaps the last two indices of any negative-volume tet
  - `TetMesh.validate() -> None`, raises `ValueError`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_tetmesh.py`:

```python
import numpy as np
import pytest

from marrow.core.tetmesh import TetMesh, repair_orientation, signed_volumes

# Unit tet: volume 1/6
UNIT_NODES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
UNIT_TET = np.array([[0, 1, 2, 3]], dtype=np.int32)


def test_signed_volume_of_unit_tet():
    vols = signed_volumes(UNIT_NODES, UNIT_TET)
    assert vols.shape == (1,)
    assert np.isclose(vols[0], 1.0 / 6.0)


def test_signed_volume_is_negative_when_wound_backwards():
    flipped = np.array([[0, 1, 3, 2]], dtype=np.int32)
    assert signed_volumes(UNIT_NODES, flipped)[0] < 0


def test_repair_orientation_makes_all_volumes_positive():
    flipped = np.array([[0, 1, 3, 2]], dtype=np.int32)
    fixed = repair_orientation(flipped, UNIT_NODES)
    assert np.all(signed_volumes(UNIT_NODES, fixed) > 0)


def test_repair_orientation_leaves_good_tets_untouched():
    fixed = repair_orientation(UNIT_TET, UNIT_NODES)
    assert np.array_equal(fixed, UNIT_TET)


def test_validate_rejects_negative_volume():
    mesh = TetMesh(UNIT_NODES, np.array([[0, 1, 3, 2]], dtype=np.int32))
    with pytest.raises(ValueError, match="negative volume"):
        mesh.validate()


def test_validate_rejects_out_of_range_index():
    mesh = TetMesh(UNIT_NODES, np.array([[0, 1, 2, 9]], dtype=np.int32))
    with pytest.raises(ValueError, match="out of range"):
        mesh.validate()


def test_validate_rejects_duplicate_node_in_tet():
    mesh = TetMesh(UNIT_NODES, np.array([[0, 1, 1, 3]], dtype=np.int32))
    with pytest.raises(ValueError, match="repeated node"):
        mesh.validate()


def test_validate_accepts_good_mesh():
    TetMesh(UNIT_NODES, UNIT_TET).validate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_tetmesh.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.core.tetmesh'`

- [ ] **Step 3: Implement**

`marrow/core/tetmesh.py`:

```python
"""Tetrahedral mesh container and orientation invariants."""

from dataclasses import dataclass

import numpy as np


def signed_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volume of every tet. Positive means correct winding."""
    p0 = nodes[tets[:, 0]]
    e1 = nodes[tets[:, 1]] - p0
    e2 = nodes[tets[:, 2]] - p0
    e3 = nodes[tets[:, 3]] - p0
    return np.einsum("ij,ij->i", np.cross(e1, e2), e3) / 6.0


def repair_orientation(tets: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Swap the last two indices of any tet with negative signed volume."""
    fixed = np.array(tets, dtype=np.int32, copy=True)
    negative = signed_volumes(nodes, fixed) < 0.0
    fixed[negative, 2], fixed[negative, 3] = fixed[negative, 3], fixed[negative, 2].copy()
    return fixed


@dataclass(frozen=True)
class TetMesh:
    nodes: np.ndarray  # (N, 3) float64
    tets: np.ndarray   # (T, 4) int32

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[0])

    def validate(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"nodes must be (N,3), got {self.nodes.shape}")
        if self.tets.ndim != 2 or self.tets.shape[1] != 4:
            raise ValueError(f"tets must be (T,4), got {self.tets.shape}")
        if self.n_tets and (self.tets.min() < 0 or self.tets.max() >= self.n_nodes):
            raise ValueError("tet node index out of range")
        for row in self.tets:
            if len(set(row.tolist())) != 4:
                raise ValueError(f"tet has a repeated node: {row.tolist()}")
        bad = int(np.count_nonzero(signed_volumes(self.nodes, self.tets) <= 0.0))
        if bad:
            raise ValueError(f"{bad} tets have negative volume or are degenerate")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_tetmesh.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add marrow/core/tetmesh.py tests/core/test_tetmesh.py
git commit -m "feat: add TetMesh container with volume and orientation invariants"
```

---

### Task 3: Cube-split tet lattice

**Files:**
- Create: `marrow/core/lattice.py`
- Test: `tests/core/test_lattice.py`

**Note on terminology:** the spec says "BCC lattice, 5 tets per cell". Those are two different schemes. What this task implements, and what the spec means, is the **5-tet cube subdivision with checkerboard parity alternation**: each occupied voxel is split into 5 tets, and the split pattern alternates by `(i+j+k) % 2` so that neighbouring cells agree on their shared face diagonals and the mesh stays conforming. Update the spec's wording when this task lands.

**Interfaces:**
- Consumes: `TetMesh`, `repair_orientation` from Task 2
- Produces:
  - `grid_dims(bounds_min, bounds_max, spacing) -> tuple[int, int, int]`
  - `build_lattice(bounds_min, spacing, cell_mask) -> TetMesh` where `cell_mask` is a bool array `(nx, ny, nz)`; only nodes touched by kept cells appear in the output

- [ ] **Step 1: Write the failing tests**

`tests/core/test_lattice.py`:

```python
import numpy as np

from marrow.core.lattice import build_lattice, grid_dims
from marrow.core.tetmesh import signed_volumes


def test_grid_dims_rounds_up_to_cover_bounds():
    dims = grid_dims(np.zeros(3), np.array([1.0, 1.0, 1.0]), 0.4)
    assert dims == (3, 3, 3)


def test_grid_dims_is_at_least_one_cell():
    assert grid_dims(np.zeros(3), np.array([0.01, 0.01, 0.01]), 1.0) == (1, 1, 1)


def test_single_cell_makes_five_tets():
    mask = np.ones((1, 1, 1), dtype=bool)
    mesh = build_lattice(np.zeros(3), 1.0, mask)
    assert mesh.n_tets == 5
    assert mesh.n_nodes == 8


def test_all_tets_have_positive_volume():
    mask = np.ones((3, 3, 3), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    assert np.all(signed_volumes(mesh.nodes, mesh.tets) > 0)
    mesh.validate()


def test_tet_volumes_sum_to_cell_volume():
    mask = np.ones((2, 2, 2), dtype=bool)
    spacing = 0.5
    mesh = build_lattice(np.zeros(3), spacing, mask)
    total = signed_volumes(mesh.nodes, mesh.tets).sum()
    assert np.isclose(total, 8 * spacing**3)


def test_empty_mask_yields_empty_mesh():
    mesh = build_lattice(np.zeros(3), 1.0, np.zeros((2, 2, 2), dtype=bool))
    assert mesh.n_tets == 0
    assert mesh.n_nodes == 0


def test_unused_nodes_are_dropped():
    mask = np.zeros((2, 1, 1), dtype=bool)
    mask[0, 0, 0] = True
    mesh = build_lattice(np.zeros(3), 1.0, mask)
    assert mesh.n_nodes == 8  # only the one kept cell's corners


def test_parity_split_is_conforming_across_neighbours():
    """Two adjacent cells must share exactly the 4 nodes of their common face."""
    mask = np.ones((2, 1, 1), dtype=bool)
    mesh = build_lattice(np.zeros(3), 1.0, mask)
    assert mesh.n_nodes == 12  # 8 + 4 new, shared face reused
    assert mesh.n_tets == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_lattice.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.core.lattice'`

- [ ] **Step 3: Implement**

`marrow/core/lattice.py`:

```python
"""Conforming tet lattice by 5-tet cube subdivision with parity alternation."""

import numpy as np

from marrow.core.tetmesh import TetMesh, repair_orientation

# Cube corner order: bit 0 = x, bit 1 = y, bit 2 = z.
_CORNER_OFFSETS = np.array(
    [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.int64
)

# Five-tet splits. The two patterns are mirror images; alternating them by
# (i+j+k) parity makes neighbouring cells agree on shared face diagonals.
_SPLIT_EVEN = np.array(
    [[0, 1, 2, 4], [1, 3, 2, 7], [1, 4, 5, 7], [2, 4, 6, 7], [1, 2, 4, 7]],
    dtype=np.int64,
)
_SPLIT_ODD = np.array(
    [[0, 1, 3, 5], [0, 3, 2, 6], [0, 5, 4, 6], [3, 5, 7, 6], [0, 3, 5, 6]],
    dtype=np.int64,
)


def grid_dims(bounds_min, bounds_max, spacing: float) -> tuple[int, int, int]:
    """Cell counts covering the bounds, at least 1 per axis."""
    extent = np.asarray(bounds_max, dtype=np.float64) - np.asarray(
        bounds_min, dtype=np.float64
    )
    counts = np.ceil(extent / float(spacing)).astype(np.int64)
    counts = np.maximum(counts, 1)
    return tuple(int(c) for c in counts)


def build_lattice(bounds_min, spacing: float, cell_mask: np.ndarray) -> TetMesh:
    """Build a conforming tet mesh over every True cell of ``cell_mask``."""
    mask = np.asarray(cell_mask, dtype=bool)
    nx, ny, nz = mask.shape
    occupied = np.argwhere(mask)  # (M, 3) cell coordinates

    if occupied.size == 0:
        return TetMesh(
            np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.int32)
        )

    # Global corner-lattice index: (nx+1, ny+1, nz+1) grid of potential nodes.
    def corner_id(ijk):
        return (ijk[..., 0] * (ny + 1) + ijk[..., 1]) * (nz + 1) + ijk[..., 2]

    # (M, 8, 3) corner coordinates for each occupied cell.
    cell_corners = occupied[:, None, :] + _CORNER_OFFSETS[None, :, :]
    cell_corner_ids = corner_id(cell_corners)  # (M, 8)

    used_ids, inverse = np.unique(cell_corner_ids.ravel(), return_inverse=True)
    local = inverse.reshape(cell_corner_ids.shape)  # (M, 8) compacted indices

    # Recover lattice coordinates of the used corners to build positions.
    zi = used_ids % (nz + 1)
    yi = (used_ids // (nz + 1)) % (ny + 1)
    xi = used_ids // ((nz + 1) * (ny + 1))
    lattice = np.stack([xi, yi, zi], axis=1).astype(np.float64)
    nodes = np.asarray(bounds_min, dtype=np.float64) + lattice * float(spacing)

    parity = (occupied.sum(axis=1) % 2).astype(bool)  # (M,)
    splits = np.where(parity[:, None, None], _SPLIT_ODD[None], _SPLIT_EVEN[None])

    # local[m, splits[m, t, c]] -> (M, 5, 4)
    tets = np.take_along_axis(local[:, None, :], splits, axis=2)
    tets = tets.reshape(-1, 4).astype(np.int32)
    tets = repair_orientation(tets, nodes)

    return TetMesh(nodes, tets)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_lattice.py -v`
Expected: 8 passed. If `test_parity_split_is_conforming_across_neighbours` fails on node count, the two split tables disagree on the shared face diagonal — recheck `_SPLIT_ODD` against `_SPLIT_EVEN` before touching anything else.

- [ ] **Step 5: Update the spec terminology**

In `docs/superpowers/specs/2026-08-11-marrow-gpu-tet-soft-body-design.md`, replace "BCC lattice, 5 tets per cell. Uniform quality by construction, no slivers." with "5-tet cube subdivision with checkerboard parity alternation, which keeps neighbouring cells conforming. Uniform quality by construction, no slivers." Replace other bare "BCC" mentions with "cube-split lattice".

- [ ] **Step 6: Commit**

```bash
git add marrow/core/lattice.py tests/core/test_lattice.py docs/superpowers/specs
git commit -m "feat: add conforming 5-tet cube-split lattice"
```

---

### Task 4: Barycentric bind and deform

**Files:**
- Create: `marrow/core/bind.py`
- Test: `tests/core/test_bind.py`

**Interfaces:**
- Consumes: `TetMesh` from Task 2
- Produces:
  - `bind_points(nodes, tets, points) -> tuple[np.ndarray, np.ndarray]`, returning `(P,)` int32 tet indices and `(P,4)` float64 weights. A point outside every tet binds to the nearest tet by centroid distance, with clamped weights.
  - `deform(nodes, tets, bind_idx, bind_w) -> np.ndarray` shape `(P,3)`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_bind.py`:

```python
import numpy as np

from marrow.core.bind import bind_points, deform
from marrow.core.lattice import build_lattice

NODES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
TETS = np.array([[0, 1, 2, 3]], dtype=np.int32)


def test_corner_binds_to_pure_weight():
    idx, w = bind_points(NODES, TETS, NODES[1][None, :])
    assert idx[0] == 0
    assert np.allclose(w[0], [0.0, 1.0, 0.0, 0.0], atol=1e-9)


def test_weights_sum_to_one():
    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.3, 0.4], [0.0, 0.0, 0.0]])
    _, w = bind_points(NODES, TETS, pts)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_interior_weights_are_non_negative():
    pts = np.array([[0.2, 0.2, 0.2]])
    _, w = bind_points(NODES, TETS, pts)
    assert np.all(w >= -1e-12)


def test_deform_reproduces_original_points_when_cage_unmoved():
    mask = np.ones((2, 2, 2), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.05, 0.95, size=(50, 3))
    idx, w = bind_points(mesh.nodes, mesh.tets, pts)
    out = deform(mesh.nodes, mesh.tets, idx, w)
    assert np.allclose(out, pts, atol=1e-9)


def test_deform_follows_a_rigid_translation():
    mask = np.ones((2, 2, 2), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    pts = np.array([[0.3, 0.4, 0.5], [0.7, 0.2, 0.9]])
    idx, w = bind_points(mesh.nodes, mesh.tets, pts)
    shift = np.array([1.0, -2.0, 0.5])
    out = deform(mesh.nodes + shift, mesh.tets, idx, w)
    assert np.allclose(out, pts + shift, atol=1e-9)


def test_point_outside_all_tets_still_binds():
    pts = np.array([[5.0, 5.0, 5.0]])
    idx, w = bind_points(NODES, TETS, pts)
    assert idx[0] == 0
    assert np.isclose(w.sum(), 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_bind.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.core.bind'`

- [ ] **Step 3: Implement**

`marrow/core/bind.py`:

```python
"""Bind arbitrary points into a tet cage and deform them with it."""

import numpy as np


def _barycentric_all(nodes: np.ndarray, tets: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Barycentric coords of one point in every tet. Returns (T, 4)."""
    p0 = nodes[tets[:, 0]]
    m = np.stack(
        [
            nodes[tets[:, 1]] - p0,
            nodes[tets[:, 2]] - p0,
            nodes[tets[:, 3]] - p0,
        ],
        axis=2,
    )  # (T, 3, 3) columns are the edge vectors
    rhs = point[None, :] - p0  # (T, 3)
    solved = np.linalg.solve(m, rhs[..., None])[..., 0]  # (T, 3)
    first = 1.0 - solved.sum(axis=1)
    return np.concatenate([first[:, None], solved], axis=1)


def bind_points(nodes: np.ndarray, tets: np.ndarray, points: np.ndarray):
    """Bind each point to a containing tet, falling back to the nearest one."""
    points = np.asarray(points, dtype=np.float64)
    n_points = points.shape[0]
    idx = np.zeros(n_points, dtype=np.int32)
    weights = np.zeros((n_points, 4), dtype=np.float64)

    centroids = nodes[tets].mean(axis=1)  # (T, 3)

    for i in range(n_points):
        bary = _barycentric_all(nodes, tets, points[i])
        inside = np.all(bary >= -1e-9, axis=1)
        if np.any(inside):
            # Prefer the most interior containment for numerical comfort.
            candidates = np.flatnonzero(inside)
            best = candidates[np.argmax(bary[candidates].min(axis=1))]
            w = bary[best]
        else:
            best = int(np.argmin(np.linalg.norm(centroids - points[i], axis=1)))
            w = np.clip(bary[best], 0.0, None)
            total = w.sum()
            w = w / total if total > 0 else np.full(4, 0.25)
        idx[i] = best
        weights[i] = w

    return idx, weights


def deform(nodes: np.ndarray, tets: np.ndarray, bind_idx: np.ndarray, bind_w: np.ndarray):
    """Interpolate point positions from the current cage node positions."""
    corners = nodes[tets[bind_idx]]  # (P, 4, 3)
    return np.einsum("pij,pi->pj", corners, bind_w)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_bind.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add marrow/core/bind.py tests/core/test_bind.py
git commit -m "feat: add barycentric bind and deform"
```

---

### Task 5: Tet graph colouring

**Files:**
- Create: `marrow/core/coloring.py`
- Test: `tests/core/test_coloring.py`

**Interfaces:**
- Consumes: nothing beyond numpy
- Produces:
  - `color_tets(tets, n_nodes) -> np.ndarray` shape `(T,)` int32
  - `color_groups(colors) -> list[np.ndarray]`, one int32 index array per colour

- [ ] **Step 1: Write the failing tests**

`tests/core/test_coloring.py`:

```python
import numpy as np

from marrow.core.coloring import color_groups, color_tets
from marrow.core.lattice import build_lattice


def test_disjoint_tets_share_one_color():
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    colors = color_tets(tets, 8)
    assert colors[0] == colors[1] == 0


def test_tets_sharing_a_node_get_different_colors():
    tets = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    colors = color_tets(tets, 7)
    assert colors[0] != colors[1]


def test_no_two_tets_in_a_color_share_a_node():
    mask = np.ones((3, 3, 3), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    colors = color_tets(mesh.tets, mesh.n_nodes)
    for group in color_groups(colors):
        seen = set()
        for tet in mesh.tets[group]:
            nodes = set(tet.tolist())
            assert not (seen & nodes), "colour group has a node collision"
            seen |= nodes


def test_every_tet_gets_a_color():
    mask = np.ones((2, 2, 2), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    colors = color_tets(mesh.tets, mesh.n_nodes)
    assert colors.shape == (mesh.n_tets,)
    assert np.all(colors >= 0)


def test_color_groups_partition_all_tets():
    mask = np.ones((2, 2, 2), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    groups = color_groups(color_tets(mesh.tets, mesh.n_nodes))
    assert sum(len(g) for g in groups) == mesh.n_tets
    assert len(set(np.concatenate(groups).tolist())) == mesh.n_tets


def test_empty_input():
    colors = color_tets(np.zeros((0, 4), dtype=np.int32), 0)
    assert colors.shape == (0,)
    assert color_groups(colors) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_coloring.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.core.coloring'`

- [ ] **Step 3: Implement**

`marrow/core/coloring.py`:

```python
"""Greedy colouring so tets in a colour never share a node.

This is what lets the GPU solve kernel do plain read-modify-write on node
positions with no atomics: within one dispatch, no two threads touch the
same node.
"""

import numpy as np


def color_tets(tets: np.ndarray, n_nodes: int) -> np.ndarray:
    """Assign each tet a colour such that a colour's tets are node-disjoint."""
    n_tets = int(tets.shape[0])
    colors = np.full(n_tets, -1, dtype=np.int32)
    if n_tets == 0:
        return colors

    # node_color_used[node] is the set of colours already claimed at that node.
    node_colors: list[set[int]] = [set() for _ in range(int(n_nodes))]

    for t in range(n_tets):
        nodes = tets[t]
        taken = set()
        for n in nodes:
            taken |= node_colors[int(n)]
        c = 0
        while c in taken:
            c += 1
        colors[t] = c
        for n in nodes:
            node_colors[int(n)].add(c)

    return colors


def color_groups(colors: np.ndarray) -> list[np.ndarray]:
    """Split tet indices into one int32 array per colour, in colour order."""
    if colors.size == 0:
        return []
    return [
        np.flatnonzero(colors == c).astype(np.int32)
        for c in range(int(colors.max()) + 1)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_coloring.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add marrow/core/coloring.py tests/core/test_coloring.py
git commit -m "feat: add node-disjoint tet colouring"
```

---

### Task 6: CPU reference solver, integration only

**Files:**
- Create: `marrow/core/solver_ref.py`
- Test: `tests/core/test_solver_integration.py`

**Interfaces:**
- Consumes: `TetMesh` from Task 2
- Produces:
  - `SolverParams` dataclass with fields `dt: float = 1/24`, `substeps: int = 10`, `gravity: tuple = (0.0, 0.0, -9.81)`, `mu: float = 1.0e4`, `lam: float = 1.0e5`, `damping: float = 0.999`
  - `SolverState` dataclass with fields `nodes`, `velocities`, `inv_mass`
  - `make_state(nodes, density=1.0, pinned=None) -> SolverState`
  - `step(state, tets, dm_inv, rest_vol, params) -> None`, mutating in place
  - `precompute(nodes, tets) -> tuple[np.ndarray, np.ndarray]` returning `(T,3,3)` `dm_inv` and `(T,)` `rest_vol`

  In this task `step` performs predict, no constraint solve, then integrate. Task 7 adds the constraint solve.

- [ ] **Step 1: Write the failing tests**

`tests/core/test_solver_integration.py`:

```python
import numpy as np

from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import SolverParams, make_state, precompute, step

CUBE = build_lattice(np.zeros(3), 1.0, np.ones((1, 1, 1), dtype=bool))


def test_precompute_shapes_and_rest_volume():
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    assert dm_inv.shape == (CUBE.n_tets, 3, 3)
    assert rest_vol.shape == (CUBE.n_tets,)
    assert np.isclose(rest_vol.sum(), 1.0)
    assert np.all(rest_vol > 0)


def test_free_fall_matches_analytic_curve():
    """With no constraints and no pins, the body is in free fall."""
    params = SolverParams(dt=1 / 24, substeps=10, damping=1.0, mu=0.0, lam=0.0)
    state = make_state(CUBE.nodes)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)

    start_z = state.nodes[:, 2].copy()
    n_frames = 12
    for _ in range(n_frames):
        step(state, CUBE.tets, dm_inv, rest_vol, params)

    t = n_frames * params.dt
    expected_drop = 0.5 * 9.81 * t * t
    actual_drop = (start_z - state.nodes[:, 2]).mean()
    # Symplectic Euler over-integrates slightly; 2% is the honest tolerance.
    assert np.isclose(actual_drop, expected_drop, rtol=0.02)


def test_pinned_nodes_never_move():
    params = SolverParams(mu=0.0, lam=0.0)
    state = make_state(CUBE.nodes, pinned=np.array([0, 1], dtype=np.int32))
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    before = state.nodes[[0, 1]].copy()
    for _ in range(5):
        step(state, CUBE.tets, dm_inv, rest_vol, params)
    assert np.allclose(state.nodes[[0, 1]], before)


def test_pinned_nodes_have_zero_inverse_mass():
    state = make_state(CUBE.nodes, pinned=np.array([3], dtype=np.int32))
    assert state.inv_mass[3] == 0.0
    assert np.all(state.inv_mass[[0, 1, 2]] > 0.0)


def test_damping_reduces_speed():
    fast = SolverParams(damping=1.0, mu=0.0, lam=0.0)
    slow = SolverParams(damping=0.5, mu=0.0, lam=0.0)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)

    a = make_state(CUBE.nodes)
    b = make_state(CUBE.nodes)
    for _ in range(5):
        step(a, CUBE.tets, dm_inv, rest_vol, fast)
        step(b, CUBE.tets, dm_inv, rest_vol, slow)

    assert np.linalg.norm(b.velocities) < np.linalg.norm(a.velocities)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_solver_integration.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.core.solver_ref'`

- [ ] **Step 3: Implement**

`marrow/core/solver_ref.py`:

```python
"""numpy XPBD reference solver.

This exists to be *correct and readable*, not fast. It is the oracle the GPU
compute kernels are diffed against, because a sign error in a compute shader
is otherwise indistinguishable from a sign error in the constraint algebra.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolverParams:
    dt: float = 1.0 / 24.0
    substeps: int = 10
    gravity: tuple = (0.0, 0.0, -9.81)
    mu: float = 1.0e4       # deviatoric stiffness
    lam: float = 1.0e5      # hydrostatic (volume) stiffness
    damping: float = 0.999


@dataclass
class SolverState:
    nodes: np.ndarray        # (N, 3) float64
    velocities: np.ndarray   # (N, 3) float64
    inv_mass: np.ndarray     # (N,)   float64, 0 means pinned
    predicted: np.ndarray = field(default=None, repr=False)


def make_state(nodes: np.ndarray, density: float = 1.0, pinned=None) -> SolverState:
    nodes = np.array(nodes, dtype=np.float64, copy=True)
    inv_mass = np.full(nodes.shape[0], 1.0 / density, dtype=np.float64)
    if pinned is not None and len(pinned):
        inv_mass[np.asarray(pinned, dtype=np.int64)] = 0.0
    return SolverState(
        nodes=nodes,
        velocities=np.zeros_like(nodes),
        inv_mass=inv_mass,
        predicted=np.zeros_like(nodes),
    )


def precompute(nodes: np.ndarray, tets: np.ndarray):
    """Rest shape inverse and rest volume per tet."""
    p0 = nodes[tets[:, 0]]
    dm = np.stack(
        [nodes[tets[:, 1]] - p0, nodes[tets[:, 2]] - p0, nodes[tets[:, 3]] - p0],
        axis=2,
    )  # (T, 3, 3)
    dm_inv = np.linalg.inv(dm)
    rest_vol = np.linalg.det(dm) / 6.0
    return dm_inv, rest_vol


def step(state: SolverState, tets, dm_inv, rest_vol, params: SolverParams) -> None:
    """Advance one frame of ``params.substeps`` XPBD substeps, in place."""
    h = params.dt / params.substeps
    gravity = np.asarray(params.gravity, dtype=np.float64)
    movable = state.inv_mass > 0.0

    for _ in range(params.substeps):
        # predict
        state.predicted[:] = state.nodes
        state.predicted[movable] += (
            state.velocities[movable] * h + gravity * (h * h)
        )

        solve_constraints(state, tets, dm_inv, rest_vol, params, h)

        # integrate
        state.velocities[movable] = (
            (state.predicted[movable] - state.nodes[movable]) / h * params.damping
        )
        state.nodes[movable] = state.predicted[movable]


def solve_constraints(state, tets, dm_inv, rest_vol, params, h) -> None:
    """No-op until Task 7 adds the neo-Hookean constraints."""
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_solver_integration.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add marrow/core/solver_ref.py tests/core/test_solver_integration.py
git commit -m "feat: add XPBD reference solver integration loop"
```

---

### Task 7: Stable neo-Hookean constraints

**Files:**
- Modify: `marrow/core/solver_ref.py` (replace the `solve_constraints` stub)
- Test: `tests/core/test_solver_constraints.py`

**Interfaces:**
- Consumes: `SolverState`, `SolverParams`, `precompute` from Task 6
- Produces: a working `solve_constraints(state, tets, dm_inv, rest_vol, params, h) -> None`. Signature is unchanged from Task 6, so Task 6's `step` needs no edit.

**Math being implemented,** per tet, following Macklin and Muller's stable neo-Hookean formulation:

- `Ds` = current shape matrix, columns `p1-p0, p2-p0, p3-p0`. Deformation gradient `F = Ds @ dm_inv`.
- Deviatoric constraint: `C_D = sqrt(sum(F*F))`, i.e. the Frobenius norm, driven to zero with **no rest offset**. Gradient with respect to `F` is `F / C_D`.
- Hydrostatic constraint: `C_H = det(F) - gamma` with `gamma = 1 + mu/lam`. Gradient with respect to `F` has columns `cross(f1,f2), cross(f2,f0), cross(f0,f1)`.
- The two are a matched pair. `gamma`'s offset exists precisely to cancel the offset-free deviatoric term at `F = I`: the gradients there are `mu*I` and `-mu*I`, so rest is stress-free. Subtracting `sqrt(3)` from `C_D` as well zeroes that term a second time, leaving the volume constraint unopposed — measured, the body then settles permanently at `det(F) = gamma` (1.097x rest volume) with a drift that does not shrink as substeps refine. Do not mix the two conventions.
- Convert a gradient in `F` to gradients in node positions: `G = dCdF @ dm_inv.T` gives columns for nodes 1,2,3; node 0's gradient is `-(g1+g2+g3)`.
- XPBD update: `alpha_tilde = (1/stiffness) / (rest_vol * h*h)`, then
  `dlambda = (-C - alpha_tilde * lam_acc) / (sum_i inv_mass_i * |g_i|^2 + alpha_tilde)`,
  and `p_i += inv_mass_i * g_i * dlambda`.

- [ ] **Step 1: Write the failing tests**

`tests/core/test_solver_constraints.py`:

```python
import numpy as np

from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import (
    SolverParams,
    make_state,
    precompute,
    solve_constraints,
    step,
)

CUBE = build_lattice(np.zeros(3), 1.0, np.ones((1, 1, 1), dtype=bool))
BLOCK = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))


def _total_volume(nodes, tets):
    from marrow.core.tetmesh import signed_volumes

    return signed_volumes(nodes, tets).sum()


def _rest_hold(substeps, frames=5):
    """Hold an undeformed cube under no gravity. Returns (max drift, volume)."""
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=substeps)
    state = make_state(CUBE.nodes)
    before = state.nodes.copy()
    for _ in range(frames):
        step(state, CUBE.tets, dm_inv, rest_vol, params)
    return (
        float(np.abs(state.nodes - before).max()),
        _total_volume(state.nodes, CUBE.tets),
    )


def test_rest_configuration_converges_to_a_fixed_point():
    """An undeformed body under no gravity holds still as substeps refine.

    The rest state is stress-free in energy: gamma = 1 + mu/lam makes the
    deviatoric and hydrostatic gradients cancel exactly at F = I. But XPBD
    projects the two constraints sequentially, so each substep leaves a
    Gauss-Seidel residual. That residual is discretisation error, not bias,
    and must fall off with substep count. A formulation that zeroed the
    deviatoric term at rest instead would leave a drift that never converges
    and settle the body at det(F) = gamma.
    """
    coarse, _ = _rest_hold(4)
    fine, fine_volume = _rest_hold(40)
    assert fine < coarse / 100.0, f"rest drift did not converge: {coarse} -> {fine}"
    assert np.isclose(fine_volume, 1.0, rtol=1e-3), (
        f"cube did not hold its rest volume: {fine_volume}"
    )


def test_stretched_body_is_pulled_back():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=20)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    state = make_state(CUBE.nodes)
    state.nodes[:, 0] *= 1.5  # stretch along x
    stretched_span = np.ptp(state.nodes[:, 0])
    for _ in range(20):
        step(state, CUBE.tets, dm_inv, rest_vol, params)
    assert np.ptp(state.nodes[:, 0]) < stretched_span


def test_volume_is_preserved_under_compression():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=20, lam=1.0e7)
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    state = make_state(BLOCK.nodes)
    rest_total = rest_vol.sum()

    state.nodes[:, 2] *= 0.6  # squash in z
    for _ in range(40):
        step(state, BLOCK.tets, dm_inv, rest_vol, params)

    assert np.isclose(_total_volume(state.nodes, BLOCK.tets), rest_total, rtol=0.10)


def test_solver_stays_finite_under_extreme_deformation():
    params = SolverParams(substeps=10)
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    state = make_state(BLOCK.nodes)
    state.nodes *= 3.0
    for _ in range(30):
        step(state, BLOCK.tets, dm_inv, rest_vol, params)
    assert np.all(np.isfinite(state.nodes)), "solver produced NaN or inf"


def test_pinned_nodes_stay_put_with_constraints_active():
    params = SolverParams(substeps=10)
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    pinned = np.array([0, 1, 2], dtype=np.int32)
    state = make_state(BLOCK.nodes, pinned=pinned)
    before = state.nodes[pinned].copy()
    for _ in range(20):
        step(state, BLOCK.tets, dm_inv, rest_vol, params)
    assert np.allclose(state.nodes[pinned], before)


def test_solve_constraints_is_a_noop_at_zero_stiffness():
    params = SolverParams(mu=0.0, lam=0.0)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    state = make_state(CUBE.nodes)
    state.predicted[:] = state.nodes + 0.1
    before = state.predicted.copy()
    solve_constraints(state, CUBE.tets, dm_inv, rest_vol, params, 1e-3)
    assert np.allclose(state.predicted, before)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_solver_constraints.py -v`
Expected: FAIL. `test_stretched_body_is_pulled_back` and `test_volume_is_preserved_under_compression` fail because `solve_constraints` currently does nothing.

- [ ] **Step 3: Replace the stub**

In `marrow/core/solver_ref.py`, replace the entire `solve_constraints` function with:

```python
def _grads_from_dcdf(dcdf: np.ndarray, dm_inv: np.ndarray) -> np.ndarray:
    """Map a gradient in F to per-node gradients. Returns (4, 3)."""
    g123 = dcdf @ dm_inv.T          # (3, 3), columns are nodes 1, 2, 3
    g1, g2, g3 = g123[:, 0], g123[:, 1], g123[:, 2]
    g0 = -(g1 + g2 + g3)
    return np.stack([g0, g1, g2, g3], axis=0)


def _apply(state, nodes_idx, grads, c_value, compliance, h, lam_acc):
    """One XPBD constraint projection. Returns the updated multiplier."""
    w = state.inv_mass[nodes_idx]
    denom = float(np.sum(w * np.einsum("ij,ij->i", grads, grads)))
    alpha_tilde = compliance / (h * h)
    denom += alpha_tilde
    if denom < 1e-20:
        return lam_acc
    dlambda = (-c_value - alpha_tilde * lam_acc) / denom
    state.predicted[nodes_idx] += grads * (w[:, None] * dlambda)
    return lam_acc + dlambda


def solve_constraints(state, tets, dm_inv, rest_vol, params, h) -> None:
    """Stable neo-Hookean: one deviatoric and one hydrostatic constraint per tet."""
    if params.mu <= 0.0 and params.lam <= 0.0:
        return

    gamma = 1.0 + (params.mu / params.lam if params.lam > 0.0 else 0.0)
    n_tets = int(tets.shape[0])

    # Multipliers reset every substep, which is standard XPBD.
    lam_dev = np.zeros(n_tets, dtype=np.float64)
    lam_hyd = np.zeros(n_tets, dtype=np.float64)

    for t in range(n_tets):
        idx = tets[t]
        if not np.any(state.inv_mass[idx] > 0.0):
            continue

        p0 = state.predicted[idx[0]]
        ds = np.stack(
            [
                state.predicted[idx[1]] - p0,
                state.predicted[idx[2]] - p0,
                state.predicted[idx[3]] - p0,
            ],
            axis=1,
        )
        f = ds @ dm_inv[t]

        # Deviatoric: resist distortion. The constraint is driven to zero with
        # no rest offset, which on its own would collapse the tet to a point.
        # gamma below is what holds it open: at F = I the two gradients are
        # mu*I and -mu*I, so the rest state is stress-free. Subtracting sqrt(3)
        # here as well would zero this term twice over and leave the volume
        # constraint inflating the body to det(F) = gamma forever.
        if params.mu > 0.0:
            c_dev = float(np.sqrt(np.sum(f * f)))
            if c_dev > 1e-12:
                grads = _grads_from_dcdf(f / c_dev, dm_inv[t])
                lam_dev[t] = _apply(
                    state, idx, grads, c_dev,
                    1.0 / (params.mu * abs(rest_vol[t])), h, lam_dev[t],
                )

        # Hydrostatic: resist volume change.
        if params.lam > 0.0:
            # F is recomputed rather than reused: the deviatoric projection
            # above has already moved state.predicted, so the stale F would
            # linearise the volume constraint about the wrong configuration.
            # This duplication is load-bearing, not an oversight.
            p0 = state.predicted[idx[0]]
            ds = np.stack(
                [
                    state.predicted[idx[1]] - p0,
                    state.predicted[idx[2]] - p0,
                    state.predicted[idx[3]] - p0,
                ],
                axis=1,
            )
            f = ds @ dm_inv[t]
            f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
            dcdf = np.stack(
                [np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=1
            )
            grads = _grads_from_dcdf(dcdf, dm_inv[t])
            c_hyd = float(np.linalg.det(f) - gamma)
            lam_hyd[t] = _apply(
                state, idx, grads, c_hyd,
                1.0 / (params.lam * abs(rest_vol[t])), h, lam_hyd[t],
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_solver_constraints.py -v`
Expected: 6 passed

If `test_volume_is_preserved_under_compression` fails by drifting the wrong way, the sign of `c_hyd` is inverted. If the body explodes, `alpha_tilde` is too small — check that compliance is divided by `h*h` and not multiplied. If `test_rest_configuration_converges_to_a_fixed_point` fails with a drift that barely changes between the coarse and fine runs, a rest offset has crept back into `c_dev`.

- [ ] **Step 5: Run the whole core suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/core -v`
Expected: all pass, 45 tests

- [ ] **Step 6: Commit**

```bash
git add marrow/core/solver_ref.py tests/core/test_solver_constraints.py
git commit -m "feat: add stable neo-Hookean XPBD constraints to reference solver"
```

---

### Task 8: Blender inside-test and cell mask

**Files:**
- Create: `marrow/blender/inside_bvh.py`
- Create: `tests/blender/run_tests.py`
- Test: `tests/blender/test_inside_bvh.py`

**Interfaces:**
- Consumes: `grid_dims` from Task 3
- Produces:
  - `cell_mask_from_object(obj, spacing) -> tuple[np.ndarray, np.ndarray]` returning the bool mask `(nx, ny, nz)` and the `bounds_min` float64 `(3,)` the mask is anchored at. A cell is kept when its centre is inside the mesh.
  - `tests/blender/run_tests.py`, the assert-based runner, since Blender's Python has no pytest

- [ ] **Step 1: Write the runner and the failing test**

`tests/blender/run_tests.py`:

```python
"""Assert-based test runner for Blender-dependent code.

Blender's bundled Python has no pytest and we will not install into it.
Run: blender -b --factory-startup --python tests/blender/run_tests.py
Exits 1 on any failure so CI and humans both notice.
"""

import importlib
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Both paths must be set BEFORE any test module is imported.
sys.path.insert(0, str(ROOT))   # so `import marrow` works
sys.path.insert(0, str(HERE))   # so test modules import as top-level names


def main():
    # Auto-discover, so adding a test file needs no edit here. Import failures
    # are counted rather than raised: an exception escaping main() would leave
    # Blender to exit 0, so a module that will not even import would read as a
    # clean run. Measured — that is exactly what happened before this guard.
    paths = sorted(HERE.glob("test_*.py"))
    modules = []
    failures = 0
    for path in paths:
        try:
            modules.append(importlib.import_module(path.stem))
        except Exception:
            failures += 1
            print(f"FAIL {path.stem} (import)")
            traceback.print_exc()

    if not paths:
        print("no test modules found")
        sys.exit(1)

    for module in modules:
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            try:
                getattr(module, name)()
                print(f"PASS {module.__name__}.{name}")
            except Exception:
                failures += 1
                print(f"FAIL {module.__name__}.{name}")
                traceback.print_exc()
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


main()
```

`tests/blender/test_inside_bvh.py`:

```python
import bpy
import numpy as np

from marrow.blender.inside_bvh import cell_mask_from_object


def _fresh_cube(size=2.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=size)
    return bpy.context.active_object


def test_cube_mask_is_fully_occupied_in_the_interior():
    obj = _fresh_cube(size=2.0)
    mask, bounds_min = cell_mask_from_object(obj, spacing=0.5)
    assert mask.dtype == bool
    assert mask.any(), "cube produced an empty mask"
    # Cube spans -1..1, so a 0.5 grid is 4x4x4 and every cell centre is inside.
    assert mask.shape == (4, 4, 4)
    assert mask.all()
    assert np.allclose(bounds_min, [-1.0, -1.0, -1.0], atol=1e-6)


def test_sphere_mask_excludes_corners():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    obj = bpy.context.active_object
    mask, _ = cell_mask_from_object(obj, spacing=0.25)
    assert mask.any()
    assert not mask.all(), "a sphere must leave the bounding-box corners empty"
    assert not mask[0, 0, 0]


def test_spacing_too_coarse_yields_empty_mask_not_a_crash():
    """A spacing far larger than the object must return cleanly, not raise."""
    obj = _fresh_cube(size=0.01)
    mask, bounds_min = cell_mask_from_object(obj, spacing=10.0)
    assert mask.shape == (1, 1, 1)
    assert mask.dtype == bool
    assert bounds_min.shape == (3,)
    assert np.all(np.isfinite(bounds_min))
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
"/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b --factory-startup \
  --python "C:/Users/user/Documents/marrow/tests/blender/run_tests.py"
```
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.blender.inside_bvh'`

- [ ] **Step 3: Implement**

`marrow/blender/inside_bvh.py`:

```python
"""Inside/outside testing against a Blender object, via BVH ray parity."""

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from marrow.core.lattice import grid_dims

_RAY = Vector((0.5773502691896258, 0.4082482904638631, 0.7071067811865476))
_EPS = 1e-6


def _world_bvh(obj):
    """BVH of the evaluated object in world space, plus its world bounds."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = obj.matrix_world
        verts = [matrix @ v.co for v in mesh.vertices]
        polys = [tuple(p.vertices) for p in mesh.polygons]
        tris = []
        for poly in polys:  # fan-triangulate, BVHTree wants tris or quads
            for i in range(1, len(poly) - 1):
                tris.append((poly[0], poly[i], poly[i + 1]))
        bvh = BVHTree.FromPolygons(verts, tris, all_triangles=True)
        coords = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()
    return bvh, coords


def _is_inside(bvh, point: Vector) -> bool:
    """Odd number of forward hits means inside."""
    hits = 0
    origin = point.copy()
    while True:
        location, _normal, _index, _dist = bvh.ray_cast(origin, _RAY)
        if location is None:
            break
        hits += 1
        origin = location + _RAY * _EPS
    return hits % 2 == 1


def cell_mask_from_object(obj, spacing: float):
    """Voxel occupancy of ``obj`` at ``spacing``, by cell-centre inside test."""
    bvh, coords = _world_bvh(obj)
    bounds_min = coords.min(axis=0)
    bounds_max = coords.max(axis=0)
    dims = grid_dims(bounds_min, bounds_max, spacing)

    mask = np.zeros(dims, dtype=bool)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                centre = bounds_min + (np.array([i, j, k]) + 0.5) * spacing
                mask[i, j, k] = _is_inside(bvh, Vector(centre.tolist()))
    return mask, bounds_min
```

- [ ] **Step 4: Run to verify it passes**

Run the same Blender command. Expected: the three `test_inside_bvh` tests print PASS, `0 failure(s)`. The runner auto-discovers `test_*.py`, so it finds only this file for now and needs no editing as later tasks add more.

- [ ] **Step 5: Commit**

```bash
git add marrow/blender/inside_bvh.py tests/blender
git commit -m "feat: add BVH inside test producing a voxel cell mask"
```

---

### Task 9: Blender storage round-trip

**Files:**
- Create: `marrow/blender/storage.py`
- Test: `tests/blender/test_storage.py`

**Interfaces:**
- Consumes: `TetMesh` from Task 2
- Produces:
  - `write_tetmesh(mesh, tetmesh, colors) -> None`
  - `read_tetmesh(mesh) -> tuple[TetMesh, np.ndarray]`
  - `write_bind(mesh, bind_idx, bind_w) -> None`
  - `read_bind(mesh) -> tuple[np.ndarray, np.ndarray]`

**Storage layout, matching the spec's table:**

| Data | Where |
|---|---|
| Tet nodes | cage mesh vertices |
| Tet connectivity | `mesh["marrow_tets"]` int list, flat, 4 per tet |
| Tet colours | `mesh["marrow_colors"]` int list, 1 per tet |
| Bind tet index | `marrow_bind_idx`, INT on POINT domain of the render mesh |
| Bind weights | `marrow_bind_w0..w3`, FLOAT on POINT domain |

- [ ] **Step 1: Write the failing test**

`tests/blender/test_storage.py`:

```python
import bpy
import numpy as np

from marrow.blender.storage import (
    read_bind,
    read_tetmesh,
    write_bind,
    write_tetmesh,
)
from marrow.core.lattice import build_lattice
from marrow.core.coloring import color_tets

MESH = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))


def _empty_mesh(name="cage"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.data.meshes.new(name)


def test_tetmesh_round_trip_is_exact():
    mesh = _empty_mesh()
    colors = color_tets(MESH.tets, MESH.n_nodes)
    write_tetmesh(mesh, MESH, colors)

    out, out_colors = read_tetmesh(mesh)
    assert out.n_nodes == MESH.n_nodes
    assert out.n_tets == MESH.n_tets
    assert np.allclose(out.nodes, MESH.nodes, atol=1e-6)
    assert np.array_equal(out.tets, MESH.tets)
    assert np.array_equal(out_colors, colors)
    out.validate()


def test_bind_round_trip_is_exact():
    mesh = _empty_mesh("render")
    n = 12
    mesh.from_pydata([(float(i), 0.0, 0.0) for i in range(n)], [], [])
    mesh.update()

    rng = np.random.default_rng(3)
    idx = rng.integers(0, MESH.n_tets, size=n).astype(np.int32)
    w = rng.random((n, 4))
    w = w / w.sum(axis=1, keepdims=True)

    write_bind(mesh, idx, w)
    out_idx, out_w = read_bind(mesh)
    assert np.array_equal(out_idx, idx)
    assert np.allclose(out_w, w, atol=1e-6)


def test_read_tetmesh_on_bare_mesh_raises():
    mesh = _empty_mesh("bare")
    try:
        read_tetmesh(mesh)
    except KeyError as exc:
        assert "marrow_tets" in str(exc)
    else:
        raise AssertionError("expected KeyError on a mesh with no Marrow data")


def test_empty_tetmesh_round_trips():
    from marrow.core.tetmesh import TetMesh

    mesh = _empty_mesh("empty")
    empty = TetMesh(np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32))
    write_tetmesh(mesh, empty, np.zeros(0, dtype=np.int32))
    out, colors = read_tetmesh(mesh)
    assert out.n_tets == 0
    assert colors.shape == (0,)
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command from Task 8. The runner picks up the new file automatically.
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.blender.storage'`

- [ ] **Step 3: Implement**

`marrow/blender/storage.py`:

```python
"""Persist tet cage and bind data on Blender datablocks.

Addons cannot register new datablock types, so cage nodes live as mesh
vertices, connectivity and colours live in ID properties, and per-point bind
data lives in POINT-domain attributes. All of it survives save and load.
"""

import numpy as np

from marrow.core.tetmesh import TetMesh

TETS_KEY = "marrow_tets"
COLORS_KEY = "marrow_colors"
BIND_IDX = "marrow_bind_idx"
BIND_W = ("marrow_bind_w0", "marrow_bind_w1", "marrow_bind_w2", "marrow_bind_w3")


def write_tetmesh(mesh, tetmesh: TetMesh, colors: np.ndarray) -> None:
    mesh.clear_geometry()
    mesh.from_pydata([tuple(p) for p in tetmesh.nodes], [], [])
    mesh.update()
    mesh[TETS_KEY] = tetmesh.tets.ravel().astype(np.int32).tolist()
    mesh[COLORS_KEY] = np.asarray(colors, dtype=np.int32).tolist()


def read_tetmesh(mesh):
    if TETS_KEY not in mesh.keys():
        raise KeyError(f"mesh has no {TETS_KEY}; it is not a Marrow cage")

    n_nodes = len(mesh.vertices)
    nodes = np.empty(n_nodes * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", nodes)
    nodes = nodes.reshape(n_nodes, 3)

    flat = np.array(mesh[TETS_KEY], dtype=np.int32)
    tets = flat.reshape(-1, 4) if flat.size else np.zeros((0, 4), dtype=np.int32)
    colors = np.array(mesh.get(COLORS_KEY, []), dtype=np.int32)
    return TetMesh(nodes, tets), colors


def _ensure_attr(mesh, name, data_type):
    attr = mesh.attributes.get(name)
    if attr is None or attr.data_type != data_type or attr.domain != "POINT":
        if attr is not None:
            mesh.attributes.remove(attr)
        attr = mesh.attributes.new(name=name, type=data_type, domain="POINT")
    return attr


def write_bind(mesh, bind_idx: np.ndarray, bind_w: np.ndarray) -> None:
    _ensure_attr(mesh, BIND_IDX, "INT").data.foreach_set(
        "value", np.asarray(bind_idx, dtype=np.int32).tolist()
    )
    for i, name in enumerate(BIND_W):
        _ensure_attr(mesh, name, "FLOAT").data.foreach_set(
            "value", np.asarray(bind_w[:, i], dtype=np.float32).tolist()
        )
    mesh.update()


def read_bind(mesh):
    n = len(mesh.vertices)
    idx = np.empty(n, dtype=np.int32)
    mesh.attributes[BIND_IDX].data.foreach_get("value", idx)

    weights = np.empty((n, 4), dtype=np.float64)
    for i, name in enumerate(BIND_W):
        column = np.empty(n, dtype=np.float32)
        mesh.attributes[name].data.foreach_get("value", column)
        weights[:, i] = column
    return idx, weights
```

- [ ] **Step 4: Run to verify it passes**

Run the Blender command. Expected: all `test_inside_bvh` and `test_storage` tests print PASS, `0 failure(s)`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add marrow/blender/storage.py tests/blender
git commit -m "feat: add tet cage and bind storage round-trip"
```

---

### Task 10: Tetrahedralize operator, cage viewer, and manifest

**Files:**
- Create: `marrow/blender/ops.py`, `marrow/blender/ui.py`, `marrow/blender_manifest.toml`
- Modify: `marrow/__init__.py`
- Test: `tests/blender/test_ops.py`

**Interfaces:**
- Consumes: everything from Tasks 2 through 9
- Produces:
  - `MARROW_OT_tetrahedralize`, `bl_idname = "marrow.tetrahedralize"`
  - `MarrowSettings` property group on `bpy.types.Object` as `obj.marrow`, with `resolution: FloatProperty` default 0.25, minimum 0.001
  - `MARROW_PT_panel` in View3D sidebar, category "Marrow"
  - `register()` / `unregister()` in `marrow/__init__.py`

- [ ] **Step 1: Write the failing test**

`tests/blender/test_ops.py`:

```python
import bpy
import numpy as np

import marrow
from marrow.blender.storage import read_bind, read_tetmesh


def _setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    return bpy.context.active_object


def test_operator_creates_a_valid_cage():
    obj = _setup()
    obj.marrow.resolution = 0.5
    assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}

    cage_obj = bpy.data.objects.get(f"{obj.name}_marrow_cage")
    assert cage_obj is not None, "no cage object was created"
    tetmesh, colors = read_tetmesh(cage_obj.data)
    tetmesh.validate()
    assert tetmesh.n_tets > 0
    assert colors.shape == (tetmesh.n_tets,)


def test_operator_binds_every_render_vertex():
    obj = _setup()
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()

    idx, w = read_bind(obj.data)
    assert idx.shape[0] == len(obj.data.vertices)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert idx.min() >= 0


def test_cage_is_hidden_and_parented():
    obj = _setup()
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()
    cage_obj = bpy.data.objects[f"{obj.name}_marrow_cage"]
    assert cage_obj.parent == obj
    assert cage_obj.display_type == "WIRE"
    assert cage_obj.hide_render


def test_resolution_too_coarse_reports_an_error_not_a_crash():
    obj = _setup()
    obj.marrow.resolution = 100.0
    # An operator that reports {"ERROR"} and returns CANCELLED surfaces in
    # Python as RuntimeError: that is Blender promoting the report, not the
    # operator crashing, so the return value never reaches the caller. What
    # matters is that the user gets the fix instruction rather than a
    # traceback from somewhere inside numpy.
    try:
        bpy.ops.marrow.tetrahedralize()
    except RuntimeError as exc:
        assert "Lower Resolution" in str(exc), f"unhelpful error text: {exc}"
    else:
        raise AssertionError("expected a reported error at 100.0 resolution")


def test_rerunning_replaces_the_previous_cage():
    obj = _setup()
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()
    bpy.ops.marrow.tetrahedralize()
    matching = [o for o in bpy.data.objects if o.name.startswith(f"{obj.name}_marrow_cage")]
    assert len(matching) == 1, f"expected one cage, found {[o.name for o in matching]}"
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command. The runner discovers the new file automatically.
Expected: FAIL, `AttributeError` on `marrow.register` or on `obj.marrow`.

- [ ] **Step 3: Implement the operator**

`marrow/blender/ops.py`:

```python
"""Marrow operators."""

import bpy
import numpy as np

from marrow.blender.inside_bvh import cell_mask_from_object
from marrow.blender.storage import write_bind, write_tetmesh
from marrow.core.bind import bind_points
from marrow.core.coloring import color_tets
from marrow.core.lattice import build_lattice

CAGE_SUFFIX = "_marrow_cage"


class MARROW_OT_tetrahedralize(bpy.types.Operator):
    bl_idname = "marrow.tetrahedralize"
    bl_label = "Tetrahedralize"
    bl_description = "Fill the selected mesh with a tetrahedral cage"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        spacing = float(obj.marrow.resolution)

        mask, bounds_min = cell_mask_from_object(obj, spacing)
        if not mask.any():
            self.report(
                {"ERROR"},
                "No cells inside the mesh. Lower Resolution in the Marrow panel "
                "until the cage fills the object.",
            )
            return {"CANCELLED"}

        tetmesh = build_lattice(bounds_min, spacing, mask)
        try:
            tetmesh.validate()
        except ValueError as exc:
            self.report({"ERROR"}, f"Invalid cage: {exc}")
            return {"CANCELLED"}

        colors = color_tets(tetmesh.tets, tetmesh.n_nodes)

        render_verts = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", render_verts)
        render_verts = render_verts.reshape(-1, 3)
        world = np.array(obj.matrix_world.to_4x4())
        world_verts = render_verts @ world[:3, :3].T + world[:3, 3]

        bind_idx, bind_w = bind_points(tetmesh.nodes, tetmesh.tets, world_verts)
        write_bind(obj.data, bind_idx, bind_w)

        cage_name = f"{obj.name}{CAGE_SUFFIX}"
        existing = bpy.data.objects.get(cage_name)
        if existing is not None:
            cage_mesh = existing.data
            bpy.data.objects.remove(existing, do_unlink=True)
            if cage_mesh.users == 0:
                bpy.data.meshes.remove(cage_mesh)

        cage_mesh = bpy.data.meshes.new(cage_name)
        write_tetmesh(cage_mesh, tetmesh, colors)
        cage_obj = bpy.data.objects.new(cage_name, cage_mesh)
        context.collection.objects.link(cage_obj)
        cage_obj.parent = obj
        cage_obj.display_type = "WIRE"
        cage_obj.hide_render = True
        cage_obj.hide_select = True

        self.report(
            {"INFO"},
            f"Marrow: {tetmesh.n_tets} tets, {tetmesh.n_nodes} nodes, "
            f"{int(colors.max()) + 1 if colors.size else 0} colours",
        )
        return {"FINISHED"}
```

- [ ] **Step 4: Implement the UI and registration**

`marrow/blender/ui.py`:

```python
"""Marrow properties and sidebar panel."""

import bpy


class MarrowSettings(bpy.types.PropertyGroup):
    resolution: bpy.props.FloatProperty(
        name="Resolution",
        description="Cage cell size in world units. Smaller fills finer detail",
        default=0.25,
        min=0.001,
        soft_max=1.0,
        unit="LENGTH",
    )


class MARROW_PT_panel(bpy.types.Panel):
    bl_label = "Marrow"
    bl_idname = "MARROW_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Marrow"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.prop(obj.marrow, "resolution")
        layout.operator("marrow.tetrahedralize", icon="MESH_ICOSPHERE")
```

`marrow/__init__.py` (replace the whole file):

```python
"""Marrow: GPU tetrahedral soft body for Blender."""

__version__ = "0.1.0"

# bpy is imported inside register()/unregister(), not at module scope. Any
# `from marrow.core.x import ...` executes this file first, so a top-level
# `import bpy` takes the entire pytest core suite down with
# ModuleNotFoundError outside Blender - measured, all six core modules failed
# at collection. Deferring is the ordinary addon idiom and costs nothing:
# Blender only ever calls these two functions, and by then bpy is present.

_registered = []


def register():
    import bpy

    from marrow.blender.ops import MARROW_OT_tetrahedralize
    from marrow.blender.ui import MARROW_PT_panel, MarrowSettings

    classes = (MarrowSettings, MARROW_OT_tetrahedralize, MARROW_PT_panel)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.marrow = bpy.props.PointerProperty(type=MarrowSettings)
    _registered[:] = classes


def unregister():
    import bpy

    del bpy.types.Object.marrow
    for cls in reversed(_registered):
        bpy.utils.unregister_class(cls)
    _registered.clear()
```

`marrow/blender_manifest.toml`:

```toml
schema_version = "1.0.0"
id = "marrow"
version = "0.1.0"
name = "Marrow"
tagline = "GPU tetrahedral soft body simulation"
maintainer = "Amsy <alanmaulanasyarif@gmail.com>"
type = "add-on"
license = ["SPDX:GPL-3.0-or-later"]
blender_version_min = "5.2.0"
tags = ["Physics"]
```

- [ ] **Step 5: Run to verify it passes**

Run the Blender command. Expected: every test across all three Blender modules prints PASS, `0 failure(s)`, exit code 0. Twelve tests total.

`marrow/__init__.py` must stay importable without Blender. `tests/core/test_no_bpy.py` only *scans* `marrow/core/`, but every core test module imports through the `marrow` package and so executes `__init__.py` for real — a top-level `import bpy` there is not a scan failure, it is a collection failure across the whole suite. That is why `register()`/`unregister()` import `bpy` lazily. Confirm with:

Run: `./.venv/Scripts/python.exe -m pytest tests/core -v`
Expected: all 45 pass.

- [ ] **Step 6: Commit**

```bash
git add marrow tests/blender
git commit -m "feat: add tetrahedralize operator, cage viewer, and addon manifest"
```

---

## Self-Review Notes

**Spec coverage.** Every Stage 1 requirement maps to a task: cube-split lattice (Task 3), inside test via BVHTree (Task 8), barycentric binding (Task 4), storage table (Task 9), colouring (Task 5), cage viewer and resolution slider (Task 10). Solver oracle requirements map to Tasks 6 and 7. Spec failure-handling rows covered here are the coarse-resolution error and the invalid-cage error (Task 10); the NaN detector, GLSL compile log, and node-count budget belong to the GPU plan, since nothing in this plan can produce them.

**Deferred to the GPU plan (build steps 3 and 4):** texture packing, the four compute kernels, oracle-versus-GPU differential tests, cache, material presets, and Vulkan backend validation.

**Corrected during Task 7 execution, 2026-08-12.** This note previously claimed the deviatoric constraint takes a `sqrt(3)` rest offset and called that "the standard formulation". It is not, and combined with `gamma = 1 + mu/lam` it is a defect: both terms then vanish at `F = I` by different means, the volume constraint runs unopposed, and the body inflates permanently to `det(F) = gamma`. Measured across substeps 4/40/400, that drift held flat at 1.6e-2 and settled volume at 1.097 against a rest volume of 1.0. The published Macklin and Muller pairing — offset-free `C_D` with `gamma` — converges instead: drift 7.3e-2, 1.5e-4, 1.7e-6, settled volume 1.00000. Marrow uses the published pairing. Task 7's rest test was rewritten as a convergence test because the residual at any finite substep count is genuine sequential-projection error, not bias, and was fault-injection checked against the old form.

**Task 3 split tables are pre-verified, not assumed.** Before this plan was finalised the two 5-tet split tables were run numerically. Results, which are exactly what Task 3's tests assert: one cell gives 5 tets and 8 nodes with total volume 1.0 and all volumes positive; a 2x1x1 pair gives 12 nodes and 10 tets with total volume 2.0; the shared plane between the two cells carries exactly one face diagonal, `(4,7)`, so both cells agree and the mesh is conforming; a 3x3x3 block at spacing 0.5 gives 64 nodes, 135 tets, total volume exactly 3.375, all positive, and zero faces shared by more than two tets. If Task 3's tests fail, suspect the implementation, not the tables.

**Two bugs were found and fixed during self-review**, recorded so they are not reintroduced. The Blender runner originally imported test modules before extending `sys.path`, which cannot work; it now sets both paths first and auto-discovers `test_*.py`, which also removed three manual `MODULES` edits that Tasks 8 to 10 would otherwise have needed. And `_apply` carried an unused `t` parameter.
