# Marrow GPU Solver Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Marrow's XPBD substep loop onto the GPU as GLSL compute kernels, and prove each kernel matches the numpy oracle already in `marrow/core/solver_ref.py`.

**Architecture:** Simulation state lives in RGBA32F/R32F images, one texel per node or per tet, indexed `i -> (i % 4096, i / 4096)`. Four kernels run per substep — `predict`, `solve` (one dispatch per constraint colour), `collide`, `integrate` — then a `skin` kernel blends render vertices out of the cage and a single readback moves only those vertices back to Python. Every kernel is diffed against the CPU oracle before the next one is written, because a sign error in GLSL is otherwise indistinguishable from a sign error in the algebra.

**Tech Stack:** Blender 5.2 `gpu` module (OpenGL 4.6 backend), GLSL compute, numpy, Python 3.13 (Blender runtime) / 3.12 (pytest).

## Scope

This plan covers **build step 3 only**. The spec groups steps 3 and 4 into one plan; splitting them is a deliberate deviation. Step 3 ends at a defensible deliverable — the GPU reproduces the oracle — which is independently testable and is the only part whose risk is technical. Step 4 (cache, material presets, UI polish) is product work sitting entirely on top of it, and folding it in would put a dozen tasks under kernels that have not run yet. That is the same mistake the spec's own decomposition section warns about. Step 4 gets a third plan once this one lands.

## Measured Facts This Plan Rests On

From `tools/spike_01_gpu_packing.py`, run 2026-08-12 on Blender 5.2.0 LTS / RTX 5050 / OpenGL 4.6. Do not re-derive these; do not design around anything that contradicts them.

| Question | Answer |
|---|---|
| `gpu.compute` barrier API | **None.** Only `dispatch` is exposed. |
| Chained write-then-read dispatches without a barrier | Correct 50/50 at 16x16. **Carried risk at scale — Task 8 re-tests it at realistic size.** |
| Integer images (`RGBA32I`) | **Unusable.** `GPUTexture.__new__` raises `Only Buffer of format 'FLOAT' is currently supported`, so integer data cannot be uploaded from Python at all. Shader-written `RGBA32I` also read back as garbage. |
| Integer indices in `RGBA32F` | Exact for every integer up to 2**24 (16,777,216). |
| `R32F` single-channel | Works. |
| One image bound `{"READ", "WRITE"}` | Works — verified with three in-place increments. |
| Six images on one shader | Works. `max_images_get()` is 8. |
| numpy upload round-trip | Exact, zero difference. |
| Push constants `FLOAT` / `INT` / `VEC3` | Work, via `shader.uniform_float` / `uniform_int`. |
| `local_group_size(64, 1)` | Works. |

**Two spec corrections follow from this and are applied by Task 1:**

1. The texture budget's `tets | RGBA32I` row is not implementable. Tet indices ride in `RGBA32F` as floats and are converted with `ivec4()` in the shader. The 2**24 headroom is ~1000x the node budget.
2. The `lambda | R32F | 2T` row is **dropped entirely**. The oracle zeroes `lam_dev`/`lam_hyd` at the top of every `solve_constraints` call and visits each tet exactly once, so the accumulated multiplier is provably always `0.0` at the point `_apply` reads it. The `alpha_tilde * lam_acc` term is therefore always zero. Carrying a texture that only ever holds zeros, plus a clear kernel to keep it that way, buys nothing. If a future change runs more than one constraint iteration per substep, the multiplier becomes live and this texture comes back — that is the only condition that reintroduces it.

Net budget: **5 images**, not 6.

## Global Constraints

- Blender 5.2.0 LTS only. `blender_version_min = "5.2.0"`.
- `marrow/core/` MUST NOT import `bpy`, `mathutils`, `gpu`, `bmesh`, `gpu_extras`, `bpy_extras`, `blf`, `aud`, or `bl_math`. Enforced by `tests/core/test_no_bpy.py`.
- `marrow/__init__.py` MUST NOT import `bpy` at module scope. `register()`/`unregister()` import it lazily. A top-level import takes the whole pytest core suite down at collection.
- No dependency outside numpy and Blender's bundled modules. No scipy, no torch, no compiled extension.
- Never `pip install` into Blender's bundled Python. Test dependencies live in the worktree `.venv` built on system Python 3.12.
- Core arrays are `float64` / `int32`. Conversion to `float32` happens only at the GPU boundary, in `marrow/gpu/textures.py`, nowhere else.
- `gpu.init()` must be called before any `gpu.*` access in background mode.
- `GPUTexture.read()` returns a Buffer shaped `[H][W][C]`. Always go through `np.asarray(...).reshape(H, W, C)` and assert on size first — an empty readback compared elementwise passes vacuously, which produced a false PASS during the step 0 spike.
- Product name is "Marrow" in every user-facing string.
- Texture width is 4096. One constant, `TEX_WIDTH`, in `marrow/core/layout.py`. Never inline it.

---

## File Structure

| Path | Responsibility |
|---|---|
| `marrow/core/layout.py` | Texel index math, colour-ordered tet permutation, pack/unpack to flat float32 arrays. Pure numpy, bpy-free, pytest-tested. |
| `marrow/gpu/__init__.py` | Namespace only |
| `marrow/gpu/textures.py` | numpy array <-> `GPUTexture`. The only place `float32` conversion happens. |
| `marrow/gpu/kernels.py` | GLSL sources and `GPUShader` construction, with compile logs surfaced verbatim |
| `marrow/gpu/solver.py` | `GPUSolver` — owns the textures, runs the substep loop, detects NaN |
| `tests/core/test_layout.py` | pytest, pure numpy |
| `tests/blender/test_textures.py` | round-trip through a real `GPUTexture` |
| `tests/blender/test_kernels_vs_oracle.py` | per-kernel parity against `solver_ref` |
| `tests/blender/test_solver_parity.py` | full-loop parity, NaN detection, barrier-at-scale |
| `tools/spike_01_gpu_packing.py` | Done. Do not modify — it is the record of the measurements above. |

---

### Task 1: Texel layout and colour ordering

**Files:**
- Create: `marrow/core/layout.py`
- Test: `tests/core/test_layout.py`

**Interfaces:**
- Consumes: `color_tets` from `marrow/core/coloring.py`
- Produces:
  - `TEX_WIDTH = 4096`
  - `texture_shape(count) -> tuple[int, int]` returning `(width, height)` covering `count` texels
  - `texel_index(i) -> tuple[int, int]` returning `(x, y)` for element `i`
  - `color_ordered(tets, colors) -> tuple[np.ndarray, np.ndarray]` returning tets permuted so each colour is contiguous, and an int32 `offsets` array of length `n_colors + 1` where colour `c` occupies `[offsets[c], offsets[c+1])`
  - `pack_nodes(nodes, inv_mass) -> np.ndarray` shape `(H, W, 4)` float32, xyz in rgb and inverse mass in a
  - `pack_tets(tets) -> np.ndarray` shape `(H, W, 4)` float32, the four node indices as floats
  - `pack_rest(dm_inv, rest_vol) -> np.ndarray` shape `(H, W, 4)` float32 over `3 * T` texels; texel `3t + j` holds column `j` of `dm_inv[t]` in rgb, and texel `3t` holds `rest_vol[t]` in a
  - `unpack_vec3(image, count) -> np.ndarray` shape `(count, 3)` float64

- [ ] **Step 1: Write the failing tests**

`tests/core/test_layout.py`:

```python
import numpy as np
import pytest

from marrow.core.coloring import color_tets
from marrow.core.lattice import build_lattice
from marrow.core.layout import (
    TEX_WIDTH,
    color_ordered,
    pack_nodes,
    pack_rest,
    pack_tets,
    texel_index,
    texture_shape,
    unpack_vec3,
)
from marrow.core.solver_ref import precompute

MESH = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))


def test_texture_width_is_the_documented_constant():
    assert TEX_WIDTH == 4096


def test_texture_shape_is_one_row_when_it_fits():
    assert texture_shape(10) == (TEX_WIDTH, 1)


def test_texture_shape_rounds_up_to_whole_rows():
    assert texture_shape(TEX_WIDTH + 1) == (TEX_WIDTH, 2)
    assert texture_shape(TEX_WIDTH * 3) == (TEX_WIDTH, 3)


def test_texture_shape_of_zero_still_has_one_row():
    """A zero-height texture cannot be allocated."""
    assert texture_shape(0) == (TEX_WIDTH, 1)


def test_texel_index_wraps_at_the_row_width():
    assert texel_index(0) == (0, 0)
    assert texel_index(TEX_WIDTH - 1) == (TEX_WIDTH - 1, 0)
    assert texel_index(TEX_WIDTH) == (0, 1)
    assert texel_index(TEX_WIDTH + 5) == (5, 1)


def test_color_ordered_groups_every_color_contiguously():
    colors = color_tets(MESH.tets, MESH.n_nodes)
    ordered, offsets = color_ordered(MESH.tets, colors)

    assert ordered.shape == MESH.tets.shape
    assert offsets.shape == (int(colors.max()) + 2,)
    assert offsets[0] == 0
    assert offsets[-1] == MESH.n_tets
    assert np.all(np.diff(offsets) >= 0)


def test_color_ordered_slices_are_node_disjoint():
    """The whole point: one colour's slice can be dispatched race-free."""
    colors = color_tets(MESH.tets, MESH.n_nodes)
    ordered, offsets = color_ordered(MESH.tets, colors)

    for c in range(len(offsets) - 1):
        seen = set()
        for tet in ordered[offsets[c]:offsets[c + 1]]:
            nodes = set(tet.tolist())
            assert not (seen & nodes), f"colour {c} slice shares a node"
            seen |= nodes


def test_color_ordered_is_a_permutation_not_a_rewrite():
    colors = color_tets(MESH.tets, MESH.n_nodes)
    ordered, _ = color_ordered(MESH.tets, colors)
    before = sorted(tuple(sorted(t)) for t in MESH.tets.tolist())
    after = sorted(tuple(sorted(t)) for t in ordered.tolist())
    assert before == after


def test_pack_nodes_carries_position_and_inverse_mass():
    inv_mass = np.arange(MESH.n_nodes, dtype=np.float64)
    img = pack_nodes(MESH.nodes, inv_mass)

    assert img.dtype == np.float32
    assert img.shape == (1, TEX_WIDTH, 4)
    x, y = texel_index(7)
    assert np.allclose(img[y, x, :3], MESH.nodes[7], atol=1e-6)
    assert np.isclose(img[y, x, 3], inv_mass[7])


def test_pack_tets_indices_survive_the_float_round_trip():
    img = pack_tets(MESH.tets)
    assert img.dtype == np.float32
    for t in (0, 5, MESH.n_tets - 1):
        x, y = texel_index(t)
        assert img[y, x].astype(np.int64).tolist() == MESH.tets[t].tolist()


def test_pack_tets_is_exact_at_the_float32_integer_limit():
    """float32 holds every integer to 2**24; the cage budget is far below."""
    big = np.array([[0, 1, 2, 2**24 - 1]], dtype=np.int32)
    img = pack_tets(big)
    x, y = texel_index(0)
    assert img[y, x].astype(np.int64).tolist() == [0, 1, 2, 2**24 - 1]


def test_pack_rest_uses_three_texels_per_tet():
    dm_inv, rest_vol = precompute(MESH.nodes, MESH.tets)
    img = pack_rest(dm_inv, rest_vol)

    t = 4
    cols = []
    for j in range(3):
        x, y = texel_index(3 * t + j)
        cols.append(img[y, x, :3])
    assert np.allclose(np.stack(cols, axis=1), dm_inv[t], atol=1e-5)

    x, y = texel_index(3 * t)
    assert np.isclose(img[y, x, 3], rest_vol[t], atol=1e-6)


def test_unpack_vec3_inverts_pack_nodes():
    inv_mass = np.ones(MESH.n_nodes)
    img = pack_nodes(MESH.nodes, inv_mass)
    out = unpack_vec3(img, MESH.n_nodes)
    assert out.shape == (MESH.n_nodes, 3)
    assert out.dtype == np.float64
    assert np.allclose(out, MESH.nodes, atol=1e-6)


def test_unpack_vec3_rejects_a_count_the_image_cannot_hold():
    img = pack_nodes(MESH.nodes, np.ones(MESH.n_nodes))
    with pytest.raises(ValueError, match="cannot hold"):
        unpack_vec3(img, TEX_WIDTH * 99)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_layout.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.core.layout'`

- [ ] **Step 3: Implement**

`marrow/core/layout.py`:

```python
"""Texel layout for the GPU solver. Pure numpy, no Blender.

Everything here exists because Blender's Python GPU API has no storage
buffers, so all bulk state rides in 2D images. A 1D layout would cap at the
32768 max texture size; 2D indexing removes the ceiling.

Integer data rides in float32 images: GPUTexture refuses any upload Buffer
that is not FLOAT, so RGBA32I is unreachable from Python. float32 represents
every integer up to 2**24 exactly, which is about a thousand times the node
budget, so tet indices are stored as floats and read back with ivec4().
"""

import numpy as np

TEX_WIDTH = 4096


def texture_shape(count: int) -> tuple[int, int]:
    """(width, height) of an image holding ``count`` texels."""
    rows = max(1, -(-int(count) // TEX_WIDTH))  # ceil division
    return (TEX_WIDTH, rows)


def texel_index(i: int) -> tuple[int, int]:
    """(x, y) of element ``i``. Mirrors the shader's texel() exactly."""
    i = int(i)
    return (i % TEX_WIDTH, i // TEX_WIDTH)


def color_ordered(tets: np.ndarray, colors: np.ndarray):
    """Permute tets so each colour is one contiguous slice.

    The solve kernel dispatches once per colour over a half-open range, so a
    colour must be a slice, not a scatter list. Returns the permuted tets and
    an offsets array where colour c is ``[offsets[c], offsets[c + 1])``.
    """
    colors = np.asarray(colors, dtype=np.int32)
    if colors.size == 0:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        )

    order = np.argsort(colors, kind="stable")
    ordered = np.asarray(tets, dtype=np.int32)[order]
    counts = np.bincount(colors, minlength=int(colors.max()) + 1)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return ordered, offsets


def _blank(count: int) -> np.ndarray:
    width, height = texture_shape(count)
    return np.zeros((height, width, 4), dtype=np.float32)


def _write(image: np.ndarray, values: np.ndarray) -> None:
    """Fill the first len(values) texels of image, row-major."""
    flat = image.reshape(-1, 4)
    flat[: values.shape[0]] = values


def pack_nodes(nodes: np.ndarray, inv_mass: np.ndarray) -> np.ndarray:
    """Positions in rgb, inverse mass in a. Zero mass means pinned."""
    nodes = np.asarray(nodes, dtype=np.float64)
    inv_mass = np.asarray(inv_mass, dtype=np.float64)
    image = _blank(nodes.shape[0])
    values = np.concatenate([nodes, inv_mass[:, None]], axis=1)
    _write(image, values.astype(np.float32))
    return image


def pack_tets(tets: np.ndarray) -> np.ndarray:
    """Four node indices per texel, as floats. Exact below 2**24."""
    tets = np.asarray(tets, dtype=np.int64)
    image = _blank(tets.shape[0])
    _write(image, tets.astype(np.float32))
    return image


def pack_rest(dm_inv: np.ndarray, rest_vol: np.ndarray) -> np.ndarray:
    """Three texels per tet: texel 3t+j is column j of dm_inv[t].

    Column-major on purpose. GLSL's mat3(c0, c1, c2) takes columns, so the
    shader can rebuild the matrix with no transpose and F = Ds * DmInv then
    matches numpy's ds @ dm_inv term for term.
    """
    dm_inv = np.asarray(dm_inv, dtype=np.float64)
    rest_vol = np.asarray(rest_vol, dtype=np.float64)
    n_tets = dm_inv.shape[0]
    image = _blank(3 * n_tets)

    values = np.zeros((3 * n_tets, 4), dtype=np.float64)
    for j in range(3):
        values[j::3, :3] = dm_inv[:, :, j]
    values[0::3, 3] = rest_vol
    _write(image, values.astype(np.float32))
    return image


def unpack_vec3(image: np.ndarray, count: int) -> np.ndarray:
    """First ``count`` texels of an image as (count, 3) float64."""
    flat = np.asarray(image).reshape(-1, 4)
    if count > flat.shape[0]:
        raise ValueError(
            f"image cannot hold {count} texels, it has {flat.shape[0]}"
        )
    return flat[:count, :3].astype(np.float64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/core/test_layout.py -v`
Expected: 14 passed

- [ ] **Step 5: Run the whole core suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/core -v`
Expected: 64 passed. `test_no_bpy.py` must still pass — `layout.py` imports only numpy.

- [ ] **Step 6: Commit**

```bash
git add marrow/core/layout.py tests/core/test_layout.py
git commit -m "feat: add GPU texel layout and colour-ordered tet permutation"
```

---

### Task 2: numpy to GPUTexture and back

**Files:**
- Create: `marrow/gpu/__init__.py`, `marrow/gpu/textures.py`
- Test: `tests/blender/test_textures.py`

**Interfaces:**
- Consumes: `texture_shape` from Task 1
- Produces:
  - `upload(image: np.ndarray, fmt: str = "RGBA32F") -> gpu.types.GPUTexture`
  - `download(tex: gpu.types.GPUTexture) -> np.ndarray` shape `(H, W, C)` float32
  - `blank(count: int, fmt: str = "RGBA32F") -> gpu.types.GPUTexture`

- [ ] **Step 1: Write the failing test**

`tests/blender/test_textures.py`:

```python
import gpu
import numpy as np

from marrow.core.layout import TEX_WIDTH, texture_shape
from marrow.gpu.textures import blank, download, upload

gpu.init()


def test_round_trip_is_bit_exact():
    rng = np.random.default_rng(1)
    src = rng.random((2, TEX_WIDTH, 4)).astype(np.float32)
    back = download(upload(src))
    assert back.shape == src.shape
    assert np.array_equal(back, src), "upload/download must not perturb data"


def test_round_trip_survives_a_multi_row_image():
    src = np.arange(3 * TEX_WIDTH * 4, dtype=np.float32).reshape(3, TEX_WIDTH, 4)
    back = download(upload(src))
    assert back.shape == (3, TEX_WIDTH, 4)
    assert np.array_equal(back, src)


def test_blank_is_zeroed_and_correctly_shaped():
    tex = blank(TEX_WIDTH + 1)
    back = download(tex)
    assert back.shape == (2, TEX_WIDTH, 4)
    assert np.all(back == 0.0)


def test_blank_single_channel():
    tex = blank(16, fmt="R32F")
    back = download(tex)
    assert back.shape[:2] == texture_shape(16)[::-1]
    assert np.all(back == 0.0)


def test_download_asserts_on_an_empty_readback():
    """A vacuous readback comparison produced a false PASS in spike 0.

    download() must never hand back a zero-size array quietly.
    """
    tex = blank(8)
    back = download(tex)
    assert back.size > 0
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
"/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b --factory-startup \
  --python tests/blender/run_tests.py
```
Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.gpu'`

- [ ] **Step 3: Implement**

`marrow/gpu/__init__.py`:

```python
"""GPU-side adapters for marrow.core. Imports gpu; never imported by core."""
```

`marrow/gpu/textures.py`:

```python
"""numpy <-> GPUTexture. The only float32 boundary in the codebase.

GPUTexture accepts an upload Buffer only in FLOAT format - integer buffers
raise outright - so every image here is float-typed, including the one
holding tet indices.
"""

import gpu
import numpy as np

_CHANNELS = {"RGBA32F": 4, "R32F": 1}


def _channels(fmt: str) -> int:
    if fmt not in _CHANNELS:
        raise ValueError(f"unsupported texture format {fmt!r}; use one of {sorted(_CHANNELS)}")
    return _CHANNELS[fmt]


def upload(image: np.ndarray, fmt: str = "RGBA32F") -> gpu.types.GPUTexture:
    """Create a texture holding ``image``, which must be (H, W, channels)."""
    channels = _channels(fmt)
    array = np.ascontiguousarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != channels:
        raise ValueError(
            f"{fmt} needs an (H, W, {channels}) array, got {array.shape}"
        )
    height, width = array.shape[0], array.shape[1]
    buffer = gpu.types.Buffer("FLOAT", array.size, array.ravel().tolist())
    return gpu.types.GPUTexture((width, height), format=fmt, data=buffer)


def download(tex: gpu.types.GPUTexture) -> np.ndarray:
    """Read a texture back as (H, W, channels) float32.

    GPUTexture.read() hands back a Buffer shaped [H][W][C], not a flat
    sequence. Reshaping explicitly and refusing an empty result is what
    stops a zero-size readback from passing an elementwise comparison
    vacuously, which is how spike 0 produced a false PASS.
    """
    array = np.asarray(tex.read(), dtype=np.float32)
    if array.size == 0:
        raise RuntimeError("texture readback returned no data")
    if array.ndim == 2:  # single-channel comes back without a channel axis
        array = array[:, :, None]
    return array


def blank(count: int, fmt: str = "RGBA32F") -> gpu.types.GPUTexture:
    """A zeroed texture with room for ``count`` elements."""
    from marrow.core.layout import texture_shape

    width, height = texture_shape(count)
    zeros = np.zeros((height, width, _channels(fmt)), dtype=np.float32)
    return upload(zeros, fmt=fmt)
```

- [ ] **Step 4: Run to verify it passes**

Run the Blender command. Expected: the 5 `test_textures` tests print PASS alongside the existing 12, `0 failure(s)`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add marrow/gpu tests/blender/test_textures.py
git commit -m "feat: add numpy to GPUTexture upload and readback"
```

---

### Task 3: Shader construction with verbatim compile logs

**Files:**
- Create: `marrow/gpu/kernels.py`
- Test: `tests/blender/test_kernels_compile.py`

**Interfaces:**
- Consumes: nothing beyond `gpu`
- Produces:
  - `TEXEL_GLSL` — the shared `texel()` helper injected into every kernel
  - `build(name, source, images, push_constants, group_size=64) -> gpu.types.GPUShader`, raising `RuntimeError` whose message contains the driver's compile log verbatim
  - `PREDICT_SRC`, `SOLVE_SRC`, `COLLIDE_SRC`, `INTEGRATE_SRC`, `SKIN_SRC` — filled in by Tasks 4 to 9; this task defines only `PREDICT_SRC`

**Why the log matters:** the spec's failure table requires "surface the shader log verbatim, never swallow". A GLSL error swallowed into a generic exception is the single most expensive failure mode in this plan, because the kernel source is the thing most likely to be wrong.

- [ ] **Step 1: Write the failing test**

`tests/blender/test_kernels_compile.py`:

```python
import gpu

from marrow.gpu.kernels import PREDICT_SRC, TEXEL_GLSL, build

gpu.init()


def test_texel_helper_is_present_in_every_build():
    assert "ivec2 texel(" in TEXEL_GLSL


def test_predict_kernel_compiles():
    shader = build(
        "predict",
        PREDICT_SRC,
        images=[
            ("RGBA32F", "FLOAT_2D", "x", {"READ"}),
            ("RGBA32F", "FLOAT_2D", "v", {"READ"}),
            ("RGBA32F", "FLOAT_2D", "p", {"WRITE"}),
        ],
        push_constants=[("FLOAT", "h"), ("VEC3", "gravity"), ("INT", "n_nodes")],
    )
    assert shader is not None


def test_a_broken_kernel_surfaces_the_driver_log():
    broken = "void main() { this_is_not_glsl(); }"
    try:
        build("broken", broken, images=[], push_constants=[])
    except RuntimeError as exc:
        text = str(exc)
        assert "broken" in text, "error must name the kernel"
        assert len(text) > 40, f"compile log looks swallowed: {text!r}"
    else:
        raise AssertionError("a syntactically invalid kernel must not compile")
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command. Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.gpu.kernels'`

- [ ] **Step 3: Implement**

`marrow/gpu/kernels.py`:

```python
"""GLSL compute kernels and their construction.

Every kernel is a 1D dispatch over elements and does its own bounds check,
so a dispatch rounded up to a whole workgroup cannot write past the end.
The texel() helper must stay identical to marrow.core.layout.texel_index -
if the two ever disagree the solver reads someone else's data and nothing
raises.
"""

import gpu

TEX_WIDTH = 4096

TEXEL_GLSL = f"""
const int TEX_WIDTH = {TEX_WIDTH};

ivec2 texel(int i)
{{
  return ivec2(i % TEX_WIDTH, i / TEX_WIDTH);
}}
"""


def build(name, source, images, push_constants, group_size: int = 64):
    """Compile one compute kernel, or raise with the driver log intact."""
    info = gpu.types.GPUShaderCreateInfo()
    info.local_group_size(group_size, 1)
    for slot, (fmt, kind, image_name, qualifiers) in enumerate(images):
        info.image(slot, fmt, kind, image_name, qualifiers=qualifiers)
    for const_type, const_name in push_constants:
        info.push_constant(const_type, const_name)
    info.compute_source(TEXEL_GLSL + source)
    try:
        return gpu.shader.create_from_info(info)
    except Exception as exc:
        raise RuntimeError(
            f"Marrow kernel {name!r} failed to compile.\n"
            f"--- driver log ---\n{exc}\n--- end log ---"
        ) from exc


PREDICT_SRC = """
void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_nodes) { return; }
  ivec2 c = texel(i);

  vec4 xi = imageLoad(x, c);
  vec4 vi = imageLoad(v, c);
  float w = xi.w;

  vec3 pos = xi.xyz;
  if (w > 0.0) {
    pos += vi.xyz * h + gravity * (h * h);
  }
  imageStore(p, c, vec4(pos, w));
}
"""
```

- [ ] **Step 4: Run to verify it passes**

Run the Blender command. Expected: the 3 `test_kernels_compile` tests PASS, `0 failure(s)`.

- [ ] **Step 5: Commit**

```bash
git add marrow/gpu/kernels.py tests/blender/test_kernels_compile.py
git commit -m "feat: add kernel builder that surfaces GLSL compile logs verbatim"
```

---

### Task 4: predict kernel matches the oracle

**Files:**
- Modify: `marrow/gpu/kernels.py` (no source change; this task only adds a harness and the test)
- Create: `tests/blender/_oracle_harness.py`
- Test: `tests/blender/test_predict_vs_oracle.py`

**Interfaces:**
- Consumes: `build`, `PREDICT_SRC` from Task 3; `pack_nodes`, `unpack_vec3` from Task 1; `make_state`, `SolverParams` from `marrow/core/solver_ref.py`
- Produces: `tests/blender/_oracle_harness.py` with
  - `CUBE` — the one-cell lattice every kernel test uses
  - `BLOCK` — the 2x2x2 lattice
  - `oracle_predict(state, params, h) -> np.ndarray` running exactly the oracle's predict step and returning predicted positions
  - `assert_close(gpu_out, cpu_out, tol, what)` raising with the worst index and both values

**Tolerance:** the oracle is float64, the GPU is float32. On a unit-scale cube one predict step is a handful of float operations, so `1e-6` absolute is generous and any real sign or indexing error blows past it by orders of magnitude. Do not loosen a tolerance to make a test pass — a parity failure here means the kernel is wrong, not the tolerance.

**Note on the leading underscore:** `run_tests.py` discovers `test_*.py` only, so `_oracle_harness.py` is imported as a helper and never collected as a test module.

- [ ] **Step 1: Write the harness and the failing test**

`tests/blender/_oracle_harness.py`:

```python
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
```

`tests/blender/test_predict_vs_oracle.py`:

```python
import gpu
import numpy as np

from _oracle_harness import CUBE, assert_close, oracle_predict
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.kernels import PREDICT_SRC, build
from marrow.gpu.textures import blank, download, upload

gpu.init()

TOL = 1e-6

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "x", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "v", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "p", {"WRITE"}),
]
PUSH = [("FLOAT", "h"), ("VEC3", "gravity"), ("INT", "n_nodes")]


def _run_predict(state, params, h):
    shader = build("predict", PREDICT_SRC, IMAGES, PUSH)
    n = state.nodes.shape[0]

    tex_x = upload(pack_nodes(state.nodes, state.inv_mass))
    tex_v = upload(pack_nodes(state.velocities, np.zeros(n)))
    tex_p = blank(n)

    shader.bind()
    shader.image("x", tex_x)
    shader.image("v", tex_v)
    shader.image("p", tex_p)
    shader.uniform_float("h", h)
    shader.uniform_float("gravity", tuple(params.gravity))
    shader.uniform_int("n_nodes", n)
    gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)

    return unpack_vec3(download(tex_p), n)


def test_predict_matches_the_oracle_from_rest():
    params = SolverParams()
    state = make_state(CUBE.nodes)
    h = params.dt / params.substeps
    assert_close(_run_predict(state, params, h), oracle_predict(state, params, h),
                 TOL, "predict from rest")


def test_predict_matches_the_oracle_with_velocity():
    params = SolverParams()
    state = make_state(CUBE.nodes)
    rng = np.random.default_rng(2)
    state.velocities[:] = rng.uniform(-2.0, 2.0, size=state.nodes.shape)
    h = params.dt / params.substeps
    assert_close(_run_predict(state, params, h), oracle_predict(state, params, h),
                 TOL, "predict with velocity")


def test_predict_leaves_pinned_nodes_where_they_are():
    params = SolverParams()
    state = make_state(CUBE.nodes, pinned=np.array([0, 3], dtype=np.int32))
    state.velocities[:] = 5.0
    h = params.dt / params.substeps
    out = _run_predict(state, params, h)
    assert np.allclose(out[[0, 3]], CUBE.nodes[[0, 3]], atol=TOL), (
        "a zero-inverse-mass node must not be integrated"
    )


def test_predict_does_not_write_past_the_node_count():
    """The bounds check is what makes a rounded-up dispatch safe."""
    params = SolverParams()
    state = make_state(CUBE.nodes)
    n = state.nodes.shape[0]
    shader = build("predict", PREDICT_SRC, IMAGES, PUSH)

    tex_x = upload(pack_nodes(state.nodes, state.inv_mass))
    tex_v = upload(pack_nodes(state.velocities, np.zeros(n)))
    tex_p = blank(n)

    shader.bind()
    shader.image("x", tex_x)
    shader.image("v", tex_v)
    shader.image("p", tex_p)
    shader.uniform_float("h", 1.0)
    shader.uniform_float("gravity", (0.0, 0.0, -9.81))
    shader.uniform_int("n_nodes", n)
    gpu.compute.dispatch(shader, 4, 1, 1)  # 256 threads for 8 nodes

    flat = download(tex_p).reshape(-1, 4)
    assert np.all(flat[n:] == 0.0), "kernel wrote past n_nodes"
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command. Expected: FAIL, `ModuleNotFoundError: No module named '_oracle_harness'` on the first run, then real parity failures if the kernel is wrong.

- [ ] **Step 3: No implementation needed**

`PREDICT_SRC` already landed in Task 3. If the parity tests fail, the kernel is wrong — fix `PREDICT_SRC`, not the tolerance. The likely culprits, in order: `gravity * (h * h)` written as `gravity * h * h` with a precedence slip, `texel()` disagreeing with `texel_index`, or the `w > 0.0` pin check inverted.

- [ ] **Step 4: Run to verify it passes**

Run the Blender command. Expected: the 4 `test_predict_vs_oracle` tests PASS, `0 failure(s)`.

- [ ] **Step 5: Commit**

```bash
git add tests/blender/_oracle_harness.py tests/blender/test_predict_vs_oracle.py
git commit -m "test: pin predict kernel against the numpy oracle"
```

---

### Task 5: solve kernel matches the oracle

This is the task the whole plan exists to de-risk. It is also the only kernel where a wrong sign produces plausible-looking motion rather than an obvious explosion.

**Files:**
- Modify: `marrow/gpu/kernels.py` (add `SOLVE_SRC`)
- Test: `tests/blender/test_solve_vs_oracle.py`

**Interfaces:**
- Consumes: `color_ordered`, `pack_tets`, `pack_rest` from Task 1; `solve_constraints`, `precompute` from `marrow/core/solver_ref.py`
- Produces: `SOLVE_SRC`, dispatched once per colour with push constants `color_begin` and `color_end`

**The math, transcribed from `marrow/core/solver_ref.py:solve_constraints`.** Per tet, in this order:

1. `Ds` = columns `p1-p0, p2-p0, p3-p0`; `F = Ds * DmInv`
2. Deviatoric, if `mu > 0`: `C_D = length(F)` (Frobenius). **No rest offset** — see the correction note below. Gradient in F is `F / C_D`. Skip if `C_D <= 1e-12`.
3. Hydrostatic, if `lam > 0`: recompute `F` from the *updated* `p`, then `C_H = det(F) - gamma` with `gamma = 1 + mu/lam`. Gradient columns are `cross(f1,f2), cross(f2,f0), cross(f0,f1)`.
4. Node gradients from a gradient in F: `G = dCdF * transpose(DmInv)` gives columns for nodes 1, 2, 3; node 0's is `-(g1+g2+g3)`.
5. Projection: `alpha_tilde = compliance / (h*h)`, `dlambda = -C / (sum_i w_i |g_i|^2 + alpha_tilde)`, `p_i += w_i * g_i * dlambda`.

**Correction carried from the first plan — do not reintroduce.** An earlier version paired `C_D = ||F|| - sqrt(3)` with `gamma = 1 + mu/lam`. Those are two different ways of making the rest state stress-free and applying both cancels the deviatoric term twice, leaving the volume constraint unopposed. Measured, the body then inflates permanently to `det(F) = gamma` with a drift that does not shrink as substeps refine — flat at 1.6e-2 across substeps 4, 40 and 400, settling at 1.097x rest volume. The published pairing is offset-free `C_D` with `gamma`, and its residual converges: 7.3e-2, 1.5e-4, 1.7e-6.

**Why `dlambda` has no `alpha_tilde * lam_acc` term:** each tet is visited exactly once per substep and the multipliers are zeroed at the top of every substep, so the accumulator is provably zero when it is read. See the "Measured Facts" section.

**Recomputing F twice is deliberate,** not an oversight. The deviatoric projection has already moved `p`, so a stale `F` would linearise the volume constraint about the wrong configuration. The oracle does the same and its comment says so.

- [ ] **Step 1: Write the failing test**

`tests/blender/test_solve_vs_oracle.py`:

```python
import gpu
import numpy as np

from _oracle_harness import BLOCK, CUBE, assert_close
from marrow.core.coloring import color_tets
from marrow.core.layout import color_ordered, pack_nodes, pack_rest, pack_tets, unpack_vec3
from marrow.core.solver_ref import SolverParams, make_state, precompute, solve_constraints
from marrow.gpu.kernels import SOLVE_SRC, build
from marrow.gpu.textures import download, upload

gpu.init()

TOL = 2e-5  # float32 across a full constraint projection on a unit-scale cage

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "tets", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "rest", {"READ"}),
]
PUSH = [
    ("FLOAT", "h"),
    ("FLOAT", "mu"),
    ("FLOAT", "lam"),
    ("INT", "color_begin"),
    ("INT", "color_end"),
]


def _run_solve(mesh, state, params, h):
    """One GPU substep of constraint solving, colour by colour."""
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, offsets = color_ordered(mesh.tets, colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)

    shader = build("solve", SOLVE_SRC, IMAGES, PUSH)
    tex_p = upload(pack_nodes(state.predicted, state.inv_mass))
    tex_t = upload(pack_tets(ordered))
    tex_r = upload(pack_rest(dm_inv, rest_vol))

    for c in range(len(offsets) - 1):
        begin, end = int(offsets[c]), int(offsets[c + 1])
        if end <= begin:
            continue
        shader.bind()
        shader.image("p", tex_p)
        shader.image("tets", tex_t)
        shader.image("rest", tex_r)
        shader.uniform_float("h", h)
        shader.uniform_float("mu", params.mu)
        shader.uniform_float("lam", params.lam)
        shader.uniform_int("color_begin", begin)
        shader.uniform_int("color_end", end)
        gpu.compute.dispatch(shader, (end - begin + 63) // 64, 1, 1)

    return unpack_vec3(download(tex_p), mesh.n_nodes)


def _run_oracle(mesh, state, params, h):
    """Same substep on the CPU, over the same colour-ordered tets."""
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, _ = color_ordered(mesh.tets, colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)
    solve_constraints(state, ordered, dm_inv, rest_vol, params, h)
    return state.predicted.copy()


def _paired_states(mesh, deform):
    """Two identical states, one for each side of the comparison."""
    a, b = make_state(mesh.nodes), make_state(mesh.nodes)
    for st in (a, b):
        st.predicted[:] = deform(mesh.nodes.copy())
    return a, b


def test_solve_matches_oracle_on_a_stretched_cube():
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    h = params.dt / params.substeps

    def stretch(nodes):
        nodes[:, 0] *= 1.3
        return nodes

    gpu_state, cpu_state = _paired_states(CUBE, stretch)
    assert_close(
        _run_solve(CUBE, gpu_state, params, h),
        _run_oracle(CUBE, cpu_state, params, h),
        TOL,
        "solve on a stretched cube",
    )


def test_solve_matches_oracle_on_a_squashed_block():
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    h = params.dt / params.substeps

    def squash(nodes):
        nodes[:, 2] *= 0.7
        return nodes

    gpu_state, cpu_state = _paired_states(BLOCK, squash)
    assert_close(
        _run_solve(BLOCK, gpu_state, params, h),
        _run_oracle(BLOCK, cpu_state, params, h),
        TOL,
        "solve on a squashed block",
    )


def test_solve_matches_oracle_with_pinned_nodes():
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    h = params.dt / params.substeps
    pinned = np.array([0, 1, 2], dtype=np.int32)

    gpu_state = make_state(BLOCK.nodes, pinned=pinned)
    cpu_state = make_state(BLOCK.nodes, pinned=pinned)
    for st in (gpu_state, cpu_state):
        st.predicted[:] = BLOCK.nodes * 1.1

    out = _run_solve(BLOCK, gpu_state, params, h)
    assert_close(out, _run_oracle(BLOCK, cpu_state, params, h), TOL, "solve with pins")
    assert np.allclose(out[pinned], BLOCK.nodes[pinned] * 1.1, atol=TOL), (
        "pinned nodes must not be moved by a constraint projection"
    )


def test_solve_at_rest_is_near_stationary():
    """Not exactly zero - the rest residual is real. It must be small."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    h = params.dt / params.substeps
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes
    out = _run_solve(CUBE, state, params, h)
    assert np.abs(out - CUBE.nodes).max() < 1e-3, (
        "rest configuration drifted far more than sequential-projection residual"
    )


def test_zero_stiffness_is_a_noop():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0)
    h = params.dt / params.substeps
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes + 0.1
    out = _run_solve(CUBE, state, params, h)
    assert np.allclose(out, CUBE.nodes + 0.1, atol=TOL)
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command. Expected: FAIL, `ImportError: cannot import name 'SOLVE_SRC'`

- [ ] **Step 3: Implement**

Append to `marrow/gpu/kernels.py`:

```python
SOLVE_SRC = """
// Stable neo-Hookean, one deviatoric and one hydrostatic constraint per tet.
// Transcribed from marrow/core/solver_ref.py:solve_constraints. The two must
// stay in step: the oracle is the only way a sign error here is detectable.

mat3 shape_matrix(ivec4 idx, vec3 p0)
{
  return mat3(
    imageLoad(p, texel(idx.y)).xyz - p0,
    imageLoad(p, texel(idx.z)).xyz - p0,
    imageLoad(p, texel(idx.w)).xyz - p0
  );
}

void project(ivec4 idx, vec3 g0, vec3 g1, vec3 g2, vec3 g3,
             float c_value, float compliance, float h_step)
{
  vec4 n0 = imageLoad(p, texel(idx.x));
  vec4 n1 = imageLoad(p, texel(idx.y));
  vec4 n2 = imageLoad(p, texel(idx.z));
  vec4 n3 = imageLoad(p, texel(idx.w));

  float denom = n0.w * dot(g0, g0) + n1.w * dot(g1, g1)
              + n2.w * dot(g2, g2) + n3.w * dot(g3, g3);
  float alpha_tilde = compliance / (h_step * h_step);
  denom += alpha_tilde;
  if (denom < 1e-20) { return; }

  // The XPBD multiplier is zeroed every substep and each tet is visited once,
  // so the usual -alpha_tilde*lambda term is provably zero here.
  float dlambda = -c_value / denom;

  imageStore(p, texel(idx.x), vec4(n0.xyz + g0 * (n0.w * dlambda), n0.w));
  imageStore(p, texel(idx.y), vec4(n1.xyz + g1 * (n1.w * dlambda), n1.w));
  imageStore(p, texel(idx.z), vec4(n2.xyz + g2 * (n2.w * dlambda), n2.w));
  imageStore(p, texel(idx.w), vec4(n3.xyz + g3 * (n3.w * dlambda), n3.w));
}

void main()
{
  int t = color_begin + int(gl_GlobalInvocationID.x);
  if (t >= color_end) { return; }

  ivec4 idx = ivec4(imageLoad(tets, texel(t)));

  vec4 r0 = imageLoad(rest, texel(3 * t));
  vec4 r1 = imageLoad(rest, texel(3 * t + 1));
  vec4 r2 = imageLoad(rest, texel(3 * t + 2));
  mat3 dm_inv = mat3(r0.xyz, r1.xyz, r2.xyz);
  float rest_vol = abs(r0.w);

  vec4 w0 = imageLoad(p, texel(idx.x));
  vec4 w1 = imageLoad(p, texel(idx.y));
  vec4 w2 = imageLoad(p, texel(idx.z));
  vec4 w3 = imageLoad(p, texel(idx.w));
  if (!(w0.w > 0.0 || w1.w > 0.0 || w2.w > 0.0 || w3.w > 0.0)) { return; }

  mat3 dm_inv_t = transpose(dm_inv);

  // --- deviatoric ---
  if (mu > 0.0) {
    vec3 p0 = imageLoad(p, texel(idx.x)).xyz;
    mat3 f = shape_matrix(idx, p0) * dm_inv;

    float c_dev = sqrt(dot(f[0], f[0]) + dot(f[1], f[1]) + dot(f[2], f[2]));
    if (c_dev > 1e-12) {
      mat3 dcdf = f / c_dev;
      mat3 g = dcdf * dm_inv_t;
      vec3 g1v = g[0];
      vec3 g2v = g[1];
      vec3 g3v = g[2];
      vec3 g0v = -(g1v + g2v + g3v);
      project(idx, g0v, g1v, g2v, g3v, c_dev, 1.0 / (mu * rest_vol), h);
    }
  }

  // --- hydrostatic ---
  // F is rebuilt from the positions the deviatoric pass just moved. Reusing
  // the stale F would linearise the volume constraint about the wrong
  // configuration. The oracle does the same.
  if (lam > 0.0) {
    vec3 p0 = imageLoad(p, texel(idx.x)).xyz;
    mat3 f = shape_matrix(idx, p0) * dm_inv;

    mat3 dcdf = mat3(cross(f[1], f[2]), cross(f[2], f[0]), cross(f[0], f[1]));
    mat3 g = dcdf * dm_inv_t;
    vec3 g1v = g[0];
    vec3 g2v = g[1];
    vec3 g3v = g[2];
    vec3 g0v = -(g1v + g2v + g3v);

    float gamma = 1.0 + mu / lam;
    float c_hyd = determinant(f) - gamma;
    project(idx, g0v, g1v, g2v, g3v, c_hyd, 1.0 / (lam * rest_vol), h);
  }
}
"""
```

- [ ] **Step 4: Run to verify it passes**

Run the Blender command. Expected: the 6 `test_solve_vs_oracle` tests PASS, `0 failure(s)`.

Diagnostics if they do not, in the order worth checking:
- Every node off by a similar amount, rest test failing badly: a rest offset has crept into `c_dev`, or `gamma` is wrong.
- One tet wildly wrong, the rest fine: `texel()` disagrees with `texel_index`, or `pack_rest`'s column-major layout was transposed.
- Signs mirrored: `dcdf * dm_inv_t` written as `dm_inv_t * dcdf`. GLSL `mat3` is column-major and the multiplication order is load-bearing.
- Pinned nodes moving: `n.w` is being read from the wrong image, or `pack_nodes` put inverse mass somewhere other than `.a`.

- [ ] **Step 5: Commit**

```bash
git add marrow/gpu/kernels.py tests/blender/test_solve_vs_oracle.py
git commit -m "feat: add stable neo-Hookean solve kernel, pinned against the oracle"
```

---

### Task 6: integrate kernel matches the oracle

**Files:**
- Modify: `marrow/gpu/kernels.py` (add `INTEGRATE_SRC`)
- Test: `tests/blender/test_integrate_vs_oracle.py`

**Interfaces:**
- Consumes: Task 3's `build`, Task 1's packers
- Produces: `INTEGRATE_SRC` with images `x` (READ, WRITE), `p` (READ), `v` (READ, WRITE) and push constants `h`, `damping`, `n_nodes`

**Oracle behaviour being matched,** from `step()`: `v = (p - x) / h * damping` then `x = p`, both only where `inv_mass > 0`.

- [ ] **Step 1: Write the failing test**

`tests/blender/test_integrate_vs_oracle.py`:

```python
import gpu
import numpy as np

from _oracle_harness import CUBE, assert_close
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import make_state
from marrow.gpu.kernels import INTEGRATE_SRC, build
from marrow.gpu.textures import download, upload

gpu.init()

TOL = 1e-5

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "x", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "p", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "v", {"READ", "WRITE"}),
]
PUSH = [("FLOAT", "h"), ("FLOAT", "damping"), ("INT", "n_nodes")]


def _run_integrate(state, predicted, h, damping):
    n = state.nodes.shape[0]
    shader = build("integrate", INTEGRATE_SRC, IMAGES, PUSH)

    tex_x = upload(pack_nodes(state.nodes, state.inv_mass))
    tex_p = upload(pack_nodes(predicted, state.inv_mass))
    tex_v = upload(pack_nodes(state.velocities, np.zeros(n)))

    shader.bind()
    shader.image("x", tex_x)
    shader.image("p", tex_p)
    shader.image("v", tex_v)
    shader.uniform_float("h", h)
    shader.uniform_float("damping", damping)
    shader.uniform_int("n_nodes", n)
    gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)

    return unpack_vec3(download(tex_x), n), unpack_vec3(download(tex_v), n)


def _oracle_integrate(state, predicted, h, damping):
    movable = state.inv_mass > 0.0
    velocities = state.velocities.copy()
    nodes = state.nodes.copy()
    velocities[movable] = (predicted[movable] - nodes[movable]) / h * damping
    nodes[movable] = predicted[movable]
    return nodes, velocities


def test_integrate_matches_the_oracle():
    h, damping = 1 / 240, 0.999
    state = make_state(CUBE.nodes)
    rng = np.random.default_rng(4)
    predicted = CUBE.nodes + rng.uniform(-0.05, 0.05, size=CUBE.nodes.shape)

    gpu_x, gpu_v = _run_integrate(state, predicted, h, damping)
    cpu_x, cpu_v = _oracle_integrate(state, predicted, h, damping)
    assert_close(gpu_x, cpu_x, TOL, "integrate positions")
    assert_close(gpu_v, cpu_v, 1e-3, "integrate velocities")


def test_integrate_scales_velocity_by_damping():
    h = 1 / 240
    state = make_state(CUBE.nodes)
    predicted = CUBE.nodes + 0.01

    _, fast_v = _run_integrate(state, predicted, h, 1.0)
    _, slow_v = _run_integrate(state, predicted, h, 0.5)
    assert np.linalg.norm(slow_v) < np.linalg.norm(fast_v)


def test_integrate_leaves_pinned_nodes_alone():
    h = 1 / 240
    pinned = np.array([2], dtype=np.int32)
    state = make_state(CUBE.nodes, pinned=pinned)
    predicted = CUBE.nodes + 0.5

    gpu_x, gpu_v = _run_integrate(state, predicted, h, 1.0)
    assert np.allclose(gpu_x[pinned], CUBE.nodes[pinned], atol=TOL)
    assert np.allclose(gpu_v[pinned], 0.0, atol=TOL)
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command. Expected: FAIL, `ImportError: cannot import name 'INTEGRATE_SRC'`

- [ ] **Step 3: Implement**

Append to `marrow/gpu/kernels.py`:

```python
INTEGRATE_SRC = """
void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_nodes) { return; }
  ivec2 c = texel(i);

  vec4 xi = imageLoad(x, c);
  if (!(xi.w > 0.0)) { return; }   // pinned: position and velocity both hold

  vec3 pi = imageLoad(p, c).xyz;
  vec3 vel = (pi - xi.xyz) / h * damping;

  imageStore(v, c, vec4(vel, 0.0));
  imageStore(x, c, vec4(pi, xi.w));
}
"""
```

- [ ] **Step 4: Run to verify it passes**

Run the Blender command. Expected: the 3 `test_integrate_vs_oracle` tests PASS.

The velocity tolerance is looser than the position one on purpose: velocity is a difference divided by `h`, so float32 cancellation is amplified by `1/h` — at `h = 1/240` that is a factor of 240 on the position error, and 1e-3 is the honest bound rather than a fudge.

- [ ] **Step 5: Commit**

```bash
git add marrow/gpu/kernels.py tests/blender/test_integrate_vs_oracle.py
git commit -m "feat: add integrate kernel, pinned against the oracle"
```

---

### Task 7: collide kernel

**Files:**
- Modify: `marrow/gpu/kernels.py` (add `COLLIDE_SRC`)
- Test: `tests/blender/test_collide.py`

**Interfaces:**
- Consumes: Task 3's `build`
- Produces: `COLLIDE_SRC` with image `p` (READ, WRITE) and push constants `ground_z`, `ground_on` (INT used as a bool), `n_nodes`

**No oracle for this one.** `solver_ref` has no collision, so this kernel is tested on its own behaviour rather than by parity. That is also why the parity tests in Task 8 run with `ground_on = 0`: adding an unmodelled force to one side of a comparison would make the comparison meaningless.

- [ ] **Step 1: Write the failing test**

`tests/blender/test_collide.py`:

```python
import gpu
import numpy as np

from _oracle_harness import CUBE
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import make_state
from marrow.gpu.kernels import COLLIDE_SRC, build
from marrow.gpu.textures import download, upload

gpu.init()

IMAGES = [("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"})]
PUSH = [("FLOAT", "ground_z"), ("INT", "ground_on"), ("INT", "n_nodes")]


def _run_collide(positions, inv_mass, ground_z, ground_on):
    n = positions.shape[0]
    shader = build("collide", COLLIDE_SRC, IMAGES, PUSH)
    tex_p = upload(pack_nodes(positions, inv_mass))

    shader.bind()
    shader.image("p", tex_p)
    shader.uniform_float("ground_z", ground_z)
    shader.uniform_int("ground_on", int(ground_on))
    shader.uniform_int("n_nodes", n)
    gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)
    return unpack_vec3(download(tex_p), n)


def test_nodes_below_the_ground_are_lifted_onto_it():
    state = make_state(CUBE.nodes)
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 0.0, True)
    assert np.all(out[:, 2] >= -1e-6)


def test_nodes_above_the_ground_are_untouched():
    state = make_state(CUBE.nodes)
    lifted = CUBE.nodes + np.array([0.0, 0.0, 5.0])
    out = _run_collide(lifted, state.inv_mass, 0.0, True)
    assert np.allclose(out, lifted, atol=1e-6)


def test_horizontal_position_is_never_changed():
    state = make_state(CUBE.nodes)
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 0.0, True)
    assert np.allclose(out[:, :2], sunk[:, :2], atol=1e-6)


def test_disabled_ground_is_a_noop():
    state = make_state(CUBE.nodes)
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 0.0, False)
    assert np.allclose(out, sunk, atol=1e-6)


def test_pinned_nodes_are_not_pushed_by_the_ground():
    """A pin outranks a collider: the user put it there deliberately."""
    state = make_state(CUBE.nodes, pinned=np.array([0], dtype=np.int32))
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 0.0, True)
    assert np.allclose(out[0], sunk[0], atol=1e-6)


def test_ground_height_is_respected():
    state = make_state(CUBE.nodes)
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 1.5, True)
    assert np.all(out[:, 2] >= 1.5 - 1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command. Expected: FAIL, `ImportError: cannot import name 'COLLIDE_SRC'`

- [ ] **Step 3: Implement**

Append to `marrow/gpu/kernels.py`:

```python
COLLIDE_SRC = """
void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_nodes) { return; }
  if (ground_on == 0) { return; }

  ivec2 c = texel(i);
  vec4 pi = imageLoad(p, c);
  if (!(pi.w > 0.0)) { return; }   // a pin outranks a collider

  if (pi.z < ground_z) {
    imageStore(p, c, vec4(pi.x, pi.y, ground_z, pi.w));
  }
}
"""
```

- [ ] **Step 4: Run to verify it passes**

Run the Blender command. Expected: the 6 `test_collide` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add marrow/gpu/kernels.py tests/blender/test_collide.py
git commit -m "feat: add ground plane collide kernel"
```

---

### Task 8: full substep loop, parity, NaN detection, barrier at scale

**Files:**
- Create: `marrow/gpu/solver.py`
- Test: `tests/blender/test_solver_parity.py`

**Interfaces:**
- Consumes: everything from Tasks 1 to 7
- Produces:
  - `GPUSolver(mesh, inv_mass, params, ground_z=0.0, ground_on=False)` — builds every shader and uploads every texture once
  - `GPUSolver.step() -> None` running one frame of `params.substeps` substeps
  - `GPUSolver.positions() -> np.ndarray` shape `(N, 3)` float64
  - `GPUSolver.MarrowNaNError` raised by `step()` when the state stops being finite

**Three things this task settles at once,** because they can only be measured on the assembled loop:

1. **Parity.** The GPU loop against `solver_ref.step` over 10 frames.
2. **NaN.** The spec requires detection at readback, a freeze, and a refusal to write into cache.
3. **The carried barrier risk.** The spike proved chained dispatches correct at 16x16, which is 256 texels. This task re-runs the same determinism check at a realistic node count. If it fails, every kernel above is still correct and the fix is a serialisation point, not a rewrite — but it must be measured, not assumed.

- [ ] **Step 1: Write the failing test**

`tests/blender/test_solver_parity.py`:

```python
import gpu
import numpy as np

from _oracle_harness import BLOCK, CUBE, assert_close
from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import SolverParams, make_state, precompute, step
from marrow.gpu.solver import GPUSolver, MarrowNaNError

gpu.init()

TOL = 1e-4  # 10 frames x 10 substeps of accumulated float32 on a unit cage


def _oracle_run(mesh, params, frames, pinned=None):
    state = make_state(mesh.nodes, pinned=pinned)
    dm_inv, rest_vol = precompute(mesh.nodes, mesh.tets)
    for _ in range(frames):
        step(state, mesh.tets, dm_inv, rest_vol, params)
    return state.nodes


def _gpu_run(mesh, params, frames, pinned=None):
    state = make_state(mesh.nodes, pinned=pinned)
    solver = GPUSolver(mesh, state.inv_mass, params)
    for _ in range(frames):
        solver.step()
    return solver.positions()


def test_free_fall_matches_the_oracle():
    params = SolverParams(mu=0.0, lam=0.0)
    assert_close(_gpu_run(CUBE, params, 10), _oracle_run(CUBE, params, 10),
                 TOL, "10 frames of free fall")


def test_constrained_block_matches_the_oracle():
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    assert_close(_gpu_run(BLOCK, params, 10), _oracle_run(BLOCK, params, 10),
                 TOL, "10 frames of constrained block")


def test_pinned_block_under_gravity_matches_the_oracle():
    params = SolverParams()
    pinned = np.array([0, 1, 2], dtype=np.int32)
    assert_close(_gpu_run(BLOCK, params, 10, pinned), _oracle_run(BLOCK, params, 10, pinned),
                 TOL, "10 frames of pinned block")


def test_pinned_body_settles_rather_than_exploding():
    params = SolverParams()
    pinned = np.arange(4, dtype=np.int32)
    out = _gpu_run(BLOCK, params, 60, pinned)
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() < 100.0, "a pinned body drifted absurdly far"


def test_nan_state_is_detected_and_raises():
    """The spec requires a freeze and a refusal, not a quiet cache write."""
    params = SolverParams()
    state = make_state(CUBE.nodes)
    solver = GPUSolver(CUBE, state.inv_mass, params)
    solver.poison_for_test()
    try:
        solver.step()
    except MarrowNaNError as exc:
        assert "NaN" in str(exc) or "not finite" in str(exc)
    else:
        raise AssertionError("a non-finite state must not pass silently")


def test_dispatch_chain_is_deterministic_at_realistic_scale():
    """The spike proved this at 256 texels. Re-check it at ~35k nodes.

    There is no barrier API, so ordering between dependent dispatches is the
    driver's to guarantee. Running the same deterministic frame twice must
    give bit-identical results; if it does not, dispatches are racing.
    """
    big = build_lattice(np.zeros(3), 0.1, np.ones((20, 20, 20), dtype=bool))
    params = SolverParams(substeps=2)
    state = make_state(big.nodes)

    first = None
    for _ in range(3):
        solver = GPUSolver(big, state.inv_mass, params)
        solver.step()
        out = solver.positions()
        if first is None:
            first = out
        else:
            assert np.array_equal(first, out), (
                f"non-deterministic across runs at {big.n_nodes} nodes, "
                f"{big.n_tets} tets - dependent dispatches are racing"
            )
    print(f"  barrier check: {big.n_nodes} nodes, {big.n_tets} tets, deterministic")
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command. Expected: FAIL, `ModuleNotFoundError: No module named 'marrow.gpu.solver'`

- [ ] **Step 3: Implement**

`marrow/gpu/solver.py`:

```python
"""The GPU substep loop.

Textures are uploaded once at construction. Per frame the loop runs
predict -> solve (one dispatch per colour) -> collide -> integrate, entirely
on the card. Nothing crosses PCIe until positions() is asked for.

There is no barrier API in Blender's gpu module, so ordering between
dependent dispatches is the driver's to provide. Measured correct at 16x16
in the spike and re-checked at realistic scale by the test suite; if that
ever regresses, the fix is a readback between stages, which is a
serialisation point rather than a redesign.
"""

import gpu
import numpy as np

from marrow.core.coloring import color_tets
from marrow.core.layout import (
    color_ordered,
    pack_nodes,
    pack_rest,
    pack_tets,
    unpack_vec3,
)
from marrow.core.solver_ref import precompute
from marrow.gpu import kernels
from marrow.gpu.textures import blank, download, upload

_GROUP = 64


class MarrowNaNError(RuntimeError):
    """The solver state stopped being finite. Freeze, report, write nothing."""


def _groups(count: int) -> int:
    return max(1, (int(count) + _GROUP - 1) // _GROUP)


class GPUSolver:
    def __init__(self, mesh, inv_mass, params, ground_z=0.0, ground_on=False):
        self.mesh = mesh
        self.params = params
        self.ground_z = float(ground_z)
        self.ground_on = bool(ground_on)
        self.n_nodes = mesh.n_nodes

        colors = color_tets(mesh.tets, mesh.n_nodes)
        ordered, self.offsets = color_ordered(mesh.tets, colors)
        dm_inv, rest_vol = precompute(mesh.nodes, ordered)

        self.tex_x = upload(pack_nodes(mesh.nodes, inv_mass))
        self.tex_p = blank(self.n_nodes)
        self.tex_v = upload(pack_nodes(np.zeros_like(mesh.nodes), np.zeros(self.n_nodes)))
        self.tex_tets = upload(pack_tets(ordered))
        self.tex_rest = upload(pack_rest(dm_inv, rest_vol))

        self.sh_predict = kernels.build(
            "predict", kernels.PREDICT_SRC,
            [("RGBA32F", "FLOAT_2D", "x", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "v", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "p", {"WRITE"})],
            [("FLOAT", "h"), ("VEC3", "gravity"), ("INT", "n_nodes")],
        )
        self.sh_solve = kernels.build(
            "solve", kernels.SOLVE_SRC,
            [("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"}),
             ("RGBA32F", "FLOAT_2D", "tets", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "rest", {"READ"})],
            [("FLOAT", "h"), ("FLOAT", "mu"), ("FLOAT", "lam"),
             ("INT", "color_begin"), ("INT", "color_end")],
        )
        self.sh_collide = kernels.build(
            "collide", kernels.COLLIDE_SRC,
            [("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"})],
            [("FLOAT", "ground_z"), ("INT", "ground_on"), ("INT", "n_nodes")],
        )
        self.sh_integrate = kernels.build(
            "integrate", kernels.INTEGRATE_SRC,
            [("RGBA32F", "FLOAT_2D", "x", {"READ", "WRITE"}),
             ("RGBA32F", "FLOAT_2D", "p", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "v", {"READ", "WRITE"})],
            [("FLOAT", "h"), ("FLOAT", "damping"), ("INT", "n_nodes")],
        )

    def step(self) -> None:
        """One frame. Raises MarrowNaNError rather than propagating garbage."""
        h = self.params.dt / self.params.substeps
        node_groups = _groups(self.n_nodes)

        for _ in range(self.params.substeps):
            self.sh_predict.bind()
            self.sh_predict.image("x", self.tex_x)
            self.sh_predict.image("v", self.tex_v)
            self.sh_predict.image("p", self.tex_p)
            self.sh_predict.uniform_float("h", h)
            self.sh_predict.uniform_float("gravity", tuple(self.params.gravity))
            self.sh_predict.uniform_int("n_nodes", self.n_nodes)
            gpu.compute.dispatch(self.sh_predict, node_groups, 1, 1)

            for c in range(len(self.offsets) - 1):
                begin, end = int(self.offsets[c]), int(self.offsets[c + 1])
                if end <= begin:
                    continue
                self.sh_solve.bind()
                self.sh_solve.image("p", self.tex_p)
                self.sh_solve.image("tets", self.tex_tets)
                self.sh_solve.image("rest", self.tex_rest)
                self.sh_solve.uniform_float("h", h)
                self.sh_solve.uniform_float("mu", self.params.mu)
                self.sh_solve.uniform_float("lam", self.params.lam)
                self.sh_solve.uniform_int("color_begin", begin)
                self.sh_solve.uniform_int("color_end", end)
                gpu.compute.dispatch(self.sh_solve, _groups(end - begin), 1, 1)

            self.sh_collide.bind()
            self.sh_collide.image("p", self.tex_p)
            self.sh_collide.uniform_float("ground_z", self.ground_z)
            self.sh_collide.uniform_int("ground_on", int(self.ground_on))
            self.sh_collide.uniform_int("n_nodes", self.n_nodes)
            gpu.compute.dispatch(self.sh_collide, node_groups, 1, 1)

            self.sh_integrate.bind()
            self.sh_integrate.image("x", self.tex_x)
            self.sh_integrate.image("p", self.tex_p)
            self.sh_integrate.image("v", self.tex_v)
            self.sh_integrate.uniform_float("h", h)
            self.sh_integrate.uniform_float("damping", self.params.damping)
            self.sh_integrate.uniform_int("n_nodes", self.n_nodes)
            gpu.compute.dispatch(self.sh_integrate, node_groups, 1, 1)

        self._guard_finite()

    def positions(self) -> np.ndarray:
        return unpack_vec3(download(self.tex_x), self.n_nodes)

    def poison_for_test(self) -> None:
        """Force a non-finite state, so the NaN guard can be tested honestly."""
        poisoned = pack_nodes(
            np.full_like(self.mesh.nodes, np.nan),
            np.ones(self.n_nodes),
        )
        self.tex_x = upload(poisoned)

    def _guard_finite(self) -> None:
        out = unpack_vec3(download(self.tex_x), self.n_nodes)
        if not np.all(np.isfinite(out)):
            bad = int(np.count_nonzero(~np.isfinite(out).all(axis=1)))
            raise MarrowNaNError(
                f"Marrow solver produced NaN or inf at {bad} of {self.n_nodes} "
                f"nodes. The frame was not written. Lower Substeps or Stiffness "
                f"in the Marrow panel and re-run."
            )
```

- [ ] **Step 4: Run to verify it passes**

Run the Blender command. Expected: the 6 `test_solver_parity` tests PASS, and the barrier check prints its node and tet counts.

If `test_dispatch_chain_is_deterministic_at_realistic_scale` fails, stop and report before continuing to Task 9. That result invalidates the loop structure, not the kernels, and the decision about how to serialise belongs to the human partner.

If the parity tests fail while every per-kernel test from Tasks 4 to 7 passes, the fault is in the loop, not the math: check that `collide` runs on `p` before `integrate` reads it, and that `substeps` is applied per frame rather than per call.

- [ ] **Step 5: Commit**

```bash
git add marrow/gpu/solver.py tests/blender/test_solver_parity.py
git commit -m "feat: add GPU substep loop with oracle parity and NaN guard"
```

---

### Task 9: skin kernel and render mesh readback

**Files:**
- Modify: `marrow/gpu/kernels.py` (add `SKIN_SRC`), `marrow/gpu/solver.py` (add skinning)
- Test: `tests/blender/test_skin.py`

**Interfaces:**
- Consumes: `bind_points`/`deform` from `marrow/core/bind.py`, `GPUSolver` from Task 8
- Produces:
  - `SKIN_SRC` with images `x` (READ), `tets` (READ), `bind` (READ), `out` (WRITE) and push constant `n_render`
  - `GPUSolver.attach_render(bind_idx, bind_w) -> None`
  - `GPUSolver.skin() -> np.ndarray` shape `(R, 3)` float64

**The readback rule.** This is the only place data comes back per frame, and it moves `R` texels, not `N`. On a 50k-tet body that is roughly 3k render vertices against 12k nodes. Reading `x` back per frame instead would be the mistake that kills most GPU-simulation-in-Python attempts.

**Bind texture layout:** one RGBA32F texel per render vertex, `.r` holding the tet index as a float and `.gba` holding barycentric weights `w1, w2, w3`. `w0` is recovered in the shader as `1 - w1 - w2 - w3`, which is exact enough for weights that sum to 1 by construction and saves a second texture.

- [ ] **Step 1: Write the failing test**

`tests/blender/test_skin.py`:

```python
import gpu
import numpy as np

from _oracle_harness import BLOCK
from marrow.core.bind import bind_points, deform
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.solver import GPUSolver

gpu.init()

TOL = 1e-5


def _solver_with_render(points, params=None):
    params = params or SolverParams(gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0)
    state = make_state(BLOCK.nodes)
    solver = GPUSolver(BLOCK, state.inv_mass, params)
    idx, w = bind_points(BLOCK.nodes, BLOCK.tets, points)
    solver.attach_render(idx, w)
    return solver, idx, w


def test_skin_at_rest_reproduces_the_bound_points():
    rng = np.random.default_rng(7)
    points = rng.uniform(0.05, 0.95, size=(40, 3))
    solver, _, _ = _solver_with_render(points)
    assert np.allclose(solver.skin(), points, atol=TOL)


def test_skin_matches_the_cpu_deform():
    rng = np.random.default_rng(8)
    points = rng.uniform(0.05, 0.95, size=(40, 3))
    solver, idx, w = _solver_with_render(points)
    solver.step()

    cpu = deform(solver.positions(), BLOCK.tets, idx, w)
    gpu_out = solver.skin()
    assert gpu_out.shape == cpu.shape
    assert np.abs(gpu_out - cpu).max() < TOL


def test_skin_follows_a_falling_cage():
    rng = np.random.default_rng(9)
    points = rng.uniform(0.05, 0.95, size=(20, 3))
    params = SolverParams(mu=0.0, lam=0.0)
    solver, _, _ = _solver_with_render(points, params)
    before = solver.skin()
    for _ in range(5):
        solver.step()
    after = solver.skin()
    assert np.all(after[:, 2] < before[:, 2]), "render points did not fall with the cage"


def test_skin_reads_back_only_render_vertices():
    """R texels, not N. The readback rule is the Python-side ceiling."""
    rng = np.random.default_rng(10)
    points = rng.uniform(0.05, 0.95, size=(7, 3))
    solver, _, _ = _solver_with_render(points)
    assert solver.skin().shape == (7, 3)
    assert BLOCK.n_nodes > 7, "fixture no longer proves anything"
```

- [ ] **Step 2: Run to verify it fails**

Run the Blender command. Expected: FAIL, `AttributeError: 'GPUSolver' object has no attribute 'attach_render'`

- [ ] **Step 3: Implement the kernel**

Append to `marrow/gpu/kernels.py`:

```python
SKIN_SRC = """
void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_render) { return; }
  ivec2 c = texel(i);

  vec4 b = imageLoad(bind, c);
  int t = int(b.r);
  vec3 w = b.gba;
  float w0 = 1.0 - w.x - w.y - w.z;

  ivec4 idx = ivec4(imageLoad(tets, texel(t)));
  vec3 pos = imageLoad(x, texel(idx.x)).xyz * w0
           + imageLoad(x, texel(idx.y)).xyz * w.x
           + imageLoad(x, texel(idx.z)).xyz * w.y
           + imageLoad(x, texel(idx.w)).xyz * w.z;

  imageStore(out_pos, c, vec4(pos, 1.0));
}
"""
```

- [ ] **Step 4: Implement the solver side**

Add to `marrow/gpu/solver.py`, inside `GPUSolver`:

```python
    def attach_render(self, bind_idx, bind_w) -> None:
        """Upload the render-vertex bind data. Call once, not per frame."""
        bind_idx = np.asarray(bind_idx, dtype=np.int64)
        bind_w = np.asarray(bind_w, dtype=np.float64)
        self.n_render = int(bind_idx.shape[0])

        packed = np.zeros((self.n_render, 4), dtype=np.float64)
        packed[:, 0] = bind_idx
        packed[:, 1:] = bind_w[:, 1:]  # w0 is recovered as 1 - w1 - w2 - w3

        from marrow.core.layout import texture_shape

        width, height = texture_shape(self.n_render)
        image = np.zeros((height, width, 4), dtype=np.float32)
        image.reshape(-1, 4)[: self.n_render] = packed.astype(np.float32)

        self.tex_bind = upload(image)
        self.tex_skin = blank(self.n_render)
        self.sh_skin = kernels.build(
            "skin", kernels.SKIN_SRC,
            [("RGBA32F", "FLOAT_2D", "x", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "tets", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "bind", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "out_pos", {"WRITE"})],
            [("INT", "n_render")],
        )

    def skin(self) -> np.ndarray:
        """Blend render vertices out of the cage and read back only those."""
        if not hasattr(self, "sh_skin"):
            raise RuntimeError("attach_render() must be called before skin()")

        self.sh_skin.bind()
        self.sh_skin.image("x", self.tex_x)
        self.sh_skin.image("tets", self.tex_tets)
        self.sh_skin.image("bind", self.tex_bind)
        self.sh_skin.image("out_pos", self.tex_skin)
        self.sh_skin.uniform_int("n_render", self.n_render)
        gpu.compute.dispatch(self.sh_skin, _groups(self.n_render), 1, 1)

        return unpack_vec3(download(self.tex_skin), self.n_render)
```

**Note the tet-index mismatch.** `GPUSolver.__init__` uploads *colour-ordered* tets, but `bind_points` in the test returns indices into the *original* tet order. Task 9's implementation must keep the original ordering available for skinning — store the permutation in `__init__` and translate `bind_idx` through it inside `attach_render`, or upload a second unpermuted tets texture. Choose one and say which in the commit message; the tests above will catch the mismatch immediately if neither is done, because `test_skin_at_rest_reproduces_the_bound_points` will bind to the wrong tets.

- [ ] **Step 5: Run to verify it passes**

Run the Blender command. Expected: the 4 `test_skin` tests PASS, `0 failure(s)`, exit code 0.

Then confirm the core suite is untouched:

Run: `./.venv/Scripts/python.exe -m pytest tests/core -v`
Expected: 64 passed.

- [ ] **Step 6: Commit**

```bash
git add marrow/gpu tests/blender/test_skin.py
git commit -m "feat: add skin kernel and render-vertex readback"
```

---

## Self-Review Notes

**Spec coverage.** Texture budget maps to Tasks 1 and 2, with two rows corrected and justified in the "Measured Facts" section. The four substep kernels map to Tasks 4 to 7, `skin` and the readback rule to Task 9, the substep loop to Task 8. The constraint model maps to Task 5, transcribed from the oracle rather than re-derived. Of the spec's failure-handling table, this plan covers GLSL compile logs (Task 3) and NaN detection with a refusal to write (Task 8). The remaining rows — open-mesh warning, node-count budget, no-GPU-context disable — are UI-layer concerns and belong to the step 4 plan, since nothing here has a UI to report through.

**Deferred to the step 4 plan:** frame cache, material presets, the modifier and its frame handler, the node budget refusal, the no-GPU-context disable at register, GUI-mode confirmation of the step 0 spike result, and Vulkan backend validation.

**Two spec rows are corrected, not implemented as written.** `tets | RGBA32I` is impossible — `GPUTexture` rejects any non-FLOAT upload buffer — so indices ride in `RGBA32F` with 2**24 of exact headroom. `lambda | R32F | 2T` is dropped because the oracle's multiplier is provably zero at every read. Both are measured, not assumed, and both are recorded at the top of this plan so the next reader does not re-derive them.

**The oracle is the spine.** Tasks 4, 5, 6 and 8 all compare against `solver_ref`. That is deliberate: the spec calls the CPU reference solver "the load-bearing testing decision", and the first plan's worst defect was a constraint-formulation error that only surfaced when a test actually ran the math. Task 5 restates that correction inline so it cannot be reintroduced from memory.

**Tolerances are justified, never tuned.** `1e-6` for a single predict step, `2e-5` for one constraint projection, `1e-4` for ten frames of accumulation, `1e-3` for velocities because a difference divided by `h` amplifies float32 cancellation by `1/h`. If a parity test fails, the kernel is wrong. Loosening a tolerance to get green is the one move this plan forbids outright.

**The barrier risk is carried, not resolved.** There is no barrier API and the spike only proved determinism at 256 texels. Task 8 re-tests at roughly 35k nodes and stops the plan if it fails, because the remedy is a structural decision rather than a bug fix.

**Known gap, deliberately left to the implementer.** Task 9 has a real ordering hazard: the solver uploads colour-ordered tets while `bind_points` indexes the original order. The task names the hazard and both viable fixes rather than picking one, because the choice depends on whether the step 4 cache wants the permutation available anyway. The tests fail loudly if it is ignored.

---

## Execution Notes, 2026-08-12

All nine tasks executed. 49 Blender tests, 64 core tests, both suites exit 0.

**The carried barrier risk resolved in the direction the plan hoped, for dispatch-to-dispatch.** Task 8's determinism check ran at 9261 nodes and 40000 tets, three runs bit-identical. The loop structure stands.

**But an unlisted sibling of that risk is real and unfixed: dispatch-to-readback.** `GPUTexture.read()` issued straight after a compute dispatch intermittently returns the pre-dispatch contents. Caught with readback output bit-identical to the uploaded input, in roughly 15-20% of full-suite runs, always in the collide module — the only place a single small dispatch is immediately followed by a readback. This is the GL rule that `imageStore` writes require a memory barrier before being read by other means; Blender's Python API never issues one and exposes no way to ask.

It is a product bug, not only a test bug: `skin()` reads back every frame, so a stale read is a dropped frame of animation.

A mitigation was tried and rejected: a no-op read-modify-write dispatch on the target image before reading, to put a real dependency in front of the readback. It crashed Blender at shutdown with `EXCEPTION_ACCESS_VIOLATION` because the sync shaders were cached in a module-level dict, which keeps GPU objects alive past context teardown. Any future attempt must not hold GPU objects in module state, and must be measured against the ~15-20% baseline rather than assumed to work — 10 clean runs is not evidence at that rate.

**Three plan defects found by execution, all fixed:**

1. Task 8's `step()` guard downloaded the whole node image every frame — the full-state readback the spec singles out as fatal. Detection moved to the readback boundary; `step()` no longer touches PCIe.
2. Task 8's oracle helper iterated tets in original order while the GPU runs colour-ordered. XPBD is Gauss-Seidel and order-dependent, so parity missed by 3.4e-4. Identified by measurement, not by loosening the tolerance: the gap was already 3.05e-4 after one frame, barely grew over 40, was identical against a float32-rounded oracle, and shrank as substeps rose. Rounding error grows with more operations; ordering error shrinks.
3. Task 5's rest test asserted an invented 1e-3 absolute bound, but the oracle's own single-substep drift on that cube is 1.61e-3. It was demanding the GPU be more correct than its own reference. Rewritten as a parity assertion.

**Deferred, unchanged:** build step 4 — cache, material presets, the modifier and frame handler, node budget refusal, no-GPU-context disable, GUI-mode confirmation of the step 0 spike, Vulkan validation.
