# Marrow: Self-Collision

Design spec, 2026-08-13. Extends the v1 spec, `2026-08-11-marrow-gpu-tet-soft-body-design.md`, which listed self-collision as an explicit non-goal.

## Summary

Surface nodes of the tet cage push each other apart when they come within a contact thickness. All-pairs on the GPU, Jacobi, no broad phase, no readback.

One new kernel, one new core function, two new settings. The substep loop gains a single dispatch between `solve` and `collide`.

## Motivation

Without self-collision a body that folds passes through itself. Every shot that motivated the recent work needs it: the reference gel-stretch clip collapses into a sheet that lands on itself, and a torn body's flaps overlap immediately.

Houdini's Vellum tet softbody exposes this as a checkbox plus a thickness, and that is the UX target. Blender's own XPBD solver treats self-collision as its acknowledged hole for cloth, so there is nothing to inherit.

## Enabling measurement

Measured 2026-08-13, Blender 5.2.0 LTS, RTX 5050.

### Integer images and atomics do not work in this API

Ten Minute Physics #15 builds a dense spatial hash: count per cell, prefix sum, fill. Counting needs `imageAtomicAdd` on an integer image. Four spikes:

| Probe | Result |
|---|---|
| `GPUTexture(format="R32I")` creation | OK |
| `imageAtomicAdd` on `R32I` in `GPUShaderCreateInfo` | compiles |
| `imageStore(img, c, ivec4(7))` on `R32I`, then read back | never lands; `read()` returns garbage |
| `GPUTexture.read()` on `R32I` | always a `Buffer(FLOAT, ...)`; there is no integer readback path |
| `gpu.types.Buffer("INT", ...)` upload | rejected: "Only Buffer of format `FLOAT` is currently supported" |
| Control: `imageStore(vec4(7.0))` on `R32F` | clean `7.0` in every texel |

The integer path is unusable end to end, not merely awkward. The reference hash cannot be ported.

### Surface node counts

Unit ball, current lattice tetrahedralizer:

| Resolution | nodes | surface nodes | surface share |
|---|---|---|---|
| 0.25 | 462 | 314 | 68% |
| 0.15 | 1696 | 848 | 50% |
| 0.10 | 5233 | 1898 | 36% |
| 0.07 | 14136 | 3866 | 27% |
| 0.05 | 37292 | 7586 | 20% |

The surface share falls as resolution rises, which is what makes all-pairs viable: the cost grows with the square of an area, not of a volume.

### All-pairs cost, measured

A Jacobi kernel with the rest-distance gate, dispatched over the node image, timed with a flush and readback forcing completion:

| surface nodes | pairs/substep | ms/substep | ms/frame at 10 substeps |
|---|---|---|---|
| 314 | 49,298 | 0.08 | 0.8 |
| 848 | 359,552 | 0.21 | 2.1 |
| 1,898 | 1,801,202 | 0.48 | **4.8** |
| 3,866 | 7,472,978 | 0.96 | 9.6 |
| 7,586 | 28,773,698 | 1.88 | 18.8 |

15 billion pair-tests per second at the top of the range. Current frame cost without self-collision is 1-2 ms, so Resolution 0.1 goes to roughly 7 ms/frame and stays interactive.

### Confirmed after implementation

Measured through the shipped `GPUSolver`, ball cages, 10 substeps:

| Resolution | cage nodes | surface nodes | off | on | delta |
|---|---|---|---|---|---|
| 0.25 | 461 | 314 | 1.9 ms | 3.0 ms | 1.1 ms |
| 0.15 | 1,707 | 848 | 2.0 ms | 4.9 ms | 2.9 ms |
| 0.10 | 5,233 | 1,898 | 2.1 ms | 8.4 ms | **6.3 ms** |
| 0.07 | 14,226 | 3,866 | 2.5 ms | 15.2 ms | 12.8 ms |

The delta runs about 30% above the microbenchmark - the bind and uniform calls happen once per substep, ten times a frame. The conclusion is unchanged.

## Design decisions

### D1. No spatial hash

The hash in the reference exists to avoid quadratic cost on a CPU in JavaScript. On this GPU, restricted to surface nodes, all-pairs is faster than building a hash would be, and it needs none of the machinery this API lacks. A hash would also need per-frame neighbour lists uploaded from the CPU, which means a full node readback per frame - the one thing `marrow/gpu/solver.py` is explicitly built to avoid.

Rejected alternatives: CPU spatial hash rebuilt per frame (scales past ~20k surface nodes, breaks the readback rule, much more code); GPU uniform grid without atomics (needs a bitonic sort over many passes; no).

### D2. Surface nodes only

Interior nodes cannot be the first point of contact on a closed cage - the surface reaches any external material first. Restricting to the boundary cuts the pair count from 5233^2 to 1898^2 at Resolution 0.1, roughly 8x, and is what makes D1 hold.

### D3. Jacobi, with a ping-pong texture

Each thread owns one node, reads every surface node, and writes only its own texel. No two threads write the same texel, so this needs no graph colouring, no atomics, and no second pass - the reference is Gauss-Seidel (writes both particles of a pair) and would need exactly the machinery that does not exist here.

Reading and writing the same image in one dispatch is a race, so the kernel writes a second position image and Python swaps `self.tex_p` with `self.tex_p2` afterwards. Every other stage binds `self.tex_p` at dispatch time, so the swap costs nothing and there is no copy pass.

**Consequence: every thread must write exactly one texel of `out_p`, always.** An early `return` for a pinned or interior node would leave a stale texel in the swapped image. Interior nodes and pinned nodes copy their position through unchanged.

### D4. Rest-distance gate

Straight from the reference. Lattice neighbours sit at exactly Resolution apart at rest; with a thickness of one Resolution they would fight the tets on every substep. So: skip a pair whose current separation is at least its rest separation, and push only to the smaller of thickness and that rest separation.

### D5. Thickness is a multiple of Resolution

One float, unitless, default 1.0. It auto-tracks when Resolution changes, and 1.0 is the padded gap that keeps the skinned render surface from visibly interpenetrating. The session multiplies it by `settings.resolution` and hands the solver an absolute distance, so the solver stays in world units like every other collider.

### D6. Placement: after `solve`, before `collide`

`predict -> solve -> self-collide -> collide -> integrate`. External colliders and pins get the last word, so a sticky contact or a ground plane outranks a self-contact. Velocity needs no special handling: `integrate` derives it from `(p - x)/h`, so a position correction becomes a velocity like every other constraint. The reference uses friction 0, so there is no separate friction pass.

### D7. Mass-weighted correction split

The reference splits a correction 0.5/0.5 because its particles have equal mass. Here a pinned node has `inv_mass == 0`, so the split is `w_i / (w_i + w_j)`: 0.5 between two free nodes, and the full correction onto the free node of a free/pinned pair. Same cost, correct against pins.

## Data layout

Four new images. The self-collide kernel binds five (`p`, `out_p`, `rest_pos`, `surf`, `surf_idx`), under the measured limit of eight per shader.

| Name | Format | Texels | Contents |
|---|---|---|---|
| `tex_p2` | RGBA32F | `n_nodes` | ping-pong target for `tex_p` |
| `tex_rest_pos` | RGBA32F | `n_nodes` | `.xyz` rest node position (the post-lift `start` array, the same configuration `dm_inv` is built from) |
| `tex_surf` | R32F | `n_surf` | `.x` cage node index of the i-th surface node |
| `tex_surf_idx` | R32F | `n_nodes` | `.x` position of this node in the surface list, or `-1` if interior |

`tex_rest` already exists and holds per-tet `dm_inv` and rest volume; the new image is per-node and separately named to avoid the collision.

Node indices stored as floats are exact below 2^24. The largest cage measured here is 37k nodes.

## Components

### `marrow/core/tetmesh.py`: `surface_nodes(tets) -> np.ndarray`

A tet face owned by exactly one tet is a boundary face. Build the `(4*n_tets, 3)` face array, sort each row so a shared face matches whichever tet names it, `np.unique(axis=0, return_counts=True)`, keep `count == 1`, and return the sorted unique node indices of those faces.

Pure numpy, no GPU, no Blender. Testable on its own.

### `marrow/gpu/kernels.py`: `SELF_COLLIDE_SRC`

Dispatched over `n_nodes`. Thread `i` loads `p[i]`; if `surf_idx[i] < 0` or `p[i].w == 0` it stores `p[i]` unchanged and stops. Otherwise it loops all `n_surf` entries of `tex_surf`, skipping itself, applying the thickness test, the rest-distance gate (D4) and the mass-weighted split (D7), accumulating into a local correction, and stores `p[i] + correction` with `.w` preserved.

### `marrow/gpu/solver.py`

`GPUSolver.__init__` gains a `self_distance=0.0` argument - an absolute world distance, 0 disables - calls `surface_nodes`, uploads the four images, builds the shader. `step()` inserts the dispatch and the ping-pong swap. When `self_distance <= 0.0` nothing is allocated and nothing is dispatched.

The name differs from the UI property deliberately: the UI value is a multiple of Resolution, the solver value is metres.

### `marrow/blender/ui.py`, `session.py`

`self_collision: BoolProperty` (default off) and `self_thickness: FloatProperty` (default 1.0, min 0.1, soft_max 3.0, unitless). The session converts to an absolute distance and passes it through, mirroring how `tear_threshold` and `stick_break` are already wired.

## Testing

Core, no GPU:

1. `surface_nodes` on a single tet returns all four nodes.
2. On a lattice cage, every returned node lies on at least one boundary face, and no interior node does.
3. A closed cage's boundary faces each appear exactly once.

GPU, in the existing Blender-run suite:

4. Two free nodes whose *rest* positions are far apart, currently at half thickness: they separate to thickness.
5. Two nodes whose *rest* separation is half thickness, currently at half thickness: unchanged - the gate holds.
6. A pinned node does not move; its free partner takes the whole correction.
7. Interior nodes and pinned nodes come out of the dispatch unchanged, position and `.w` both (guards the ping-pong write-through).
8. Self-collision off dispatches nothing and reproduces the existing golden trajectory bit for bit.

Integration:

9. A tall soft column dropped so it buckles onto itself: minimum pairwise surface-node distance stays above 0.8x thickness with the feature on, and falls far below it with the feature off.

## Ceilings

Deliberate simplifications, each with its upgrade path.

- **Velocity clamp.** Built 2026-08-14 after a stiff thin slab dropped onto the ground wadded its impact corner: `integrate` caps the carried velocity at `0.2 * thickness / h`, thickness the larger of the active self/body contact distances, 0 disabling it so no-contact trajectories stay bit identical. The cap limits the velocity carried into the next predict; the position corrections of the substep that produced it stand.
- **Quadratic in surface nodes.** Fine to about 20k surface nodes (roughly Resolution 0.03 on a unit ball), where it would reach ~130 ms/frame. Past that, a CPU spatial hash with per-frame neighbour upload, accepting the readback cost.
- **Node-node only.** No edge-edge or node-triangle contact. A feature thinner than the node spacing can slip through between nodes; the thickness default of one Resolution is what keeps that from happening in practice.
- **Jacobi converges slower than Gauss-Seidel.** With many simultaneous contacts some residual penetration remains at the end of a substep. Substeps mitigate it.
- **Self-collision only, not body-to-body.** Two separate Marrow objects still pass through each other. Unchanged from v1.
