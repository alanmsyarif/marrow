# Marrow: GPU Tetrahedral Soft Body for Blender 5.2

Design spec, 2026-08-11.

## Summary

Marrow is a Blender 5.2 addon that simulates volumetric soft bodies on the GPU. It fills a surface mesh with a tetrahedral cage, solves a stable neo-Hookean XPBD constraint system in GLSL compute shaders, and deforms the original render mesh by barycentric interpolation from that cage.

Written in pure Python against Blender's bundled interpreter. No compiled extension, no external dependency, no per-platform build matrix.

## Motivation

Blender 5.2 ships an XPBD solver node, but it covers hair, cloth and particles only. There is no volumetric soft body; the legacy `SOFT_BODY` modifier is a surface spring lattice with no volume preservation.

The gap is structural rather than a missing feature. The GSoC 2022 final report (Aarnav Dhanuka) states that Blender has no standard format for volumetric tetrahedral meshes, which is what most soft body solvers operate on. Two separate GSoC efforts wrote a surface-to-tet pipeline, Matt Overby in 2020 and Dhanuka in 2022, and neither landed.

Evidence gathered 2026-08-11:

- Blender 5.2 XPBD Solver node documents hair, cloth, particles. No volumetric mode.
- The official "Experimental Physics Feedback for Blender 5.2 LTS" devtalk thread is hair and cloth centric. Self-collision is the acknowledged hole; volumetric soft body is not on the track at all.
- extensions.blender.org public API returns 1346 extensions. Zero match `xpbd`, `softbody`, or `jelly`. Two taglines mention "soft body": RigFlex and Zeeks Hair, both surface jiggle, neither volumetric.
- PHENIX (GPU-native particle system, v5 RC on Superhive, May 2026) lists Jelly as one of nine material presets on a Constraint Advanced node. The words tetrahedral, FEM, volumetric, interior and volume constraint appear nowhere in its documentation. It is a particle system constraining particles.
- ZOZO's ppf-contact-solver covers FEM deformables with penetration-free contact, shipped as a Blender 5.0 addon. It competes on contact accuracy and offline quality, not on interactive iteration.

Marrow competes on interactivity and on owning the tet data layer, not on contact accuracy.

### Enabling measurement

Measured on the target machine 2026-08-11, Blender 5.2.0 LTS, RTX 5050:

| Capability | Status |
|---|---|
| `gpu.compute.dispatch(shader, x, y, z)` | present |
| `GPUShaderCreateInfo.compute_source` / `.local_group_size` | present |
| Backend | OpenGL 4.6 |
| Storage buffers (SSBO) in the Python API | **absent** |
| `image` and `uniform_buf` in `GPUShaderCreateInfo` | present |
| Max images per shader | 8 |
| Max texture size | 32768 |
| Bundled Python | 3.13.13, numpy 2.3.4, no scipy, no torch |

GLSL compute is therefore reachable from a pure-Python addon. The absence of SSBOs is the single hardest constraint and shapes the entire data layout.

### Prior measurement that constrains the design

Measured 2026-08-07: `Custom Effector` closures in Blender's bundled dynamics asset library evaluate once per frame at every stage, not once per substep. With Substeps=8 and Constraint Steps=3, a per-substep hook would have fired 24 times per frame; it fired once.

Consequence: Marrow cannot live inside Blender's XPBD node as a custom effector. It is a standalone modifier with its own substep loop and its own cache.

## Non-goals

Explicitly out of v1:

- Self-collision. Expensive, and Blender is already building it for cloth.
- Body-to-body collision.
- Arbitrary mesh colliders. v1 ships ground plane and pinned vertex group only.
- Plasticity, fracture, tearing.
- Anisotropic fiber directions.
- Per-region material zones.
- Conforming (surface-exact) tetrahedralization. See the cube-split lattice decision below.

## Architecture

Six components, one direction of data flow.

| # | Component | Location | Runs |
|---|---|---|---|
| 1 | Tetrahedralizer | CPU, numpy | Once, at bake |
| 2 | Tet data layer | Blender mesh + attributes | Persisted in .blend |
| 3 | GPU packer | CPU to GPUTexture | Once per sim start |
| 4 | Solver kernels | GLSL compute | S substeps per frame |
| 5 | Frame driver + cache | Python | Per frame |
| 6 | UI panel | Python | - |

Flow: surface mesh, tetrahedralize once, pack to float32 textures once, then per frame run S substep dispatches entirely on the card, read back surface vertices only, write to the evaluated mesh.

Symbols used throughout: **N** = tet cage nodes, **T** = tets, **R** = render mesh vertices, **S** = substeps per frame, **C** = constraint colors.

**Readback rule.** Interior tet nodes never cross PCIe. A 50k-tet body has roughly 12k nodes but perhaps 3k render vertices, so only a quarter of the state moves. Per-frame full-state readback is what kills most GPU-simulation-in-Python attempts.

**Constraint coloring.** XPBD Gauss-Seidel requires parallel-safe batches or threads race on shared nodes. Tets are graph-colored on the CPU at bake time and color offsets are stored in the data layer. Each color becomes one dispatch. Coloring costs nothing at runtime.

## Stage 1: tetrahedralizer and data layer

### Decision: cube-split lattice with barycentric embedding, not conforming Delaunay

Conforming tetrahedralization must perform boundary recovery, forcing the tet mesh to exactly reproduce the input surface. That step is fragile on real Blender meshes (non-manifold, self-intersecting, ngons, loose geometry) and is a plausible reason both GSoC attempts stalled. It also replaces the render mesh, breaking UVs, shape keys and material slots.

Marrow uses a cube-split lattice with checkerboard parity alternation instead:

1. Voxel grid over the object bounds, driven by a single resolution slider.
2. 5-tet cube subdivision with checkerboard parity alternation, which keeps neighbouring cells conforming. Uniform quality by construction, no slivers.
3. Keep cells that are inside the surface or straddle it. Inside test via `mathutils.bvhtree.BVHTree` ray parity, which ships with Blender and needs no scipy.
4. Bind each render vertex to its containing tet and store barycentric weights.
5. The render mesh deforms by barycentric interpolation from the cage.

Original topology is never modified. Messy input is tolerated because the only question asked of the mesh is inside or outside. Sliver tets, the classic FEM stability killer, cannot occur.

**Accepted cost.** A cube-split lattice cage does not hug concave detail, so thin or highly detailed models need a higher resolution setting than a conforming mesh would. This is a slider, not a failure mode.

### Storage in the .blend

Addons cannot register new datablocks, so tet data rides on existing ones.

| Data | Home |
|---|---|
| Tet nodes | Mesh vertices of a hidden cage mesh |
| Tet connectivity, 4 ints each | ID-property blob on the cage mesh |
| Node mass, pin weight | POINT-domain attributes on the cage |
| Bind tet index, barycentric weights | POINT attributes on the render mesh |
| Color offsets | ID-property on the cage mesh |
| Rest inverse matrices | Recomputed at load, not stored |

Survives save and load, inspectable in the spreadsheet, no custom datablock required.

Stage 1 is independently useful and ships alone: a tetrahedralizer plus cage viewer plus a documented data layer is a tool other addons can build on.

## Stage 2: GPU solver

### Texture budget

Six of the eight available images are used.

| Image | Format | Size | Holds |
|---|---|---|---|
| `x` | RGBA32F | N nodes | position xyz, inverse mass in w |
| `p` | RGBA32F | N | predicted position |
| `v` | RGBA32F | N | velocity |
| `tets` | RGBA32I | T tets | 4 node indices, exact fit |
| `restInv` | RGBA32F | 3T | Dm inverse, 9 floats across 3 texels, rest volume in the spare w |
| `lambda` | R32F | 2T | XPBD multiplier accumulator |

All images are 2D with index mapping `i -> (i % 4096, i / 4096)`. A 1D layout would cap at 32768 nodes; 2D removes the ceiling.

### Substep loop

Four kernels per substep:

1. `predict` over N nodes: `p = x + dt*v + dt*dt*g`
2. `solve` over T tets, dispatched once per color. Coloring guarantees no two tets in a color share a node, so read-modify-write is race-free and no atomics are needed.
3. `collide` over N nodes: ground plane, pins.
4. `integrate` over N nodes: `v = (p - x) / dt * damping`, `x = p`

Then `skin` over R render vertices, barycentric blend from the cage, followed by one RGBA32F readback of R texels into the mesh via `foreach_set`.

**Dispatch count, corrected against measurement 2026-08-11.** This spec originally estimated C=6 colours and "roughly 90 dispatches per frame". That was wrong by roughly 6x. Measured on the implemented cube-split lattice: 32 colours at 2³, 34 at 3³, 37 at 4³ and 5³. The count is geometry-imposed, not an artefact of the greedy colouring — an interior lattice node is shared by 32 tets, which is a hard lower bound on the chromatic number, so the greedy result lands only 2 to 5 colours above optimal. Choosing a different colouring algorithm cannot meaningfully reduce it.

So the real figure at S=10, C≈37 is `10 × (1 predict + 37 solve + 1 collide + 1 integrate) ≈ 400` dispatches per frame, plus one `skin` and one readback. Still workable — these are tiny kernels and dispatch overhead on a modern driver is on the order of microseconds — but it is a per-frame cost worth measuring early rather than assuming negligible, and it makes substep count the most expensive knob in the UI. If it proves too slow, the lever is fewer substeps or a coarser cage, not a better colouring.

### Constraint model

Stable neo-Hookean, two constraints per tet: hydrostatic (volume) and deviatoric (distortion). Better conditioned than a raw volume-plus-distance formulation, and it maps to exactly two user-facing sliders.

| Slider | Maps to |
|---|---|
| Stiffness | Deviatoric compliance |
| Volume Preservation | Hydrostatic compliance |
| Damping | Velocity scale in `integrate` |
| Substeps | Loop count per frame |

## Failure handling

Every failure names its fix rather than raising a traceback.

| Failure | Response |
|---|---|
| Open or non-manifold mesh, inside test unreliable | Detect, warn, offer voxel-fill fallback |
| Zero tets generated, resolution too coarse for a thin mesh | Error naming the slider to raise |
| Node count over budget | Refuse with the actual count, never hang |
| GLSL compile failure | Surface the shader log verbatim, never swallow |
| Solver produces NaN | Detect at readback, freeze, report the frame, refuse to write into cache |
| No usable GPU context | Disable at register with a plain message |

## Testing

**The CPU reference solver is the load-bearing testing decision.** A small numpy implementation of identical math serves as a test oracle. GPU kernels are otherwise undebuggable, because a wrong sign in a compute shader is indistinguishable from a wrong sign in the constraint derivation. With an oracle, per-step positions on a 5-tet mesh are diffed and the faulty side is identified immediately.

Remaining tests are headless and cheap:

- A cube tetrahedralizes to a known tet count, tet volumes sum to the mesh volume, and no tet is inverted.
- Every barycentric bind sums to 1 with no negative weights.
- No two tets within a color share a node, checked exhaustively.
- Free fall matches the analytic curve.
- A pinned body under gravity settles rather than exploding.
- An incompressible cube under load preserves volume within tolerance.

## Build order

Numbered steps below are build sequence. They are distinct from the two architectural stages above: Stage 1 (tet layer) is delivered by steps 0 to 1, Stage 2 (GPU solver) by steps 2 to 4.

0. **Spike.** Does `gpu.compute.dispatch` work from a `frame_change_post` handler? One day, before any other code.
1. Tetrahedralizer, data layer, cage viewer, tests. Ships alone.
2. CPU reference solver in numpy. The oracle.
3. GPU kernels, validated against the oracle.
4. Cache, UI, material presets.

**Decomposition.** Steps 0 to 2 form the first implementation plan and end at a shippable Stage 1 plus a working oracle. Steps 3 to 4 form a second plan, written after the spike has resolved risk 1. Attempting all five steps in one plan would put unvalidated GPU context assumptions underneath a dozen tasks.

## Open risks

1. ~~**GPU context from a frame handler.**~~ **RESOLVED 2026-08-11 by `tools/spike_00_gpu_context.py`.** `gpu.compute.dispatch` runs correctly from inside a `frame_change_post` handler. A compute shader writing `gl_GlobalInvocationID` into an RGBA32F image was verified by readback, shape `(8,8,4)`, exact per-texel match, both from plain script context and from the handler. No draw-handler or modal-timer fallback is needed.

   Two caveats carried forward. Tested in background mode (`blender -b`) on the OpenGL backend; GUI mode has a live window and draw loop and should be confirmed once there is a UI. And `gpu.init()` must be called before any `gpu.*` access in background mode.

   Note for the test suite: `GPUTexture.read()` returns a Buffer shaped `[H][W][C]`, not a flat sequence. Verify readbacks through `np.asarray(...).reshape(H, W, C)` and assert on size first — an empty readback compared with `all(... for ... in zip(...))` passes vacuously, which produced a false PASS on the first run of this spike.
2. **Readback is the Python-side ceiling.** Trivial at 3k render vertices, painful at 500k. The solver stays fast; the bridge does not. Measure early and publish the honest number.
3. **PHENIX Jelly internals are undisclosed.** Its documentation contains no tetrahedral or FEM vocabulary, which strongly suggests a particle constraint network rather than volumetric FEM, but this is inference and not confirmation.
4. **Backend portability.** Measured on OpenGL. Blender is moving to Vulkan; kernels must be validated on the Vulkan backend before release.
5. **Superhive name check outstanding.** extensions.blender.org is verified clear across 1346 extensions for `marrow`, `aspic` and `viscera`. Superhive has no public API and blocks automated fetches, so its catalogue should be checked by hand before launch.

## Environment

- Blender 5.2.0 LTS, `C:\Program Files\Blender Foundation\Blender 5.2`
- Bundled Python 3.13.13, numpy 2.3.4, no scipy, no torch
- GPU: NVIDIA RTX 5050, OpenGL 4.6 backend
- Repo: `C:\Users\user\Documents\marrow`, product name and repo name matching
