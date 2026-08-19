# Marrow: Tetrahedral Fiber Constraints

Date: 2026-08-19

## Summary

An anisotropic constraint per tet, aligned to a fiber direction baked from a
curve, driven by a procedural travelling wave. Contraction along the fiber
plus the existing hydrostatic pass produces the sideways bulge, and
contraction plus ground friction produces locomotion. No keyframes.

## Motivation

The README lists it as a limitation in as many words: *"No plasticity,
anisotropy or per-region materials."* Every tet in Marrow today is
isotropic. It resists distortion equally in all directions and there is no
way to say "this material is stronger along its length", let alone "this
material shortens along its length on cue".

That single missing term is what separates a passive jelly from a creature.
A snake, a tentacle, a gut, a worm - all of them are a tube with fibers
running down it, contracting in a wave. The technique comes from Houdini's
Vellum tetrahedral fiber constraints driven by a `materialW` direction
attribute, where the activation is animated by expression rather than
keyframes.

Marrow already has everything else that scene needs: a tet cage, an XPBD
solver with a volume term, ground contact and Coulomb friction. The fiber
constraint is the missing piece, and it is one pass.

## Design decisions

### D1. Anisotropic constraint on F, not a modulated rest shape

The constraint is `C = |F a| - s`, where `a` is the rest-space unit fiber
direction and `s` is the activation. It projects through the existing
`project()` helper, inside the existing per-tet loop, under the existing
graph colouring. One new image, six new push constants, no new dispatch
structure.

The alternative considered was modulating each tet's rest shape - rescaling
`dm_inv` along `a` each frame and letting the current deviatoric and
hydrostatic passes do the work. It adds no solve cost, which is tempting.
It was rejected because `rest` becomes a per-frame mutable texture,
`rest_vol` drifts away from the value the compliance normalization assumes,
and the hydrostatic `gamma` term (`kernels.py`, the `1.0 + mu / lam` line)
would have to track a moving target. That trades a cheap GPU pass for a
per-frame upload and a correctness hazard in the one piece of algebra that
currently has no moving parts.

A third option - explicit node-pair fiber edges - was rejected for needing
new topology, its own graph colouring and more storage, while constraining
lengths rather than deformation and therefore behaving worse under volume
preservation.

### D2. The fiber term sits between deviatoric and hydrostatic

Order is load-bearing. The hydrostatic pass runs last and rebuilds `F` from
the positions the passes before it moved, so it gets the final word on
volume. That is precisely what converts "shortened along the fiber" into
"bulged sideways". Running fiber after hydrostatic would let each
contraction steal volume that nothing puts back.

### D3. Fiber direction comes from a curve, sampled once at Tetrahedralize

Each tet takes the tangent of the nearest point on a user-supplied Curve
object as its direction, and the arclength at that point as its wave phase.
One sample yields both, and the phase channel is what makes the wave travel
rather than pulse in unison.

A curve handles a rest pose that is coiled or S-shaped, which a fixed object
axis cannot - fibers derived from an axis on a curled snake point through
its own side walls. It needs no vector painting, which is the other way to
get arbitrary directions and is awkward in Blender.

The sample happens at Tetrahedralize and is frozen. This is not only
simpler, it is required: `a` is a rest-space direction, because `F` maps
rest to world. An animated curve has no meaning as a fiber source. Changing
the curve therefore requires re-tetrahedralizing, and the panel says so.

### D4. Activation is procedural, computed on the GPU

Amplitude, wavelength, speed and waveform are push constants. Each tet
computes its own activation from its baked phase and a simulation clock:

```
phase      = fiber.w / wave_len - wave_time * wave_speed
pulse      = smooth ? 0.5 * (1 - cos(2*pi*fract(phase))) : step(0.5, fract(phase))
activation = 1 - wave_amp * pulse
```

Nothing is uploaded per frame. The repo currently uploads nothing per frame
and this feature does not change that. The sliders are ordinary Blender
properties, so they keyframe and take drivers, which is the flexibility a
per-tet expression channel would have bought at the price of a CPU pass and
a PCIe transfer every frame.

The square waveform is the direct analogue of the `(@Frame%10)/10` modulo
expression the technique originally came from; smooth is the organic
version.

### D5. The clock is per-substep, not per-frame

`GPUSolver` gains `sim_time`, starting at zero and advancing by `h` at the
end of `substep_constraints`, so every colour dispatched within one substep
sees one value. A per-frame clock would step the wave in visible stairs at
low substep counts.

The oracle mirrors this with a local `t` in `step()`, advanced by `h` after
each substep.

A live restart resets `sim_time` to zero along with the rest of the solver
state, so the wave starts from the same phase every run.

### D6. Wave travel direction and speed are readable numbers

The wave travels toward increasing arclength; a negative speed reverses it.
Travel velocity is `wave_length * wave_speed` in world units per second,
which is the number the user actually dials against ground friction, and it
belongs in the tooltip.

### D7. A torn tet has no fiber

Fiber is skipped when `is_torn`, alongside the deviatoric term. Tearing in
Marrow means the material goes slack; torn muscle that still pulls would
contradict that.

### D8. Fiber off is the current code path

`fiber_k = 0` skips the branch entirely, and the oracle reproduces today's
results bit-identically. The cost on a body with the feature off is one
texel load per tet per colour per substep.

## Data layout

Per-tet, `T * 4` floats:

| Channel | Meaning |
|---|---|
| `xyz` | Rest-space unit fiber direction. Zero means this tet has no fiber and the constraint is skipped. |
| `w` | Arclength along the curve at the sampled point, in world units. |

Stored as a flat float ID property `marrow_fiber` on the **cage** mesh,
alongside `marrow_tets` and `marrow_blend`. Per-tet data cannot live in a
POINT attribute, because the cage mesh has one vertex per node and not one
per tet, and the existing per-tet arrays already use this route.

Uploaded once at solver build as `tex_fiber`, an RGBA32F texture through the
existing verified-upload path. It is never mutated, so it needs none of the
re-check machinery the bind texture carries.

De-tetrahedralize strips `marrow_fiber` with everything else.

## Components

### `marrow/core/fiber.py` (new, pure numpy)

Takes a sampled polyline (`(S, 3)` points) and tet centroids, returns
`(T, 4)`. For each centroid: nearest point on the polyline, segment tangent
normalized, cumulative arclength at that point. Degenerate segments yield a
zero direction rather than a NaN. No `bpy`, so it is testable outside
Blender and is covered by the existing `test_no_bpy.py`.

### `marrow/core/solver_ref.py`

`SolverParams` gains `fiber_k=0.0`, `wave_amp=0.0`, `wave_len=1.0`,
`wave_speed=0.0`, `waveform=0`, all inert by default. `step()` and
`solve_constraints()` gain `fiber=None` as a keyword argument, so no
existing call site changes. `step()` owns the clock.

The early bail at the top of `solve_constraints` - currently
`if params.mu <= 0.0 and params.lam <= 0.0: return` - must also test
`fiber_k`, or a fiber-only test solves nothing and passes for the wrong
reason.

The projection:

```python
if params.fiber_k > 0.0 and fiber is not None:
    a = fiber[i, :3]
    if float(a @ a) > 0.5:
        cycle = (fiber[i, 3] / params.wave_len - t * params.wave_speed) % 1.0
        pulse = (0.5 * (1.0 - np.cos(2.0 * np.pi * cycle))
                 if params.waveform == 0 else float(cycle >= 0.5))
        s = 1.0 - params.wave_amp * pulse
        fa = f @ a
        length = float(np.linalg.norm(fa))
        if length > 1e-12:
            grads = _grads_from_dcdf(np.outer(fa / length, a), dm_inv[i])
            _apply(state, idx, grads, length - s,
                   1.0 / (params.fiber_k * rest_vol[i]), h, 0.0)
```

GLSL `fract` and numpy `% 1.0` agree on negatives, which matters because
`wave_time * wave_speed` drives the phase negative almost immediately.

### `marrow/gpu/kernels.py`

One new block in `SOLVE_SRC`, between the deviatoric and hydrostatic
sections:

```glsl
  // --- fiber ---
  // Transversely isotropic term: C = |F a| - s, with a the rest-space fiber
  // direction and s the activation. s < 1 shortens the tet along a; the
  // hydrostatic pass below then has to put that volume somewhere, which is
  // the bulge. A zero direction means this tet was never assigned a fiber.
  if (fiber_k > 0.0 && !is_torn) {
    vec4 fb = imageLoad(fiber, texel(t));
    vec3 a = fb.xyz;
    if (dot(a, a) > 0.5) {
      float phase = fb.w / wave_len - wave_time * wave_speed;
      float cycle = fract(phase);
      // Smooth is muscle; square is the literal (@Frame%10)/10 blink.
      float pulse = (waveform == 0)
        ? 0.5 * (1.0 - cos(6.2831853 * cycle))
        : step(0.5, cycle);
      float s = 1.0 - wave_amp * pulse;

      vec3 fa = f * a;
      float len = length(fa);
      if (len > 1e-12) {
        mat3 dcdf = outerProduct(fa / len, a);
        mat3 g = dcdf * dm_inv_t;
        vec3 g1v = g[0];
        vec3 g2v = g[1];
        vec3 g3v = g[2];
        vec3 g0v = -(g1v + g2v + g3v);
        project(idx, g0v, g1v, g2v, g3v, len - s, 1.0 / (fiber_k * rest_vol), h);
      }
    }
  }
```

The fiber block rebuilds `F` (`mat3 ff`) from the positions the deviatoric
projection has just moved, rather than reusing the `f` computed at the top of
`main()`. There is no pre-existing staleness here to match: `f` is built
before any `project()` call in the dispatch, and a colour's tets are
node-disjoint, so the deviatoric pass reads a fresh `F`. Reusing it for the
fiber term would be the first stale one, and it would linearise the
constraint about a configuration the solver has already left. This is the
same reason the hydrostatic block recomputes under its own comment calling
that recompute load-bearing rather than an oversight. The oracle recomputes
in the same place, so parity holds.

### `marrow/gpu/solver.py`

The solve kernel gains one image (`fiber`, RGBA32F, READ) and six push
constants (`fiber_k`, `wave_amp`, `wave_len`, `wave_speed`, `wave_time`,
`waveform`). The solve kernel's twelve push constants - ten floats and two
ints, 48 bytes of scalar data - stay under the 128-byte Vulkan floor, and
the driver emits no size warning when it builds.

The kernel that will need a UBO at port time is `collide`, not this one, and
it needed one before this feature. `COLLIDE_PUSH` (`kernels.py`) carries two
`MAT4`s, which are the whole 128-byte floor on their own; the driver warns
its way up to `the constants added so far already reach 184 bytes` as the
four scalars after them are declared. No push-constant list in `kernels.py`
changed on this branch, so fibers add no pressure of their own.

`sim_time` is initialized to zero and advanced in `substep_constraints`.

### `marrow/gpu/textures.py`

`tex_fiber`, packed from the `(T, 4)` array, uploaded once and verified like
the others.

### `marrow/blender/ops.py`

Tetrahedralize evaluates the fiber curve to a polyline in the soft body's
object space, calls `core.fiber`, and writes `marrow_fiber`. With no curve
set, nothing is written. De-tetrahedralize removes the key.

### `marrow/blender/storage.py`

`FIBER_KEY = "marrow_fiber"`, with read and write helpers matching the
existing per-tet array pattern, and inclusion in the strip list.

### `marrow/blender/session.py`

Reads the five numeric settings into `SolverParams` and binds `tex_fiber`.
This puts them on the existing restart path for free: return to the start
frame and the wave re-reads with no rebake.

### `marrow/blender/ui.py`

Six properties on `MarrowSettings`:

| Property | Default | Meaning |
|---|---|---|
| `fiber_enabled` | off | Box header toggle |
| `fiber_curve` | none | Object pointer, `poll` restricted to `CURVE` |
| `fiber_stiffness` | 1e4 | Fiber compliance, same scale as Stiffness |
| `wave_amplitude` | 0.3 | 0.3 shortens to 70% at peak. Clamped 0..0.9; 1.0 asks a tet to reach zero length |
| `wave_length` | 1.0 | Wave period in curve arclength, world units |
| `wave_speed` | 1.0 | Cycles per second |
| `waveform` | Smooth | Smooth or Square |

Gating has three states. No `marrow_fiber` on the cage: the box draws a
"Tetrahedralize with a curve set" note and the pass never dispatches. Data
present but `fiber_enabled` off: `fiber_k` is zero and the kernel branch is
dead. Both: live.

## Testing

### `tests/core/` - no `bpy`, bare pytest

`test_fiber.py`
- Straight polyline along +X: every tet reads direction +X, phase equal to
  its centroid x.
- Curved polyline: direction tracks the tangent.
- Zero-length segment: zero direction, no NaN.

`test_solver_fiber.py`
- `s < 1` shortens a tet measured along `a`, and only along `a`.
- With `lam` high, volume holds while fiber length drops.
- `fiber_k = 0` reproduces the current solve bit-identically.
- Square and smooth diverge; square below half-cycle gives exactly `s = 1`.
- Two tets at different arclength peak at different times.
- A torn tet ignores fiber.

### `tests/blender/` - inside Blender, on 5.2

`test_fiber_vs_oracle.py` is the load-bearing one. Same shape as
`test_solve_vs_oracle.py`: a random cage with random fibers, several
substeps, GPU positions diffed against numpy at the suite's existing
tolerance. A sign error in the `outerProduct` dies here or nowhere.

`test_fiber_storage.py` round-trips `marrow_fiber` through save and load and
confirms De-tetrahedralize strips it.

`test_fiber_ui.py` covers the three gating states.

Nothing needs adding for kernel compilation, but not via
`test_kernels_compile.py` - that file builds `PREDICT_SRC` and a
deliberately broken shader, and never touches `SOLVE_SRC`. The solve
kernel's build is covered anyway: `test_solve_vs_oracle` and
`test_fiber_vs_oracle` call `build()` on `SOLVE_SRC` directly, and
`test_fiber_solver` compiles it by way of a real session. A driver
rejecting `outerProduct` or the push-constant count surfaces there.

GPU tests run on Blender 5.2. 4.5 fails them all for environmental reasons.

### Not a test

Locomotion is the acceptance criterion but makes a slow, flaky assertion.
`tools/fiber_demo.py` builds the scene instead - a tube, a curve down its
spine, ground plane, friction up - following the `tools/estimate_cage.py`
precedent.

## Ceilings

- **One fiber field per body.** No independent muscle groups, no per-region
  activation. A second body is the way to get a second wave.
- **The wave is one sine or one square.** No layered waves, no per-tet
  authored patterns, no vertex-group mask on amplitude. All of these are
  additive later against the same baked channel.
- **The curve is frozen at Tetrahedralize.** Changing it means
  re-tetrahedralizing; there is no re-bake operator in this version.
- **`mu` and `lam` stay global.** Fiber adds anisotropy, not per-region
  materials, which remains a separate limitation.
- **Thin bodies still need a fine cage.** The cube-split lattice does not
  hug a thin tube, so a snake wants Adaptive or a small Resolution or its
  fibers smear across the tube walls.
- **No False Color mode for activation.** The wave is visible only through
  motion in this version.
