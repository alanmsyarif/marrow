# Marrow: Body-to-Body Collision

Design spec, 2026-08-13. Follows `2026-08-13-marrow-self-collision-design.md`, whose kernel this reuses almost entirely, and lifts the last collision non-goal from the v1 spec.

## Summary

Two tetrahedralised Marrow objects collide, and both deform. Surface nodes of one body push surface nodes of the other, all pairs, on the GPU.

The contact math is already written. What is missing is that nothing in the addon advances two bodies together, so the bulk of this change is to the stepping model, not to the physics.

## Motivation

Self-collision stops a body passing through itself. Two bodies still pass through each other, which rules out every shot with more than one soft object in it. Houdini's Vellum solves all bodies in one shared solve, so this is table stakes for the comparison.

## The problem: nothing steps two bodies together

Measured against the code as it stands:

- Each object owns a `MarrowSession` with its own `GPUSolver`. No state is shared.
- `GPUSolver.step()` runs every substep internally. A caller can only advance a body one whole frame at a time.
- `handlers.on_frame_change` walks `SESSIONS` in dictionary order and advances each independently.
- `MARROW_OT_bake` bakes the active object over the entire frame range before any other object starts.

So the only coupling reachable without a restructure is one whole frame of lag. At 24 fps a body moving 5 m/s travels 0.2 m per frame, against a default thickness of one Resolution - 0.1 m at Resolution 0.1. It would tunnel straight through. Substep-level interleaving is not optional.

One thing already works in our favour: solvers hold world-space positions. Collider specs are built from `matrix_world`, and `write_to_mesh` converts world to object space on the way out. Two bodies therefore already share a frame of reference and need no transform plumbing.

## Design decisions

### D1. Split `step()` into `substep()`, and route everything through a group driver

```python
def step(self, others=()):
    h = self.params.dt / self.params.substeps
    for _ in range(self.params.substeps):
        self.substep(h, others)

def substep(self, h, others=()):
    self.substep_constraints(h, others)   # predict, solve colours,
    self.substep_integrate(h)             # self-collide, body-collide,
                                          # colliders / then integrate
```

The two halves are separate because a group has to run every member's constraints before any member integrates - see D2a.

The group driver then interleaves at substep granularity:

```python
for _ in range(substeps):
    for body in members:
        body.substep(h, others=[m for m in members if m is not body])
```

**A solo body is a group of one.** Every body routes through the driver, so there is one code path rather than a fast path and a coupled path that drift apart. The regression test that matters is that a group of one is bit-identical to the current solver.

Body-collision runs after self-collision and before external colliders, for the same reason self-collision does: a pin, a ground plane or a sticky grab gets the last word.

### D2. The cross-body kernel reads the other body's `tex_x`, not its `tex_p`

`tex_x` is always a complete, integrated state. `tex_p` is a work-in-progress whose meaning depends on how far through its own substep the other body happens to be, which would make the result depend on dictionary order in a way that is invisible from the outside.

The cost is one substep of lag: with 10 substeps at 24 fps, a body at 5 m/s has moved 0.02 m by the time it is seen, against a 0.1 m thickness at Resolution 0.1.

### D2a. Corrected during implementation: constraints for everyone, then integration for everyone

This section originally called the within-substep behaviour "Gauss-Seidel with respect to each other - asymmetric, convergent". The symmetry test proved that wrong, and it was a real defect rather than a bad assertion.

Only `integrate` writes `tex_x`. Running each body's whole substep in turn therefore means the body walked second reads the body walked first's *already integrated* state. Measured on two equal unit tets overlapping by half the thickness: the first body moved 0.050 and the second 0.025. The first sees the whole overlap and takes half of it, the second sees only the remainder and takes half of that - a persistent **two to one** split, decided by nothing but list order. Two identical blobs would visibly squash by different amounts, and swapping their names would swap which.

`substep()` is therefore split again, into `substep_constraints()` and `substep_integrate()`, and the driver runs:

```python
for _ in range(substeps):
    for body in members:
        body.substep_constraints(h, others)
    for body in members:
        body.substep_integrate(h)
```

Nobody has written `tex_x` when the contact passes run, so every body sees the same snapshot of every other and the split is exactly the mass ratio. This costs nothing: no copies, no extra images, and for a group of one the dispatch order is unchanged.

### D3. Two-way coupling falls out of the mass split

No new mechanism. Body A computes `w_A / (w_A + w_B)` and moves only its own nodes; body B computes `w_B / (w_A + w_B)` and moves only its own. The shares sum to one, so the pair separates by exactly the thickness, and a pinned body (`w = 0`) pushes without being pushed. The other body's inverse mass rides in `tex_x.w`, which is already there.

### D4. No rest-distance gate across bodies

Two bodies have no shared rest configuration, so there is nothing to compare against and every contact inside the thickness pushes. This is the one real behavioural difference from the self-collision kernel, and with the self-skip also gone the cross kernel is strictly the simpler of the two.

Rejected: folding both into one kernel behind a `same_body` flag. It saves roughly 30 lines and costs a mode that changes three separate behaviours - the gate, the self-skip, and which image the partner is sampled from.

### D5. One checkbox, no pairing list

**Collide With Bodies** per object. Every object with it enabled collides with every other one. No pairing UI: a list of "objects I collide with" implies a direction, and this relationship is mutual by construction.

Rejected: adding Marrow objects to the existing Colliders list. That list is for analytic primitives shaped by a transform, and its entries are strictly one-way.

### D6. Thickness is the existing slider

The same physical quantity - a contact gap in multiples of Resolution - so it drives self-collision and body collision both. The panel row enables when either toggle is on.

Two bodies can have different Resolutions and therefore different absolute thicknesses. The group takes the **largest** member's value and uses it for every pair, so both sides of a contact agree on the gap they are opening. Without this, A pushing to 0.1 while B pushes to 0.2 leaves B doing all the work.

### D7. Bake is group-wide

Baking one member of a group simulates and caches every member. A two-way bake of one body alone would be meaningless: the other body would be absent from its own contact.

### D8. A group shares the highest Substeps in it

Interleaving requires a common substep count. Taking the maximum is the safe direction - more substeps is stabler, never less. It does mean a body's own Substeps setting can be raised by the company it keeps, so Marrow prints this once when it happens rather than changing the physics silently.

## Components

### `marrow/gpu/kernels.py`: `BODY_COLLIDE_SRC`

Dispatched over this body's `n_nodes`. Thread `i` loads `p[i]`; if it is interior or pinned it writes `p[i]` through unchanged. Otherwise it loops the other body's surface list, reading positions and inverse masses from `x_other`, applies the thickness test and the mass-weighted split, and stores `p[i] + correction`.

Five images: `p`, `out_p`, `x_other`, `surf_other`, `surf_idx`. Push constants: `thickness`, `n_nodes`, `n_surf_other`.

Jacobi and ping-ponged exactly as the self-collide pass is, so every thread writes one texel and the caller swaps `tex_p` with `tex_p2`.

### `marrow/gpu/solver.py`

- `step(others=())` becomes a loop over the extracted `substep(h, others=())`.
- `_dispatch_body_collision(node_groups, other)`, one dispatch per other body.
- Surface textures (`tex_surf`, `tex_surf_idx`, `tex_p2`) are built when **either** self-collision or body collision is on. `tex_rest_pos` stays specific to self-collision, which is the only thing that reads it.
- New `body_distance` attribute, absolute world units, mirroring `self_distance`.

### `marrow/blender/group.py`

New. Owns the restart and catch-up rules that live in `MarrowSession.ensure_frame` today, applied once for a whole group instead of once per session:

- `groups_for(sessions)` - partitions into one multi-body group of everything with body collision on, plus a group of one for each remaining session.
- `advance(members, frame, frame_start)` - decides which frames to simulate, then for each frame drives `substeps` rounds of `substep()` across all members before asking each to skin and cache.

`MarrowSession._step_and_cache` splits in two: the group drives the solver, the session keeps `cache_frame()` - skin, the non-finite check, and the cache write.

### `marrow/blender/ui.py`, `session.py`, `ops.py`

`body_collision: BoolProperty`, default off. `session.body_distance` computed the same way `self_distance` is, from `self_thickness * resolution`. `MARROW_OT_bake` builds sessions for every group member and bakes the group.

## Testing

GPU, in the existing Blender-run suite:

1. Two blocks driven into each other: **both** move. Neither is left undeformed.
2. Equal bodies get equal and opposite corrections, and the pair ends the thickness apart.
3. A pinned body pushes the other and does not move itself.
4. Interior and pinned nodes come out of the cross-body dispatch unchanged - the ping-pong write-through, again, for the new kernel.
5. Body collision off is bit-identical to a run without it.
6. `substep()` called `n` times equals `step()`, bit for bit.
7. **A group of one is bit-identical to the pre-refactor solver.** The most important test here: every solo body now routes through the new driver.
8. Two bodies far apart never touch, and cost nothing beyond the dispatch.

Session and bake level:

9. Baking one member of a group caches frames for every member.
10. A group takes the highest Substeps among its members.

## Ceilings

- **No broad phase across bodies either.** Cost is surface_A x surface_B per pair, and N bodies is N^2 pairs. Two or three bodies at Resolution 0.1 is comfortable; a crowd is not.
- **Bodies authored already overlapping are shoved apart hard on the first frame.** The same failure as starting inside a sticky collider, and undiagnosable from the viewport. No depenetration pass; the ground plane remains the only thing that fixes its own starting state.
- **One substep of lag** reading `tex_x`, per D2.
- **Node-node contact only**, inherited from self-collision. A body thinner than the other's node spacing can slip through.
- **A group shares its highest Substeps**, per D8.
- **No friction between bodies.** Contact is pure separation, so bodies slide against each other freely.
