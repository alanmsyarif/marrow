# Marrow: Mesh Colliders

Design spec, 2026-08-13. Lifts the v1 non-goal "arbitrary mesh colliders. v1 ships ground plane and pinned vertex group only."

## Summary

A collider can be any mesh. Its shape is baked once into a signed distance field held in a 3D texture, sampled in the collide kernel, and animated for free by the object transform that already drives the analytic primitives.

## Motivation

Colliders today are a ground plane, a unit sphere and a unit box, each shaped entirely by the object's transform. Anything that is not one of those three has to be approximated by one, which rules out a hand, a floor with a lip, a bowl, or any modelled prop. Houdini Vellum collides against arbitrary geometry, and this is the most common thing a user will try first.

## Enabling measurement

Measured 2026-08-13, Blender 5.2.0 LTS, RTX 5050.

### 3D textures work

The whole design depends on holding a volume on the GPU. Blender's Python API was an open question, having already refused integer images and storage buffers.

| Probe | Result |
|---|---|
| `GPUTexture((16,16,16), format="R32F")` | OK |
| `info.image(0, "R32F", "FLOAT_3D", ...)` | compiles |
| `FLOAT_2D_ARRAY` image slot | also compiles |
| upload with data, sample in a kernel, read back | **bit-exact** |

No slice-packing into a 2D atlas is needed. This is the one measurement that decides the approach.

### Baking is cheap, and the robust sign is the cheap one

`BVHTree.find_nearest` over a voxel grid, and the two candidate ways to sign the result:

| Mesh | Grid | `find_nearest` | ray parity | normal-sign disagreement |
|---|---|---|---|---|
| Suzanne | 24^3 = 13,824 | 0.02 s | 0.01 s | 361 voxels, 2.61% |
| Suzanne | 32^3 = 32,768 | 0.05 s | 0.03 s | 943 voxels, 2.88% |
| Torus | 24^3 | 0.05 s | 0.02 s | 4 voxels, 0.03% |
| Torus | 32^3 | 0.05 s | 0.03 s | 8 voxels, 0.02% |

Ray parity costs 0.6x what `find_nearest` does, so the robust sign is also the cheaper one and there is no tradeoff to make. The nearest-face-normal shortcut, which most SDF bakers use, is wrong on nearly 3% of Suzanne's voxels - the pseudonormal problem at edges and creases.

Total bake for a 32^3 grid is about 0.08 s, and about 0.65 s at 64^3.

## Design decisions

### D1. A signed distance field in a 3D R32F texture

Distance is negative inside. Depenetration walks the gradient to the zero isosurface, which handles concavity, holes and thin features the same way it handles a convex blob - none of which a primitive or a nearest-vertex cloud can do.

Rejected: treating the collider's vertices as a point cloud and reusing `BODY_COLLIDE_SRC`. It would have been nearly free, but a point cloud has no inside, so a node deeper than the thickness is not pushed anywhere and passes straight through, and a coarse mesh leaks between its own vertices.

### D2. Baked in local space, so animation is free

The SDF covers the collider's local-space bounding box, padded. The object transform then places, rotates and scales it, exactly as it does the unit sphere and the unit box. **A collider that moves needs no rebake** - the existing `to_local` and `to_world` push constants already do that work, including for sticky contacts, whose anchor is recorded in local space precisely so it rides the transform.

### D3. The bounds mapping is folded into `to_local`, so no push constant is added

`COLLIDE_SRC` already exceeds its push-constant budget - the driver warns that "the constants added so far already reach 176 bytes" against a supported 128. Passing grid bounds and dimensions would make that worse.

Instead the CPU composes the padded-bbox-to-unit-cube mapping into the matrices it already sends:

```
to_local = grid_from_local @ world.inverted()
to_world = world @ grid_from_local.inverted()
```

The kernel then sees a collider that is "an SDF filling `[0,1]^3`", the same shape of contract as "a unit sphere". Zero new push constants.

A consequence worth stating plainly: the SDF holds distance in **grid** units, not world units. That is not an approximation. Walking `-d * normalize(grad)` lands on the zero isosurface in grid space, and the surface in grid space is the surface in world space under any invertible transform. Non-uniform scale changes which direction is shortest, but not where the surface is.

### D4. Sign by ray parity, magnitude by nearest point

Per the measurement above. `_is_inside` in `marrow/blender/inside_bvh.py` already does parity and is already trusted for deciding cage occupancy, so this reuses it rather than adding a second, weaker inside test to the codebase.

### D5. Resolution is derived, not exposed

Grid cell size is the body's **Resolution**, clamped to 16..96 cells per axis. The SDF only has to resolve detail the cage can represent, so a finer grid than the cage is wasted, and a coarser one is what the user asked for when they raised Resolution. One less setting to explain, and it auto-tracks.

96^3 of R32F is 3.5 MB, which bounds the worst case.

### D6. Bakes are cached

Keyed by the mesh datablock name, its vertex count and the grid dimensions. A live restart rebuilds solvers on every return to the start frame, and re-paying 0.08 s per collider each time is avoidable.

The vertex count in the key is a cheap guard against the obvious edit. It is not a full change detection - see the ceilings.

### D7. Mesh is the default shape for new collider slots

It works for any object, which is the request. Sphere and Box stay, as cheaper and exactly-correct options for when the shape genuinely is one. Existing slots keep whatever they were saved with.

## Components

### `marrow/blender/sdf.py`

New. `bake(obj, cell_size)` returns `(field, grid_from_local)`:

- Build a BVH of the evaluated object in **local** space.
- Pad the local bounding box by two cells, so the zero isosurface is never clipped by the grid edge and the gradient has room at the boundary.
- Grid dimensions from the padded box and the cell size, clamped per axis to 16..96.
- For every voxel centre: magnitude from `BVHTree.find_nearest`, sign from `inside_bvh._is_inside`.
- `grid_from_local` maps the padded box onto `[0,1]^3`.

Cached in a module dict per D6.

### `marrow/gpu/kernels.py`

`COLLIDE_SRC` gains `kind == 3`:

- The node is already in grid space, since D3 folded the mapping into `to_local`.
- Outside `[0,1]^3`, there is no contact.
- Trilinear sample the field. If the distance is not negative, there is no contact.
- Gradient by central differences, one texel apart.
- Push to the surface: `lp -= d * normalize(grad)`, then the existing `to_world` line returns it to world space.

One new image, `sdf`, `R32F`, `FLOAT_3D`, READ. Total three images on this shader, against a measured limit of eight.

### `marrow/gpu/solver.py`

Collider specs carry an optional field array. `_dispatch_colliders` uploads each one once at construction and binds it per dispatch. Non-mesh colliders bind a 1x1x1 dummy, because a declared image must be bound.

### `marrow/blender/session.py`, `ops.py`, `ui.py`

`shape` enum gains `MESH`, and it becomes the default. `_collider_specs` bakes for mesh colliders and composes the matrices per D3.

## Testing

1. A baked SDF of a sphere mesh matches the analytic sphere distance to within a voxel.
2. A node inside a mesh collider is pushed out to its surface.
3. A node outside one is left alone.
4. **A node in the hole of a torus is not pushed.** The test this design exists for: a bounding box, a convex hull or a vertex cloud all fail it, and only a real SDF passes.
5. A mesh collider built from a sphere and an analytic sphere collider agree, within the grid tolerance.
6. A mesh collider that is moved between steps pushes from its new position, with no rebake.
7. Sticky works against a mesh collider, and the anchor rides the transform.
8. A second solver build against the same collider reuses the cached bake.
9. Concave contact: a node inside a bowl's cavity is pushed to the cavity wall, not out through the bottom.

## Ceilings

- **A deforming collider is baked once.** Shape keys, an armature or Geometry Nodes on the collider are captured at build time and never revisited. Transform animation is free; deformation is not. The cache key notices a vertex count change, nothing subtler.
- **Features thinner than a voxel are missed**, since the field is sampled at cell centres. Lower Resolution, which raises the grid.
- **Non-uniform scale changes the push direction**, though not the surface it lands on, per D3.
- **The starting-state problem is unchanged.** A body authored inside a mesh collider is depenetrated on frame one with the same violence as the sticky-collider case already documented in the README. Only the ground plane fixes its own starting state.
- Memory is 3.5 MB per collider at the 96^3 cap, held for the life of the session.
