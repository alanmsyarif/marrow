# Marrow

GPU tetrahedral soft body for Blender 5.2.

Marrow fills a mesh with a tetrahedral cage, solves a stable neo-Hookean XPBD system in GLSL compute shaders, and deforms the original render mesh by barycentric interpolation from that cage. Pure Python against Blender's bundled interpreter, with no compiled extension, no external dependency and no per-platform build.

Blender 5.2 ships XPBD for hair, cloth and particles. There is no volumetric soft body; the legacy `SOFT_BODY` modifier is a surface spring lattice with no volume preservation. Marrow fills that gap.

## Install

```
blender --command extension install-file -r user_default --enable dist/marrow-0.6.1.zip
```

Or in Blender: **Edit > Preferences > Get Extensions > Install from Disk**.

To build the package from source:

```
blender --command extension build --source-dir marrow --output-dir dist
```

## Quick start

1. Select a mesh. Open the **Marrow** tab in the 3D viewport sidebar (`N`).
2. Set **Resolution** and press **Tetrahedralize**. A hidden wire cage is created and parented to your object.
3. Optionally add something to collide against: under **Simulation > Colliders**, press **+** and pick an object. Toggle the magnet on that slot to make it [sticky](#sticky-colliders), so material it touches is dragged along with it rather than only pushed away.
4. Press play.

That is the whole loop. **Live is on by default**: each frame is simulated as the timeline reaches it and cached, so scrubbing back is free and a second pass costs nothing.

To change how it behaves, edit any slider and **return to the start frame**. That restarts the simulation with the new settings, with no freeing and no rebaking. Measured on an RTX 5050, a live frame costs 1 to 2 ms from a 125-node cage up to a 2,744-node one, so the edit loop stays interactive.

### Getting your object back

**De-tetrahedralize**, in the Cage box, restores the shape you modelled, deletes the cage, strips every attribute Marrow wrote and releases the session. The mesh comes back bit-identical.

It works because Tetrahedralize records the rest shape in a `marrow_rest` attribute first. That also means re-tetrahedralizing after playing the timeline rebuilds from the original shape rather than from the deformed pose, so changing Resolution mid-project is safe.

Note that **Free Bake is not this**. Free Bake discards the cache and the GPU memory and leaves the object tetrahedralised and deformed, ready to simulate again.

Editing the mesh's topology between the two invalidates the stored shape, since the attribute is per-point. Re-tetrahedralize after any Edit Mode change.

### Live or Bake

| | |
|---|---|
| **Live** (default) | Simulates as you play, caching as it goes. Returning to the start frame restarts and re-reads the sliders. |
| **Bake** | Simulates the whole scene range up front. The result is then fixed: replaying it, including from the start frame, never re-simulates, so it survives slider changes until you free it. |
| **Free** | Discards the cache and releases GPU memory. |

Turning **Live** off stops simulation entirely for that object.

A skip of up to 8 frames is caught up, so playback that drops frames does not stall. A larger jump plays whatever is cached and otherwise leaves the mesh where it is, because chasing hundreds of frames inside a frame handler would lock the UI.

## Settings

| Setting | What it does |
|---|---|
| **Resolution** | Cage cell size in world units. Smaller fills finer detail and costs more. |
| **Substeps** | XPBD substeps per frame. More is stabler and slower, and this is the most expensive knob here. |
| **Stiffness** | Resistance to distortion (deviatoric compliance). |
| **Volume Preservation** | Resistance to volume change (hydrostatic compliance). |
| **Damping** | Velocity retained each substep. 1.0 is undamped. |
| **Ground Plane** / **Ground Height** | An infinite horizontal plane the body cannot fall through. |
| **Tearing** / **Tear Strain** | Largest stretch a tet survives, in any direction. 1.5 fails at 1.5x rest length. |
| **Self Collision** | Stop the body passing through itself where it folds. |
| **Collide With Bodies** | Collide with other Marrow objects that also have it on. Both deform. |
| **Thickness** | Contact gap for both of the above, as a multiple of Resolution. |
| **Colliders** | The list of objects this body collides against, and whether each one is sticky. |
| **Stick Break** | How far material may drag a sticky contact before it lets go. 0 never lets go. |

### Colliders

Each slot picks an object and how to treat it. **Mesh** is the default and uses the object's actual shape, so anything works — a bowl, a hand, a floor with a lip. **Sphere** and **Box** are cheaper and exactly round or square, for when the shape really is one.

A mesh collider is baked once into a signed distance field, in the object's own local space. That means moving, rotating and scaling it costs nothing — the field rides the transform exactly as the primitives do. It also means concavity works properly: a node in the hole of a torus is correctly outside the solid, which a bounding box or a convex hull would get wrong.

The field's grid tracks your **Resolution**, since it only needs to resolve detail the cage can represent. There is no separate setting.

**A deforming collider is not re-baked.** Shape keys, an armature or Geometry Nodes on the collider are captured once when the simulation starts and never revisited. Transform animation is free; deformation is not.


Colliders belong to the body being simulated, so you set them up without leaving it. In **Simulation > Colliders**, press **+**, pick an object in the slot, and choose **Sphere** or **Box**. Press **-** to remove the selected slot.

The shape is a unit primitive driven entirely by the picked object's transform, so a default Blender sphere or cube maps exactly, and position, rotation and scale all animate. An **Empty works just as well as a mesh**, and is often tidier: a primitive collider needs a transform and nothing else.

Collider transforms are re-sampled every frame in both live and baked modes, so a falling ball genuinely lands on a jelly rather than sitting still.

An empty slot, or a body pointed at itself, is skipped rather than treated as an error. Both are just a half-finished edit.

#### Sticky colliders

The magnet toggle on a collider slot makes it **sticky**. Material that touches it is held to the surface and dragged along as the collider moves, instead of only being pushed out of it.

Plain non-penetration can only push, so a collider that lifts away leaves the body behind. Sticky is what makes squash-and-stretch possible: press a plate into a blob, lift it, and the material is drawn up into a column. The contact point is recorded in the collider's own local space, so the anchor rides the animated transform for free and rotation and scale come along with it.

**Stick Break** is how far the material may drag a contact point before it lets go, in world units. Zero never lets go. There is no universal value: the distance a contact settles at depends on Stiffness and Substeps, so tune it against the shot.

> **Do not start a body already overlapping a sticky collider.** Every buried node is grabbed on the first frame and welded to whichever face happened to be nearest. They scatter across different faces and turn the body inside out. Measured on a sphere half-buried in a sticky box: 219 of 461 nodes seized immediately, 12% of tets inverted, render mesh shredded. Move the collider clear at the start frame, or leave Sticky off. A collider that presses in *during* the simulation is fine, and is the intended way to set a stretch shot up.

### Tearing

Tearing is **constraint failure**, not fracture. A tetrahedron past the strain threshold stops resisting distortion, permanently, so the material necks, stretches and pulls apart. It does **not** split into separate pieces with a visible gap: the render mesh is never modified, which is what keeps your UVs, shape keys and material slots intact.

**Tear Strain is the largest principal stretch a tet survives**, so it reads the same in every direction: 1.5 means failure once anything is pulled to 1.5x its rest length, whether that is a pull along one axis, a uniform swell or a shear. Rotation is not strain and never tears.

One consequence worth knowing: a volume-preserving squash stretches the material sideways, and that counts. Press a blob to a quarter of its height and it has stretched 2x laterally, which fails a 1.5 threshold. If a heavy press is tearing material you wanted intact, raise Tear Strain or switch Tearing off for that shot.

Two things a torn tet still does:

- **It keeps a volume constraint**, targeting the volume it had at the instant it broke. Broken material is not new material. Without this the cage inflated without bound, measured at 3.1x on a stretch test.
- **It cannot be the last tet holding a node.** A node whose every tet has torn has no constraint at all: it free-falls, and because the render topology is fixed it drags a spike behind it instead of becoming debris. The tear is refused instead. On a stretch test this took 324 orphaned nodes to zero while still tearing four fifths of the cage.

### Self collision

Off by default, because it is the second most expensive knob after Substeps. Measured on an RTX 5050, cost per frame at 10 substeps:

| Resolution | cage nodes | surface nodes | off | on |
|---|---|---|---|---|
| 0.25 | 461 | 314 | 1.9 ms | 3.0 ms |
| 0.15 | 1,707 | 848 | 2.0 ms | 4.9 ms |
| 0.10 | 5,233 | 1,898 | 2.1 ms | 8.4 ms |
| 0.07 | 14,226 | 3,866 | 2.5 ms | 15.2 ms |

Only nodes on the surface of the cage take part, so the cost grows with the square of an area rather than of a volume. Every surface node is tested against every other one; there is no spatial hash, because integer images and `imageAtomicAdd` do not work in Blender's Python GPU API and a CPU-side hash would mean reading the whole cage back every frame.

**Thickness** is the contact gap, as a multiple of Resolution. 1.0 keeps the skinned render surface from visibly interpenetrating. Below about 1.0 a fold can slip between cage nodes, since contact is node against node and nothing checks the space between them.

Parts of the body that are already close together in the **rest** pose are left alone. This is what stops the cage fighting its own tets, whose nodes sit exactly one Resolution apart, but it also means a shape authored with two surfaces already touching will let those two surfaces keep passing through each other.

Fast motion can still tunnel: a node moving more than the thickness in one substep can cross to the far side before it is ever tested. Raise Substeps or Thickness if you see it.

### Colliding two bodies

Turn **Collide With Bodies** on for each object you want in the pile. There is no pairing list: everything with it on collides with everything else that has it on, and both sides deform. The correction splits by mass, so a pinned body pushes without being pushed.

Because they push each other, they have to be **simulated together**. Three consequences worth knowing before you use it:

- **Baking one bakes all of them.** A two-way bake of a single body would leave the other absent from its own contact.
- **The group runs at its highest Substeps.** A body's own setting can be raised by the company it keeps. Marrow says so on the console when it happens.
- **Adding a body mid-playback restarts the group.** There is no sound way to splice a body into a simulation already in progress.

Contact uses the same **Thickness** as self-collision, and the group takes the largest value among its members so both sides of a contact agree on the gap. Unlike self-collision there is no rest-pose exemption: two bodies share no rest state, so every contact inside the thickness is a real one.

Cost is surface nodes of one body times surface nodes of the other, per pair, with N bodies making N² pairs. Two or three bodies at Resolution 0.1 is comfortable. A crowd is not.

### Ground plane

A cage that starts below the ground plane is lifted onto it, rigidly, before the first frame, and Marrow says so on the console.

This is not cosmetic. Collision resolves penetration by moving the predicted position, and the integrator reads that move as velocity of depth divided by the substep length. Mid-simulation that is harmless, because a substep can only sink a node so far. The starting state has no such bound: a unit ball authored straddling the plane left its first substep at 226 m/s, which is past any tear threshold and shreds the body. Lifting rigidly rather than clamping each node matters too, since clamping flattens the buried half and the stored energy launches it nearly as hard.

## How it works

| Stage | Where |
|---|---|
| Tetrahedralize | CPU, numpy, once |
| Tet data | Mesh vertices, ID properties and POINT attributes, surviving save and load |
| Rest shape | A `marrow_rest` POINT attribute, so De-tetrahedralize can undo the whole thing |
| Pack to textures | CPU to `GPUTexture`, once per simulation start |
| Solve | GLSL compute, 6 kernels x substeps x constraint colours |
| Skin and readback | GPU blend, then only the render vertices cross PCIe |

The cage uses a **cube-split lattice with checkerboard parity**, not conforming Delaunay. Boundary recovery is fragile on real meshes and replaces the render mesh; this approach never touches your topology and tolerates messy input, because the only question asked of the mesh is inside or outside. The accepted cost is that the cage does not hug concave detail, so thin models need a finer Resolution.

Tets are graph-coloured at build time so each colour dispatches race-free with no atomics. Interior cage nodes never cross PCIe; only render vertices are read back.

## Limitations

- **Contact is node against node only.** No edge-edge or node-triangle contact, for either self-collision or body-to-body, and neither scales past roughly 20,000 surface nodes. Collision against a *collider* is a distance field and does not have this limit.
- **A deforming collider is baked once**, and features thinner than one SDF cell are missed. See [Colliders](#colliders).
- **No friction anywhere.** Contact only ever separates, so bodies slide against each other and against themselves freely.
- **No pinning yet.** The solver supports it (zero inverse mass) but nothing exposes it. A sticky collider is the only way to hold material in place today.
- **A body must not start inside a sticky collider.** See [Sticky colliders](#sticky-colliders). Only the ground plane depenetrates its starting state.
- **Resolution changes the physics, not just the detail.** Every cage node carries the same mass regardless of cell size, so a finer cage makes the same object heavier while Stiffness stays put, and it sags further. Going from 0.25 to 0.1 on a unit sphere takes it from 461 to 5,104 mass units. Expect to re-tune Stiffness and Volume Preservation after a Resolution change rather than treating them as absolute.
- **The cache lives in memory, not in the .blend.** Reopening a file means playing again from the start; live rebuilds the cache as you go.
- **No plasticity, anisotropy or per-region materials.**
- Measured on the OpenGL backend. Blender is moving to Vulkan, and the kernels need revalidating there.

### A note on GPU reliability

Blender's Python GPU API exposes no memory barrier; `gpu.compute` offers only `dispatch`. Marrow works around this. Readbacks are verified against a generation mark stamped by the writing kernel, uploads are read back and confirmed, and the bind texture is re-checked before every dispatch because it has been observed losing its contents after a verified upload. When a repair happens Marrow says so on the console rather than staying quiet. If you see corrupt frames, please report them along with that console output.

## Development

Core geometry and solver maths live in `marrow/core/` and never import `bpy`, which is what makes them testable outside Blender.

```
# Core suite (pytest, needs a venv on system Python 3.12 with numpy + pytest)
.venv/Scripts/python -m pytest tests/core

# Blender suite (Blender's bundled Python has no pytest, so this is an
# assert-based runner; it exits non-zero on failure)
blender -b --factory-startup --python tests/blender/run_tests.py
```

**Run the Blender suite on 5.2, and check which binary you invoked.** Background mode only has a GPU context from 5.2 on. Point this at 4.5 and every GPU test fails with `GPU functions for drawing are not available in background mode`, and a windowed 4.5 driven by `--python` at startup fails each readback with `StaleReadError: a RGBA32F upload never became visible`. Neither says anything about the code, and on a machine with several Blender versions installed it is an easy hour to lose. The suite is 168 tests and they all pass on 5.2.

Running a single module rather than `run_tests.py` needs a `gpu.init()` of your own first: 5.2 requires it, and several modules rely on some earlier module in the full run having already called it.

A numpy reference solver in `marrow/core/solver_ref.py` is the test oracle: every GPU kernel is diffed against it, because a wrong sign in a compute shader is otherwise indistinguishable from a wrong sign in the constraint derivation.

The addon must use relative imports only. Installed as an extension its package is `bl_ext.user_default.marrow`, not `marrow`, so an absolute self-import fails at register time on a user's machine while every test still passes. `tests/core/test_packaging.py` guards this.

Never `pip install` into Blender's bundled Python.

## Licence

GPL-3.0-or-later.
