# Marrow

GPU tetrahedral soft body for Blender 5.2.

Marrow fills a mesh with a tetrahedral cage, solves a stable neo-Hookean XPBD system in GLSL compute shaders, and deforms the original render mesh by barycentric interpolation from that cage. Pure Python against Blender's bundled interpreter, with no compiled extension, no external dependency and no per-platform build.

Blender 5.2 ships XPBD for hair, cloth and particles. There is no volumetric soft body; the legacy `SOFT_BODY` modifier is a surface spring lattice with no volume preservation. Marrow fills that gap.

## Install

```
blender --command extension install-file -r user_default --enable dist/marrow-0.1.0.zip
```

Or in Blender: **Edit > Preferences > Get Extensions > Install from Disk**.

To build the package from source:

```
blender --command extension build --source-dir marrow --output-dir dist
```

## Quick start

1. Select a mesh. Open the **Marrow** tab in the 3D viewport sidebar (`N`).
2. Set **Resolution** and press **Tetrahedralize**. A hidden wire cage is created and parented to your object.
3. Optionally add something to collide against: under **Simulation > Colliders**, press **+** and pick an object.
4. Press play.

That is the whole loop. **Live is on by default**: each frame is simulated as the timeline reaches it and cached, so scrubbing back is free and a second pass costs nothing.

To change how it behaves, edit any slider and **return to the start frame**. That restarts the simulation with the new settings, with no freeing and no rebaking. Measured on an RTX 5050, a live frame costs 1 to 2 ms from a 125-node cage up to a 2,744-node one, so the edit loop stays interactive.

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
| **Tearing** / **Tear Strain** | Material past this stretch ratio fails permanently. 1.5 tears at 50% strain. |
| **Colliders** | The list of objects this body collides against. |

### Colliders

Colliders belong to the body being simulated, so you set them up without leaving it. In **Simulation > Colliders**, press **+**, pick an object in the slot, and choose **Sphere** or **Box**. Press **-** to remove the selected slot.

The shape is a unit primitive driven entirely by the picked object's transform, so a default Blender sphere or cube maps exactly, and position, rotation and scale all animate. An **Empty works just as well as a mesh**, and is often tidier: a primitive collider needs a transform and nothing else.

Collider transforms are re-sampled every frame in both live and baked modes, so a falling ball genuinely lands on a jelly rather than sitting still.

An empty slot, or a body pointed at itself, is skipped rather than treated as an error. Both are just a half-finished edit.

### Tearing

Tearing is **constraint failure**, not fracture. A tetrahedron past the strain threshold stops resisting, permanently, so the material necks, stretches and pulls apart. It does **not** split into separate pieces with a visible gap: the render mesh is never modified, which is what keeps your UVs, shape keys and material slots intact.

## How it works

| Stage | Where |
|---|---|
| Tetrahedralize | CPU, numpy, once |
| Tet data | Mesh vertices, ID properties and POINT attributes, surviving save and load |
| Pack to textures | CPU to `GPUTexture`, once per simulation start |
| Solve | GLSL compute, 4 kernels x substeps x constraint colours |
| Skin and readback | GPU blend, then only the render vertices cross PCIe |

The cage uses a **cube-split lattice with checkerboard parity**, not conforming Delaunay. Boundary recovery is fragile on real meshes and replaces the render mesh; this approach never touches your topology and tolerates messy input, because the only question asked of the mesh is inside or outside. The accepted cost is that the cage does not hug concave detail, so thin models need a finer Resolution.

Tets are graph-coloured at build time so each colour dispatches race-free with no atomics. Interior cage nodes never cross PCIe; only render vertices are read back.

## Limitations

- **No self-collision or body-to-body collision.** Colliders are sphere, box and ground plane only.
- **No pinning yet.** The solver supports it (zero inverse mass) but nothing exposes it, so every body falls freely.
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

A numpy reference solver in `marrow/core/solver_ref.py` is the test oracle: every GPU kernel is diffed against it, because a wrong sign in a compute shader is otherwise indistinguishable from a wrong sign in the constraint derivation.

The addon must use relative imports only. Installed as an extension its package is `bl_ext.user_default.marrow`, not `marrow`, so an absolute self-import fails at register time on a user's machine while every test still passes. `tests/core/test_packaging.py` guards this.

Never `pip install` into Blender's bundled Python.

## Licence

GPL-3.0-or-later.
