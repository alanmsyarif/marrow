# Marrow

GPU tetrahedral soft body for Blender 5.2.

Marrow fills a mesh with a tetrahedral cage, solves a stable neo-Hookean XPBD system in GLSL compute shaders, and deforms the original render mesh by barycentric interpolation from that cage. Pure Python against Blender's bundled interpreter, with no compiled extension, no external dependency and no per-platform build.

Blender 5.2 ships XPBD for hair, cloth and particles. There is no volumetric soft body; the legacy `SOFT_BODY` modifier is a surface spring lattice with no volume preservation. Marrow fills that gap.

## Install

```
blender --command extension install-file -r user_default --enable dist/marrow-1.5.0.zip
```

Or in Blender: **Edit > Preferences > Get Extensions > Install from Disk**.

To build the package from source:

```
blender --command extension build --source-dir marrow --output-dir dist
```

## Quick start

1. Select a mesh. Open the **Marrow** tab in the 3D viewport sidebar (`N`).
2. Set **Resolution** and press **Tetrahedralize**. A hidden wire cage is created and parented to your object.
3. Optionally add something to collide against: select it, then the soft body, and under **Simulation > Colliders** press **+**. Toggle the magnet on its row to make it [sticky](#sticky-colliders), so material it touches is dragged along with it rather than only pushed away.
4. Press play.

That is the whole loop. **Live is on by default**: each frame is simulated as the timeline reaches it and cached, so scrubbing back is free and a second pass costs nothing.

To change how it behaves, edit any slider and **return to the start frame**. That restarts the simulation with the new settings, with no freeing and no rebaking. Measured on an RTX 5050, a live frame costs 1 to 2 ms from a 125-node cage up to a 2,744-node one, so the edit loop stays interactive.

### Getting your object back

**De-tetrahedralize**, in the Cage box, restores the shape you modelled, deletes the cage, strips every attribute Marrow wrote and releases the session. The mesh comes back bit-identical.

It works because Tetrahedralize records the rest shape in a `marrow_rest` attribute first. That also means re-tetrahedralizing after playing the timeline rebuilds from the original shape rather than from the deformed pose, so changing Resolution mid-project is safe.

Note that **Free Bake is not this**. Free Bake discards the cache and the GPU memory and leaves the object tetrahedralised and deformed, ready to simulate again.

Editing the mesh's topology between the two invalidates the stored shape, since the attribute is per-point. Re-tetrahedralize after any Edit Mode change.

### Adaptive cages

Uniform cages resolve the thinnest feature they must capture and pay for it everywhere: a walking character whose ankles need 0.03 cells pays 0.03 through the whole torso, roughly (0.25/0.03)³ ≈ 580x the nodes. **Adaptive**, in the Cage box, replaces the uniform grid with an octree that refines towards the surface - **Resolution** stays the coarse bulk cell size, **Min Size** is how small cells get. One rule gives three behaviours: a boundary layer at Min Size everywhere on the surface, thin features filled at Min Size so they keep at least two cells across, and deep interior left at Resolution. Where fine cells meet coarser ones, the extra face nodes are glued to the coarse face by bilinear interpolation, so the cage bends as one piece.

The trade is the boundary layer: every surface cell sits at Min Size, so an adaptive cage is only cheap where the shape has genuine bulk to leave coarse, and building one takes longer than the uniform fill it replaces. On a chunky cantilever test shape the adaptive cage deflects like the uniform-at-Min-Size one to within 15% at over 4x fewer nodes. Adaptive and uniform cages behave identically once built - same solver, same settings - and Adaptive off stays bit-identical to the old uniform path.

### Live or Bake

| | |
|---|---|
| **Live** (default) | Simulates as you play, caching as it goes. Returning to the start frame restarts and re-reads the sliders. Scrubbing *below* the start frame resets the body to rest. |
| **Bake** | Simulates the whole scene range up front. The result is then fixed: replaying it, including from the start frame or below it, never re-simulates, so it survives slider changes until you free it. |
| **Free** | Discards the cache and releases GPU memory. |

Moving the playhead **before the start frame** resets a live body: the cache is dropped, the mesh snaps back to its rest shape and False Color repaints at rest. Paused or playing makes no difference. A baked body is exempt and keeps replaying its cache.

Turning **Live** off stops simulation entirely for that object.

A skip of up to 8 frames is caught up, so playback that drops frames does not stall. A larger jump plays whatever is cached and otherwise leaves the mesh where it is, because chasing hundreds of frames inside a frame handler would lock the UI. A body that has never simulated is the exception: with no history to protect, the first jump catches up from the start frame in one go, so pressing play mid-timeline just works.

## Settings

| Setting | What it does |
|---|---|
| **Resolution** | Cage cell size in world units. Smaller fills finer detail and costs more. |
| **Adaptive** | Follow the surface instead of filling uniformly. See [Adaptive cages](#adaptive-cages). |
| **Min Size** | Smallest adaptive cell; thin features fill at this size. Only active with Adaptive. |
| **Substeps** | XPBD substeps per frame. More is stabler and slower, and this is the most expensive knob here. |
| **Stiffness** | Resistance to distortion (deviatoric compliance). |
| **Volume Preservation** | Resistance to volume change (hydrostatic compliance). |
| **Damping** | Velocity retained each substep. 1.0 is undamped. |
| **Fiber** | Contract along baked fiber directions. See [Fiber](#fiber). |
| **Curve** | Curve running along the body. Its tangent is the fiber direction, its arclength the wave phase. |
| **Fiber Stiffness** | Resistance to stretch along the fiber, and how hard it pulls. |
| **Amplitude** | Peak contraction. 0.3 shortens to 70% of rest length. |
| **Wavelength** | Distance between crests, along the curve. Floored at 1e-4 - both the oracle and the kernel divide by it unguarded, so this is not a soft suggestion. |
| **Speed** | Cycles per second. Travel velocity is Wavelength x Speed. |
| **Waveform** | Smooth cosine, or hard on/off square. |
| **Friction** | Resistance to sliding, for the ground plane, self-collision and body-to-body contact. 0 slides freely. Colliders carry their own value instead. See [Friction](#friction). |
| **Ground Plane** / **Ground Height** | An infinite horizontal plane the body cannot fall through. |
| **Tearing** / **Tear Strain** | Largest stretch a tet survives, in any direction. 1.5 fails at 1.5x rest length. |
| **Self Collision** | Stop the body passing through itself where it folds. |
| **Collide With Bodies** | Collide with other Marrow objects that also have it on. Both deform. |
| **Thickness** | Contact gap for both of the above, as a multiple of Resolution. |
| **Attachment** | An armature or other deforming modifiers drive the sim from the inside. See [Attachment](#attachment). |
| **Attach Stiffness** | How hard the flesh follows the animation. 1.0 rides it exactly; lower lags, jiggles and overshoots. |
| **Pin Group** | Vertex group whose weight holds material in place. 1.0 pins solid; below that is a heavier node, not a partial hold. See [Pinning](#pinning). |
| **Follows Animation** | Let the pinned region ride the animation instead of staying put. Still rigid. Needs Attachment. See [Pinning](#pinning). |
| **Colliders** | The collection of objects this body collides against. Shape, Sticky and Friction are set on each object. |
| **Stick Break** | How far material may drag a sticky contact before it lets go. 0 never lets go. |
| **False Color** | Off / Stretch rainbow display of how far the material is stretched. See [False color](#false-color). |

**Mass is a property of the object, not of the cage.** Each node carries the volume of material it represents at a fixed density (64 mass units per cubic metre), so an object weighs the same however finely it is tetrahedralized, and Stiffness and Volume Preservation keep their meaning across a Resolution change. The density is chosen so the average node at the default Resolution of 0.25 weighs the 1 mass unit older versions gave every node. A very fine cage can still read a touch firmer at the same Substeps — a fixed iteration budget converges more constraints less — so raise Substeps if a resolution change needs to match to the last percent.

### Colliders

Each collider carries its own **Shape**. **Mesh** is the default and uses the object's actual shape, so anything works — a bowl, a hand, a floor with a lip. **Sphere** and **Box** are cheaper and exactly round or square, for when the shape really is one.

A mesh collider is baked once into a signed distance field, in the object's own local space. That means moving, rotating and scaling it costs nothing — the field rides the transform exactly as the primitives do. It also means concavity works properly: a node in the hole of a torus is correctly outside the solid, which a bounding box or a convex hull would get wrong.

The field's grid tracks your **Resolution**, since it only needs to resolve detail the cage can represent. There is no separate setting.

**A deforming collider is not re-baked.** Shape keys, an armature or Geometry Nodes on the collider are captured once when the simulation starts and never revisited. Transform animation is free; deformation is not.


Colliders come from a **collection**. The body points at one, and every object in it is a collider — nested collections included, so you can group them however the shot wants.

The quickest way in: select the objects you want to collide against, then the soft body last so it is active, and press **+** in **Simulation > Colliders**. Marrow makes a `<body> Colliders` collection if there is none yet, and links them in. **-** unlinks the highlighted row. You can also just point the field at a collection you already have, and drag objects in and out of it in the outliner.

Because the settings live on the collider and not on the body, an object dropped into two bodies' collections is described once. Change its Shape and both bodies follow.

The shape is a unit primitive driven entirely by the picked object's transform, so a default Blender sphere or cube maps exactly, and position, rotation and scale all animate. An **Empty works just as well as a mesh**, and is often tidier: a primitive collider needs a transform and nothing else.

Collider transforms are re-sampled every frame in both live and baked modes, so a falling ball genuinely lands on a jelly rather than sitting still.

An empty collection, or a body sitting in its own collider collection, is skipped rather than treated as an error. Both are just a half-finished edit.

### Friction

Contact resists sliding, not just interpenetration. **Friction** is one coefficient per contact surface: `0` slides freely, around `0.5` grips on a gentle slope, `1` and above holds almost anywhere it can get a grip.

There are two places to set it, and which one applies depends on what the material is touching. Each collider in the list carries its **own** Friction, set on the collider object beside its Shape and Sticky — so a slippery floor and a grippy hand can be in the same scene. Everything with no collider slot of its own — the ground plane, the body against itself, and contact with other Marrow bodies — reads the body's **Friction** slider instead.

One coefficient covers both static and sliding friction. Below the limit the entire sideways step is given back and the material simply holds; above it the contact slips, at a rate the coefficient sets. The limit scales with how hard the contact is pressed, which is what makes it read as weight: the same slider grips harder under a heavy landing than a light touch.

Friction is applied as a position correction inside the contact pass, from the correction that pass already computed — so it needs no separate normal, and no second pass. Both the ground plane and the collider primitives measure it against a collider held still for the substep. Self-collision and body-to-body measure the *pair*, so two parts of one body travelling together are not braked for merely touching.

`0` is the default everywhere and is bit-identical to having no friction at all, so nothing that was authored before this existed moves differently.

**Sticky is not friction.** Friction resists sliding along a surface; Sticky welds material to it and drags it wherever the collider goes. A sticky collider ignores Friction entirely, and the panel greys the value out to say so.

#### Sticky colliders

The magnet toggle on a collider makes it **sticky**. Material that touches it is held to the surface and dragged along as the collider moves, instead of only being pushed out of it.

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

**Tearing is what ends a sticky stretch.** With it off, material held by a sticky collider that pulls away necks without limit into one unbroken spike, because nothing in the solver can ever fail. Stick Break does not substitute: it measures how far a contact point *drags across the collider*, which barely moves in a straight pull, so a stretch that should snap keeps stretching. If a stretch shot smears instead of separating, switch Tearing on.

### Fiber

Every tet in Marrow is otherwise isotropic: it resists distortion equally in all directions. **Fiber** adds one direction that is different, and drives it - the tet is told to shorten along its fiber, and the volume-preservation constraint puts the material it displaces out sideways. That sideways bulge is what makes it read as a muscle instead of a shrinking rod.

Directions come from a **Curve** you point at the body. Each tet takes the tangent at the nearest point on that curve as its fiber direction, and the arclength at that point as its place in the wave, so the contraction travels from one end of the body to the other instead of the whole thing pulsing at once. Both are sampled at Tetrahedralize and frozen - a fiber direction is a property of the rest shape, so an animated curve has no meaning as a source. Change the curve and tetrahedralize again.

**Setting the curve takes two Tetrahedralize passes.** The Curve field lives below the "Tetrahedralize to simulate" gate, so it does not exist until the object already has a cage. The flow is Tetrahedralize, then set Curve, then Tetrahedralize again - the first pass only builds the cage the field needs to appear on, and the second is the one that samples the curve and bakes fibers.

**A bevelled or cyclic curve bakes no fibers, and says nothing about it.** The curve is evaluated to a polyline that must resolve to one open path: a bevel or extrude turns it into a tube, a cyclic curve into a ring, and neither has an unambiguous direction to hand a fiber. Either one bakes nothing, silently - the panel keeps reading "Tetrahedralize to bake fibers" exactly as if no curve had been set at all. If fibers refuse to bake, check the curve for a bevel or `Cyclic U` before looking anywhere else.

The wave itself is procedural, not keyframed. **Wavelength** and **Speed** set its shape and how fast it travels: the crest moves at Wavelength x Speed in world units per second, and a negative Speed sends it the other way. **Amplitude** is how hard it squeezes - 0.3 shortens a fiber to 70% of its rest length at the crest. **Waveform** picks a smooth cosine or a hard on/off square.

Contraction alone moves nothing. Locomotion is contraction plus grip, so a crawling body needs [Friction](#friction) above zero and something to push against - `tools/fiber_demo.py` builds a working scene to start from.

A torn tet loses its fiber along with its stiffness. Torn muscle does not pull.

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

### Attachment

**Attachment** makes an armature - or any deforming modifier - drive the simulation from the inside instead of bending the result afterwards. Every frame Marrow evaluates the object's modifier stack against the stored rest shape and pulls each cage node towards where the stack puts it, so the bones lead and the flesh follows with the full XPBD behaviour: inertia, jiggle, collision and tearing all stay in the loop.

**Attach Stiffness** is how hard the flesh follows. 1.0 rides the animation exactly; lower values let the bones lead and the flesh lag, jiggle and overshoot.

While Attachment is on, modifiers that keep the vertex count - Armature, Simple Deform and the like - are **muted in the display**, viewport and render. They feed the targets, and the written simulation is the display; leaving them shown would deform the result a second time - measured at exactly twice the bone travel. Modifiers that change the count - Subdivision, Decimate - are the opposite: their vertices have no per-base-vertex meaning, so they cannot feed the targets, and they stay shown to smooth the simulated shape. Toggling Attachment off or De-tetrahedralize hands the original visibility back. Toggling Attachment on or changing Attach Stiffness mid-simulation restarts the sim on the next frame instead of waiting for a trip to the start frame. The targets come from Blender's evaluated mesh, so linear and dual quaternion skinning both work without Marrow reimplementing either.

Contacts keep the last word: the attachment pull runs before collision, so a ground plane or collider still stops a body the bone drags into it.

Attachment needs the object to keep its vertex count: the weight table is per base vertex, which is why count-changing modifiers are kept out of the targets and left on the display only. Weights are synthesized once against the rest shape, in object space, so moving or rotating the object afterwards cannot scramble them; like the rest of Marrow the simulation itself stays in the world frame it was tetrahedralized in. Tetrahedralize fills the cage from the modelled shape whichever frame it runs on - the capture parks the modifier stack, so a pose playing at tet time cannot wedge the posed silhouette between the lattice and the bind. Editing the mesh needs a re-tetrahedralize, same rule as the bind data.

### Pinning

**Pin Group** holds material in place. Point it at a vertex group on the object and paint that group wherever the body must not move; weight 1.0 is a solid pin.

The group is painted on the render mesh, because that is the geometry you can see and select - cage nodes are interior lattice points and carry no vertex groups of their own. Each node reads the group through the same k-nearest-vertex map the attachment pass uses, so a node is held exactly when the surface around it is, and pinning works whether or not Attachment is on.

A pin is zero inverse mass, and every pass in the solver already defers to it. Predict skips the gravity step, integrate holds the position and the velocity both, and the collider and attachment kernels each check it first: a pin outranks a collider, and a pin outranks the armature. Nothing pushes a pinned node anywhere.

**A weight below 1.0 is not a partial hold.** It scales inverse mass, so 0.5 is a node with twice the mass, not a node half held - gravity is an acceleration, so a heavy node still falls at g. What the falloff buys is the boundary. The blend across the edge of a painted region makes the transition nodes progressively heavier instead of dropping a hard mass discontinuity into the middle of the material, which shows as a stress ring. Paint the region you want held at a solid 1.0 and let the blend feather the edge.

Pins anchor in the world frame the body was tetrahedralized in - the same rule as the bind and attachment data - so moving the object afterwards does not drag the pin along with it. That is the point on an animated character: a pinned foot stays nailed where it was while the rest of the body is free to be driven. Changing the Pin Group or Follows Animation mid-simulation takes effect on the next frame, the same as Attachment - all three change what the solver is built from. Repainting weights *inside* a group that is already selected does not: the setting itself has not changed, so nothing knows to rebuild. Return to the start frame in Live, or Free and bake again after a Bake.

**A pin holds; Attachment carries it.** By default a pin is a *static* anchor, so hooking a pinned region to an Empty and moving it does nothing - the pin is exactly what the solver holds still, and the attachment kernel skips a node with no inverse mass. Tick **Follows Animation** and the pin rides the animation instead: still rigid, still outranking every collider, but driven rather than frozen.

Measured on a cube whose top face is hooked to an Empty moved 3.0 in X, tracking how far the cage top and bottom travel:

| Setup | Top follows | Bottom |
|---|---|---|
| Pin Group only, no Attachment | -0.004 | 0.102 |
| Attachment 0.5, no pin | 2.600 | 0.929 |
| Pin + Attachment, Follows Animation **off** | 0.240 | 0.092 |
| Pin + Attachment, Follows Animation **on** | **2.874** | 0.815 |

Row 4 is the shape you usually want: the pinned region tracks the hook almost exactly while the rest of the body lags and jiggles behind it. Row 3 is the trap - a static pin and Attachment fight each other, and the pin wins, dragging the whole thing back.

**Set Attach Stiffness to 0 to let a driven pin carry the body.** This is the setting the mode is built for, and it is worth understanding why. Attachment aims *every* node at its evaluated position, and for material the animation does not reach, that aim is the rest pose - so the same pass that supplies a pin its target is otherwise nailing the rest of the body down. At 0 the pass runs for the pins alone and leaves the free material entirely to the simulation.

Measured on a 23,697-node cage, a 27-vertex painted region hooked to an Empty travelling 1.473:

| Attach Stiffness | Pin travel | Body travel |
|---|---|---|
| 0.50 | 1.473 | 0.026 |
| 0.05 | 1.473 | 0.040 |
| 0.01 | 1.473 | 0.374 |
| **0.00 (pins only)** | 1.473 | **1.552** |

The pin tracks perfectly at every setting. What changes is whether the body comes with it - and nothing about mass, gravity, ground contact or material Stiffness was ever the limit; releasing the attachment grip was the whole of it. A high Attach Stiffness also switches the physics off entirely: at 1.0 the pass hard-snaps every node to its evaluated position after the elastic solve, so Marrow just replays the modifier stack and Stiffness stops mattering - measured identical body travel to three decimals across a 400x change in it.

Use a stiffness above 0 only when you want the *whole* body to follow the animation with soft-body lag, which is what Attachment is for on its own. Mixing that with a driven pin fights itself.

**Follows Animation needs Attachment on**, because the attachment pass is where per-node targets come from; the checkbox greys out otherwise. Targets are sampled once per frame, the same treatment collider transforms get, so a driven pin reaches the frame's position on the first substep and holds. That is the same front-loading Attachment at stiffness 1.0 already applies to every node in the body.

Leave Follows Animation **off** to nail a region in world space while an armature drives the rest - a foot planted on the floor while the body moves over it. That is the case the flag exists to preserve.

One more reason to turn Attachment on here: modifiers are only muted in the display while Attachment is on. With a pin alone, a Hook or Armature stays shown, so Marrow writes the simulation into the mesh and the modifier then bends that result a second time. The stretched spike this produces is display only - the simulation underneath is unharmed.

### Ground plane

A cage that starts below the ground plane is lifted onto it, rigidly, before the first frame, and Marrow says so on the console.

This is not cosmetic. Collision resolves penetration by moving the predicted position, and the integrator reads that move as velocity of depth divided by the substep length. Mid-simulation that is harmless, because a substep can only sink a node so far. The starting state has no such bound: a unit ball authored straddling the plane left its first substep at 226 m/s, which is past any tear threshold and shreds the body. Lifting rigidly rather than clamping each node matters too, since clamping flattens the buried half and the stored energy launches it nearly as hard.

### False color

**False Color**, in the Display box, rainbow-shades the render surface by how far the material is stretched, in the style of Vellum's false color mode. The value is the edge stretch ratio: 1 at rest, hot where pulled past rest length, cold where compressed.

While stretch display is active a generated emission material sits in slot 0; choosing **Off** puts the object's own material back exactly as it was, including on an object that had no material at all, and De-tetrahedralize cleans up too. The scalar is computed per tet on the CPU from the cached cage positions, so switching the mode on after a bake still colours every cached frame.

## How it works

| Stage | Where |
|---|---|
| Tetrahedralize | CPU, numpy, once |
| Tet data | Mesh vertices, ID properties and POINT attributes, surviving save and load |
| Rest shape | A `marrow_rest` POINT attribute, so De-tetrahedralize can undo the whole thing |
| Pack to textures | CPU to `GPUTexture`, once per simulation start |
| Solve | GLSL compute, 6 kernels x substeps x constraint colours, plus a hanging-node blend pass between the elastic solve and attachment on adaptive cages |
| Skin and readback | GPU blend, then only the render vertices cross PCIe |

The cage uses a **cube-split lattice with checkerboard parity**, not conforming Delaunay. Boundary recovery is fragile on real meshes and replaces the render mesh; this approach never touches your topology and tolerates messy input, because the only question asked of the mesh is inside or outside. The accepted cost is that the cage does not hug concave detail, so thin models need a finer Resolution - or [Adaptive](#adaptive-cages), which puts the fine cells only where the shape is thin.

Tets are graph-coloured at build time so each colour dispatches race-free with no atomics. Interior cage nodes never cross PCIe; only render vertices are read back.

### Long runs

**Tetrahedralize and Bake report progress and can be cancelled.** Both show the stage and a percentage in the status bar while they work, and **Esc** stops them. Neither blocks the window any more, so a dense cage no longer looks like a hang — which it did, and people killed Blender over it.

The two cancel differently, because the work is different. Esc during Tetrahedralize discards everything: nothing is written to the file until the cage is complete, so there is no half-built state to leave behind. Esc during a Bake **keeps the frames it already simulated** — the cache is keyed by frame, so a bake stopped at 96 of 250 is playable to 96 and is not a wasted wait.

Cost scales sharply with Resolution: halving it is roughly eight times the cage. On a 34,000-vertex mesh, Resolution 0.25 builds in about 7 seconds, 0.12 in 22, and 0.08 in 66 — and the voxel pass, which asks inside-or-outside about every cell in the bounding box, is most of that at fine settings. Tetrahedralize now also names the cage size when it finishes, and warns there if the cage is over the node budget, rather than letting Bake be the first to mention it.

`tools/estimate_cage.py` reports what a setting will cost before you commit to it: run it from the Scripting tab with the object selected.

## Limitations

- **Contact is node against node only.** No edge-edge or node-triangle contact, for either self-collision or body-to-body, and neither scales past roughly 20,000 surface nodes. Collision against a *collider* is a distance field and does not have this limit.
- **A deforming collider is baked once**, and features thinner than one SDF cell are missed. See [Colliders](#colliders).
- **Friction does not ride a moving collider.** Contact friction resists sliding, but it measures the node against a collider treated as still for the substep, so a plate sliding sideways under a body does not drag it along. Sticky is how a moving collider carries material. Self-collision and body-to-body both measure the pair properly and have no such limit.
- **A body must not start inside a sticky collider.** See [Sticky colliders](#sticky-colliders). Only the ground plane depenetrates its starting state.
- **The cache lives in memory, not in the .blend.** Reopening a file means playing again from the start; live rebuilds the cache as you go.
- **No plasticity or per-region materials.** Fiber adds anisotropy along one baked direction; `mu` and `lam` are still global.
- **Attachment weights are synthesized once**, against the rest shape. Editing the mesh without re-tetrahedralizing leaves them stale, same rule as the bind data.
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

**Run the Blender suite on 5.2, and check which binary you invoked.** Background mode only has a GPU context from 5.2 on. Point this at 4.5 and every GPU test fails with `GPU functions for drawing are not available in background mode`, and a windowed 4.5 driven by `--python` at startup fails each readback with `StaleReadError: a RGBA32F upload never became visible`. Neither says anything about the code, and on a machine with several Blender versions installed it is an easy hour to lose. The suite is 254 tests on 5.2. A full run occasionally reports a few failures that pass when their module is run alone, and the failing set changes between runs - see [A note on GPU reliability](#a-note-on-gpu-reliability). Re-run before reading a red full run as a regression.

Running a single module rather than `run_tests.py` needs a `gpu.init()` of your own first: 5.2 requires it, and several modules rely on some earlier module in the full run having already called it.

A numpy reference solver in `marrow/core/solver_ref.py` is the test oracle: every GPU kernel is diffed against it, because a wrong sign in a compute shader is otherwise indistinguishable from a wrong sign in the constraint derivation.

The addon must use relative imports only. Installed as an extension its package is `bl_ext.user_default.marrow`, not `marrow`, so an absolute self-import fails at register time on a user's machine while every test still passes. `tests/core/test_packaging.py` guards this.

Never `pip install` into Blender's bundled Python.

## Licence

GPL-3.0-or-later.
