# Kinematic pins

A pin is a static anchor. `Pin Group` sets a cage node's inverse mass to
zero, and every pass in the solver reads that as "this node does not move" -
predict skips the gravity step, integrate holds position and velocity both,
and the collider and attachment kernels each check it first. That is what
holds material against gravity, and it is also why hooking a pinned region
to an Empty and moving it does nothing.

This adds the other half: a pin that is still rigid, but rides the
animation instead of the origin.

## The problem

Measured on a cube whose top face is hooked to an Empty moved 3.0 in X,
tracking how far the cage top follows:

| Setup | Top follows | Bottom |
|---|---|---|
| Pin Group only | -0.004 | 0.102 |
| Attachment 1.0, no pin | 2.762 | 0.238 |
| Attachment 0.5, no pin | 2.600 | 0.929 |
| Attachment 1.0 **and** Pin Group | 1.322 | 0.238 |

Row 1 is the report: the pin does not follow. Row 4 is worse - the two
fight, and the pin drags the hook back to less than half the travel
attachment alone achieves.

Neither is a defect. Attachment is soft and uniform across the whole body;
Pin is rigid but frozen. **A moving rigid anchor is not expressible**, and
that is the gap.

## Design

### What drives it

The object's own evaluated shape, through the attachment target path that
already exists. A Hook to an Empty, an armature, a shape key - anything
count-preserving that Blender can evaluate - already expresses "this region
follows this thing", and `sample_targets` already blends the evaluated mesh
into one target per cage node every frame.

No new target machinery, no new object slot. A kinematic pin is attachment
at infinite stiffness, restricted to the painted region.

### How the user asks for it

A `pin_follows` boolean, **Follows Animation**, under Pin Group in the
panel. Off is today's frozen anchor; on rides the evaluated mesh.

Opt-in rather than automatic. Automatic - kinematic whenever Attachment is
on - needs no new UI and could be defended, since row 4 above shows the
current combination is already broken. But it would delete the static pin
on an animated body: a pinned foot nailed to the floor while the armature
drives the rest. That case is real, it is documented in the README, and
automatic offers no way to ask for it back.

Kinematic pins need Attachment on with a stiffness above zero, because that
is what builds the target texture. The checkbox greys out otherwise.

### The mechanism

Per substep the solver runs predict, solve, blend, **attach**, self
collision, body collision, colliders, integrate. Two guards stand between
a pin and the animation, and both are already exactly where the change
belongs.

**Attach** (`ATTACH_SRC`) opens `if (!(pi.w > 0.0)) { return; }` - a pin
outranks the armature. Under a kinematic pin it stores the target instead:

```glsl
if (!(pi.w > 0.0)) {
  if (kinematic != 0) { imageStore(p, c, vec4(target_pos, pi.w)); }
  return;
}
```

Guarded by a new `kinematic` uniform, so a static pin keeps returning.

**Integrate** (`INTEGRATE_SRC`) opens `if (!(xi.w > 0.0)) { return; }`, so
a pinned node's `x` never advances - without this the attach store above
would be discarded every substep. It writes the position through and zeroes
the velocity:

```glsl
if (!(xi.w > 0.0)) {
  imageStore(v, c, vec4(0.0));
  imageStore(x, c, vec4(imageLoad(p, c).xyz, xi.w));
  return;
}
```

Velocity is zeroed rather than differenced. A kinematic pin is driven, not
simulated; `(p - x) / h` would hand it a velocity that predict never reads
anyway, since predict skips the integration step for a pinned node.

Both stores sit behind the same `kinematic` uniform.

The first draft of this design left integrate unguarded, on the argument
that the store is provably a no-op for a static pin: `p == x` exactly at
integrate time, since predict copies it, `project()` scales every
correction by `n.w` which is zero, the blend kernel scales by `hx.w` which
is zero, and all three contact passes return early on the same guard.

That argument is correct and the conclusion was still wrong.
`test_integrate_leaves_pinned_nodes_alone` feeds the kernel a `p` half a
unit from `x` for a pinned node - a state the pipeline cannot produce - and
asserts the node does not move. It is not testing the no-op; it is pinning
down a **backstop**, so that "a pin does not move" is a property of the
integrator itself rather than of every pass that happens to run before it.
In a module whose own README documents textures losing their contents after
a verified upload, that guarantee is worth more than one saved uniform.
Guarded, the existing test passes untouched.

### Attach Stiffness 0 means pins only

Shipped after the first version of this design, from measurement rather
than foresight, and it is what makes the feature usable.

Attachment aims *every* node at its evaluated position. For material the
animation does not reach, that position is the REST pose - so the one pass
that supplies a driven pin its target is simultaneously nailing the rest of
the body down. Measured on a 23,697-node cage with a 27-vertex painted
region hooked to an Empty travelling 1.473:

| Attach Stiffness | Pin travel | Body travel |
|---|---|---|
| 0.50 | 1.473 | 0.026 |
| 0.05 | 1.473 | 0.040 |
| 0.01 | 1.473 | 0.374 |
| 0.00 | 1.473 | 1.552 |

The pin tracks perfectly throughout. Nothing about mass, gravity, ground
contact or material stiffness was the limit - the attachment grip was all
of it. But stiffness 0 used to disable the pass outright, so the only way
to reach the useful region was to type a magic near-zero number, and the
natural value for it was the one that switched the feature off.

So stiffness 0 alongside a driven pin now runs the pass for the pins alone:
a `drive_free` uniform, off, makes the attach kernel return for every node
with inverse mass, after the pinned store above. Free material is left
entirely to the elastic solve.

Two consequences worth stating:

- **The free cage starts at rest.** With a full attachment the whole cage
  starts from the targets so the body begins posed. Pins-only starts only
  the driven nodes there; starting the free material at the pose would
  apply exactly the displacement this mode exists to avoid.
- **`attach_active` is one predicate, deliberately.** Whether the pass runs
  is needed both to build the solver and to decide whether a bake walks the
  scene forward so targets change. Written out twice, the second copy kept
  the old `stiffness > 0` test and pins-only baked every frame against the
  start pose - the animation invisible to the one mode built to follow it.

### Contact ordering

The contact passes run after attach and all skip zero inverse mass, so
nothing overwrites a kinematic pin's stored target. A kinematic pin
therefore outranks colliders, self collision and body collision, which is
the correct reading: it is being driven, so it wins.

### Targets stay frame-constant

`set_targets` uploads once per frame, not per substep - "the pose is
constant across a frame's substeps, the same treatment the colliders'
transforms get". A kinematic pin therefore reaches the frame's target on
substep 1 and holds for the rest.

Deliberately not interpolated. That front-loading is exactly what
attachment at stiffness 1.0 already does to **every node in the body** -
`attach_compliance` documents "k = 1 is exactly zero compliance - a hard
snap", and `test_a_hard_attachment_rides_the_bone` asserts it rides the
bone exactly. Teleporting a subset is strictly gentler than something that
already ships and passes.

If a fast hook does tear in practice, the fix is cheap and can be added
then: `set_targets` already re-uploads a fresh texture each frame, so
keeping the previous one is a reference and a lerp, not another upload.

### Rejected alternatives

**A separate kinematic kernel**, dispatched after attach, leaving both
existing kernels untouched. Its whole selling point is not modifying
kernels the oracle already diffs against - and it fails to deliver, because
`x` still has to advance, so integrate changes either way. That leaves an
extra image binding and an extra dispatch bought for nothing.

**CPU re-upload of `tex_x` each frame.** No GLSL at all, and `set_targets`
plus `poison_for_test` show that reassigning a texture wholesale is an
established move. But it needs a full cage readback every frame, straight
against "interior cage nodes never cross PCIe", through the `read_stable`
path the README already documents as unreliable enough to need generation
marks.

## Components

| Unit | Responsibility |
|---|---|
| `MarrowSettings.pin_follows` | The panel flag. Greys out unless Attachment is on. |
| `MarrowSession.pin_kinematic` | Mirrors the flag, refreshed on restart like the rest. |
| `GPUSolver(pin_kinematic=...)` | Feeds the `kinematic` uniform to both the attach and integrate dispatches. |
| `ATTACH_SRC` | Stores the target for a pinned node when kinematic. |
| `INTEGRATE_SRC` | Advances `x` and zeroes `v` for pinned nodes when kinematic; otherwise returns as before. |
| `solver_ref.solve_attachment` | Numpy mirror of the attach change. |
| `solver_ref.step` | Numpy mirror of the integrate change. |

## Testing

The repo's rule is that every GPU kernel is diffed against
`marrow/core/solver_ref.py`, because a wrong sign in a compute shader is
otherwise indistinguishable from a wrong sign in the constraint derivation.
That holds here.

1. **Static pins are untouched.** With the flag off a pin stays frozen
   even while its targets move, and `test_integrate_leaves_pinned_nodes_alone`
   continues to hold the integrator to its backstop with no edit.
2. **A kinematic pin reaches its target.** Pinned nodes land on the frame's
   target, not near it.
3. **A kinematic pin drags material.** Free neighbours follow; the far side
   hangs. This is the row-1 failure from the table, inverted.
4. **A kinematic pin outranks a collider.** A collider pushing against a
   driven pin does not move it.
5. **Oracle parity.** GPU against numpy for a kinematic pin, alongside the
   existing parity tests.
6. **Kinematic without Attachment is inert**, not an error.
7. **The panel offers the checkbox** and greys it without Attachment.

## Out of scope

- Substep interpolation of targets. See above.
- Per-node attachment stiffness. Pin is the rigid case; a general
  per-region stiffness map is a different feature.
- A pin target that is an object transform rather than the evaluated mesh.
  The Hook already covers it.
