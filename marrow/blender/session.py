"""Per-object simulation session: a GPU solver plus a baked frame cache.

Baking runs the solver forward once and stores the skinned render-vertex
positions for every frame. Playback is then a dictionary lookup and a
foreach_set, which is what makes scrubbing interactive - nothing simulates
while the user drags the timeline.

The cache lives in memory and is not written into the .blend. Persisting it
would need either a large ID-property blob or a sidecar file, and neither is
required to make the addon work end to end. Rebaking is cheap. Deliberate v1
limit.
"""

import numpy as np
from mathutils import Matrix

from ..blender.storage import (
    TETS_KEY,
    has_legacy_blend,
    read_bind,
    read_fiber,
    read_tetmesh,
)
from ..core.solver_ref import SolverParams
from ..core.tetmesh import MASS_DENSITY, node_volumes
from ..gpu.solver import GPUSolver, MarrowNaNError

CAGE_SUFFIX = "_marrow_cage"

# Refuse rather than hang. The spec requires the actual count in the message.
MAX_NODES = 200_000


class MarrowSession:
    """Owns the GPU state and the baked cache for one object."""

    def __init__(self, obj, params: SolverParams = None, ground_z=0.0, ground_on=False,
                 collider_objects=None, tear_threshold=0.0, stick_break=0.0,
                 self_distance=0.0, body_distance=0.0, friction=0.0,
                 attach_enabled=False, attach_stiffness=0.0, pin_group=None,
                 pin_kinematic=None, region_group=None, region_softest=None):
        # A caller who supplies params owns them; only a session built from
        # the panel follows the panel. Otherwise a restart would silently
        # overwrite explicitly chosen settings.
        self.follow_settings = params is None
        self.params = params or SolverParams()
        self.ground_z = float(ground_z)
        self.ground_on = bool(ground_on)
        self.collider_objects = list(collider_objects or [])
        self.tear_threshold = float(tear_threshold)
        self.stick_break = float(stick_break)
        self.self_distance = float(self_distance)
        self.body_distance = float(body_distance)
        # Contact friction for the ground, self-collision and body-to-body.
        # A collider slot carries its own, read in _collider_specs.
        self.friction = float(friction)
        # Attachment feeds the object's animation into the sim as targets.
        # Weights and the first target set are built in _build_solver, so a
        # restart (which re-reads the panel and rebuilds) picks up toggles.
        self.attach_enabled = bool(attach_enabled)
        self.attach_stiffness = float(attach_stiffness)
        self.attach_idx = None
        self.attach_w = None
        # Drives the mesh-collider SDF grid: the field only has to resolve
        # detail the cage can represent, so it tracks Resolution rather than
        # adding a setting of its own.
        self.resolution = float(
            getattr(getattr(obj, "marrow", None), "resolution", 0.25)
        )
        # Vertex group whose weight holds material still. None means "read
        # the panel", the same rule Resolution follows, so the handler path
        # - MarrowSession(obj) with no kwargs - picks a pin up without the
        # caller having to know about it.
        self.pin_group = (
            str(getattr(getattr(obj, "marrow", None), "pin_group", ""))
            if pin_group is None
            else str(pin_group)
        )
        # Whether that pin rides the animation or stays put. Same None-reads-
        # the-panel rule as pin_group above.
        self.pin_kinematic = (
            bool(getattr(getattr(obj, "marrow", None), "pin_follows", False))
            if pin_kinematic is None
            else bool(pin_kinematic)
        )
        # Vertex group whose weight says how stiff the material is, and the
        # multiplier an unpainted vertex gets. Same None-reads-the-panel
        # rule as pin_group above.
        self.region_group = (
            str(getattr(getattr(obj, "marrow", None), "region_group", ""))
            if region_group is None
            else str(region_group)
        )
        self.region_softest = (
            float(getattr(getattr(obj, "marrow", None), "region_softest", 0.1))
            if region_softest is None
            else float(region_softest)
        )
        # Per-tet stiffness multiplier, or None for a uniform body. Built in
        # _build_solver, so a restart picks up a repainted group.
        self.region = None
        # Live mode simulates forward as the timeline plays, caching as it
        # goes, so scrubbing back is still a cache lookup.
        self.live = False
        # A baked cache is authoritative: replaying it must never re-simulate,
        # even from the start frame. A live cache is disposable.
        self.baked = False
        self._last_simulated = None
        self.object_name = obj.name
        self._cache: dict[int, np.ndarray] = {}
        # Cage node positions per frame, alongside the skinned render
        # positions. The false-color display derives its per-tet metric from
        # these on the CPU, so a mode switched on after a bake still colours
        # every cached frame.
        self._cache_nodes: dict[int, np.ndarray] = {}
        self._freed = False

        cage_obj = _cage_of(obj)
        tetmesh, _colors = read_tetmesh(cage_obj.data)
        if tetmesh.n_nodes > MAX_NODES:
            raise ValueError(
                f"Marrow cage has {tetmesh.n_nodes} nodes, over the {MAX_NODES} "
                f"budget. Raise Resolution in the Marrow panel to coarsen the cage."
            )

        self.tetmesh = tetmesh
        # An adaptive cage carries hanging nodes that only the blend pass
        # could hold together, and that pass is gone. Refusing is the honest
        # answer: without the glue those nodes are free, and the cage comes
        # apart on the first substep in a way that looks like a solver bug.
        if has_legacy_blend(cage_obj.data):
            raise ValueError(
                f"{obj.name!r} has an adaptive cage, which Marrow no longer "
                f"supports - the uniform cage covers thin features now and "
                f"costs fewer nodes. Run Tetrahedralize again to rebuild it."
            )
        # None on a cage tetrahedralized without a fiber curve. The solver
        # allocates a blank row per tet either way, so the fiber pass is
        # dead rather than absent.
        self.fiber = read_fiber(cage_obj.data)
        # The world matrix the solver runs in. Replaced at every build with
        # the object current transform, so a restart follows a move.
        self.sim_world = np.eye(4)
        self.bind_idx, self.bind_w = read_bind(obj.data)
        self._build_solver()

    def _collider_specs(self):
        """(kind, to_local, to_world, sticky, field, friction) at this frame.

        Primitives are unit-sized in local space, so the object transform is
        the whole description - position, orientation and size all come from
        it and animate for free.

        A mesh collider works the same way, with a signed distance field
        standing in for the primitive. Its field is baked in local space, so
        the transform animates it for free too and ``field`` is the same
        cached array every frame. The bounding-box-to-unit-cube mapping is
        composed into the matrices here rather than sent as push constants,
        because the collide kernel's push block already overflows.
        """
        from . import sdf

        specs = []
        for entry in self.collider_objects:
            if isinstance(entry, tuple):
                collider, shape = entry[0], entry[1]
                sticky = bool(entry[2]) if len(entry) > 2 else False
                mu = float(entry[3]) if len(entry) > 3 else 0.0
            else:
                collider, shape, sticky, mu = entry, "SPHERE", False, 0.0
            if collider is None:
                continue
            world = collider.matrix_world.copy()

            if shape == "MESH":
                field, grid = sdf.bake(collider, self.resolution)
                if field is None:
                    # Nothing to collide against - an empty or curve-only
                    # object. Skipping beats colliding with a stale shape.
                    continue
                grid_m = Matrix([list(row) for row in grid])
                specs.append(
                    (3, grid_m @ world.inverted(), world @ grid_m.inverted(),
                     sticky, field, mu)
                )
                continue

            kind = 1 if shape == "SPHERE" else 2
            specs.append((kind, world.inverted(), world, sticky, None, mu))
        return specs

    @property
    def attach_active(self) -> bool:
        """Whether the attachment pass runs at all.

        Stiffness above zero pulls the free material towards the animation.
        Stiffness zero still runs the pass when a pin is driven, because
        that is where the pin gets its target - it just stops gripping
        everything else.

        One predicate, deliberately, because two places need it: building
        the solver, and deciding whether a bake has to walk the scene
        forward so the targets change. Written out twice, the second copy
        kept the old `stiffness > 0` test and pins-only baked every frame
        against the start pose - the animation invisible to the one mode
        built entirely to follow it.
        """
        return self.attach_enabled and (
            self.attach_stiffness > 0.0 or self.pin_kinematic
        )

    def _compute_inv_mass(self) -> None:
        """Lumped inverse mass for every cage node, pins included.

        Mass is material: each node carries the volume it represents at a
        fixed density, so re-tetrahedralizing finer makes the same object,
        not a heavier one. A pin group then scales that: weight 1 leaves
        zero inverse mass, which predict, integrate and every contact pass
        already read as "this node does not move".

        Called from _build_solver rather than __init__ so a restart -
        refresh_from_object then _build_solver - picks up a repainted group
        instead of holding the weights the session was born with.
        """
        mass = node_volumes(self.tetmesh.nodes, self.tetmesh.tets) * MASS_DENSITY
        self.inv_mass = 1.0 / np.maximum(mass, 1e-12)
        if not self.pin_group:
            return
        import bpy

        from .attach import node_group_weights

        obj = bpy.data.objects.get(self.object_name)
        if obj is None:
            return
        weights = node_group_weights(obj, self.tetmesh, self.pin_group)
        if weights is None:
            return
        scale = np.clip(1.0 - weights, 0.0, 1.0)
        # A fully painted blend row sums to 1 only to float precision, so a
        # solid pin can leave 1e-16 of inverse mass behind - and predict
        # only asks `w > 0.0`, so that speck still takes the whole gravity
        # step and the "pinned" body sails away. Snap it to a real zero.
        scale[scale < 1e-6] = 0.0
        self.inv_mass *= scale

    def _compute_region(self) -> None:
        """Per-tet stiffness multiplier from the painted group, or None.

        Weight 1 keeps the panel's Stiffness and Volume Preservation as they
        are, weight 0 drops to ``region_softest``. Anchored at the top on
        purpose: the sliders stay the material the user tuned, and the group
        says where the body gives. The alternative - multiplier equal to the
        raw weight - turns every unpainted vertex to zero stiffness, so
        painting one stiff region would liquefy the rest of the body.

        Called from _build_solver rather than __init__ for the same reason
        _compute_inv_mass is: a restart has to see a repainted group.
        """
        self.region = None
        if not self.region_group:
            return
        import bpy

        from ..core.attach import tet_scalar
        from .attach import node_group_weights

        obj = bpy.data.objects.get(self.object_name)
        if obj is None:
            return
        weights = node_group_weights(obj, self.tetmesh, self.region_group)
        if weights is None:
            return
        softest = min(max(self.region_softest, 0.0), 1.0)
        node_scale = softest + (1.0 - softest) * np.clip(weights, 0.0, 1.0)
        self.region = tet_scalar(node_scale, self.tetmesh.tets)

    def _sample_frame(self):
        """The world matrix the solver is about to run in, and the rigid move
        from the cage bind frame into it.

        Cage nodes are stored in the world frame Tetrahedralize saw, and that
        frame is kept on the cage as the inverse of matrix_parent_inverse. If
        the object has been moved or turned since, the two disagree, and the
        body simulates and draws where it used to be while its origin sits
        somewhere else.

        Sampled here rather than per frame, which is the entire difference
        between "move the object and scrub to the start" and "the object
        transform is an animation channel". _refresh_targets reuses whatever
        this returned, so an animated transform still cannot drive anything.

        Returns (world, delta) as 4x4 arrays, both identity-safe.
        """
        import bpy

        obj = bpy.data.objects.get(self.object_name)
        cage = find_cage(obj) if obj is not None else None
        if obj is None or cage is None:
            return np.eye(4), np.eye(4)
        world = np.array(obj.matrix_world.to_4x4())
        delta = np.array(
            (obj.matrix_world @ cage.matrix_parent_inverse).to_4x4()
        )
        linear = delta[:3, :3]
        # Rotation and translation only. A scaled delta would resize the cage
        # under a rest shape measured before it, which silently changes both
        # the mass and what Stiffness means - refuse rather than quietly
        # simulate a different object.
        if not np.allclose(linear @ linear.T, np.eye(3), atol=1e-4):
            raise ValueError(
                f"{obj.name!r} has been scaled since Tetrahedralize. Marrow "
                f"can follow a move or a rotation, but not a scale - apply "
                f"the scale, or Tetrahedralize again."
            )
        return world, delta

    def _rebased(self, delta):
        """The cage and the fiber directions, moved into the solver frame."""
        from ..core.tetmesh import TetMesh

        if np.allclose(delta, np.eye(4), atol=1e-9):
            return self.tetmesh, self.fiber
        nodes = self.tetmesh.nodes @ delta[:3, :3].T + delta[:3, 3]
        fiber = self.fiber
        if fiber is not None:
            fiber = fiber.copy()
            # Directions rotate, arclength and side do not - one is a length
            # along the curve and the other a normalised offset, and neither
            # cares where the body is standing.
            fiber[:, :3] = fiber[:, :3] @ delta[:3, :3].T
        return TetMesh(nodes, self.tetmesh.tets), fiber

    def _build_solver(self) -> None:
        self._compute_inv_mass()
        self._compute_region()
        self.sim_world, delta = self._sample_frame()
        mesh, fiber = self._rebased(delta)
        attach_targets = None
        attach_stiffness = 0.0
        if self.attach_active:
            # Prepared here rather than in __init__ so the restart path -
            # refresh_from_object then _build_solver - honours a toggle
            # flipped since the session was created.
            import bpy

            from .attach import ensure_weights, sample_targets

            obj = bpy.data.objects.get(self.object_name)
            if obj is None:
                raise ValueError(f"object {self.object_name!r} no longer exists")
            self.attach_idx, self.attach_w = ensure_weights(obj, self.tetmesh)
            attach_targets = sample_targets(
                obj, self.attach_idx, self.attach_w, self.sim_world
            )
            attach_stiffness = self.attach_stiffness
        self.solver = GPUSolver(
            mesh,
            self.inv_mass,
            self.params,
            ground_z=self.ground_z,
            ground_on=self.ground_on,
            colliders=self._collider_specs(),
            tear_threshold=self.tear_threshold,
            stick_break=self.stick_break,
            self_distance=self.self_distance,
            body_distance=self.body_distance,
            friction=self.friction,
            attach_stiffness=attach_stiffness,
            attach_targets=attach_targets,
            pin_kinematic=self.pin_kinematic,
            fiber=fiber,
            region=self.region,
        )
        self.solver.attach_render(self.bind_idx, self.bind_w)

    def _refresh_targets(self) -> None:
        """Sample this frame's evaluated shape and hand it to the solver.

        Called once per frame by the group driver, alongside the collider
        transform resample. Silent no-op when the pass is off.
        """
        if self.attach_idx is None or getattr(self.solver, "sh_attach", None) is None:
            return
        import bpy

        from .attach import sample_targets

        obj = bpy.data.objects.get(self.object_name)
        if obj is None:
            return
        self.solver.set_targets(
            sample_targets(obj, self.attach_idx, self.attach_w, self.sim_world)
        )

    def refresh_from_object(self) -> None:
        """Re-read the panel settings so a restart picks up edited sliders.

        Without this, changing Stiffness or Substeps would do nothing until
        the session was freed and rebuilt by hand.
        """
        import bpy

        if not self.follow_settings:
            return
        obj = bpy.data.objects.get(self.object_name)
        if obj is None or getattr(obj, "marrow", None) is None:
            return
        settings = obj.marrow

        from .ops import _params_from, collider_objects_of

        # The bake path builds its params through _params_from directly, so
        # reading the sliders a second time here would be a copy to keep in
        # step - and the two silently diverging is how a setting ends up
        # working live and not in a bake.
        self.params = _params_from(settings)
        self.ground_z = float(settings.ground_z)
        self.ground_on = bool(settings.ground_enabled)
        self.tear_threshold = (
            float(settings.tear_threshold) if settings.tearing_enabled else 0.0
        )
        self.stick_break = float(settings.stick_break)
        self.friction = float(settings.friction)
        self.attach_enabled = bool(settings.attach_enabled)
        self.attach_stiffness = float(settings.attach_stiffness)
        self.pin_group = str(settings.pin_group)
        self.pin_kinematic = bool(settings.pin_follows)
        # Read here as well as in __init__, or a session already alive
        # keeps the group it was born with and a panel change silently
        # does nothing until the object is freed by hand.
        self.region_group = str(settings.region_group)
        self.region_softest = float(settings.region_softest)
        # The panel holds a multiple of Resolution; the solver wants metres.
        self.resolution = float(settings.resolution)
        thickness = float(settings.self_thickness) * float(settings.resolution)
        self.self_distance = thickness if settings.self_collision else 0.0
        self.body_distance = thickness if settings.body_collision else 0.0
        self.collider_objects = collider_objects_of(obj)

    def _check_live(self) -> None:
        if self._freed:
            raise RuntimeError(
                f"Marrow session for {self.object_name!r} has been freed; "
                f"bake again to recreate it."
            )

    @property
    def baked_range(self):
        if not self._cache:
            return None
        return (min(self._cache), max(self._cache))

    def bake(self, frame_start: int, frame_end: int, scene=None) -> int:
        """Simulate frame_start..frame_end inclusive, caching each frame.

        A group of one. Baking bodies that collide goes through
        group.bake with every member, since a two-way bake of one body alone
        would leave the other absent from its own contact.
        """
        from . import group

        return group.bake([self], frame_start, frame_end, scene=scene)

    # Small skips are tolerated so playback that drops a frame does not
    # permanently stall the simulation. A large jump is not chased - catching
    # up hundreds of frames inside a frame handler would lock the UI.
    MAX_CATCHUP = 8

    def cache_frame(self, frame: int):
        """Skin the current solver state and cache it as ``frame``.

        The group driver owns the stepping, because bodies that collide have
        to be interleaved substep by substep. This is the half that stays
        with the session: it reads back and stores its own frame.
        """
        positions = self.solver.skin()
        # skin() already raises on NaN. Readback is not fully reliable -
        # there is no barrier API - so refuse to cache anything suspect
        # rather than keep a bad frame permanently.
        if not np.all(np.isfinite(positions)):
            raise MarrowNaNError(
                f"Marrow produced a non-finite frame at {frame}. Nothing was "
                f"cached. Raise Substeps, or lower Stiffness and Volume "
                f"Preservation."
            )
        self._cache[frame] = positions.astype(np.float32)
        self._cache_nodes[frame] = self.solver.positions().astype(np.float32)
        self._last_simulated = frame
        return self._cache[frame]

    def _clear_cache(self) -> None:
        self._cache.clear()
        self._cache_nodes.clear()

    def ensure_frame(self, frame: int, frame_start: int):
        """Positions for ``frame``, simulating forward when live.

        Returns None when the frame cannot be served: before the start, or
        after a jump too large to catch up with.

        The work happens in group.py, because a body that collides with
        another has to be advanced alongside it. A body with no partners is
        a group of one and takes the same path.
        """
        from . import group

        return group.advance(self, frame, frame_start)

    def frame_positions(self, frame: int):
        """Cached world-space render positions, or None if not baked."""
        return self._cache.get(int(frame))

    def write_to_mesh(self, obj, frame: int, frame_start: int = None) -> bool:
        """Write a frame into the mesh, simulating it first when live.

        False means the frame could not be served and the mesh was left alone.
        """
        if frame_start is None:
            positions = self.frame_positions(frame)
        else:
            positions = self.ensure_frame(frame, frame_start)
        if positions is None:
            return False

        # skin() works in the solver frame; mesh vertices are object space.
        #
        # Divided by the frame the solver was BUILT in, not the object
        # current one. Those are the same matrix until someone moves the
        # object mid-playback, and then this is what decides what that looks
        # like: dividing by the live transform would draw the body sliding
        # the opposite way, because the simulation has not moved. Dividing by
        # the build frame carries the simulation along rigidly instead, the
        # way moving any other object behaves, and the next restart rebases
        # it properly.
        world_to_local = np.linalg.inv(self.sim_world)
        local = positions @ world_to_local[:3, :3].T + world_to_local[:3, 3]

        obj.data.vertices.foreach_set("co", local.ravel().astype(np.float64))
        obj.data.update()
        self._write_false_color(obj, int(frame))
        return True

    def _write_false_color(self, obj, frame: int) -> None:
        """Refresh the stretch attribute when a mode is active.

        The metric is rotation- and translation-invariant, so comparing the
        world-space cached cage against the world-space rest cage is safe
        whatever the object transform does.
        """
        mode = getattr(getattr(obj, "marrow", None), "false_color", "OFF")
        if mode == "OFF":
            return
        nodes = self._cache_nodes.get(frame)
        if nodes is None:
            # The only frame served without cached nodes is a reset below the
            # start frame, and a reset leaves the cage at rest. Comparing rest
            # against rest paints the whole surface at 1, which is the honest
            # answer - keeping the last frame's colours would show a deformed
            # body over a mesh that is back at rest. No readback needed: the
            # rest cage is already here on the CPU.
            nodes = self.tetmesh.nodes
        from ..core import metric
        from . import false_color

        tet_values = metric.tet_stretch(
            nodes, self.tetmesh.tets, self.tetmesh.nodes
        )
        false_color.write_attribute(
            obj.data, metric.vertex_values(tet_values, self.bind_idx)
        )

    def free(self) -> None:
        """Drop GPU references. Idempotent."""
        self.solver = None
        self._clear_cache()
        self._freed = True


def find_cage(obj):
    """The object's Marrow cage object, or None.

    The cage is parented to the object at tetrahedralize time, so look among
    its children rather than by name: renaming the object does not rename the
    cage, and a name lookup silently loses it - or worse, finds an unrelated
    object that happens to reuse the expected name. The TETS_KEY check is
    what tells a real cage apart from something merely named like one.
    """
    import bpy

    for child in obj.children:
        data = getattr(child, "data", None)
        if (
            child.name.endswith(CAGE_SUFFIX)
            and data is not None
            and TETS_KEY in data.keys()
        ):
            return child
    # Fallback for a cage that was unparented by hand.
    cage = bpy.data.objects.get(f"{obj.name}{CAGE_SUFFIX}")
    if cage is not None:
        data = getattr(cage, "data", None)
        if data is not None and TETS_KEY in data.keys():
            return cage
    return None


def _cage_of(obj):
    cage = find_cage(obj)
    if cage is None:
        raise ValueError(
            f"{obj.name!r} has no Marrow cage. Run Tetrahedralize in the "
            f"Marrow panel first."
        )
    return cage
