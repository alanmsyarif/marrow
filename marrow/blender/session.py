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

from ..blender.storage import TETS_KEY, read_bind, read_tetmesh
from ..core.solver_ref import SolverParams
from ..gpu.solver import GPUSolver, MarrowNaNError

CAGE_SUFFIX = "_marrow_cage"

# Refuse rather than hang. The spec requires the actual count in the message.
MAX_NODES = 200_000


class MarrowSession:
    """Owns the GPU state and the baked cache for one object."""

    def __init__(self, obj, params: SolverParams = None, ground_z=0.0, ground_on=False,
                 collider_objects=None, tear_threshold=0.0, stick_break=0.0,
                 self_distance=0.0, body_distance=0.0):
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
        # Drives the mesh-collider SDF grid: the field only has to resolve
        # detail the cage can represent, so it tracks Resolution rather than
        # adding a setting of its own.
        self.resolution = float(
            getattr(getattr(obj, "marrow", None), "resolution", 0.25)
        )
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
        self.bind_idx, self.bind_w = read_bind(obj.data)
        # Pinning is not exposed yet; every node is free.
        self.inv_mass = np.ones(tetmesh.n_nodes, dtype=np.float64)
        self._build_solver()

    def _collider_specs(self):
        """(kind, to_local, to_world, sticky, field) at the current frame.

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
            else:
                collider, shape, sticky = entry, "SPHERE", False
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
                     sticky, field)
                )
                continue

            kind = 1 if shape == "SPHERE" else 2
            specs.append((kind, world.inverted(), world, sticky, None))
        return specs

    def _build_solver(self) -> None:
        self.solver = GPUSolver(
            self.tetmesh,
            self.inv_mass,
            self.params,
            ground_z=self.ground_z,
            ground_on=self.ground_on,
            colliders=self._collider_specs(),
            tear_threshold=self.tear_threshold,
            stick_break=self.stick_break,
            self_distance=self.self_distance,
            body_distance=self.body_distance,
        )
        self.solver.attach_render(self.bind_idx, self.bind_w)

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

        self.params = SolverParams(
            substeps=int(settings.substeps),
            mu=float(settings.stiffness),
            lam=float(settings.volume_preservation),
            damping=float(settings.damping),
        )
        self.ground_z = float(settings.ground_z)
        self.ground_on = bool(settings.ground_enabled)
        self.tear_threshold = (
            float(settings.tear_threshold) if settings.tearing_enabled else 0.0
        )
        self.stick_break = float(settings.stick_break)
        # The panel holds a multiple of Resolution; the solver wants metres.
        self.resolution = float(settings.resolution)
        thickness = float(settings.self_thickness) * float(settings.resolution)
        self.self_distance = thickness if settings.self_collision else 0.0
        self.body_distance = thickness if settings.body_collision else 0.0
        from .ops import collider_objects_of

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

        # skin() works in world space; mesh vertices are object space.
        world_to_local = np.array(obj.matrix_world.inverted())
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
