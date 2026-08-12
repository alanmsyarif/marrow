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

from ..blender.storage import read_bind, read_tetmesh
from ..core.solver_ref import SolverParams
from ..gpu.solver import GPUSolver, MarrowNaNError

CAGE_SUFFIX = "_marrow_cage"

# Refuse rather than hang. The spec requires the actual count in the message.
MAX_NODES = 200_000


class MarrowSession:
    """Owns the GPU state and the baked cache for one object."""

    def __init__(self, obj, params: SolverParams = None, ground_z=0.0, ground_on=False,
                 collider_objects=None, tear_threshold=0.0):
        # A caller who supplies params owns them; only a session built from
        # the panel follows the panel. Otherwise a restart would silently
        # overwrite explicitly chosen settings.
        self.follow_settings = params is None
        self.params = params or SolverParams()
        self.ground_z = float(ground_z)
        self.ground_on = bool(ground_on)
        self.collider_objects = list(collider_objects or [])
        self.tear_threshold = float(tear_threshold)
        # Live mode simulates forward as the timeline plays, caching as it
        # goes, so scrubbing back is still a cache lookup.
        self.live = False
        # A baked cache is authoritative: replaying it must never re-simulate,
        # even from the start frame. A live cache is disposable.
        self.baked = False
        self._last_simulated = None
        self.object_name = obj.name
        self._cache: dict[int, np.ndarray] = {}
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
        """(kind, to_local, to_world) per collider, sampled at the current frame.

        Primitives are unit-sized in local space, so the object transform is
        the whole description - position, orientation and size all come from
        it and animate for free.
        """
        specs = []
        for entry in self.collider_objects:
            collider, shape = entry if isinstance(entry, tuple) else (entry, "SPHERE")
            if collider is None:
                continue
            kind = 1 if shape == "SPHERE" else 2
            world = collider.matrix_world.copy()
            specs.append((kind, world.inverted(), world))
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

        Rebuilds the solver first so a bake always starts from rest rather
        than from wherever a previous bake left the cage.
        """
        self._check_live()
        self._cache.clear()
        self.refresh_from_object()
        self._build_solver()
        self._last_simulated = int(frame_end)
        self.live = False
        self.baked = True

        for frame in range(int(frame_start), int(frame_end) + 1):
            if scene is not None and self.collider_objects:
                # Re-sample collider transforms so animated colliders work.
                # Without this a falling ball would sit still for the whole bake.
                scene.frame_set(frame)
                self.solver.colliders = self._collider_specs()
            self.solver.step()
            positions = self.solver.skin()
            # skin() already raises on NaN. Readback is not fully reliable -
            # there is no barrier API - so refuse to cache anything suspect
            # rather than bake a bad frame in permanently.
            if not np.all(np.isfinite(positions)):
                raise MarrowNaNError(
                    f"Marrow produced a non-finite frame at {frame}. Nothing "
                    f"was cached. Raise Substeps, or lower Stiffness and "
                    f"Volume Preservation, and bake again."
                )
            self._cache[frame] = positions.astype(np.float32)

        return len(self._cache)

    # Small skips are tolerated so playback that drops a frame does not
    # permanently stall the simulation. A large jump is not chased - catching
    # up hundreds of frames inside a frame handler would lock the UI.
    MAX_CATCHUP = 8

    def _step_and_cache(self, frame: int):
        if self.collider_objects:
            # The handler runs after the frame changed, so collider transforms
            # are already at the new frame.
            self.solver.colliders = self._collider_specs()
        self.solver.step()
        positions = self.solver.skin()
        if not np.all(np.isfinite(positions)):
            raise MarrowNaNError(
                f"Marrow produced a non-finite frame at {frame}. Live "
                f"simulation stopped. Raise Substeps, or lower Stiffness and "
                f"Volume Preservation."
            )
        self._cache[frame] = positions.astype(np.float32)
        self._last_simulated = frame
        return self._cache[frame]

    def ensure_frame(self, frame: int, frame_start: int):
        """Positions for ``frame``, simulating forward when live.

        Returns None when the frame cannot be served: before the start, or
        after a jump too large to catch up with.
        """
        self._check_live()
        frame, frame_start = int(frame), int(frame_start)

        # A baked cache is played back, never regenerated.
        if self.baked:
            return self._cache.get(frame)

        if not self.live or frame < frame_start:
            return self._cache.get(frame)

        # Returning to the start always restarts, even if that frame is
        # already cached. That is what makes edited sliders take effect
        # without having to free the cache by hand.
        if self._last_simulated is None or frame == frame_start:
            self._cache.clear()
            self.refresh_from_object()
            self._build_solver()
            self._last_simulated = frame_start - 1
        else:
            cached = self._cache.get(frame)
            if cached is not None:
                return cached

        gap = frame - self._last_simulated
        if gap <= 0 or gap > self.MAX_CATCHUP:
            return None

        result = None
        for step_frame in range(self._last_simulated + 1, frame + 1):
            result = self._step_and_cache(step_frame)
        return result

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
        return True

    def free(self) -> None:
        """Drop GPU references. Idempotent."""
        self.solver = None
        self._cache.clear()
        self._freed = True


def _cage_of(obj):
    import bpy

    cage = bpy.data.objects.get(f"{obj.name}{CAGE_SUFFIX}")
    if cage is None:
        raise ValueError(
            f"{obj.name!r} has no Marrow cage. Run Tetrahedralize in the "
            f"Marrow panel first."
        )
    return cage
