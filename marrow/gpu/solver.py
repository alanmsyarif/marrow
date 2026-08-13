"""The GPU substep loop.

Textures are uploaded once at construction. Per frame the loop runs
predict -> solve (one dispatch per colour) -> collide -> integrate, entirely
on the card. Nothing crosses PCIe until positions() or skin() is asked for.

There is no barrier API in Blender's gpu module, so ordering between
dependent dispatches is the driver's to provide. Dispatch-to-dispatch is
reliable - measured bit-identical across runs at 9261 nodes and 40000 tets.
Dispatch-to-readback is NOT: GPUTexture.read() straight after a dispatch can
return pre-dispatch contents, so positions() and skin() call textures.flush()
first. Do not "fix" an ordering problem by inserting a readback between
stages - the readback is itself the unordered path.

**Deviation from the plan text, deliberate.** The plan had step() call a
finiteness guard that downloaded the whole node image every frame. That is
the full-state readback the spec singles out as what kills GPU simulation in
Python - it would move N texels per frame to check something, defeating the
readback rule the whole architecture is built on. Detection lives at the
readback boundary instead: positions() and skin() validate exactly what they
already had to move. step() touches PCIe not at all.
"""

import gpu
import numpy as np

from ..core.coloring import color_tets
from ..core.layout import (
    color_order,
    color_ordered,
    pack_nodes,
    pack_rest,
    pack_scalar,
    pack_tets,
    texture_shape,
    unpack_vec3,
)
from ..core.solver_ref import precompute
from ..core.tetmesh import surface_nodes
from ..gpu import kernels
from ..gpu.textures import (
    blank,
    download,
    flush,
    make_flush_shader,
    StaleReadError,
    read_marked,
    read_stable,
    upload,
    upload_verified,
)

_GROUP = 64


class MarrowNaNError(RuntimeError):
    """The solver state stopped being finite. Freeze, report, write nothing."""


def _groups(count: int) -> int:
    return max(1, (int(count) + _GROUP - 1) // _GROUP)


def _guard_finite(values: np.ndarray, what: str, total: int) -> np.ndarray:
    """Refuse to hand back a state that is not finite."""
    finite = np.isfinite(values).all(axis=1)
    if not finite.all():
        bad = int(np.count_nonzero(~finite))
        raise MarrowNaNError(
            f"Marrow solver produced NaN or inf at {bad} of {total} {what}. "
            f"The frame was not written. Raise Substeps, or lower Stiffness "
            f"and Volume Preservation, in the Marrow panel and re-run."
        )
    return values


class GPUSolver:
    def __init__(self, mesh, inv_mass, params, ground_z=0.0, ground_on=False,
                 colliders=None, tear_threshold=0.0, stick_break=0.0,
                 self_distance=0.0):
        self.mesh = mesh
        self.params = params
        self.ground_z = float(ground_z)
        self.ground_on = bool(ground_on)
        # An absolute world distance, not the UI's multiple of Resolution.
        # Zero disables self-collision and allocates nothing.
        self.self_distance = float(self_distance)
        # Each entry is (kind, to_local, to_world) with kind 1 sphere, 2 box,
        # optionally followed by a sticky flag.
        self.colliders = list(colliders or [])
        self.tear_threshold = float(tear_threshold)
        self.stick_break = float(stick_break)
        self.n_nodes = mesh.n_nodes

        start = self._lift_out_of_ground(mesh.nodes)

        colors = color_tets(mesh.tets, mesh.n_nodes)
        ordered, self.offsets = color_ordered(mesh.tets, colors)
        self._tet_order = color_order(colors)   # colour-ordered slot -> mesh tet
        dm_inv, rest_vol = precompute(start, ordered)

        self.tex_x = upload_verified(pack_nodes(start, inv_mass))
        self.tex_p = blank(self.n_nodes)
        self.tex_v = upload_verified(
            pack_nodes(np.zeros_like(mesh.nodes), np.zeros(self.n_nodes))
        )
        self.tex_tets = upload_verified(pack_tets(ordered))
        self.tex_rest = upload_verified(pack_rest(dm_inv, rest_vol))
        # Unpermuted, for the skin kernel's bind indices.
        self.tex_tets_orig = upload_verified(pack_tets(mesh.tets))
        # One flag per tet. Zero means intact; set once, never cleared.
        self.tex_torn = blank(mesh.n_tets, fmt="R32F")
        # One counter per node: how many of its tets are still intact. The tear
        # rule reads it to refuse to orphan a node - see SOLVE_SRC. Incidence
        # is the same whichever tet order it is counted over.
        self.tex_live = upload_verified(
            pack_scalar(
                np.bincount(
                    np.asarray(mesh.tets, dtype=np.int64).ravel(),
                    minlength=self.n_nodes,
                )
            ),
            fmt="R32F",
        )
        # One texel per node: .xyz the contact point in the holding collider's
        # local space, .w that collider's id. Zero means the node is free.
        self.tex_stick = blank(self.n_nodes)

        self.sh_self = None
        surf = surface_nodes(mesh.tets) if self.self_distance > 0.0 else None
        if surf is not None and surf.shape[0] > 1:
            self.n_surf = int(surf.shape[0])
            # Where each node sits in the surface list, -1 for interior. The
            # kernel needs the index and not merely a flag, so a thread can
            # skip itself without comparing node ids.
            idx = np.full(self.n_nodes, -1.0)
            idx[surf] = np.arange(self.n_surf)
            self.tex_surf = upload_verified(pack_scalar(surf), fmt="R32F")
            self.tex_surf_idx = upload_verified(pack_scalar(idx), fmt="R32F")
            # Rest node positions. tex_rest is per-tet dm_inv and rest volume;
            # this is the per-node configuration those were built from, which
            # the rest-distance gate compares against.
            self.tex_rest_pos = upload_verified(
                pack_nodes(start, np.zeros(self.n_nodes))
            )
            self.tex_p2 = blank(self.n_nodes)
            self.sh_self = kernels.build(
                "self_collide", kernels.SELF_COLLIDE_SRC,
                [("RGBA32F", "FLOAT_2D", "p", {"READ"}),
                 ("RGBA32F", "FLOAT_2D", "out_p", {"WRITE"}),
                 ("RGBA32F", "FLOAT_2D", "rest_pos", {"READ"}),
                 ("R32F", "FLOAT_2D", "surf", {"READ"}),
                 ("R32F", "FLOAT_2D", "surf_idx", {"READ"})],
                [("FLOAT", "thickness"), ("INT", "n_nodes"), ("INT", "n_surf")],
            )

        self.sh_predict = kernels.build(
            "predict", kernels.PREDICT_SRC,
            [("RGBA32F", "FLOAT_2D", "x", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "v", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "p", {"WRITE"})],
            [("FLOAT", "h"), ("VEC3", "gravity"), ("INT", "n_nodes")],
        )
        self.sh_solve = kernels.build(
            "solve", kernels.SOLVE_SRC,
            [("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"}),
             ("RGBA32F", "FLOAT_2D", "tets", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "rest", {"READ"}),
             ("R32F", "FLOAT_2D", "torn", {"READ", "WRITE"}),
             ("R32F", "FLOAT_2D", "live", {"READ", "WRITE"})],
            [("FLOAT", "h"), ("FLOAT", "mu"), ("FLOAT", "lam"),
             ("FLOAT", "tear_threshold"),
             ("INT", "color_begin"), ("INT", "color_end")],
        )
        self.sh_collide = kernels.build(
            "collide", kernels.COLLIDE_SRC,
            [("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"}),
             ("RGBA32F", "FLOAT_2D", "stick", {"READ", "WRITE"})],
            [("FLOAT", "ground_z"), ("INT", "kind"), ("INT", "n_nodes"),
             ("MAT4", "to_local"), ("MAT4", "to_world"),
             ("INT", "collider_id"), ("INT", "sticky"), ("FLOAT", "break_dist")],
        )
        # Instance state, never module state - see make_flush_shader.
        self.sh_flush = make_flush_shader("RGBA32F")
        self.sh_flush_r32f = make_flush_shader("R32F")
        self._skin_mark = 0
        self.sh_integrate = kernels.build(
            "integrate", kernels.INTEGRATE_SRC,
            [("RGBA32F", "FLOAT_2D", "x", {"READ", "WRITE"}),
             ("RGBA32F", "FLOAT_2D", "p", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "v", {"READ", "WRITE"})],
            [("FLOAT", "h"), ("FLOAT", "damping"), ("INT", "n_nodes")],
        )

    def _lift_out_of_ground(self, nodes: np.ndarray) -> np.ndarray:
        """Rigidly raise a cage that starts below the ground plane.

        The collide pass depenetrates by moving the predicted position, and
        integrate then reads that move as velocity: depth / h, where h is
        dt / substeps. During the simulation that is harmless - a substep can
        only sink a node by v * h, so the velocity it reads back is the
        velocity that put it there. The starting state is the one case with
        no such bound. A unit ball authored straddling the ground plane leaves
        its first substep at 226 m/s (measured, 10 substeps), which is past
        any tear threshold and shreds the body into spikes.

        Translating is what makes this safe. Clamping each node onto the plane
        instead would flatten the buried half, and the elastic energy stored in
        that pancake launches it nearly as hard - measured 11 m/s and 22 torn
        tets on the same ball. A translation leaves the rest shape, the bind
        weights and the tet winding all untouched.

        ponytail: ground plane only. A cage authored inside a sphere or box
        collider has the same unbounded first substep; depenetrate per
        collider if that ever comes up.
        """
        if not self.ground_on or nodes.shape[0] == 0:
            return nodes
        lift = self.ground_z - float(nodes[:, 2].min())
        if lift <= 0.0:
            return nodes
        # Said out loud: the body is not where the user put it any more.
        print(
            f"marrow: cage started {lift:.4f} below the ground plane, "
            f"lifted onto it"
        )
        return nodes + np.array([0.0, 0.0, lift])

    def step(self) -> None:
        """One frame of params.substeps substeps. Reads nothing back."""
        h = self.params.dt / self.params.substeps
        node_groups = _groups(self.n_nodes)

        for _ in range(self.params.substeps):
            self.sh_predict.bind()
            self.sh_predict.image("x", self.tex_x)
            self.sh_predict.image("v", self.tex_v)
            self.sh_predict.image("p", self.tex_p)
            self.sh_predict.uniform_float("h", h)
            self.sh_predict.uniform_float("gravity", tuple(self.params.gravity))
            self.sh_predict.uniform_int("n_nodes", self.n_nodes)
            gpu.compute.dispatch(self.sh_predict, node_groups, 1, 1)

            for c in range(len(self.offsets) - 1):
                begin, end = int(self.offsets[c]), int(self.offsets[c + 1])
                if end <= begin:
                    continue
                self.sh_solve.bind()
                self.sh_solve.image("p", self.tex_p)
                self.sh_solve.image("tets", self.tex_tets)
                self.sh_solve.image("rest", self.tex_rest)
                self.sh_solve.image("torn", self.tex_torn)
                self.sh_solve.image("live", self.tex_live)
                self.sh_solve.uniform_float("h", h)
                self.sh_solve.uniform_float("tear_threshold", self.tear_threshold)
                self.sh_solve.uniform_float("mu", self.params.mu)
                self.sh_solve.uniform_float("lam", self.params.lam)
                self.sh_solve.uniform_int("color_begin", begin)
                self.sh_solve.uniform_int("color_end", end)
                gpu.compute.dispatch(self.sh_solve, _groups(end - begin), 1, 1)

            self._dispatch_self_collision(node_groups)
            self._dispatch_colliders(node_groups)

            self.sh_integrate.bind()
            self.sh_integrate.image("x", self.tex_x)
            self.sh_integrate.image("p", self.tex_p)
            self.sh_integrate.image("v", self.tex_v)
            self.sh_integrate.uniform_float("h", h)
            self.sh_integrate.uniform_float("damping", self.params.damping)
            self.sh_integrate.uniform_int("n_nodes", self.n_nodes)
            gpu.compute.dispatch(self.sh_integrate, node_groups, 1, 1)

    def _dispatch_self_collision(self, node_groups) -> None:
        """Push apart surface nodes that have come within self_distance.

        Runs before the colliders so a pin, a ground plane or a sticky grab
        gets the last word on a node that is in both kinds of contact.

        The kernel cannot correct in place - it reads every surface node and
        writes its own, so one image cannot be both. It writes tex_p2 and the
        two swap here. Every other stage binds self.tex_p at dispatch time,
        so the swap costs nothing and no copy pass is needed.
        """
        if self.sh_self is None:
            return
        self.sh_self.bind()
        self.sh_self.image("p", self.tex_p)
        self.sh_self.image("out_p", self.tex_p2)
        self.sh_self.image("rest_pos", self.tex_rest_pos)
        self.sh_self.image("surf", self.tex_surf)
        self.sh_self.image("surf_idx", self.tex_surf_idx)
        self.sh_self.uniform_float("thickness", self.self_distance)
        self.sh_self.uniform_int("n_nodes", self.n_nodes)
        self.sh_self.uniform_int("n_surf", self.n_surf)
        gpu.compute.dispatch(self.sh_self, node_groups, 1, 1)
        self.tex_p, self.tex_p2 = self.tex_p2, self.tex_p

    def _dispatch_colliders(self, node_groups) -> None:
        """One dispatch per collider, ground plane included.

        Looping colliders inside the kernel would need an array uniform.
        Dispatching per collider reuses the plain push-constant path that is
        already measured to work, and there are only ever a handful.
        """
        from mathutils import Matrix

        identity = Matrix.Identity(4)
        jobs = []
        if self.ground_on:
            # The ground never sticks, and it runs first so that a sticky
            # collider's anchor wins over it rather than the other way round.
            jobs.append((0, identity, identity, False, 0))
        for index, entry in enumerate(self.colliders, start=1):
            kind, to_local, to_world = entry[:3]
            sticky = bool(entry[3]) if len(entry) > 3 else False
            # The id is the slot position, not the loop counter, so it stays
            # the same frame to frame - an anchor recorded last substep has to
            # still name the same collider this one.
            jobs.append((kind, to_local, to_world, sticky, index))

        # Zero means "off" in the panel, but the kernel wants a distance it can
        # compare against, so an unbreakable hold is a distance nothing reaches.
        break_dist = self.stick_break if self.stick_break > 0.0 else 1.0e30

        for kind, to_local, to_world, sticky, collider_id in jobs:
            self.sh_collide.bind()
            self.sh_collide.image("p", self.tex_p)
            self.sh_collide.image("stick", self.tex_stick)
            self.sh_collide.uniform_float("ground_z", self.ground_z)
            self.sh_collide.uniform_int("kind", int(kind))
            self.sh_collide.uniform_int("n_nodes", self.n_nodes)
            self.sh_collide.uniform_int("collider_id", int(collider_id))
            self.sh_collide.uniform_int("sticky", 1 if sticky else 0)
            self.sh_collide.uniform_float("break_dist", break_dist)
            # Must be a mathutils Matrix. A flat 16-float list is accepted
            # without error and silently applies the wrong transform.
            self.sh_collide.uniform_float("to_local", to_local)
            self.sh_collide.uniform_float("to_world", to_world)
            gpu.compute.dispatch(self.sh_collide, node_groups, 1, 1)

    def positions(self) -> np.ndarray:
        """Full cage state. Debug and test path - not the per-frame route."""
        flush(self.sh_flush, self.tex_x)
        # tex_x's alpha holds inverse mass, so there is nowhere to stamp a
        # mark; fall back to reading until two reads agree, re-flushing between.
        out = unpack_vec3(
            read_stable(self.tex_x, nudge=lambda: flush(self.sh_flush, self.tex_x)),
            self.n_nodes,
        )
        return _guard_finite(out, "cage nodes", self.n_nodes)

    def torn_flags(self) -> np.ndarray:
        """One flag per tet, 1.0 where torn, in mesh tet order. Diagnostic path."""
        flush(self.sh_flush_r32f, self.tex_torn)
        flat = read_stable(
            self.tex_torn, nudge=lambda: flush(self.sh_flush_r32f, self.tex_torn)
        ).reshape(-1)
        # The texture stores the volume ratio a tet broke at, not a bare flag,
        # so anything non-zero is torn. Callers want the flag.
        ordered_flags = (flat[: self.mesh.n_tets] > 0.0).astype(np.float64)
        # The solve kernel indexes tets by colour-ordered position, so that is
        # the order the texture is in. Callers think in mesh order, and the two
        # only agree when every tet happens to land in colour 0. Anything that
        # maps a flag back onto mesh.tets needs this.
        out = np.empty_like(ordered_flags)
        out[self._tet_order] = ordered_flags
        return out

    def poison_for_test(self) -> None:
        """Force a non-finite state, so the NaN guard can be tested honestly."""
        self.tex_x = upload(
            pack_nodes(np.full_like(self.mesh.nodes, np.nan), np.ones(self.n_nodes))
        )

    def attach_render(self, bind_idx, bind_w) -> None:
        """Upload the render-vertex bind data. Call once, not per frame."""
        bind_idx = np.asarray(bind_idx, dtype=np.int64)
        bind_w = np.asarray(bind_w, dtype=np.float64)
        self.n_render = int(bind_idx.shape[0])

        packed = np.zeros((self.n_render, 4), dtype=np.float64)
        packed[:, 0] = bind_idx
        packed[:, 1:] = bind_w[:, 1:]  # w0 is recovered as 1 - w1 - w2 - w3

        width, height = texture_shape(self.n_render)
        image = np.zeros((height, width, 4), dtype=np.float32)
        image.reshape(-1, 4)[: self.n_render] = packed.astype(np.float32)

        # Keep the CPU copy: this texture has been observed to lose its
        # contents after a verified upload while its siblings stayed intact,
        # so skin() re-checks it before every dispatch.
        self._bind_image = image
        self.tex_bind = upload_verified(image)
        self.tex_skin = blank(self.n_render)
        self.sh_skin = kernels.build(
            "skin", kernels.SKIN_SRC,
            [("RGBA32F", "FLOAT_2D", "x", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "tets", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "bind", {"READ"}),
             ("RGBA32F", "FLOAT_2D", "out_pos", {"WRITE"})],
            [("INT", "n_render"), ("FLOAT", "mark")],
        )

    def skin(self) -> np.ndarray:
        """Blend render vertices out of the cage and read back only those.

        This is the per-frame readback: R texels, never N. Bind indices refer
        to the original tet order, so the unpermuted tets texture is bound
        here rather than the colour-ordered one used by solve.
        """
        if not hasattr(self, "sh_skin"):
            raise RuntimeError("attach_render() must be called before skin()")

        # Re-dispatch on a stale read, do not merely re-read. Measured: 16
        # consecutive reads of a stale texture all returned its initial
        # contents, while re-issuing the dispatch recovered immediately.
        for attempt in range(1, 9):
            self._skin_mark += 1
            mark = float(self._skin_mark)

            # Verifying the OUTPUT is not enough - the bind texture itself has
            # been caught holding data that is not what was uploaded, with
            # tets and x in the same solver still correct. A corrupt input
            # produces a confidently-marked, completely wrong result that no
            # amount of re-dispatching can fix.
            current_bind = download(self.tex_bind)
            if not np.array_equal(current_bind, self._bind_image):
                print("marrow: bind texture lost its contents, re-uploading")
                self.tex_bind = upload_verified(self._bind_image)

            self.sh_skin.bind()
            self.sh_skin.image("x", self.tex_x)
            self.sh_skin.image("tets", self.tex_tets_orig)
            self.sh_skin.image("bind", self.tex_bind)
            self.sh_skin.image("out_pos", self.tex_skin)
            self.sh_skin.uniform_int("n_render", self.n_render)
            self.sh_skin.uniform_float("mark", mark)
            gpu.compute.dispatch(self.sh_skin, _groups(self.n_render), 1, 1)

            flush(self.sh_flush, self.tex_skin)
            array = read_marked(self.tex_skin, mark, self.n_render)
            if array is not None:
                if attempt > 1:
                    # Visible on purpose: a silent retry would let this rot.
                    print(f"marrow: skin readback recovered on attempt {attempt}")
                out = unpack_vec3(array, self.n_render)
                return _guard_finite(out, "render vertices", self.n_render)

        raise StaleReadError(
            f"Marrow: the skin texture never reported its generation mark "
            f"after 8 re-dispatches. The GPU queue appears wedged."
        )
