"""Advancing bodies that collide with each other.

Two bodies stepped a whole frame at a time only ever see each other's
end-of-frame state. At 24fps a body at 5 m/s covers 0.2m in a frame against a
default thickness of 0.1m at Resolution 0.1, so it would pass clean through.
Everything here exists to interleave bodies at substep granularity instead.

Every session goes through this module, including one with no partners. A
solo body is a group of one, which keeps a single code path rather than a
fast path and a coupled path that drift apart as one of them is maintained.

Membership is sorted by object name, never by who asked. Dictionary order
would make a three-body pile settle differently depending on which mesh the
frame handler happened to reach first.
"""

import bpy

from ..core.progress import drain

_warned: set = set()


def _couples(session) -> bool:
    return (
        not session._freed
        and not session.baked
        and session.live
        and float(getattr(session, "body_distance", 0.0)) > 0.0
    )


def members_of(session) -> list:
    """Every live session this one must be advanced alongside."""
    if float(getattr(session, "body_distance", 0.0)) <= 0.0:
        return [session]

    from .handlers import SESSIONS

    group = [s for s in SESSIONS.values() if _couples(s)]
    if session not in group:
        group.append(session)
    return sorted(group, key=lambda s: s.object_name)


def _substeps_of(members) -> int:
    """The count the whole group runs at.

    Interleaving needs a common substep length. The maximum is the safe
    direction - more substeps is stabler, never less - but it does mean a
    body's own setting can be raised by the company it keeps, so say so.
    """
    counts = [int(m.solver.params.substeps) for m in members]
    highest = max(counts)
    for member, own in zip(members, counts):
        if own == highest:
            continue
        key = (member.object_name, own, highest)
        if key not in _warned:
            _warned.add(key)
            print(
                f"marrow: {member.object_name!r} collides with a body at "
                f"{highest} substeps, so it runs at {highest} rather than "
                f"its own {own}"
            )
    return highest


def _step_group(members, frame: int) -> None:
    """One frame for every member, interleaved substep by substep."""
    solvers = [m.solver for m in members]
    substeps = _substeps_of(members)
    # dt is the scene's frame length and the same for every body.
    h = members[0].solver.params.dt / substeps

    gap = thickness_for(members)
    for member in members:
        if member.collider_objects:
            member.solver.colliders = member._collider_specs()
        # Attachment targets are sampled once per frame, like collider
        # transforms, and held constant across the frame's substeps. A
        # no-op for members without the pass.
        member._refresh_targets()
        # Both sides of a contact must open to the same distance.
        member.solver.body_distance = gap

    for _ in range(substeps):
        # Constraints for everyone, then integration for everyone. Only
        # integrate writes the tex_x that the body-collision pass samples its
        # partner from, so splitting the substep this way is what stops the
        # body walked first from absorbing more of every contact than the
        # body walked last.
        for i, member in enumerate(members):
            member.solver.substep_constraints(
                h, [s for j, s in enumerate(solvers) if j != i]
            )
        for member in members:
            member.solver.substep_integrate(h)

    for member in members:
        member.cache_frame(frame)


def _restart(members, frame_start: int) -> None:
    for member in members:
        member._clear_cache()
        member.refresh_from_object()
        member._build_solver()
        member._last_simulated = frame_start - 1


def advance(session, frame: int, frame_start: int):
    """Positions for ``frame``, simulating the group forward when live.

    Returns None when the frame cannot be served: before the start, or after
    a jump too large to catch up with - unless nothing has been simulated
    yet, in which case the jump is caught up from the start instead.
    """
    session._check_live()
    frame, frame_start = int(frame), int(frame_start)

    # A baked cache is played back, never regenerated - a scrub below the
    # start frame included, which is the one thing that resets a live body.
    if session.baked or not session.live:
        return session._cache.get(frame)

    members = members_of(session)

    # Flipping Attachment or its stiffness must rebuild on the next frame
    # change, not wait for a manual trip to the start frame: the mute hides
    # the driving modifiers the instant the box is ticked, so a body that is
    # muted but not yet bone-driven reads as broken.
    #
    # The pin settings ride along for a plainer reason - they sit in the same
    # panel box as Attachment, so leaving them out made one checkbox take
    # effect at once and the checkbox under it silently do nothing until the
    # timeline was scrubbed back. Measured: Follows Animation ticked at frame
    # 15, played on to 24, solver still running the old flag. Both of these
    # change what _build_solver produces, which is the same reason Attachment
    # qualifies.
    for m in members:
        obj = bpy.data.objects.get(m.object_name)
        settings = getattr(obj, "marrow", None) if obj is not None else None
        if settings is not None and (
            bool(settings.attach_enabled) != m.attach_enabled
            or float(settings.attach_stiffness) != m.attach_stiffness
            or str(settings.pin_group) != m.pin_group
            or bool(settings.pin_follows) != m.pin_kinematic
        ):
            _restart(members, frame_start)
            break

    if frame < frame_start:
        # There is no frame to serve down here, and leaving the last
        # simulated pose on screen reads as a simulated state when it is a
        # stale one. Reset instead, and hand back the rest shape so the
        # viewport shows it. Playback running or paused makes no difference:
        # the handler sees a scrub either way.
        _restart(members, frame_start)
        return session.solver.skin()

    # Returning to the start always restarts, even if that frame is already
    # cached. That is what makes edited sliders take effect without having to
    # free the cache by hand. Members that disagree on how far they have got
    # restart too - that is a body which has just joined the group, and there
    # is no sound way to splice it into a simulation already in progress.
    marks = {m._last_simulated for m in members}
    if None in marks or len(marks) > 1 or frame == frame_start:
        _restart(members, frame_start)
    elif all(m._cache.get(frame) is not None for m in members):
        return session._cache.get(frame)

    last = members[0]._last_simulated
    gap = frame - last
    if gap <= 0:
        return None
    if gap > session.MAX_CATCHUP and any(m._cache for m in members):
        # A long scrub jump into an already-simulated body is not chased:
        # catching up hundreds of frames inside a frame handler would stall
        # the UI, and a half-simulated mesh is worse than an untouched one.
        return None
    # A large gap with an empty cache is a body that never simulated landing
    # mid-timeline - a file reopened at frame 88, play pressed there. It used
    # to sit at rest forever; with no history to protect, catching up from
    # the start once is exactly what a bake would have done.
    for step_frame in range(last + 1, frame + 1):
        _step_group(members, step_frame)
    return session._cache.get(frame)


def bake(members, frame_start: int, frame_end: int, scene=None) -> int:
    """Simulate frame_start..frame_end for a whole group, caching each frame.

    Rebuilds every solver first so a bake always starts from rest rather than
    from wherever a previous bake left the cages.
    """
    return drain(bake_iter(members, frame_start, frame_end, scene=scene))


def bake_iter(members, frame_start: int, frame_end: int, scene=None):
    """bake as a generator, yielding 0..1 after each simulated frame.

    One frame per yield is the natural slice: it is the unit the cache is
    keyed by, so a bake stopped partway leaves whole frames rather than a
    half-integrated one. The operator uses that - Esc keeps what has been
    simulated instead of discarding the wait.
    """
    members = list(members)
    samples_scene = any(
        m.collider_objects or m.attach_active for m in members
    )
    if scene is not None and samples_scene:
        # Rebuilds sample the current frame - collider transforms and the
        # attachment start pose - so park the scene at the start frame
        # first or a bake begun mid-timeline would seed from the wrong pose.
        scene.frame_set(int(frame_start))
    for member in members:
        member._check_live()
        member._cache.clear()
        member.refresh_from_object()
        member._build_solver()
        member.live = False
        member.baked = True

    first, last = int(frame_start), int(frame_end)
    total = max(last - first + 1, 1)
    reached = first - 1
    try:
        for frame in range(first, last + 1):
            if scene is not None and samples_scene:
                # Re-sample collider transforms and attachment targets so
                # animation works. Without this a falling ball would sit still
                # for the whole bake.
                scene.frame_set(frame)
            _step_group(members, frame)
            reached = frame
            if frame < last:
                yield (frame - first + 1) / total
    finally:
        # Also runs when the generator is closed partway, so an interrupted
        # bake still reports how far it actually got and the frames already
        # cached stay playable.
        for member in members:
            member._last_simulated = reached

    yield 1.0
    return len(members[0]._cache)


def partners_in_scene(obj) -> list:
    """Other objects that would share ``obj``'s group, for a bake.

    Bake cannot read the live SESSIONS dictionary - it clears it - so group
    membership at bake time comes from the scene instead.
    """
    import bpy

    from .session import find_cage

    settings = getattr(obj, "marrow", None)
    if settings is None or not settings.body_collision:
        return []
    found = []
    for other in bpy.data.objects:
        if other is obj or other.type != "MESH":
            continue
        other_settings = getattr(other, "marrow", None)
        if other_settings is None or not other_settings.body_collision:
            continue
        if find_cage(other) is None:
            continue
        found.append(other)
    return sorted(found, key=lambda o: o.name)


def thickness_for(sessions) -> float:
    """The gap a group opens at every contact.

    Two bodies can differ in Resolution and so in absolute thickness. Both
    sides have to agree or the larger one does all the work, so the group
    takes the largest.
    """
    distances = [float(getattr(s, "body_distance", 0.0)) for s in sessions]
    return max(distances) if distances else 0.0
