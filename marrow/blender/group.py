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
        member._cache.clear()
        member.refresh_from_object()
        member._build_solver()
        member._last_simulated = frame_start - 1


def advance(session, frame: int, frame_start: int):
    """Positions for ``frame``, simulating the group forward when live.

    Returns None when the frame cannot be served: before the start, or after
    a jump too large to catch up with.
    """
    session._check_live()
    frame, frame_start = int(frame), int(frame_start)

    # A baked cache is played back, never regenerated.
    if session.baked or not session.live or frame < frame_start:
        return session._cache.get(frame)

    members = members_of(session)

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
    if gap <= 0 or gap > session.MAX_CATCHUP:
        return None

    for step_frame in range(last + 1, frame + 1):
        _step_group(members, step_frame)
    return session._cache.get(frame)


def bake(members, frame_start: int, frame_end: int, scene=None) -> int:
    """Simulate frame_start..frame_end for a whole group, caching each frame.

    Rebuilds every solver first so a bake always starts from rest rather than
    from wherever a previous bake left the cages.
    """
    members = list(members)
    for member in members:
        member._check_live()
        member._cache.clear()
        member.refresh_from_object()
        member._build_solver()
        member.live = False
        member.baked = True

    samples_colliders = any(m.collider_objects for m in members)
    for frame in range(int(frame_start), int(frame_end) + 1):
        if scene is not None and samples_colliders:
            # Re-sample collider transforms so animated colliders work.
            # Without this a falling ball would sit still for the whole bake.
            scene.frame_set(frame)
        _step_group(members, frame)

    for member in members:
        member._last_simulated = int(frame_end)
    return len(members[0]._cache)


def partners_in_scene(obj) -> list:
    """Other objects that would share ``obj``'s group, for a bake.

    Bake cannot read the live SESSIONS dictionary - it clears it - so group
    membership at bake time comes from the scene instead.
    """
    import bpy

    from .session import CAGE_SUFFIX

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
        if bpy.data.objects.get(f"{other.name}{CAGE_SUFFIX}") is None:
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
