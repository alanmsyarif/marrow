# Marrow Simulation Driver and Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn the working GPU solver into a usable addon — settings, a bake that drives the solver over a frame range, playback by scrubbing, and the spec's remaining failure rows.

**Architecture:** A `MarrowSession` owns a `GPUSolver` plus a baked cache of render-vertex positions, one float32 array per frame. Baking runs the solver forward and stores `skin()` output. A `frame_change_post` handler writes the cached frame into the render mesh with `foreach_set`. Nothing simulates during playback — scrubbing is a cache lookup, which is what makes it interactive.

**Tech Stack:** Blender 5.2 `bpy`, numpy, the existing `marrow.gpu` stack.

## Scope

Build step 4, trimmed to what a working build needs. Material presets are **cut** — they are a dict of slider values, they change no behaviour, and they can land any time. Vulkan validation and GUI-mode spike confirmation are environment work, not build work.

## Global Constraints

- Everything from the previous plans still holds: `marrow/core/` stays bpy-free; `marrow/__init__.py` imports `bpy` only inside `register()`/`unregister()`.
- **Never hold GPU objects in module-level state.** Module globals are collected after Blender tears down the GPU context — measured, that crashes Blender at shutdown. Sessions live in a module dict keyed by object name, but the dict holds sessions whose GPU objects are released explicitly by `free()`.
- Readback is ~4% unreliable (no barrier API). Bake must validate what it stores; a NaN or a stale frame must not enter the cache.
- Product name is "Marrow" in user-facing strings.

---

## File Structure

| Path | Responsibility |
|---|---|
| `marrow/blender/session.py` | `MarrowSession`: solver + per-frame cache, bake, playback write |
| `marrow/blender/handlers.py` | `frame_change_post` registration, cache playback |
| `marrow/blender/ops.py` | add `MARROW_OT_bake`, `MARROW_OT_free` |
| `marrow/blender/ui.py` | simulation settings and bake buttons |
| `marrow/__init__.py` | register the new classes and the handler |
| `tests/blender/test_session.py` | bake, playback, NaN refusal, budget refusal |
| `tests/blender/test_bake_ops.py` | end-to-end through the operators |

---

### Task 1: MarrowSession — solver ownership, bake, cache

**Files:** Create `marrow/blender/session.py`; test `tests/blender/test_session.py`

**Interfaces:**
- `MAX_NODES = 200_000`
- `MarrowSession(obj)` built from a tetrahedralised object, raising `ValueError` naming the count if the cage exceeds `MAX_NODES`
- `.bake(frame_start, frame_end) -> int` returning frames stored; refuses to store a non-finite frame
- `.frame_positions(frame) -> np.ndarray | None`
- `.write_to_mesh(obj, frame) -> bool`
- `.free() -> None` dropping GPU references
- `.baked_range -> tuple[int, int] | None`

- [ ] **Step 1:** Write `tests/blender/test_session.py` covering: bake stores one array per frame; positions differ from rest after baking under gravity; `frame_positions` outside the baked range returns `None`; a cage over `MAX_NODES` is refused with the count in the message; `write_to_mesh` moves the render vertices; `free()` makes the session unusable rather than silently wrong.
- [ ] **Step 2:** Run the Blender suite, confirm `ModuleNotFoundError: marrow.blender.session`.
- [ ] **Step 3:** Implement `session.py`.
- [ ] **Step 4:** Run, confirm green.
- [ ] **Step 5:** Commit.

---

### Task 2: Frame handler and playback

**Files:** Create `marrow/blender/handlers.py`; extend `tests/blender/test_session.py`

**Interfaces:**
- `SESSIONS: dict[str, MarrowSession]` module-level, holding sessions only (GPU objects released via `free()`)
- `register_handler()` / `unregister_handler()`, both idempotent
- `on_frame_change(scene, depsgraph=None)` writing the cached frame for every session

- [ ] **Step 1:** Tests: handler registration is idempotent; scrubbing to a baked frame updates the mesh; scrubbing outside the baked range leaves the mesh alone; `unregister_handler` removes exactly Marrow's handler and leaves others.
- [ ] **Step 2:** Run, confirm red.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run, confirm green.
- [ ] **Step 5:** Commit.

---

### Task 3: Simulation settings and bake operators

**Files:** Modify `marrow/blender/ui.py`, `marrow/blender/ops.py`, `marrow/__init__.py`; test `tests/blender/test_bake_ops.py`

**Interfaces:**
- `MarrowSettings` gains `substeps` (int, default 10, min 1, max 100), `stiffness` (float, default 1e4), `volume_preservation` (float, default 1e5), `damping` (float, default 0.999, 0..1), `ground_enabled` (bool), `ground_z` (float)
- `MARROW_OT_bake` (`marrow.bake`) and `MARROW_OT_free` (`marrow.free`)
- Panel gains a Simulation box with the sliders and both buttons

- [ ] **Step 1:** Tests: bake on an untetrahedralised object reports an error naming Tetrahedralize; bake after tetrahedralize stores the scene frame range; the mesh moves when scrubbing after bake; free clears the session.
- [ ] **Step 2:** Run, confirm red.
- [ ] **Step 3:** Implement operators and settings.
- [ ] **Step 4:** Run, confirm green.
- [ ] **Step 5:** Commit.

---

### Task 4: Remaining spec failure rows

**Files:** Modify `marrow/__init__.py`, `marrow/blender/ops.py`; extend tests

**Rows still open from the spec's failure table:**

| Failure | Response |
|---|---|
| Node count over budget | Refuse with the actual count, never hang |
| No usable GPU context | Disable at register with a plain message |
| Solver produces NaN | Already handled by `MarrowNaNError`; bake must surface it as an operator error, not a traceback |

- [ ] **Step 1:** Tests: a bake whose solver raises `MarrowNaNError` reports `{"ERROR"}` rather than propagating; `gpu_available()` returns False and `register()` still succeeds when the GPU probe fails.
- [ ] **Step 2:** Run, confirm red.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run, confirm green.
- [ ] **Step 5:** Commit.

---

## Self-Review Notes

**Cut deliberately:** material presets (no behaviour change), Vulkan validation and GUI-mode spike confirmation (environment, not build), and any further work on the ~4% stale readback beyond making bake validate what it stores.

**The cache is in memory, not in the .blend.** Persisting it would mean either a large ID-property blob or an external file, and neither is needed to make the addon work end to end. Rebaking is fast. This is a deliberate v1 limit, not an oversight.
