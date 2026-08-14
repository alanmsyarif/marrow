# Marrow: De-tetrahedralize

Design spec, 2026-08-13.

## Summary

One operator that takes a simulated object back to the mesh the user started with: restore the original vertex positions, strip the cage and every attribute Marrow wrote, and release the session.

The restore is the part that does not exist yet. Everything else is deletion.

## Motivation

Marrow is currently a one-way door. `MARROW_OT_tetrahedralize` writes bind attributes onto the mesh and builds a cage object, and `MarrowSession.write_to_mesh` writes simulated positions straight into `obj.data.vertices`. Nothing anywhere records what the mesh looked like before.

Two consequences, both live today:

1. **`Free Bake` leaves the object deformed.** It drops the GPU session and the cache, and the mesh keeps whatever pose the last simulated frame left it in. There is no way back short of undo.
2. **Re-tetrahedralizing a deformed object bakes the deformation in permanently.** The operator reads `obj.data.vertices` to build the bind, so changing Resolution after playing the timeline silently makes the deformed pose the new rest shape. Do it twice and the drift compounds.

Both are the same missing piece: a rest pose.

## Design decisions

### D1. Rest positions live in a POINT attribute on the render mesh

`marrow_rest`, FLOAT_VECTOR, POINT domain, in object space. The same mechanism `storage.py` already uses for `marrow_bind_idx` and the four weight attributes, so it survives save and load, follows the mesh datablock, and needs no new registration.

Rejected: a duplicate mesh datablock as a backup. Heavier, and it introduces a second copy of the topology that can fall out of sync with the original. Rejected: an ID property holding a flat list - attributes are the established path here and are `foreach_get`-able.

Object space, not world space, because that is what `obj.data.vertices` holds and a restore should not depend on the object transform having stayed put.

### D2. Tetrahedralize restores before it rebuilds

If `marrow_rest` is already present, the operator writes it back into the mesh before reading vertices to build the cage and the bind. That makes changing Resolution mid-project safe, and it is the whole fix for the second bug above.

Consequence worth stating: tetrahedralizing an object that already has a cage always rebuilds from the **original** shape, never from the current one. There is deliberately no way to promote a deformed pose to the new rest shape - that is what Apply on a modifier is for, and Marrow is not a modifier.

### D3. De-tetrahedralize removes everything, in a fixed order

1. Free the session and drop it from `handlers.SESSIONS`.
2. Clear `live_enabled`, so the frame handler does not immediately rebuild one.
3. Restore vertex positions from `marrow_rest`.
4. Remove `marrow_rest`, `marrow_bind_idx` and the four `marrow_bind_w*` attributes.
5. Delete the cage object, and its mesh if nothing else uses it.

The session goes first. Restoring positions while a live session is still in `SESSIONS` would have the next frame change overwrite them.

Cage deletion reuses the removal already written inline in `MARROW_OT_tetrahedralize`, which is extracted so both callers share it rather than the second one being written again slightly differently.

### D4. It is safe to run on an object that was never tetrahedralized

Reports and cancels rather than raising. Each removal step is individually tolerant of the thing already being absent, so a half-removed object - a cage deleted by hand in the outliner, say - is still cleanable.

### D5. Naming

Label **De-tetrahedralize**, sitting under Tetrahedralize in the Cage box, drawn only when a cage exists. Description: "Remove the cage and restore the object's original shape".

Not "Free", which is taken and means something narrower: `MARROW_OT_free` discards the cache and the GPU memory but leaves the object tetrahedralised and ready to simulate again.

`Free Bake` is left as it is. Restoring the rest pose there would be a surprise for anyone freeing memory mid-session, and the two operators reading differently is the point.

## Components

### `marrow/blender/storage.py`

`REST_KEY = "marrow_rest"`. Three functions:

- `write_rest(mesh)` - capture current vertex positions into the attribute. No-op overwrite is fine; callers decide when.
- `read_rest(mesh)` - `(N, 3)` float64, or `None` when the attribute is absent.
- `clear_marrow_data(mesh)` - remove the rest and bind attributes, tolerating any of them being missing.

`_ensure_attr` already handles the create-or-replace case and gains `FLOAT_VECTOR` use.

### `marrow/blender/ops.py`

- `remove_cage(obj)` - extracted from the tetrahedralize operator, deletes `{name}_marrow_cage` and its mesh if unused. Returns whether it found one.
- `MARROW_OT_tetrahedralize` - restore from `marrow_rest` if present, then `write_rest` before building the bind.
- `MARROW_OT_detetrahedralize` - D3, in order.

### `marrow/blender/ui.py`

A De-tetrahedralize button in the Cage box, drawn only when the cage object exists.

## Testing

Blender suite:

1. Tetrahedralize writes `marrow_rest`, matching the pre-tetrahedralize vertex positions exactly.
2. **Round trip: tetrahedralize, simulate frames that visibly deform the mesh, de-tetrahedralize, and the vertices are bit-identical to the original.** The test this feature exists for.
3. De-tetrahedralize removes the cage object, its mesh datablock, and all six attributes.
4. De-tetrahedralize drops the session from `SESSIONS` and clears `live_enabled`.
5. De-tetrahedralize on a plain mesh reports and cancels, and does not raise.
6. De-tetrahedralize on an object whose cage was deleted by hand still cleans the attributes.
7. Re-tetrahedralizing after a deforming simulation produces the same cage as tetrahedralizing the undeformed original - the D2 bug, which fails before this change.
8. A second de-tetrahedralize is a no-op that reports rather than raising.

## Ceilings

- **Topology must not change between tetrahedralize and de-tetrahedralize.** The rest attribute is per-point, so adding or deleting vertices in Edit Mode invalidates it. Blender drops attribute values for new points, which restores garbage rather than failing loudly. Not guarded; re-tetrahedralize after editing.
- **Undo is still the only route back from a manual mesh edit**, not this.
