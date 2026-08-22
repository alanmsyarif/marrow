# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary reader is a **Blender artist mid-shot**: Marrow is already
installed, a soft body is behaving wrong, and they are looking for the one
setting that fixes it. They arrive with a symptom, not a topic — a tentacle
tip spiking, a body falling through a collider, a wave that reads like
clockwork, fibers that refuse to bake. Lookup speed and troubleshooting
outrank orientation.

A second, smaller reader is deciding whether Marrow can do what a shot needs
before committing to it. They are served by the same content read top-down,
not by separate marketing.

## Product Purpose

Marrow is a Blender 5.2 extension that fills a mesh with a tetrahedral cage,
solves a stable neo-Hookean XPBD system in GLSL compute shaders, and deforms
the original render mesh by barycentric interpolation from that cage.

Blender ships XPBD for hair, cloth and particles and has no volumetric soft
body; the legacy `SOFT_BODY` modifier is a surface spring lattice with no
volume preservation. Marrow fills that gap.

Success for the documentation is that a reader with a symptom reaches the
paragraph that explains it without reading anything else.

## Positioning

**It never modifies your topology.** The cage is a separate hidden object and
the render mesh is deformed by interpolation from it, so UVs, shape keys and
material slots survive, and De-tetrahedralize returns the modelled mesh bit
for bit. A conforming-Delaunay tetrahedralizer cannot make that promise.

Second: **pure Python against Blender's bundled interpreter.** No compiled
extension, no external dependency, no per-platform build, and never a
`pip install` into Blender's Python.

## Operating Context

Read on a desktop beside a running Blender, usually with a shot open and a
simulation misbehaving. The reader alt-tabs in, finds a setting, alt-tabs
back. The vocabulary is Blender's: viewport sidebar, vertex group, weight
paint, collection, modifier stack, armature, Geometry Nodes, point cache.

## Capabilities and Constraints

- 29 documented sections covering cage construction, contact, material
  failure, fiber activation, attachment, pinning, force fields and internals.
- Every performance and behaviour claim in the documentation is a measurement,
  most on an RTX 5050, several against a float64 numpy oracle at 2e-5.
- Honest limitations are documented, not hidden: node-against-node contact
  only, deforming colliders baked once, no plasticity, OpenGL-only validation.
- The site is a **single static HTML file**, `docs/index.html`, served from
  GitHub Pages with no build step, no framework and no bundler.
- Google Fonts is the only permitted external request; everything else must be
  inline. The page must open correctly straight off the filesystem.

## Brand Commitments

Name: **Marrow**. Voice: measured, specific, unhedged — it states what was
measured and what it measured against, names its own failures plainly, and
never claims what it has not tested. No logo, no wordmark, no existing brand
system.

## Evidence on Hand

- `README.md` — 462 lines, the complete source of truth for every claim.
- Measured tables: cage coverage, self-collision cost, fiber noise
  (three separate tables), pinning travel, attach stiffness.
- Named failure numbers: 219 of 461 nodes seized on a sticky overlap,
  226 m/s off a ground-plane straddle, &minus;64.37 against &minus;5.11 on
  a failure test.
- **No images of any kind exist.** No renders, no viewport captures, no
  footage. The site must carry itself on typography, structure and diagrams
  it draws itself, and must not fabricate or imply imagery that does not
  exist.

## Product Principles

1. **A symptom must reach its paragraph.** Structure serves lookup before it
   serves reading order.
2. **The measurements are the credibility.** Anything that buries, decorates
   past legibility, or paraphrases a measured number is a regression.
3. **State limits as loudly as capabilities.** The limitations section is
   product, not disclaimer.
4. **Never fabricate evidence.** No stand-in imagery, no invented benchmarks,
   no implied users.
5. **Stay one static file.** Instant open, no build, no dependency.

## Accessibility & Inclusion

No product-specific standard was established. Baseline applies: legible
contrast in both themes, visible keyboard focus, honoured
`prefers-reduced-motion`, and no information carried by colour alone —
the measured tables in particular must read without it.
