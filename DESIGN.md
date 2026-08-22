# Design

Written from the built surface, not ahead of it. The only implementation is
`docs/index.html`, a single static file with no build step.

Direction: **The Anatomical Plate**, seed key `1aefbc06`. Visitor mode:
**Read**. The reader arrives mid-shot with a symptom, so the sheet opens on
the body cut open and a diagnostic key rather than on an introduction.

## Ground and light

**Single theme, dark, deliberately.** The use scene decides it: a Blender
artist at a desk in the evening, Blender's dark viewport filling the main
monitor, the docs on the second screen. A light sheet beside that viewport is
a flashbulb. There is no theme toggle, because a committed world does not
offer to be a different one.

Every colour is painted explicitly from a token, so the page holds whatever
ground a host paints behind it.

## Palette

Specimen ground, bone ink, ochre keys, carmine injected.

| Token | Value | Role |
| --- | --- | --- |
| `--ground` | `#0C110F` | the sheet. Specimen-jar green-black, never pure black |
| `--plate` | `#121815` | a plate's field: figures, tables, code, marginalia |
| `--plate-2` | `#182019` | table heads, inline code, row hover |
| `--rule` | `#26302B` | plate borders and section rules |
| `--rule-soft` | `#1B2320` | rules inside a table body |
| `--ink` | `#E9E4D6` | bone. Body text, figure contours. 15.4:1 on ground |
| `--ink-2` | `#A6AEA4` | fascia. Secondary prose, key entries. 8.3:1 |
| `--ink-3` | `#828D86` | faint labels, captions, rig lines. 5.2:1 |
| `--ochre` | `#D2A24C` | **accent.** Keys, leader lines, links, the read position |
| `--ochre-dim` | `#8A6C33` | the lattice, hairline leaders, list markers |
| `--ochre-wash` | `#1D1D14` | the lit row in a legend |
| `--carmine` | `#DD6157` | **alarm.** Injected: hazards, the fiber axis, hot stretch |
| `--carmine-dim` | `#8E3A34` | hazard borders |
| `--carmine-wash` | `#1E1210` | hazard field |

**Strategy: restrained.** Neutrals carry the sheet; ochre is the only
structural colour; carmine appears in exactly six places, all of which are
things that ruin a shot.

**Nothing is encoded by colour alone.** Every hazard carries a drawn triangle
and a label; every measured plate carries a roman numeral and a rig line;
the read position in the key carries a leader line as well as a hue.

## Type

| Role | Face | Used for |
| --- | --- | --- |
| Display | **Schibsted Grotesk** 500/700 | wordmark, headings, all UI labels, table type |
| Text | **Literata** 400/600, italic | body prose, captions, latin terms |
| Data | **Spline Sans Mono** 400/500 | code, measured numbers, plate numerals, rig lines |

Mono is for code, data and measurement only — never as a costume for
"technical". Nav, labels and buttons are Schibsted Grotesk.

Scale: wordmark `clamp(52px, 8.5vw, 96px)` at `-0.04em`; h2
`clamp(26px, 3.1vw, 34px)`; h3 19px; h4 11px uppercase tracked `.15em`. Body
16.5px / 1.7 at a 68ch measure. Latin part names are italic; their glosses
are not.

## Components

- **Plate.** A rule-topped header carrying a roman numeral, a title in tracked
  caps, and — right-aligned — the rig the numbers came off. Then the figure or
  table. Then an optional caption at `--ink-3`. Plates run I to IX; I and IV
  are drawn, the rest are measured tables.
- **The key.** The left rail is the plate's legend, not a generic sidebar.
  A hairline ochre leader scales out from the rule to the entry you are
  reading (`transform: scaleX`, never a width transition). Grouped by region
  of the body of work; sift field filters entries and hides empty groups.
- **Diagnostic key.** Symptom, reading, and where to turn. The first section,
  because it is what the primary reader came for.
- **Marginalia** (`.aside`) and **hazards** (`.hazard`): a bordered field with
  a drawn diamond or warning triangle and a tracked caps label. Hazards take
  the carmine border and field. No coloured rail above 1px anywhere.
- **Specimen tables.** Full-bleed inside their plate, `overflow-x: auto`,
  tabular numerals, right-aligned numeric columns, row hover.

## Motion

One authored moment: **Plate I's five leader lines draw in once** on load,
staggered 90 ms apart, exponential ease-out. Nothing else animates on
entrance.

Two continuous elements, both of which are content rather than decoration:
Plate IV runs the actual fiber activation on a canvas, and the key's leader
scales to follow the read position. `prefers-reduced-motion` kills every
transition, skips the draw-on, and renders Plate IV as one static frame.

## Signature interaction

Hovering **or focusing** a numbered term in Plate I's legend dims the figure
to 22% and lifts its part back to full. Pointer and keyboard behave
identically, because the legend entries are real buttons.

## Browser surfaces

Themed rather than left default: `::selection` (ochre on ground), caret
colour, scrollbar track and thumb (both `scrollbar-color` and the WebKit
pseudo-elements), `:focus-visible` rings, underline offset and thickness, and
`font-variant-numeric: tabular-nums` on every numeric column.

## Constraints this world is built under

- One static file. No build step, no framework, no bundler.
- Google Fonts is the only external request. Real fallback stacks are declared.
- **No imagery exists for this product.** Every figure on the sheet is drawn
  in SVG or canvas from the mechanism it describes. Nothing is a stand-in for
  a render that does not exist, and nothing implies one.
- The measured tables are the credibility. Any change that buries a
  measurement, drops its rig, or paraphrases a number is a regression.

## Known and accepted

- The left rail is the least-transformed element on the sheet. It stays a
  rail because lookup speed is the one thing the redesign was not allowed to
  cost; the leader-line device is what makes it the plate's legend rather
  than a docs sidebar.
- Plate IV is blank without JavaScript. Its caption carries the explanation.
- The mechanical detector ran in degraded mode (its HTML parser modules are
  not installed), so computed-contrast and selector checks were done by hand
  against the values in the table above rather than by the tool.
