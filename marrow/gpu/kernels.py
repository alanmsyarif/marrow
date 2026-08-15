"""GLSL compute kernels and their construction.

Every kernel is a 1D dispatch over elements and does its own bounds check,
so a dispatch rounded up to a whole workgroup cannot write past the end.
The texel() helper must stay identical to marrow.core.layout.texel_index -
if the two ever disagree the solver reads someone else's data and nothing
raises.
"""

import gpu

TEX_WIDTH = 4096

TEXEL_GLSL = f"""
const int TEX_WIDTH = {TEX_WIDTH};

ivec2 texel(int i)
{{
  return ivec2(i % TEX_WIDTH, i / TEX_WIDTH);
}}
"""


def build(name, source, images, push_constants, group_size: int = 64):
    """Compile one compute kernel, or raise with the driver log intact."""
    info = gpu.types.GPUShaderCreateInfo()
    # All three sizes stated. Leaving z to its -1 default emits a bare
    # `local_size_z` with no value, which some drivers accept and NVIDIA's
    # GLSL compiler rejects outright: "C3011: layout qualifier 'local_size_z',
    # requires 'a non-negative integer'". Every kernel here fails to build on
    # those cards without this.
    info.local_group_size(group_size, 1, 1)
    for slot, (fmt, kind, image_name, qualifiers) in enumerate(images):
        info.image(slot, fmt, kind, image_name, qualifiers=qualifiers)
    for const_type, const_name in push_constants:
        info.push_constant(const_type, const_name)
    info.compute_source(TEXEL_GLSL + source)
    try:
        return gpu.shader.create_from_info(info)
    except Exception as exc:
        raise RuntimeError(
            f"Marrow kernel {name!r} failed to compile.\n"
            f"--- driver log ---\n{exc}\n--- end log ---"
        ) from exc


PREDICT_SRC = """
void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_nodes) { return; }
  ivec2 c = texel(i);

  vec4 xi = imageLoad(x, c);
  vec4 vi = imageLoad(v, c);
  float w = xi.w;

  vec3 pos = xi.xyz;
  if (w > 0.0) {
    pos += vi.xyz * h + gravity * (h * h);
  }
  imageStore(p, c, vec4(pos, w));
  // Contact marks from the previous substep are stale; the contact passes
  // rewrite the ones that fire this substep.
  imageStore(mark, c, vec4(0.0));
}
"""

SOLVE_SRC = """
// Stable neo-Hookean, one deviatoric and one hydrostatic constraint per tet.
// Transcribed from marrow/core/solver_ref.py:solve_constraints. The two must
// stay in step: the oracle is the only way a sign error here is detectable.

mat3 shape_matrix(ivec4 idx, vec3 p0)
{
  return mat3(
    imageLoad(p, texel(idx.y)).xyz - p0,
    imageLoad(p, texel(idx.z)).xyz - p0,
    imageLoad(p, texel(idx.w)).xyz - p0
  );
}

void project(ivec4 idx, vec3 g0, vec3 g1, vec3 g2, vec3 g3,
             float c_value, float compliance, float h_step)
{
  vec4 n0 = imageLoad(p, texel(idx.x));
  vec4 n1 = imageLoad(p, texel(idx.y));
  vec4 n2 = imageLoad(p, texel(idx.z));
  vec4 n3 = imageLoad(p, texel(idx.w));

  float denom = n0.w * dot(g0, g0) + n1.w * dot(g1, g1)
              + n2.w * dot(g2, g2) + n3.w * dot(g3, g3);
  float alpha_tilde = compliance / (h_step * h_step);
  denom += alpha_tilde;
  if (denom < 1e-20) { return; }

  // The XPBD multiplier is zeroed every substep and each tet is visited once,
  // so the usual -alpha_tilde*lambda term is provably zero here.
  float dlambda = -c_value / denom;

  imageStore(p, texel(idx.x), vec4(n0.xyz + g0 * (n0.w * dlambda), n0.w));
  imageStore(p, texel(idx.y), vec4(n1.xyz + g1 * (n1.w * dlambda), n1.w));
  imageStore(p, texel(idx.z), vec4(n2.xyz + g2 * (n2.w * dlambda), n2.w));
  imageStore(p, texel(idx.w), vec4(n3.xyz + g3 * (n3.w * dlambda), n3.w));
}

// Largest principal stretch: the biggest singular value of F, which is the
// square root of the largest eigenvalue of F^T F.
//
// This is the honest answer to "how far is this stretched", and the tear
// threshold is quoted as a stretch ratio, so this is what it has to measure.
// ||F||_F was the old test and it conflates all three directions at once: it
// reads sqrt(3) at rest, so a 1.5 threshold sounds like 50% but a real
// volume-preserving uniaxial pull did not fail until 143%.
//
// Closed form for a symmetric 3x3 rather than an iterative SVD - one acos
// beats a loop whose iteration count can differ between drivers.
float max_principal_stretch(mat3 f)
{
  mat3 a = transpose(f) * f;
  float p1 = a[1][0] * a[1][0] + a[2][0] * a[2][0] + a[2][1] * a[2][1];
  float q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
  if (p1 <= 1e-20) {                       // already diagonal
    return sqrt(max(max(a[0][0], a[1][1]), a[2][2]));
  }
  float d0 = a[0][0] - q;
  float d1 = a[1][1] - q;
  float d2 = a[2][2] - q;
  float p = sqrt((d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1) / 6.0);
  mat3 b = (a - q * mat3(1.0)) / p;
  float r = clamp(determinant(b) * 0.5, -1.0, 1.0);
  // The largest of the three roots. F^T F is positive semi-definite so this
  // cannot really be negative, but rounding can push it a hair under zero.
  float eig = q + 2.0 * p * cos(acos(r) / 3.0);
  return sqrt(max(eig, 0.0));
}

void main()
{
  int t = color_begin + int(gl_GlobalInvocationID.x);
  if (t >= color_end) { return; }

  // The torn image carries more than a flag: for a torn tet it holds the
  // volume ratio the tet had at the instant it broke. Zero means intact.
  float torn_vol = imageLoad(torn, texel(t)).r;
  bool is_torn = torn_vol > 0.0;

  ivec4 idx = ivec4(imageLoad(tets, texel(t)));

  vec4 r0 = imageLoad(rest, texel(3 * t));
  vec4 r1 = imageLoad(rest, texel(3 * t + 1));
  vec4 r2 = imageLoad(rest, texel(3 * t + 2));
  mat3 dm_inv = mat3(r0.xyz, r1.xyz, r2.xyz);
  float rest_vol = abs(r0.w);

  vec4 w0 = imageLoad(p, texel(idx.x));
  vec4 w1 = imageLoad(p, texel(idx.y));
  vec4 w2 = imageLoad(p, texel(idx.z));
  vec4 w3 = imageLoad(p, texel(idx.w));
  if (!(w0.w > 0.0 || w1.w > 0.0 || w2.w > 0.0 || w3.w > 0.0)) { return; }

  mat3 dm_inv_t = transpose(dm_inv);

  vec3 p0 = imageLoad(p, texel(idx.x)).xyz;
  mat3 f = shape_matrix(idx, p0) * dm_inv;

  // Checked before either projection and independently of mu, so a body with
  // no deviatoric stiffness can still tear. tear_threshold reads directly as a
  // stretch ratio now: 1.5 means "fails once something is pulled to 1.5x its
  // rest length". Zero or less disables tearing, which is what the
  // oracle-parity tests run with.
  if (!is_torn && tear_threshold > 0.0
      && max_principal_stretch(f) > tear_threshold) {
    float l0 = imageLoad(live, texel(idx.x)).r;
    float l1 = imageLoad(live, texel(idx.y)).r;
    float l2 = imageLoad(live, texel(idx.z)).r;
    float l3 = imageLoad(live, texel(idx.w)).r;

    // Never tear the last intact tet holding a node. A node with no intact tet
    // has no constraint of any kind left: it free-falls, and since the render
    // mesh topology is fixed it drags a spike behind it rather than becoming
    // separate debris. Measured on a stretch shot that tore 1174 of 1400 tets:
    // 324 of 461 nodes orphaned and material streaming 34 units past the
    // plate. With this rule, none, and the shot still shreds.
    //
    // Safe without atomics for the same reason the projections are: a colour's
    // tets are node-disjoint, so no two threads in this dispatch share a
    // counter.
    if (l0 > 1.5 && l1 > 1.5 && l2 > 1.5 && l3 > 1.5) {
      // Record the volume ratio it broke at, not a bare flag. The hydrostatic
      // pass below holds a torn tet there: it may not inflate, and it may not
      // suck itself back to rest volume either. The floor keeps the value
      // positive so it still reads as "torn", even for a tet caught inverted.
      torn_vol = clamp(determinant(f), 0.05, 20.0);
      imageStore(torn, texel(t), vec4(torn_vol));
      imageStore(live, texel(idx.x), vec4(l0 - 1.0));
      imageStore(live, texel(idx.y), vec4(l1 - 1.0));
      imageStore(live, texel(idx.z), vec4(l2 - 1.0));
      imageStore(live, texel(idx.w), vec4(l3 - 1.0));
      is_torn = true;
    }
  }

  // --- deviatoric ---
  // A torn tet stops resisting distortion, for good. That is what tearing
  // means here: the material goes slack instead of springing back.
  if (mu > 0.0 && !is_torn) {
    float c_dev = sqrt(dot(f[0], f[0]) + dot(f[1], f[1]) + dot(f[2], f[2]));
    if (c_dev > 1e-12) {
      mat3 dcdf = f / c_dev;
      mat3 g = dcdf * dm_inv_t;
      vec3 g1v = g[0];
      vec3 g2v = g[1];
      vec3 g3v = g[2];
      vec3 g0v = -(g1v + g2v + g3v);
      project(idx, g0v, g1v, g2v, g3v, c_dev, 1.0 / (mu * rest_vol), h);
    }
  }

  // --- hydrostatic ---
  // Kept even when torn. Breaking material does not create matter, and a torn
  // tet with no volume constraint at all inflates without bound - measured 3.1x
  // cage volume on a stretch test that tore a sixth of its tets.
  //
  // A torn tet targets the volume it broke at rather than its rest volume.
  // Aiming at rest volume instead would have torn material actively suck
  // itself back in, which is a spring, and a torn tet is meant to have no
  // spring left in it.
  //
  // F is rebuilt from the positions the deviatoric pass just moved. Reusing the
  // stale F would linearise the volume constraint about the wrong
  // configuration. The oracle does the same.
  if (lam > 0.0) {
    vec3 hp0 = imageLoad(p, texel(idx.x)).xyz;
    mat3 hf = shape_matrix(idx, hp0) * dm_inv;

    mat3 dcdf = mat3(cross(hf[1], hf[2]), cross(hf[2], hf[0]), cross(hf[0], hf[1]));
    mat3 g = dcdf * dm_inv_t;
    vec3 g1v = g[0];
    vec3 g2v = g[1];
    vec3 g3v = g[2];
    vec3 g0v = -(g1v + g2v + g3v);

    // gamma exists only to cancel the deviatoric term at rest. A torn tet has
    // no deviatoric term left, so leaving it at 1 + mu/lam would have torn
    // material creep 10% larger every substep and never stop.
    float gamma = is_torn ? torn_vol : (1.0 + mu / lam);
    float c_hyd = determinant(hf) - gamma;
    project(idx, g0v, g1v, g2v, g3v, c_hyd, 1.0 / (lam * rest_vol), h);
  }
}
"""

INTEGRATE_SRC = """
void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_nodes) { return; }
  ivec2 c = texel(i);

  vec4 xi = imageLoad(x, c);
  if (!(xi.w > 0.0)) { return; }   // pinned: position and velocity both hold

  vec3 pi = imageLoad(p, c).xyz;
  vec3 vel = (pi - xi.xyz) / h * damping;

  // Velocity clamp, from the reference self-collision: a node that a contact
  // pass corrected this substep may not keep a velocity that crosses more
  // than 0.2 of a contact thickness in a substep, or fast material tunnels
  // through thin features and wads up instead of folding. Scoped to marked
  // nodes so free fall is never capped - a global cap turned every drop into
  // slow motion. Only the velocity carried into the next predict is limited;
  // the position corrections of this substep stand. max_vel of 0 disables it,
  // which keeps no-contact trajectories bit identical.
  float speed = length(vel);
  if (max_vel > 0.0 && imageLoad(mark, c).r > 0.0 && speed > max_vel) {
    vel *= max_vel / speed;
  }

  imageStore(v, c, vec4(vel, 0.0));
  imageStore(x, c, vec4(pi, xi.w));
}
"""

COLLIDE_SRC = """
// One dispatch per collider. Looping colliders inside the kernel would need
// an array uniform; dispatching per collider reuses the plain push-constant
// path that is already measured to work, and colliders are few.
//
// kind 0 = ground plane (world space, uses ground_z)
// kind 1 = unit sphere in the collider's local space
// kind 2 = unit box in the collider's local space
//
// Primitives are unit-sized in local space and shaped entirely by the
// object's transform, so a default Blender UV sphere (radius 1) or cube
// (size 2, spanning -1..1) maps exactly with no extra parameters.
//
// A sticky collider also grabs. The `stick` image is one texel per node:
// .w holds the id of the collider holding it (0 = free) and .xyz holds the
// contact point in that collider's LOCAL space. Local space is the whole
// trick - the anchor then rides the object's animated transform for free,
// so a plate that lifts drags the material with it. Non-penetration alone
// can only push, so without this a lifting collider leaves the body behind.

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_nodes) { return; }

  ivec2 c = texel(i);
  vec4 pi = imageLoad(p, c);
  if (!(pi.w > 0.0)) { return; }   // a pin outranks a collider

  vec3 pos = pi.xyz;

  if (kind == 0) {
    if (pos.z < ground_z) { pos.z = ground_z; }
    imageStore(p, c, vec4(pos, pi.w));
    return;
  }

  vec4 held = imageLoad(stick, c);
  int holder = int(held.w);

  // First grab wins. Without this a node caught between two sticky colliders
  // would be yanked back and forth by whichever dispatched last.
  if (holder != 0 && holder != collider_id) { return; }

  if (holder == collider_id) {
    vec3 target = (to_world * vec4(held.xyz, 1.0)).xyz;
    // The solve pass ran before this one, so `pos` is where the material
    // wants the node and this distance is how hard it is pulling on the
    // contact. break_dist arrives as a huge number when breaking is off.
    if (distance(pos, target) <= break_dist) {
      imageStore(p, c, vec4(target, pi.w));
      return;
    }
    // Pulled free. Release, then fall through to plain non-penetration so a
    // node that lets go is not left sitting inside the collider.
    imageStore(stick, c, vec4(0.0));
  }

  vec3 lp = (to_local * vec4(pos, 1.0)).xyz;
  bool inside = false;

  if (kind == 1) {
    float d = length(lp);
    if (d < 1.0) {
      inside = true;
      // Dead centre has no defined push direction; pick one rather than
      // divide by zero and produce NaN.
      lp = (d > 1e-6) ? (lp / d) : vec3(0.0, 0.0, 1.0);
    }
  } else if (kind == 3) {
    // Mesh collider. lp is already in grid space: the CPU folded the
    // bounding-box-to-unit-cube mapping into to_local, so no push constant
    // had to be added to a block that already overflows its 128 byte budget.
    ivec3 dim = imageSize(sdf);
    vec3 g = lp * vec3(dim) - 0.5;
    vec3 lo = floor(g);
    ivec3 b = ivec3(lo);
    if (all(greaterThanEqual(b, ivec3(0))) && all(lessThan(b + 1, dim))) {
      vec3 f = g - lo;
      // Trilinear, so the surface is smooth rather than voxel-stepped.
      float d000 = imageLoad(sdf, b + ivec3(0, 0, 0)).r;
      float d100 = imageLoad(sdf, b + ivec3(1, 0, 0)).r;
      float d010 = imageLoad(sdf, b + ivec3(0, 1, 0)).r;
      float d110 = imageLoad(sdf, b + ivec3(1, 1, 0)).r;
      float d001 = imageLoad(sdf, b + ivec3(0, 0, 1)).r;
      float d101 = imageLoad(sdf, b + ivec3(1, 0, 1)).r;
      float d011 = imageLoad(sdf, b + ivec3(0, 1, 1)).r;
      float d111 = imageLoad(sdf, b + ivec3(1, 1, 1)).r;
      float d = mix(mix(mix(d000, d100, f.x), mix(d010, d110, f.x), f.y),
                    mix(mix(d001, d101, f.x), mix(d011, d111, f.x), f.y), f.z);
      if (d < 0.0) {
        inside = true;
        // The exact gradient of the same trilinear interpolant, from the
        // eight values already loaded. Central differences were tried first
        // and are wrong: they evaluate at the cell corner, not at the
        // sample point, so a node half a cell in from the corner is pushed
        // diagonally. Measured on a sphere, a node at (0, 0, 0.5) that
        // should rise straight up came out at (-0.13, -0.13, 0.93).
        vec3 grad = vec3(
          mix(mix(d100 - d000, d110 - d010, f.y),
              mix(d101 - d001, d111 - d011, f.y), f.z),
          mix(mix(d010 - d000, d110 - d100, f.x),
              mix(d011 - d001, d111 - d101, f.x), f.z),
          mix(mix(d001 - d000, d101 - d100, f.x),
              mix(d011 - d010, d111 - d110, f.x), f.y));
        float glen = length(grad);
        // A flat patch of field has no direction to offer; leave the node
        // where it is rather than divide by zero and poison it with NaN.
        if (glen > 1e-9) { lp -= grad * (d / glen); }
        else { inside = false; }
      }
    }
  } else if (kind == 2) {
    vec3 a = abs(lp);
    if (a.x < 1.0 && a.y < 1.0 && a.z < 1.0) {
      inside = true;
      vec3 gap = vec3(1.0) - a;   // distance to each face
      if (gap.x <= gap.y && gap.x <= gap.z) {
        lp.x = (lp.x >= 0.0) ? 1.0 : -1.0;
      } else if (gap.y <= gap.z) {
        lp.y = (lp.y >= 0.0) ? 1.0 : -1.0;
      } else {
        lp.z = (lp.z >= 0.0) ? 1.0 : -1.0;
      }
    }
  }

  pos = (to_world * vec4(lp, 1.0)).xyz;

  // KNOWN LIMITATION - a body authored deeply overlapping a sticky collider
  // shreds. Every buried node is grabbed on frame one and welded to whichever
  // face happened to be nearest, which scatters them across faces and turns
  // the body inside out. Measured on a sphere half-buried in a sticky box:
  // 219 of 461 nodes seized immediately, 12% of tets inverted.
  //
  // Refusing to grab nodes that start inside was tried and is wrong: a plate
  // authored already pressed into the body is the legitimate version of the
  // same geometry, and it is how a squash-and-stretch shot is set up. Depth is
  // the real discriminator - a fresh contact is shallow, an authored
  // intersection is deep - but that needs a scale-aware threshold, so for now
  // do not start a body inside a sticky collider.
  //
  // Grab on contact, anchored to the surface point the node was just pushed
  // onto rather than to where it was found. An anchor on the surface stays on
  // the surface, so the hold does not drift into the collider over time.
  if (sticky != 0 && inside) {
    imageStore(stick, c, vec4(lp, float(collider_id)));
  }

  imageStore(p, c, vec4(pos, pi.w));
}
"""


# `out` is a reserved word in GLSL, so the destination image is `out_pos`.
# Declared once and shared with the tests. Three copies of this list existed
# and adding the sdf image broke the two that were not the real one.
COLLIDE_IMAGES = [
    ("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "stick", {"READ", "WRITE"}),
    ("R32F", "FLOAT_3D", "sdf", {"READ"}),
]
COLLIDE_PUSH = [
    ("FLOAT", "ground_z"),
    ("INT", "kind"),
    ("INT", "n_nodes"),
    ("MAT4", "to_local"),
    ("MAT4", "to_world"),
    ("INT", "collider_id"),
    ("INT", "sticky"),
    ("FLOAT", "break_dist"),
]


SELF_COLLIDE_SRC = """
// Self-collision between surface nodes of the cage. All pairs, Jacobi.
//
// No spatial hash. Ten Minute Physics 15 builds one to escape O(n^2) on a
// CPU in JavaScript; here the pairs are restricted to the boundary of the
// cage and run on the card, where 29M pair tests measure 1.9ms. A hash would
// need imageAtomicAdd on an integer image, and integer images do not work in
// Blender's Python GPU API at all - imageStore never lands and read() only
// ever hands back a FLOAT buffer.
//
// Jacobi, not Gauss-Seidel: each thread owns node i, reads every surface
// node j, and writes only its own texel of out_p. Nothing needs colouring or
// atomics. Reading and writing one image in a single dispatch would race, so
// the caller ping-pongs p with out_p afterwards - which is exactly why EVERY
// thread must write one texel. An early return leaves a stale texel behind
// and the node jumps to wherever the other buffer last had it.

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_nodes) { return; }
  ivec2 c = texel(i);
  vec4 pi = imageLoad(p, c);

  // Interior nodes cannot be the first point of contact on a closed cage,
  // and a pin outranks a self-contact. Both still write through.
  int si = int(imageLoad(surf_idx, c).r);
  if (si < 0 || !(pi.w > 0.0)) {
    imageStore(out_p, c, pi);
    imageStore(mark_out, c, vec4(0.0));
    return;
  }

  vec3 rest_i = imageLoad(rest_pos, c).xyz;
  vec3 fix = vec3(0.0);

  for (int s = 0; s < n_surf; ++s) {
    if (s == si) { continue; }
    ivec2 cj = texel(int(imageLoad(surf, texel(s)).r));
    vec4 pj = imageLoad(p, cj);

    vec3 d = pj.xyz - pi.xyz;
    float d2 = dot(d, d);
    if (d2 >= thickness * thickness || d2 < 1e-12) { continue; }

    // Rest-distance gate, from the reference. Lattice neighbours sit one
    // Resolution apart at rest, which is the default thickness, so without
    // this every node would fight its own tets on every substep.
    vec3 r = imageLoad(rest_pos, cj).xyz - rest_i;
    float rest2 = dot(r, r);
    if (d2 >= rest2) { continue; }

    float mind = (rest2 < thickness * thickness) ? sqrt(rest2) : thickness;
    float dist = sqrt(d2);
    // Mass weighted, where the reference splits a flat half: a pinned
    // partner has zero inverse mass and takes none of the correction, so
    // this node has to take all of it.
    float share = pi.w / (pi.w + pj.w);
    fix -= d * ((mind - dist) / dist) * share;
  }
  // The mark is what scopes the integrate velocity clamp to nodes that are
  // actually in contact, so free fall keeps its speed.
  float hit = dot(fix, fix) > 1e-24 ? 1.0 : 0.0;
  imageStore(mark_out, c, vec4(hit, 0.0, 0.0, 0.0));
  imageStore(out_p, c, vec4(pi.xyz + fix, pi.w));
}
"""

BODY_COLLIDE_SRC = """
// Collision against another Marrow body. One dispatch per other body.
//
// The self-collide kernel with two things removed: there is no self to skip,
// and no rest-distance gate, because two bodies share no rest configuration
// and every contact inside the thickness is a real one.
//
// The partner is sampled from x_other, its integrated position, and never
// from its p. p is a work in progress whose meaning depends on how far
// through its own substep the other body happens to be, which would make the
// result depend on the order the group driver walks its members - invisible
// from the outside and impossible to reproduce. x_other costs one substep of
// lag instead: 0.02m at 5 m/s with 10 substeps, against a 0.1m thickness.
//
// Two-way coupling needs no mechanism of its own. This body takes
// w_self / (w_self + w_other) of the correction and the other body, running
// its own dispatch, takes the rest. The shares sum to one, so the pair opens
// to exactly the thickness and a pinned partner pushes without moving.

void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_nodes) { return; }
  ivec2 c = texel(i);
  vec4 pi = imageLoad(p, c);

  // Write through, always - see SELF_COLLIDE_SRC on the ping-pong.
  int si = int(imageLoad(surf_idx, c).r);
  if (si < 0 || !(pi.w > 0.0)) {
    imageStore(out_p, c, pi);
    imageStore(mark_out, c, vec4(imageLoad(mark_in, c).r, 0.0, 0.0, 0.0));
    return;
  }

  vec3 fix = vec3(0.0);
  for (int s = 0; s < n_surf_other; ++s) {
    vec4 pj = imageLoad(x_other, texel(int(imageLoad(surf_other, texel(s)).r)));

    vec3 d = pj.xyz - pi.xyz;
    float d2 = dot(d, d);
    if (d2 >= thickness * thickness || d2 < 1e-12) { continue; }

    float dist = sqrt(d2);
    float share = pi.w / (pi.w + pj.w);
    fix -= d * ((thickness - dist) / dist) * share;
  }
  // Carry the self-collision mark through and add this pass's own contact.
  float hit = max(imageLoad(mark_in, c).r, dot(fix, fix) > 1e-24 ? 1.0 : 0.0);
  imageStore(mark_out, c, vec4(hit, 0.0, 0.0, 0.0));
  imageStore(out_p, c, vec4(pi.xyz + fix, pi.w));
}
"""

SKIN_SRC = """
void main()
{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= n_render) { return; }
  ivec2 c = texel(i);

  vec4 b = imageLoad(bind, c);
  int t = int(b.r);
  vec3 w = b.gba;
  float w0 = 1.0 - w.x - w.y - w.z;

  ivec4 idx = ivec4(imageLoad(tets, texel(t)));
  vec3 pos = imageLoad(x, texel(idx.x)).xyz * w0
           + imageLoad(x, texel(idx.y)).xyz * w.x
           + imageLoad(x, texel(idx.z)).xyz * w.y
           + imageLoad(x, texel(idx.w)).xyz * w.z;

  // Alpha carries a generation mark so the CPU can tell a current
  // readback from a stale one. See textures.read_marked.
  imageStore(out_pos, c, vec4(pos, mark));
}
"""
