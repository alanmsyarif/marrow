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
    info.local_group_size(group_size, 1)
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

void main()
{
  int t = color_begin + int(gl_GlobalInvocationID.x);
  if (t >= color_end) { return; }

  // A torn tet is gone for good: it contributes no constraint ever again,
  // which is what makes the material go slack there instead of springing back.
  if (imageLoad(torn, texel(t)).r > 0.5) { return; }

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

  // --- deviatoric ---
  if (mu > 0.0) {
    vec3 p0 = imageLoad(p, texel(idx.x)).xyz;
    mat3 f = shape_matrix(idx, p0) * dm_inv;

    float c_dev = sqrt(dot(f[0], f[0]) + dot(f[1], f[1]) + dot(f[2], f[2]));

    // c_dev is sqrt(3) at rest, so tear_threshold reads as a stretch ratio:
    // 1.5 means "tear at 50% strain". Zero or less disables tearing, which is
    // what the oracle-parity tests run with.
    if (tear_threshold > 0.0 && c_dev > tear_threshold * 1.7320508) {
      imageStore(torn, texel(t), vec4(1.0));
      return;
    }

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
  // F is rebuilt from the positions the deviatoric pass just moved. Reusing
  // the stale F would linearise the volume constraint about the wrong
  // configuration. The oracle does the same.
  if (lam > 0.0) {
    vec3 p0 = imageLoad(p, texel(idx.x)).xyz;
    mat3 f = shape_matrix(idx, p0) * dm_inv;

    mat3 dcdf = mat3(cross(f[1], f[2]), cross(f[2], f[0]), cross(f[0], f[1]));
    mat3 g = dcdf * dm_inv_t;
    vec3 g1v = g[0];
    vec3 g2v = g[1];
    vec3 g3v = g[2];
    vec3 g0v = -(g1v + g2v + g3v);

    float gamma = 1.0 + mu / lam;
    float c_hyd = determinant(f) - gamma;
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
  } else {
    vec3 lp = (to_local * vec4(pos, 1.0)).xyz;

    if (kind == 1) {
      float d = length(lp);
      if (d < 1.0) {
        // Dead centre has no defined push direction; pick one rather than
        // divide by zero and produce NaN.
        lp = (d > 1e-6) ? (lp / d) : vec3(0.0, 0.0, 1.0);
      }
    } else if (kind == 2) {
      vec3 a = abs(lp);
      if (a.x < 1.0 && a.y < 1.0 && a.z < 1.0) {
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
  }

  imageStore(p, c, vec4(pos, pi.w));
}
"""


# `out` is a reserved word in GLSL, so the destination image is `out_pos`.
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
