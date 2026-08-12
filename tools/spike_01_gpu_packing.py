"""Marrow build step 3 pre-spike: the API facts the GPU kernel plan rests on.

Step 0 proved `gpu.compute.dispatch` runs from a frame handler. It did not
prove the data layout in the spec is expressible. Six questions gate the
kernel design, and guessing any of them would put a dozen tasks on top of an
assumption:

  A. Is there a memory barrier between dependent dispatches, and does a
     write-then-read chain give the right answer without one?
  B. Do integer images work (RGBA32I / INT_2D)? `tets` needs one.
  C. Does single-channel R32F work? `lambda` needs one.
  D. Can one image be bound READ and WRITE in the same kernel? `x` and `p`
     are read-modify-written by the solve kernel.
  E. Can six images be bound to one shader at once? The budget says six.
  F. Can a numpy array be uploaded into a GPUTexture, and read back intact?

Run headless:
    blender -b --factory-startup --python tools/spike_01_gpu_packing.py

Exit code 0 means the spike answered. Read RESULT lines for verdicts.
"""

import sys
import traceback

import gpu
import numpy as np

RESULTS = {}


def report(key, ok, detail=""):
    RESULTS[key] = (ok, detail)
    print(f"RESULT {key}: {'PASS' if ok else 'FAIL'} {detail}")


def probe(key, fn):
    """Run one question. An exception is an answer, not a crash."""
    try:
        ok, detail = fn()
        report(key, ok, detail)
    except Exception as exc:
        report(key, False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# A. barrier API and dependent-dispatch correctness
# --------------------------------------------------------------------------

SRC_WRITE = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  imageStore(dst, c, vec4(7.0, 0.0, 0.0, 0.0));
}
"""

SRC_DOUBLE = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  float v = imageLoad(src, c).r;
  imageStore(dst, c, vec4(v * 2.0, 0.0, 0.0, 0.0));
}
"""

W = H = 16


def _shader(src, images):
    info = gpu.types.GPUShaderCreateInfo()
    info.local_group_size(1, 1)
    for slot, (fmt, kind, name, quals) in enumerate(images):
        info.image(slot, fmt, kind, name, qualifiers=quals)
    info.compute_source(src)
    return gpu.shader.create_from_info(info)


def q_barrier_api():
    names = sorted(a for a in dir(gpu.compute) if not a.startswith("_"))
    has = [n for n in names if "barrier" in n.lower()]
    return bool(has), f"gpu.compute exposes {names}; barrier-like: {has or 'NONE'}"


def q_dependent_chain():
    """Write 7 into a, then double it into b, 50 times. Expect 14 every time."""
    writer = _shader(SRC_WRITE, [("R32F", "FLOAT_2D", "dst", {"WRITE"})])
    doubler = _shader(
        SRC_DOUBLE,
        [
            ("R32F", "FLOAT_2D", "src", {"READ"}),
            ("R32F", "FLOAT_2D", "dst", {"WRITE"}),
        ],
    )
    tex_a = gpu.types.GPUTexture((W, H), format="R32F")
    tex_b = gpu.types.GPUTexture((W, H), format="R32F")

    wrong = 0
    for _ in range(50):
        writer.bind()
        writer.image("dst", tex_a)
        gpu.compute.dispatch(writer, W, H, 1)

        doubler.bind()
        doubler.image("src", tex_a)
        doubler.image("dst", tex_b)
        gpu.compute.dispatch(doubler, W, H, 1)

        out = np.asarray(tex_b.read()).reshape(H, W)
        if not np.allclose(out, 14.0):
            wrong += 1
    return wrong == 0, f"{50 - wrong}/50 chained dispatches correct without an explicit barrier"


# --------------------------------------------------------------------------
# B / C. integer and single-channel formats
# --------------------------------------------------------------------------

SRC_INT = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  imageStore(dst, c, ivec4(c.x, c.y, 11, 12));
}
"""


def q_int_image():
    sh = _shader(SRC_INT, [("RGBA32I", "INT_2D", "dst", {"WRITE"})])
    tex = gpu.types.GPUTexture((W, H), format="RGBA32I")
    sh.bind()
    sh.image("dst", tex)
    gpu.compute.dispatch(sh, W, H, 1)
    arr = np.asarray(tex.read()).reshape(H, W, 4)
    got = arr[2, 3].tolist()
    return got == [3, 2, 11, 12], f"texel(x=3,y=2)={got} expected=[3, 2, 11, 12] dtype={arr.dtype}"


def q_r32f_image():
    sh = _shader(SRC_WRITE, [("R32F", "FLOAT_2D", "dst", {"WRITE"})])
    tex = gpu.types.GPUTexture((W, H), format="R32F")
    sh.bind()
    sh.image("dst", tex)
    gpu.compute.dispatch(sh, W, H, 1)
    arr = np.asarray(tex.read())
    return bool(np.allclose(arr.reshape(H, W), 7.0)), f"shape={arr.shape} all 7.0"


# --------------------------------------------------------------------------
# D. read-modify-write on one image
# --------------------------------------------------------------------------

SRC_RMW = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  vec4 cur = imageLoad(io, c);
  imageStore(io, c, cur + vec4(1.0, 0.0, 0.0, 0.0));
}
"""


def q_read_write_same_image():
    sh = _shader(SRC_RMW, [("RGBA32F", "FLOAT_2D", "io", {"READ", "WRITE"})])
    zeros = np.zeros((H, W, 4), dtype=np.float32)
    buf = gpu.types.Buffer("FLOAT", zeros.size, zeros.ravel().tolist())
    tex = gpu.types.GPUTexture((W, H), format="RGBA32F", data=buf)
    for _ in range(3):
        sh.bind()
        sh.image("io", tex)
        gpu.compute.dispatch(sh, W, H, 1)
    arr = np.asarray(tex.read()).reshape(H, W, 4)
    return bool(np.allclose(arr[..., 0], 3.0)), f"after 3 increments r={arr[0, 0, 0]} expected=3.0"


# --------------------------------------------------------------------------
# E. six images on one shader
# --------------------------------------------------------------------------

SRC_SIX = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  float s = imageLoad(i0, c).r + imageLoad(i1, c).r + imageLoad(i2, c).r
          + imageLoad(i3, c).r + imageLoad(i4, c).r;
  imageStore(o5, c, vec4(s, 0.0, 0.0, 0.0));
}
"""


def q_six_images():
    images = [("R32F", "FLOAT_2D", f"i{n}", {"READ"}) for n in range(5)]
    images.append(("R32F", "FLOAT_2D", "o5", {"WRITE"}))
    sh = _shader(SRC_SIX, images)

    ones = np.ones((H, W), dtype=np.float32)
    texes = []
    for _ in range(5):
        buf = gpu.types.Buffer("FLOAT", ones.size, ones.ravel().tolist())
        texes.append(gpu.types.GPUTexture((W, H), format="R32F", data=buf))
    out = gpu.types.GPUTexture((W, H), format="R32F")

    sh.bind()
    for n, t in enumerate(texes):
        sh.image(f"i{n}", t)
    sh.image("o5", out)
    gpu.compute.dispatch(sh, W, H, 1)
    arr = np.asarray(out.read()).reshape(H, W)
    return bool(np.allclose(arr, 5.0)), f"sum of five unit images = {arr[0, 0]} expected=5.0"


# --------------------------------------------------------------------------
# F. numpy upload round-trip
# --------------------------------------------------------------------------


def q_numpy_upload():
    rng = np.random.default_rng(0)
    src = rng.random((H, W, 4)).astype(np.float32)
    buf = gpu.types.Buffer("FLOAT", src.size, src.ravel().tolist())
    tex = gpu.types.GPUTexture((W, H), format="RGBA32F", data=buf)
    back = np.asarray(tex.read()).reshape(H, W, 4)
    ok = np.allclose(back, src, atol=1e-6)
    return bool(ok), f"max abs diff {float(np.abs(back - src).max()):.3e}"


# --------------------------------------------------------------------------
# B2. the path `tets` actually uses: CPU uploads ints, a shader reads them.
# B tested shader-write then CPU-read, which is not what the data layer does.
# --------------------------------------------------------------------------

SRC_INT_READ = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  ivec4 t = imageLoad(idx, c);
  imageStore(dst, c, vec4(float(t.x), float(t.y), float(t.z), float(t.w)));
}
"""


def q_int_upload_read_by_shader():
    src = np.arange(H * W * 4, dtype=np.int32).reshape(H, W, 4)
    buf = gpu.types.Buffer("INT", src.size, src.ravel().tolist())
    tex_i = gpu.types.GPUTexture((W, H), format="RGBA32I", data=buf)
    out = gpu.types.GPUTexture((W, H), format="RGBA32F")

    sh = _shader(
        SRC_INT_READ,
        [
            ("RGBA32I", "INT_2D", "idx", {"READ"}),
            ("RGBA32F", "FLOAT_2D", "dst", {"WRITE"}),
        ],
    )
    sh.bind()
    sh.image("idx", tex_i)
    sh.image("dst", out)
    gpu.compute.dispatch(sh, W, H, 1)

    back = np.asarray(out.read()).reshape(H, W, 4)
    ok = np.allclose(back, src.astype(np.float32))
    return bool(ok), (
        f"uploaded ints seen by shader: texel(0,0)={back[0, 0].tolist()} "
        f"expected={src[0, 0].tolist()}, max diff "
        f"{float(np.abs(back - src).max()):.3e}"
    )


def q_float_indices_are_exact():
    """Fallback if integer images are unusable: indices as float32.

    float32 represents every integer up to 2**24 exactly, so a cage would have
    to exceed 16.7M nodes before an index lost precision.
    """
    probe_values = np.array([0, 1, 4095, 65535, 2**24 - 1, 2**24], dtype=np.int64)
    as_float = probe_values.astype(np.float32)
    exact = as_float.astype(np.int64) == probe_values
    first_bad = None if exact.all() else int(probe_values[~exact][0])
    return bool(exact[:-1].all()), (
        f"exact up to 2**24-1; first inexact value {first_bad}"
    )


# --------------------------------------------------------------------------
# G. push constants. Without them every dt/gravity/compliance change means a
# shader recompile, and the per-colour solve dispatch has no way to be told
# which colour it is running.
# --------------------------------------------------------------------------

SRC_PUSH = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  imageStore(dst, c, vec4(dt * 100.0 + float(color_begin), gravity.z, 0.0, 0.0));
}
"""


def q_push_constants():
    info = gpu.types.GPUShaderCreateInfo()
    info.local_group_size(1, 1)
    info.push_constant("FLOAT", "dt")
    info.push_constant("INT", "color_begin")
    info.push_constant("VEC3", "gravity")
    info.image(0, "RGBA32F", "FLOAT_2D", "dst", qualifiers={"WRITE"})
    info.compute_source(SRC_PUSH)
    sh = gpu.shader.create_from_info(info)

    tex = gpu.types.GPUTexture((W, H), format="RGBA32F")
    sh.bind()
    sh.uniform_float("dt", 0.25)
    sh.uniform_int("color_begin", 5)
    sh.uniform_float("gravity", (0.0, 0.0, -9.81))
    sh.image("dst", tex)
    gpu.compute.dispatch(sh, W, H, 1)

    arr = np.asarray(tex.read()).reshape(H, W, 4)
    got = arr[0, 0, :2].tolist()
    ok = np.allclose(got, [30.0, -9.81], atol=1e-4)
    return bool(ok), f"dt*100+color_begin, gravity.z = {got} expected=[30.0, -9.81]"


def q_local_group_size_64():
    """Real kernels want a wide group, not 1x1."""
    info = gpu.types.GPUShaderCreateInfo()
    info.local_group_size(64, 1)
    info.image(0, "R32F", "FLOAT_2D", "dst", qualifiers={"WRITE"})
    info.compute_source(SRC_WRITE)
    sh = gpu.shader.create_from_info(info)
    tex = gpu.types.GPUTexture((256, 4), format="R32F")
    sh.bind()
    sh.image("dst", tex)
    gpu.compute.dispatch(sh, 256 // 64, 4, 1)
    arr = np.asarray(tex.read()).reshape(4, 256)
    return bool(np.allclose(arr, 7.0)), "local_group_size(64,1) over 256x4 filled correctly"


def main():
    gpu.init()
    print(f"backend={gpu.platform.backend_type_get()} "
          f"max_images={gpu.capabilities.max_images_get()}")

    probe("A1_barrier_api", q_barrier_api)
    probe("A2_dependent_chain", q_dependent_chain)
    probe("B_int_image_RGBA32I", q_int_image)
    probe("B2_int_upload_read_by_shader", q_int_upload_read_by_shader)
    probe("B3_float_indices_exact", q_float_indices_are_exact)
    probe("C_single_channel_R32F", q_r32f_image)
    probe("D_read_write_same_image", q_read_write_same_image)
    probe("E_six_images_one_shader", q_six_images)
    probe("F_numpy_upload_roundtrip", q_numpy_upload)
    probe("G_push_constants", q_push_constants)
    probe("H_local_group_size_64", q_local_group_size_64)

    print("\n=== SPIKE 01 SUMMARY ===")
    for k, (ok, detail) in RESULTS.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'} {detail}")


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
