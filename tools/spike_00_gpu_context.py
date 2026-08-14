"""Marrow build step 0 spike.

Answers the question that gates the whole architecture:
can `gpu.compute.dispatch` run from a `frame_change_post` handler,
or only from a draw context?

Run headless:
    blender -b --factory-startup --python tools/spike_00_gpu_context.py

Exit code 0 means the spike answered; read RESULT lines for the verdict.
"""

import sys
import traceback

import bpy
import gpu

RESULTS = {}


def report(key, ok, detail=""):
    RESULTS[key] = (ok, detail)
    print(f"RESULT {key}: {'PASS' if ok else 'FAIL'} {detail}")


COMPUTE_SRC = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  imageStore(img, c, vec4(float(c.x), float(c.y), 42.0, 1.0));
}
"""

WIDTH, HEIGHT = 8, 8


def build_shader():
    info = gpu.types.GPUShaderCreateInfo()
    info.local_group_size(1, 1)
    info.image(0, "RGBA32F", "FLOAT_2D", "img", qualifiers={"WRITE"})
    info.compute_source(COMPUTE_SRC)
    return gpu.shader.create_from_info(info)


def run_dispatch(shader, tex):
    """Bind and dispatch. Raises on failure so callers can classify it."""
    shader.bind()
    shader.image("img", tex)
    gpu.compute.dispatch(shader, WIDTH, HEIGHT, 1)


def verify(tex):
    """Read the image back and confirm the kernel actually ran.

    GPUTexture.read() returns a Buffer exposing the buffer protocol, shaped
    [H][W][channels] rather than a flat sequence. Go through numpy so the
    shape is explicit and a wrong shape fails loudly instead of silently.
    """
    import numpy as np

    arr = np.asarray(tex.read())
    if arr.size == 0:
        return False, f"readback empty, shape={arr.shape}"
    try:
        arr = arr.reshape(HEIGHT, WIDTH, 4)
    except ValueError:
        return False, f"unexpected readback shape={arr.shape} size={arr.size}"

    texel = arr[2, 3].tolist()  # row y=2, column x=3
    expected = [3.0, 2.0, 42.0, 1.0]
    ok = np.allclose(texel, expected, atol=1e-5)
    return ok, f"shape={arr.shape} texel(x=3,y=2)={texel} expected={expected}"


def main():
    # 0. init
    try:
        gpu.init()
        report("gpu_init", True)
    except Exception as exc:
        report("gpu_init", False, repr(exc))
        return

    print("PROBE GPUShader methods:", sorted(a for a in dir(gpu.types.GPUShader) if not a.startswith("_")))

    # 1. shader compiles
    try:
        shader = build_shader()
        report("compile", True)
    except Exception as exc:
        report("compile", False, repr(exc))
        traceback.print_exc()
        return

    tex = gpu.types.GPUTexture((WIDTH, HEIGHT), format="RGBA32F")

    # 2. dispatch from plain script context
    try:
        run_dispatch(shader, tex)
        ok, detail = verify(tex)
        report("dispatch_script_context", ok, detail)
    except Exception as exc:
        report("dispatch_script_context", False, repr(exc))
        traceback.print_exc()

    # 3. THE question: dispatch from a frame_change_post handler
    tex2 = gpu.types.GPUTexture((WIDTH, HEIGHT), format="RGBA32F")
    handler_state = {}

    def on_frame(scene, depsgraph=None):
        try:
            run_dispatch(shader, tex2)
            handler_state["ok"], handler_state["detail"] = verify(tex2)
        except Exception as exc:
            handler_state["ok"] = False
            handler_state["detail"] = repr(exc)

    bpy.app.handlers.frame_change_post.append(on_frame)
    try:
        bpy.context.scene.frame_set(2)
    finally:
        bpy.app.handlers.frame_change_post.remove(on_frame)

    if not handler_state:
        report("dispatch_frame_handler", False, "handler never fired")
    else:
        report("dispatch_frame_handler", handler_state["ok"], handler_state["detail"])

    print("\n=== SPIKE SUMMARY ===")
    for k, (ok, detail) in RESULTS.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'} {detail}")


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
