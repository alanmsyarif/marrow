"""Is there a usable GPU compute context?

The spec requires Marrow to disable itself with a plain message rather than
fail obscurely when there is no usable GPU. Probing by actually compiling a
trivial compute shader is the honest test: capability flags like
compute_shader_support_get() are deprecated and report True everywhere, and
gpu.init() succeeding does not prove a compute shader will build.
"""

_PROBE_SRC = """
void main()
{
  imageStore(probe, ivec2(gl_GlobalInvocationID.xy), vec4(1.0));
}
"""


def gpu_available() -> bool:
    """True when a compute shader can actually be built on this machine."""
    try:
        import gpu

        gpu.init()
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(1, 1)
        info.image(0, "RGBA32F", "FLOAT_2D", "probe", qualifiers={"WRITE"})
        info.compute_source(_PROBE_SRC)
        gpu.shader.create_from_info(info)
        return True
    except Exception:
        return False
