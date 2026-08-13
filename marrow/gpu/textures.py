"""numpy <-> GPUTexture. The only float32 boundary in the codebase.

GPUTexture accepts an upload Buffer only in FLOAT format - integer buffers
raise outright - so every image here is float-typed, including the one
holding tet indices.
"""

import gpu
import numpy as np

_CHANNELS = {"RGBA32F": 4, "R32F": 1}


def _channels(fmt: str) -> int:
    if fmt not in _CHANNELS:
        raise ValueError(f"unsupported texture format {fmt!r}; use one of {sorted(_CHANNELS)}")
    return _CHANNELS[fmt]


def upload(image: np.ndarray, fmt: str = "RGBA32F") -> gpu.types.GPUTexture:
    """Create a texture holding ``image``, which must be (H, W, channels)."""
    channels = _channels(fmt)
    array = np.ascontiguousarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != channels:
        raise ValueError(
            f"{fmt} needs an (H, W, {channels}) array, got {array.shape}"
        )
    height, width = array.shape[0], array.shape[1]
    buffer = gpu.types.Buffer("FLOAT", array.size, array.ravel().tolist())
    return gpu.types.GPUTexture((width, height), format=fmt, data=buffer)


def upload3d(field: np.ndarray, fmt: str = "R32F") -> gpu.types.GPUTexture:
    """Create a 3D texture from a (nz, ny, nx) array.

    C-order ravel puts x fastest, which is the order the sampler indexes
    ivec3(x, y, z) in - verified bit-exact on a round trip through a kernel.
    """
    array = np.ascontiguousarray(field, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"a 3D texture needs an (nz, ny, nx) array, got {array.shape}")
    nz, ny, nx = array.shape
    buffer = gpu.types.Buffer("FLOAT", array.size, array.ravel().tolist())
    return gpu.types.GPUTexture((nx, ny, nz), format=fmt, data=buffer)


_FLUSH_SRC = """
void main()
{
  ivec2 c = ivec2(gl_GlobalInvocationID.xy);
  imageStore(tgt, c, imageLoad(tgt, c));
}
"""


def make_flush_shader(fmt: str = "RGBA32F"):
    """Build a no-op read-modify-write kernel used to order a readback.

    Deliberately NOT cached at module level. Blender tears the GPU context
    down before module globals are collected, so a GPUShader held in module
    state is freed against a dead context - measured, that crashes Blender at
    shutdown with EXCEPTION_ACCESS_VIOLATION in MSVCP140.dll. Instance-held
    shaders are freed early enough and are fine, which is why GPUSolver can
    hold five of them without trouble. Callers own the object.
    """
    from ..gpu.kernels import build

    return build(
        f"flush_{fmt}",
        _FLUSH_SRC,
        [(fmt, "FLOAT_2D", "tgt", {"READ", "WRITE"})],
        [],
        group_size=1,
    )


def flush(shader, tex: gpu.types.GPUTexture) -> None:
    """Make prior imageStore writes visible to a following readback.

    Blender exposes no memory barrier - gpu.compute offers only dispatch, and
    nothing in gpu, gpu.state, gpu.texture, gpu.shader or gpu.types provides
    sync, flush, finish or fence. Measured: dispatch-to-dispatch ordering is
    reliable, but GPUTexture.read() straight after a dispatch intermittently
    returns the pre-dispatch contents, caught with readback bit-identical to
    the uploaded input in roughly one suite run in six.

    Dispatching a read-modify-write over the same image puts a genuine data
    dependency in front of the read, which is the barrier we cannot request
    directly.
    """
    shader.bind()
    shader.image("tgt", tex)
    gpu.compute.dispatch(shader, tex.width, tex.height, 1)


class StaleReadError(RuntimeError):
    """A texture never reported the data we had just written into it."""


def read_marked(tex, mark: float, count: int, max_reads: int = 16) -> np.ndarray:
    """Read a texture whose alpha channel carries a known generation mark.

    The kernel that wrote this texture stamped `mark` into every texel's
    alpha. If the read comes back without it, the write is not visible yet
    and we simply read again.

    This is the cheap, sound version of the problem. An earlier attempt polled
    a separate 1x1 "fence" texture instead and was useless - measured, it
    returned the new mark on the very first poll every single time while the
    real texture was still stale. Visibility is per texture, so only the
    target itself can answer for the target.
    """
    array = download(tex)
    flat = array.reshape(-1, array.shape[2])
    if count == 0 or np.all(flat[:count, 3] == np.float32(mark)):
        return array
    return None


def read_stable(tex, nudge=None, max_reads: int = 16) -> np.ndarray:
    """Read until two consecutive reads agree.

    For textures with no spare channel to carry a mark. Costs two reads in the
    common case, which is why marked reads are preferred on the per-frame path.
    """
    previous = download(tex)
    for attempt in range(2, max_reads + 1):
        current = download(tex)
        # equal_nan matters: a NaN state is exactly what the solver's own
        # guard needs to see, and NaN != NaN would make it never settle.
        if np.array_equal(previous, current, equal_nan=True):
            return current
        previous = current
    raise StaleReadError(f"texture never settled across {max_reads} reads")


def upload_verified(image: np.ndarray, fmt: str = "RGBA32F", max_tries: int = 8):
    """Upload, then read back and confirm the data actually landed.

    Staleness runs both directions. A dispatch reading a freshly uploaded
    texture can see the memory that was there before the upload - caught with
    a rest-pose skin readback that was correctly generation-marked, so the
    kernel had genuinely run, yet produced positions 21 units out. The cage
    it read from still held recycled contents.

    This only runs at setup, never per frame, so a verifying read is cheap
    insurance rather than a cost on the hot path.
    """
    array = np.ascontiguousarray(image, dtype=np.float32)
    for attempt in range(1, max_tries + 1):
        tex = upload(array, fmt=fmt)
        back = download(tex)
        if np.array_equal(back, array if array.ndim == 3 else array[:, :, None]):
            if attempt > 1:
                print(f"marrow: texture upload landed on attempt {attempt}")
            return tex
    raise StaleReadError(
        f"a {fmt} upload never became visible after {max_tries} attempts"
    )


def download(tex: gpu.types.GPUTexture) -> np.ndarray:
    """Read a texture back as (H, W, channels) float32.

    GPUTexture.read() hands back a Buffer shaped [H][W][C], not a flat
    sequence. Reshaping explicitly and refusing an empty result is what
    stops a zero-size readback from passing an elementwise comparison
    vacuously, which is how spike 0 produced a false PASS.
    """
    array = np.asarray(tex.read(), dtype=np.float32)
    if array.size == 0:
        raise RuntimeError("texture readback returned no data")
    if array.ndim == 2:  # single-channel comes back without a channel axis
        array = array[:, :, None]
    return array


def blank(count: int, fmt: str = "RGBA32F") -> gpu.types.GPUTexture:
    """A zeroed texture with room for ``count`` elements."""
    from ..core.layout import texture_shape

    width, height = texture_shape(count)
    zeros = np.zeros((height, width, _channels(fmt)), dtype=np.float32)
    return upload(zeros, fmt=fmt)
