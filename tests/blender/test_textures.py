import gpu
import numpy as np

from marrow.core.layout import TEX_WIDTH, texture_shape
from marrow.gpu.textures import blank, download, upload

gpu.init()


def test_round_trip_is_bit_exact():
    rng = np.random.default_rng(1)
    src = rng.random((2, TEX_WIDTH, 4)).astype(np.float32)
    back = download(upload(src))
    assert back.shape == src.shape
    assert np.array_equal(back, src), "upload/download must not perturb data"


def test_round_trip_survives_a_multi_row_image():
    src = np.arange(3 * TEX_WIDTH * 4, dtype=np.float32).reshape(3, TEX_WIDTH, 4)
    back = download(upload(src))
    assert back.shape == (3, TEX_WIDTH, 4)
    assert np.array_equal(back, src)


def test_blank_is_zeroed_and_correctly_shaped():
    tex = blank(TEX_WIDTH + 1)
    back = download(tex)
    assert back.shape == (2, TEX_WIDTH, 4)
    assert np.all(back == 0.0)


def test_blank_single_channel():
    tex = blank(16, fmt="R32F")
    back = download(tex)
    assert back.shape[:2] == texture_shape(16)[::-1]
    assert np.all(back == 0.0)


def test_download_asserts_on_an_empty_readback():
    """A vacuous readback comparison produced a false PASS in spike 0.

    download() must never hand back a zero-size array quietly.
    """
    tex = blank(8)
    back = download(tex)
    assert back.size > 0
