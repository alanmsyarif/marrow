import gpu

from marrow.gpu.kernels import (
    PREDICT_IMAGES,
    PREDICT_PUSH,
    PREDICT_SRC,
    TEXEL_GLSL,
    build,
)

gpu.init()


def test_texel_helper_is_present_in_every_build():
    assert "ivec2 texel(" in TEXEL_GLSL


def test_predict_kernel_compiles():
    shader = build(
        "predict",
        PREDICT_SRC,
        images=PREDICT_IMAGES,
        push_constants=PREDICT_PUSH,
    )
    assert shader is not None


def test_a_broken_kernel_surfaces_the_driver_log():
    broken = "void main() { this_is_not_glsl(); }"
    try:
        build("broken", broken, images=[], push_constants=[])
    except RuntimeError as exc:
        text = str(exc)
        assert "broken" in text, "error must name the kernel"
        assert len(text) > 40, f"compile log looks swallowed: {text!r}"
    else:
        raise AssertionError("a syntactically invalid kernel must not compile")
