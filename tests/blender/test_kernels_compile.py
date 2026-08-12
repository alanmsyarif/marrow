import gpu

from marrow.gpu.kernels import PREDICT_SRC, TEXEL_GLSL, build

gpu.init()


def test_texel_helper_is_present_in_every_build():
    assert "ivec2 texel(" in TEXEL_GLSL


def test_predict_kernel_compiles():
    shader = build(
        "predict",
        PREDICT_SRC,
        images=[
            ("RGBA32F", "FLOAT_2D", "x", {"READ"}),
            ("RGBA32F", "FLOAT_2D", "v", {"READ"}),
            ("RGBA32F", "FLOAT_2D", "p", {"WRITE"}),
        ],
        push_constants=[("FLOAT", "h"), ("VEC3", "gravity"), ("INT", "n_nodes")],
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
