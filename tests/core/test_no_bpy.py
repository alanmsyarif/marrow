"""marrow.core must never depend on Blender. This is what makes it testable."""
import ast
import pathlib

CORE = pathlib.Path(__file__).resolve().parents[2] / "marrow" / "core"
FORBIDDEN = {"bpy", "mathutils", "gpu", "bmesh", "gpu_extras", "bpy_extras", "blf", "aud", "bl_math"}


def _imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


def test_core_has_no_blender_imports():
    offenders = {}
    for path in CORE.rglob("*.py"):
        bad = _imported_roots(path) & FORBIDDEN
        if bad:
            offenders[path.name] = sorted(bad)
    assert offenders == {}, f"marrow.core must not import Blender modules: {offenders}"


def test_core_package_imports_standalone():
    import marrow.core  # noqa: F401


# Every Blender-bundled root module FORBIDDEN is expected to cover. Kept as a
# separate literal so shrinking FORBIDDEN fails here instead of passing quietly.
_EXPECTED_FORBIDDEN = frozenset(
    {
        "bpy",
        "mathutils",
        "gpu",
        "bmesh",
        "gpu_extras",
        "bpy_extras",
        "blf",
        "aud",
        "bl_math",
    }
)


def test_forbidden_set_covers_blender_only_modules():
    """Regression guard: these are Blender-bundled and must never reach core.

    Checks the whole expected set rather than a hand-picked subset. The
    previous version listed only 7 of the 9 names, so dropping "aud" or
    "bl_math" from FORBIDDEN would have gone unnoticed. Superset, not
    equality, so genuinely new names can still be added to FORBIDDEN.
    """
    missing = _EXPECTED_FORBIDDEN - FORBIDDEN
    assert not missing, f"FORBIDDEN lost Blender-only modules: {sorted(missing)}"
