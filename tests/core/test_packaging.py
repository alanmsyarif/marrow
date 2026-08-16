"""The addon must import as a package under any name.

Installed as an extension, Marrow's package is not called "marrow" - it is
bl_ext.user_default.marrow. Any absolute `from marrow.x import y` inside the
addon therefore raises ModuleNotFoundError at register() time, on a user's
machine, in a build that passed every test.

Nothing else in the suite can catch this: the tests put the repo root on
sys.path, so `marrow` always resolves for them. Only an install exercises the
real name. This test stands in for that.
"""

import ast
import pathlib

ADDON = pathlib.Path(__file__).resolve().parents[2] / "marrow"


def _absolute_self_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "marrow" or alias.name.startswith("marrow."):
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which is what we want.
            if node.level == 0 and node.module and (
                node.module == "marrow" or node.module.startswith("marrow.")
            ):
                offenders.append(f"from {node.module} import ...")
    return offenders


def test_addon_uses_relative_imports_only():
    bad = {}
    for path in sorted(ADDON.rglob("*.py")):
        found = _absolute_self_imports(path)
        if found:
            bad[str(path.relative_to(ADDON))] = found
    assert bad == {}, (
        f"absolute self-imports break the installed extension, whose package "
        f"is bl_ext.user_default.marrow, not marrow: {bad}"
    )


def test_the_guard_can_actually_see_an_offender(tmp_path):
    """Prove the detector works, so a silent pass means something."""
    sample = tmp_path / "offender.py"
    sample.write_text("from marrow.core.tetmesh import TetMesh\nimport marrow.gpu\n")
    assert len(_absolute_self_imports(sample)) == 2

    clean = tmp_path / "clean.py"
    clean.write_text("from ..core.tetmesh import TetMesh\nfrom . import kernels\n")
    assert _absolute_self_imports(clean) == []


def test_version_is_read_from_the_manifest_not_copied():
    """One version string lives in blender_manifest.toml. __init__ reads it.

    Parsed by hand here rather than with tomllib, so this is not just the
    addon's own parse compared against itself. Catches the drift that already
    happened once: manifest bumped, __version__ literal left behind.
    """
    import marrow

    declared = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in (ADDON / "blender_manifest.toml").read_text(encoding="utf-8").splitlines()
        if line.startswith("version")
    )
    assert marrow.__version__ == declared
