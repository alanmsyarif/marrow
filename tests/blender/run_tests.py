"""Assert-based test runner for Blender-dependent code.

Blender's bundled Python has no pytest and we will not install into it.
Run: blender -b --factory-startup --python tests/blender/run_tests.py
Exits 1 on any failure so CI and humans both notice.
"""

import importlib
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Both paths must be set BEFORE any test module is imported.
sys.path.insert(0, str(ROOT))   # so `import marrow` works
sys.path.insert(0, str(HERE))   # so test modules import as top-level names


def _release_gpu_state(modules):
    """Drop every GPU object before the interpreter exits.

    Blender tears the GPU context down before module globals are collected,
    and a GPUTexture or GPUShader freed against the dead context crashes the
    process at shutdown - EXCEPTION_ACCESS_VIOLATION in MSVCP140.dll,
    measured, exit code 11. That turns a green suite into a nonzero exit for
    nothing. Test modules keep solvers and textures in module globals, so
    drop those references here, while the context is still alive. Best
    effort: a sweep failure must never mask the test result.
    """
    try:
        import gc

        import gpu

        from marrow.blender import handlers
        from marrow.blender.session import MarrowSession
        from marrow.gpu.solver import GPUSolver

        handlers.free_all()
        holders = (
            GPUSolver,
            MarrowSession,
            gpu.types.GPUTexture,
            gpu.types.GPUShader,
        )
        for module in modules:
            for name, value in list(vars(module).items()):
                try:
                    if isinstance(value, holders):
                        delattr(module, name)
                except Exception:
                    pass
        gc.collect()
    except Exception as exc:
        print(f"warning: GPU state sweep failed: {exc}")


def main():
    # Auto-discover, so adding a test file needs no edit here. Import failures
    # are counted rather than raised: an exception escaping main() would leave
    # Blender to exit 0, so a module that will not even import would read as a
    # clean run. Measured — that is exactly what happened before this guard.
    paths = sorted(HERE.glob("test_*.py"))
    modules = []
    failures = 0
    for path in paths:
        try:
            modules.append(importlib.import_module(path.stem))
        except Exception:
            failures += 1
            print(f"FAIL {path.stem} (import)")
            traceback.print_exc()

    if not paths:
        print("no test modules found")
        sys.exit(1)

    try:
        for module in modules:
            for name in sorted(n for n in dir(module) if n.startswith("test_")):
                try:
                    getattr(module, name)()
                    print(f"PASS {module.__name__}.{name}")
                except Exception:
                    failures += 1
                    print(f"FAIL {module.__name__}.{name}")
                    traceback.print_exc()
    except Exception:
        failures += 1
        traceback.print_exc()
    finally:
        _release_gpu_state(modules)
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


main()
