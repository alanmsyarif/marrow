"""Assert-based test runner for Blender-dependent code.

Blender's bundled Python has no pytest and we will not install into it.
Run: blender -b --factory-startup --python tests/blender/run_tests.py
Exits 1 on any failure so CI and humans both notice.

**One Blender per test module.** Running all of them in a single process was
measured to invent failures that are not there. The four contact modules,
run alone, failed zero times across nine full repetitions; run after the
other forty-odd, they failed one to three times a run, and the failing set
rotated. Bisecting what ran first gave 0, 6, 4, 0, 11, 6 - the count going
back DOWN rules out state corruption and leaves resource pressure: every
module holds solvers, shaders and textures alive in its globals for the
whole run, dozens of live solvers at about 25 textures each, and readbacks
start coming back wrong. One captured failure read GPU [0, 0, 0] where the
oracle wanted 1.083, which is a blank texture rather than numerical drift.

Sweeping between modules in one process was tried first and crashes: modules
hold references to each other's GPU objects, which is why the sweep below is
exit-only. A process per module is the version that works.

The cost is Blender's startup, once per module. The benefit is that a red
test means the code is wrong.

KNOWN, and what is left after this: test_velocity_clamp fails its last test
about three runs in five, and only when its own two siblings have run first
- alone, or after just the first, it passes every time. It reports 26.4 m/s
against a 0.960 cap, which is the node not being marked as in contact rather
than the clamp being wrong, and the same scenario in a fresh process marks
and caps it correctly. So it is one module, one ordered pair, and a coin
weighted three to two, instead of a rotating set drawn from six modules that
nobody could bisect. gc between the tests does not help, so it is not Python
holding the references.
"""

import importlib
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Both paths must be set BEFORE any test module is imported.
sys.path.insert(0, str(ROOT))   # so `import marrow` works
sys.path.insert(0, str(HERE))   # so test modules import as top-level names

# The worker prints this so the driver never has to count PASS lines. Blender
# interleaves driver warnings into stdout mid-line - "Push constants hPASS
# test_live..." is a real example - so anything parsed out of that stream
# undercounts. One line, printed last, with the numbers already totalled.
RESULT = "MARROW-RESULT"


def _script_args():
    """Whatever followed `--` on the Blender command line."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


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


def run_modules(names):
    """Run the named modules in THIS process. Returns (passed, failed)."""
    # Blender 5.2 needs this before any GPU call, and only some test modules
    # do it themselves - the rest used to ride on whichever module happened
    # to run first. That worked only because everything shared one process.
    # Owning it here is what makes a module runnable on its own.
    try:
        import gpu

        gpu.init()
    except Exception as exc:
        print(f"warning: gpu.init() failed: {exc}")

    modules = []
    passed = failed = 0
    for name in names:
        try:
            modules.append(importlib.import_module(name))
        except Exception:
            # Counted rather than raised: an exception escaping here would
            # leave Blender to exit 0, so a module that will not even import
            # would read as a clean run. Measured - that is what happened
            # before this guard.
            failed += 1
            print(f"FAIL {name} (import)")
            traceback.print_exc()

    try:
        for module in modules:
            for test in sorted(n for n in dir(module) if n.startswith("test_")):
                try:
                    getattr(module, test)()
                    passed += 1
                    print(f"PASS {module.__name__}.{test}")
                except Exception:
                    failed += 1
                    print(f"FAIL {module.__name__}.{test}")
                    traceback.print_exc()
    except Exception:
        failed += 1
        traceback.print_exc()
    finally:
        _release_gpu_state(modules)
    return passed, failed


def drive(names):
    """Run each module in its own Blender. Returns the number of failures."""
    import bpy

    blender = bpy.app.binary_path
    passed = failed = 0
    broken = []
    started = time.time()
    for i, name in enumerate(names, 1):
        proc = subprocess.run(
            [blender, "-b", "--factory-startup", "--python", str(HERE / "run_tests.py"),
             "--", "--module", name],
            capture_output=True, text=True,
        )
        line = next(
            (ln for ln in reversed(proc.stdout.splitlines()) if RESULT in ln), ""
        )
        if line:
            ok, bad = (int(n) for n in line.split(RESULT)[1].split())
            passed += ok
            failed += bad
            mark = "ok  " if bad == 0 else "FAIL"
        else:
            # No result line means the process died before it could print one:
            # a segfault, or a shader that took the context down. That is a
            # real failure and used to take the whole suite with it.
            ok, bad = 0, 1
            failed += 1
            broken.append(name)
            mark = "DIED"
        print(f"[{i:2}/{len(names)}] {mark} {name}", flush=True)
        if line and bad:
            for ln in proc.stdout.splitlines():
                if ln.startswith("FAIL "):
                    print(f"          {ln}")
        elif not line:
            for ln in proc.stdout.strip().splitlines()[-6:]:
                print(f"          {ln}")

    print(f"\n{passed} passed, {failed} failed, "
          f"{len(names)} modules, {time.time() - started:.0f}s")
    if broken:
        print("died before reporting: " + ", ".join(broken))
    return failed


def main():
    args = _script_args()
    names = sorted(p.stem for p in HERE.glob("test_*.py"))
    if not names:
        print("no test modules found")
        sys.exit(1)

    if "--module" in args:
        one = args[args.index("--module") + 1]
        ok, bad = run_modules([one])
        print(f"{RESULT} {ok} {bad}")
        sys.exit(1 if bad else 0)

    if "--in-process" in args:
        # The old behaviour, kept for a quick loop on one change. It is
        # faster and it lies: see the module docstring.
        ok, bad = run_modules(names)
        print(f"\n{bad} failure(s)")
        sys.exit(1 if bad else 0)

    sys.exit(1 if drive(names) else 0)


main()
