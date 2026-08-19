"""Running a chunked pass to completion when nobody is watching.

The slow passes are written as generators so the Tetrahedralize operator can
pull one slice at a time and let Blender repaint in between - a blocking run
takes over a minute on a dense cage and Windows reports the window as Not
Responding, which reads as a crash. Everything else - the tests, the bake
path, any script - just wants the answer, so it drains the generator here.

One implementation, two entry points. The alternative was a blocking copy of
each algorithm beside the chunked one, and the two would drift.
"""


def drain(work):
    """Exhaust a progress generator and return what it returns."""
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value
