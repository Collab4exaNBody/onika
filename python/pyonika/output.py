"""
Capture output written by onika's C++ code.

onika logs through C++ streams (``lout``, ``lerr``) and printf, which write
directly to the process file descriptors 1 and 2: Python's ``sys.stdout``
(Jupyter, loggers, test runners) never sees them.  ``capture_output()``
redirects those descriptors while a block runs.
"""

import contextlib
import os
import sys
import tempfile

from . import _core

__all__ = ["capture_output"]


@contextlib.contextmanager
def capture_output(stream=None, stderr=True):
    """Redirect C++ (and Python) output written during the block.

    The captured text is written to ``stream`` (default: ``sys.stdout``) when
    the block exits; pass an ``io.StringIO`` to keep it::

        buf = io.StringIO()
        with pyonika.capture_output(buf):
            pyonika.run(ctx)
        log = buf.getvalue()

    ``stderr=False`` leaves file descriptor 2 untouched.
    """
    target = sys.stdout if stream is None else stream
    fds = (1, 2) if stderr else (1,)
    _core._flush_output()
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except (AttributeError, ValueError):
            pass
    with tempfile.TemporaryFile() as tmp:
        saved = [os.dup(fd) for fd in fds]
        try:
            for fd in fds:
                os.dup2(tmp.fileno(), fd)
            yield
        finally:
            _core._flush_output()
            for s in (sys.stdout, sys.stderr):
                try:
                    s.flush()
                except (AttributeError, ValueError):
                    pass
            for fd, old in zip(fds, saved):
                os.dup2(old, fd)
                os.close(old)
            tmp.seek(0)
            target.write(tmp.read().decode(errors="replace"))
