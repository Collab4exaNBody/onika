"""
Build and smoke-run checks of an input file, each in a fresh run directory.

The run directory holds the input and a symbolic link to every entry of the
user's work directory, so that files referenced with relative paths are found
while outputs written by the run do not land among the user's files.

* ``check_build``: build the simulation graph without running it (operator
  names, batch structure, slot types)
* ``smoke_run``: build and run with the caller's overrides (typically a couple
  of iterations and no outputs), returning the thermodynamic state when the
  application provides a reader
"""

import os
import shutil
import tempfile

from .worker import run_worker

__all__ = ["STRICT_CONFIG", "check_build", "smoke_run", "merge"]

# unknown parameters are errors instead of being ignored
STRICT_CONFIG = {"debug": {"strict_slots": True}}


def merge(base, extra):
    """Recursive merge of two YAML-like dicts (``extra`` wins)."""
    out = dict(base or {})
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


class _RunDir:
    def __init__(self, text, workdir, name):
        self.path = tempfile.mkdtemp(prefix="onika-run-")
        if workdir and os.path.isdir(workdir):
            for entry in os.listdir(workdir):
                if entry != name:
                    os.symlink(os.path.join(os.path.abspath(workdir), entry), os.path.join(self.path, entry))
        self.input = os.path.join(self.path, name)
        with open(self.input, "w") as f:
            f.write(text)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def check_build(package, text, workdir=None, config=None, timeout=120):
    with _RunDir(text, workdir, "input.msp") as rd:
        payload = {"input": rd.input, "config": merge(STRICT_CONFIG, config)}
        result = run_worker(package, "check", payload, timeout=timeout, cwd=rd.path)
    result["stage"] = "build"
    return result


def smoke_run(package, text, workdir=None, overrides=None, config=None, timeout=300, thermo_reader=None):
    with _RunDir(text, workdir, "input.msp") as rd:
        payload = {"input": rd.input, "overrides": overrides, "config": merge(STRICT_CONFIG, config),
                   "thermo_reader": thermo_reader}
        result = run_worker(package, "smoke", payload, timeout=timeout, cwd=rd.path)
    result["stage"] = "smoke"
    return result
