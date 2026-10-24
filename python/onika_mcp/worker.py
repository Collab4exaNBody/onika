"""
Run onika work in a separate process.

onika can still abort the whole process (fatal errors inside OpenMP regions,
crashes in operators), so an MCP server never runs simulations itself: it calls
``run_worker()``, which starts ``python -m onika_mcp.worker`` with a timeout
and returns a JSON-able dict, even when the child crashes.

Child commands (the application package, e.g. ``pyexastamp``, is imported first
so that its environment and extractors are set up):

* ``env``: ONIKA_* search paths set up by the application package
* ``catalog``: every registered operator with its documentation and slots
* ``defaults``: operator definitions loaded from the default configuration
* ``check``: build the graph of an input file without running it
* ``smoke``: build and run an input file for a few iterations
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time

__all__ = ["run_worker", "parse_fatal_errors"]

_FATAL_RE = re.compile(r"\*+\n\*+ FATAL ERROR \*+\n\*+\n(.*?)\*{20,}", re.S)


def parse_fatal_errors(text):
    """Messages of the onika FATAL ERROR blocks found in ``text``."""
    return [m.strip() for m in _FATAL_RE.findall(text or "")]


def _warnings(log):
    return sorted({l.strip() for l in log.splitlines() if l.strip().startswith("Warning:")})


# ---------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------

def run_worker(package, command, payload=None, timeout=300, cwd=None):
    """Run ``command`` in a child Python process and return its result dict.

    The result always has ``ok`` (bool); on failure ``error`` holds the
    explanation (onika fatal error message, crash signal or timeout) and ``log``
    the end of the child's output.
    """
    payload = payload or {}
    with tempfile.TemporaryDirectory(prefix="onika-worker-") as tmp:
        result_file = os.path.join(tmp, "result.json")
        payload_file = os.path.join(tmp, "payload.json")
        with open(payload_file, "w") as f:
            json.dump(payload, f)
        argv = [sys.executable, "-m", "onika_mcp.worker", package, command, payload_file, result_file]
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
        start = time.monotonic()
        try:
            proc = subprocess.run(argv, cwd=cwd or tmp, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            log = (e.stdout or "") if isinstance(e.stdout, str) else ""
            return {"ok": False, "error": f"timeout after {timeout} s", "log": log[-4000:]}
        elapsed = time.monotonic() - start
        log = (proc.stdout or "") + (proc.stderr or "")
        if os.path.exists(result_file):
            with open(result_file) as f:
                result = json.load(f)
        else:
            fatal = parse_fatal_errors(log)
            if fatal:
                error = fatal[-1]
            elif proc.returncode < 0:
                error = f"process crashed (signal {-proc.returncode})"
            else:
                error = f"process exited with code {proc.returncode} without result"
            result = {"ok": False, "error": error}
        result.setdefault("warnings", _warnings(log))
        result["log"] = log[-4000:]
        result["elapsed"] = round(elapsed, 2)
        return result


# ---------------------------------------------------------------------------
# child side
# ---------------------------------------------------------------------------

def _error_text(exc):
    text = str(exc)
    fatal = parse_fatal_errors(text)
    if fatal:
        return fatal[-1]
    lines = [l for l in text.splitlines() if l.strip() and not set(l.strip()) <= {"*"} and "FATAL ERROR" not in l]
    return "\n".join(lines).strip()


def _cmd_env(app, payload):
    keys = ("ONIKA_PLUGIN_PATH", "ONIKA_CONFIG_PATH", "ONIKA_DATA_PATH")
    # install locations exported by the application package (e.g. XSP_DATA_DIR)
    paths = {k: v for k, v in vars(app).items() if k.endswith("_DIR") and isinstance(v, str)}
    return {"ok": True, "env": {k: os.environ.get(k, "") for k in keys}, "paths": paths}


def _cmd_catalog(app, payload):
    import pyonika
    ctx = app.init(["catalog", "--help"])   # loads plugins, builds nothing
    operators = {}
    for name in pyonika.available_operators():
        try:
            operators[name] = pyonika.describe_operator(name)
        except Exception as e:  # e.g. operators that need a grid flavor to be instantiated
            operators[name] = {"name": name, "error": _error_text(e)}
    pyonika.end(ctx)
    return {"ok": True, "operators": operators}


def _start(app, payload):
    argv = app.command_line(payload["input"], program="worker",
                            overrides=payload.get("overrides"), config=payload.get("config"))
    ctx = app.init(argv)
    if ctx.error_code >= 0:
        raise RuntimeError(f"input stopped early (error code {ctx.error_code})")
    return ctx


def _cmd_defaults(app, payload):
    import pyonika
    config_dir = os.environ.get("ONIKA_CONFIG_PATH", "")
    ctx = app.init(["defaults", os.path.join(config_dir, "main-config.msp")])
    defaults = pyonika.get_operator_defaults()
    pyonika.end(ctx)
    return {"ok": True, "defaults": defaults}


def _cmd_check(app, payload):
    ctx = _start(app, payload)
    app.end(ctx)
    return {"ok": True}


def _cmd_smoke(app, payload):
    import pyonika
    ctx = _start(app, payload)
    log = io.StringIO()
    with pyonika.capture_output(log):
        app.run(ctx)
    result = {"ok": True}
    reader = payload.get("thermo_reader")
    if reader and hasattr(app, reader):
        thermo = getattr(app, reader)(ctx)
        if thermo:
            result["thermo"] = {k: v for k, v in thermo.items() if isinstance(v, (int, float))}
    app.end(ctx)
    result["run_log"] = log.getvalue()[-6000:]
    return result


_COMMANDS = {"env": _cmd_env, "catalog": _cmd_catalog, "defaults": _cmd_defaults,
             "check": _cmd_check, "smoke": _cmd_smoke}


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("command", choices=sorted(_COMMANDS))
    parser.add_argument("payload")
    parser.add_argument("result")
    args = parser.parse_args()
    with open(args.payload) as f:
        payload = json.load(f)
    try:
        app = __import__(args.package)
        result = _COMMANDS[args.command](app, payload)
    except Exception as e:
        result = {"ok": False, "error": _error_text(e), "error_type": type(e).__name__}
    with open(args.result, "w") as f:
        json.dump(result, f, default=str)


if __name__ == "__main__":
    _main()
