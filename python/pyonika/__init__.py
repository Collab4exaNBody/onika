"""
pyonika -- Python interface to the onika simulation platform.

The compiled bindings live in ``pyonika._core``; this package re-exports them
and adds the runtime helpers shared by every onika-based application:

* ``pyonika.env.setup_env(prefixes)`` sets the ONIKA_* variables from install
  prefixes (done here for onika itself; applications add their own prefix),
* ``pyonika.mpi.ensure_mpi_initialized()`` initializes MPI once per process so
  that ``init()/run()/end()`` can be called several times,
* ``pyonika.capture_output()`` redirects output written by C++ code to Python.

Errors raised by onika (unknown operator, fatal_error() in a single-process
run, ...) are ``pyonika.OnikaError``, a subclass of ``RuntimeError``.
"""

from . import _core
from ._core import (  # noqa: F401
    init,
    run,
    run_node,
    end,
    make_operator,
    available_operators,
    get_operator_defaults,
    set_operator_defaults,
    build_simulation_graph,
    slot_as_array,
    ApplicationContext,
    OperatorNode,
    OperatorSlotBase,
    OnikaError,
    _slot_extractor_api,
    _register_slot_extractor_fn,
)
from ._paths import ONIKA_INSTALL_DIR
from .env import setup_env
from .mpi import ensure_mpi_initialized, install_abort_on_exception
from .output import capture_output

__all__ = [
    "init", "run", "run_node", "end",
    "make_operator", "available_operators", "get_operator_defaults", "set_operator_defaults",
    "build_simulation_graph", "slot_as_array",
    "ApplicationContext", "OperatorNode", "OperatorSlotBase", "OnikaError",
    "ONIKA_INSTALL_DIR", "setup_env", "ensure_mpi_initialized", "capture_output",
    "describe_operator", "command_line", "run_file",
]

setup_env([ONIKA_INSTALL_DIR])
ensure_mpi_initialized()
install_abort_on_exception()


def _cmdline_options(prefix, values):
    """Turn a nested dict into onika command line options.

    onika maps ``--a-b value`` to ``configuration: {a: {b: value}}``; values
    are parsed as YAML.
    """
    argv = []
    for k, v in values.items():
        key = f"{prefix}-{k}" if prefix else str(k)
        if isinstance(v, dict):
            argv += _cmdline_options(key, v)
        else:
            if isinstance(v, bool):
                v = "true" if v else "false"
            argv += [f"--{key}", str(v)]
    return argv


def command_line(msp_file, *args, program="pyonika", overrides=None, config=None):
    """Build the argv list used by ``init()`` for an input file.

    ``overrides`` are merged on top of the input file's operator definitions
    (through onika's ``configuration.set`` block), e.g.
    ``{"global": {"dt": 1e-3, "max_iteration": 10}}``.  ``config`` sets
    ``configuration`` options, e.g. ``{"omp_num_threads": 4}``.  Extra
    positional ``args`` are appended verbatim.
    """
    argv = [program, str(msp_file)]
    argv += _cmdline_options("set", overrides or {})
    argv += _cmdline_options("", config or {})
    argv += list(args)
    return argv


def run_file(msp_file, *args, program="pyonika", overrides=None, config=None):
    """Run an input file end to end: init, run, end.

    See ``command_line()`` for ``overrides`` and ``config``.
    Returns the context's error code (0 on success).
    """
    ctx = init(command_line(msp_file, *args, program=program, overrides=overrides, config=config))
    code = ctx.error_code
    if code < 0:
        run(ctx)
        code = 0
    end(ctx)
    return code


def describe_operator(name):
    """Documentation of a registered operator and of its slots, as a dict.

    Same as ``make_operator(name).describe()``: ``name``, ``documentation`` and
    ``slots``, each slot with ``name``, ``type``, ``direction``, ``required``,
    ``documentation`` and default ``value``.  Plugins must be loaded, i.e.
    ``init()`` must have been called.
    """
    return make_operator(name).describe()
