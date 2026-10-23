"""
MPI pre-initialization for Python sessions.

onika's ``initialize_mpi()`` treats an already initialized MPI as externally
owned and then never calls ``MPI_Finalize``.  Initializing MPI once from
Python therefore allows several ``init()/run()/end()`` cycles in the same
process; MPI is finalized by an ``atexit`` handler.

The MPI library is the one onika is linked against: it is located with
``dladdr`` on ``MPI_Initialized`` as resolved from the already loaded
``pyonika._core`` extension, which works on Linux and macOS.
"""

import atexit
import ctypes
import os
import sys

__all__ = ["ensure_mpi_initialized", "mpi_library", "install_abort_on_exception"]

# MPI standard: MPI_THREAD_SINGLE < FUNNELED < SERIALIZED < MULTIPLE.
# Open MPI and MPICH both use 0..3 in this order.
_MPI_THREAD_MULTIPLE = 3

_lib = None


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]


def mpi_library():
    """Return a ctypes handle on the MPI library used by onika, or None."""
    global _lib
    if _lib is not None:
        return _lib
    from . import _core
    try:
        core = ctypes.CDLL(_core.__file__)
        addr = ctypes.cast(core.MPI_Initialized, ctypes.c_void_p).value
    except (OSError, AttributeError):
        return None
    libc = ctypes.CDLL(None)
    info = _DlInfo()
    if not libc.dladdr(ctypes.c_void_p(addr), ctypes.byref(info)) or not info.dli_fname:
        return None
    path = os.fsdecode(info.dli_fname)
    try:
        _lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None
    return _lib


def ensure_mpi_initialized():
    """Initialize MPI (MPI_THREAD_MULTIPLE) if needed and register finalization.

    Returns True if MPI is initialized after the call.
    """
    lib = mpi_library()
    if lib is None:
        return False
    flag = ctypes.c_int(0)
    lib.MPI_Initialized(ctypes.byref(flag))
    if flag.value:
        return True
    provided = ctypes.c_int(0)
    lib.MPI_Init_thread(None, None, ctypes.c_int(_MPI_THREAD_MULTIPLE), ctypes.byref(provided))

    def _finalize():
        done = ctypes.c_int(0)
        lib.MPI_Finalized(ctypes.byref(done))
        if not done.value:
            lib.MPI_Finalize()
    atexit.register(_finalize)
    return True


def install_abort_on_exception():
    """Abort all MPI processes when a Python exception is not caught.

    With several MPI processes, an uncaught exception on one process would end
    it while the others wait forever in a collective operation.  The installed
    ``sys.excepthook`` prints the traceback as usual, then calls ``MPI_Abort``
    when more than one process runs.  Installed by ``import pyonika``.
    """
    from . import _core
    previous = sys.excepthook

    def _hook(exc_type, exc, tb):
        previous(exc_type, exc, tb)
        if _core._mpi_size() > 1:
            sys.stderr.flush()
            _core._mpi_abort(1)

    if getattr(sys.excepthook, "_pyonika_mpi_abort", False):
        return
    _hook._pyonika_mpi_abort = True
    sys.excepthook = _hook
