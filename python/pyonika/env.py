"""
Runtime environment for onika-based applications.

onika reads three environment variables when ``init()`` is called:

* ``ONIKA_PLUGIN_PATH`` -- colon-separated directories scanned for plugins,
* ``ONIKA_CONFIG_PATH`` -- a single directory holding ``main-config.msp``,
* ``ONIKA_DATA_PATH``   -- colon-separated directories searched for data files.

``setup_env()`` fills them from one or more install prefixes (onika, then
exaNBody, then the end-user application, ...), so Python users do not need to
source ``setup-env.sh``.  Variables already set by the user before the first
call are respected.
"""

import os

__all__ = ["setup_env", "prefixes"]

# Values of the variables before any setup_env() call: these come from the
# user (e.g. a sourced setup-env.sh) and always take precedence.
_USER_ENV = {k: os.environ.get(k) for k in ("ONIKA_PLUGIN_PATH", "ONIKA_CONFIG_PATH", "ONIKA_DATA_PATH")}

_prefixes = []


def prefixes():
    """Install prefixes registered so far, lowest layer (onika) first."""
    return list(_prefixes)


def _split(value):
    return [p for p in (value or "").split(":") if p]


def _merge(current, extra):
    parts = _split(current)
    for p in extra:
        if p not in parts:
            parts.append(p)
    return ":".join(parts)


def _data_dirs(prefix):
    """Data directories of an install prefix: share/<subdir> and data/."""
    dirs = []
    share = os.path.join(prefix, "share")
    if os.path.isdir(share):
        dirs += sorted(os.path.join(share, d) for d in os.listdir(share)
                       if os.path.isdir(os.path.join(share, d)))
    data = os.path.join(prefix, "data")
    if os.path.isdir(data):
        dirs.append(data)
    return dirs


def setup_env(prefix_list):
    """Register install prefixes and update the ONIKA_* environment variables.

    Prefixes are given from the lowest layer to the application, e.g.
    ``[onika_prefix, exanbody_prefix, exastamp_prefix]``.  Plugin and data
    directories of every prefix are appended; the configuration directory is
    the ``data/config`` of the last prefix that has one (the application's
    ``main-config.msp`` includes the lower layers' defaults).

    Must be called before ``pyonika.init()``.  Calling it again with more
    prefixes extends the environment.
    """
    for p in prefix_list:
        p = os.path.abspath(p)
        if p not in _prefixes:
            _prefixes.append(p)

    if not _USER_ENV["ONIKA_PLUGIN_PATH"]:
        os.environ["ONIKA_PLUGIN_PATH"] = _merge(
            os.environ.get("ONIKA_PLUGIN_PATH"),
            [os.path.join(p, "plugins") for p in _prefixes if os.path.isdir(os.path.join(p, "plugins"))])

    if not _USER_ENV["ONIKA_CONFIG_PATH"]:
        configs = [os.path.join(p, "data", "config") for p in _prefixes
                   if os.path.isfile(os.path.join(p, "data", "config", "main-config.msp"))]
        if configs:
            os.environ["ONIKA_CONFIG_PATH"] = configs[-1]

    if not _USER_ENV["ONIKA_DATA_PATH"]:
        dirs = ["."]
        for p in _prefixes:
            dirs += _data_dirs(p)
        os.environ["ONIKA_DATA_PATH"] = _merge(os.environ.get("ONIKA_DATA_PATH"), dirs)
