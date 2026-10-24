"""
Operator catalog: documentation and slots of every registered operator.

Built once by a worker process (plugins must be loaded) and cached as JSON,
keyed by the plugin and configuration directories and their modification
times, so that a rebuilt or reinstalled application refreshes it.
"""

import difflib
import hashlib
import json
import os

from .worker import run_worker

__all__ = ["Catalog"]


_CACHE_VERSION = 2   # bump when the cached content changes


def _fingerprint(*search_paths):
    h = hashlib.sha1(str(_CACHE_VERSION).encode())
    for d in sorted({p for sp in search_paths for p in sp.split(":") if p}):
        h.update(d.encode())
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                st = os.stat(os.path.join(d, f))
                h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


class Catalog:
    """Operators of an application: ``catalog.get(name)``, ``catalog.search(text)``."""

    def __init__(self, package, cache_dir=None):
        self.package = package
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "onika-mcp")
        self._operators = None
        self.cache_key = None   # identifies the install: f"{package}-{fingerprint}"

    @property
    def data(self):
        if self._operators is None:
            self._operators = self._load()
        return self._operators

    @property
    def operators(self):
        return self.data["operators"]

    @property
    def defaults(self):
        """Operator definitions of the default configuration (main-config.msp and includes)."""
        return self.data["defaults"]

    @property
    def data_dirs(self):
        """Directories searched for data files (ONIKA_DATA_PATH of the application)."""
        return [d for d in self.data["env"]["ONIKA_DATA_PATH"].split(":") if d]

    @property
    def paths(self):
        """Install directories exported by the application package (``*_DIR`` variables)."""
        return self.data.get("paths", {})

    def find_data_file(self, name):
        if os.path.isabs(name):
            return name if os.path.isfile(name) else None
        for d in self.data_dirs:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return os.path.abspath(p)
        return None

    def _load(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        env = run_worker(self.package, "env", timeout=120)
        if not env.get("ok"):
            raise RuntimeError(f"cannot import {self.package}: {env.get('error')}")
        fingerprint = _fingerprint(env["env"]["ONIKA_PLUGIN_PATH"], env["env"]["ONIKA_CONFIG_PATH"])
        self.cache_key = f"{self.package}-{fingerprint}"
        path = os.path.join(self.cache_dir, f"{self.cache_key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        result = run_worker(self.package, "catalog", timeout=600)
        if not result.get("ok"):
            raise RuntimeError(f"cannot build the operator catalog: {result.get('error')}")
        defaults = run_worker(self.package, "defaults", timeout=600)
        if not defaults.get("ok"):
            raise RuntimeError(f"cannot read the default configuration: {defaults.get('error')}")
        data = {"operators": result["operators"], "defaults": defaults["defaults"], "env": env["env"],
                "paths": env.get("paths", {})}
        with open(path, "w") as f:
            json.dump(data, f)
        return data

    def names(self):
        return sorted(self.operators)

    def get(self, name):
        return self.operators.get(name)

    def slot_names(self, name):
        op = self.get(name)
        if not op or "slots" not in op:
            return None
        return {s["name"] for s in op["slots"]}

    def closest(self, name, n=3):
        return difflib.get_close_matches(name, self.names(), n=n, cutoff=0.6)

    def search(self, text, limit=20):
        """Operators whose name or documentation contains every word of ``text``."""
        words = [w.lower() for w in text.split()]
        hits = []
        for name, op in self.operators.items():
            blob = (name + " " + op.get("documentation", "")).lower()
            if all(w in blob for w in words):
                hits.append({"name": name, "documentation": op.get("documentation", "")[:200]})
        hits.sort(key=lambda h: (not h["name"].lower().startswith(words[0]) if words else False, h["name"]))
        return hits[:limit]
