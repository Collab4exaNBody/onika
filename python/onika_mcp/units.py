"""
Static checking of onika quantities ("5.0 ang", "0.1 ps^-1", "50. m/s").

Mirrors onika/src/physics/units.cpp (quantity_from_string, make_quantity) and
the unit table of onika/include/onika/physics/units.h, and adds what onika does
not check: the physical dimension expected by a parameter.
"""

import math
import re

__all__ = ["UNITS", "DIMENSIONS", "parse_quantity", "check_quantity", "UnitError"]

# base dimensions: length, mass, time, charge, temperature, amount, luminosity, angle
_L, _M, _T, _Q, _K, _N, _J, _A = range(8)


def _dim(**powers):
    d = [0.0] * 8
    for name, p in powers.items():
        d[{"L": _L, "M": _M, "T": _T, "Q": _Q, "K": _K, "N": _N, "J": _J, "A": _A}[name]] = p
    return tuple(d)


_ENERGY = _dim(M=1, L=2, T=-2)

# symbol -> (SI factor, dimension), from onika/include/onika/physics/units.h
UNITS = {
    "m": (1.0, _dim(L=1)), "mm": (1e-3, _dim(L=1)), "um": (1e-6, _dim(L=1)),
    "nm": (1e-9, _dim(L=1)), "ang": (1e-10, _dim(L=1)),
    "kg": (1.0, _dim(M=1)), "g": (1e-3, _dim(M=1)), "Da": (1.66053906660e-27, _dim(M=1)),
    "h": (3600.0, _dim(T=1)), "s": (1.0, _dim(T=1)), "us": (1e-6, _dim(T=1)),
    "ns": (1e-9, _dim(T=1)), "ps": (1e-12, _dim(T=1)), "fs": (1e-15, _dim(T=1)),
    "C": (1.0, _dim(Q=1)), "e-": (1.602176634e-19, _dim(Q=1)),
    "K": (1.0, _dim(K=1)), "°": (1.0, _dim(K=1)),
    "mol": (1.0, _dim(N=1)), "particle": (1.0 / 6.02214076e23, _dim(N=1)),
    "cd": (1.0, _dim(J=1)),
    "rad": (1.0, _dim(A=1)), "degree": (math.pi / 180.0, _dim(A=1)),
    "J": (1.0, _ENERGY), "kJ": (1e3, _ENERGY), "eV": (1.602176634e-19, _ENERGY),
    "cal": (4.184, _ENERGY), "kcal": (4.184e3, _ENERGY),
}

# named dimensions used by block schemas
DIMENSIONS = {
    "none": _dim(),
    "length": _dim(L=1),
    "mass": _dim(M=1),
    "time": _dim(T=1),
    "inverse_time": _dim(T=-1),
    "velocity": _dim(L=1, T=-1),
    "charge": _dim(Q=1),
    "temperature": _dim(K=1),
    "energy": _ENERGY,
    "pressure": _dim(M=1, L=-1, T=-2),
    "force": _dim(M=1, L=1, T=-2),
    "angle": _dim(A=1),
}

# frequent mistakes -> valid symbol
_SUGGEST = {"A": "ang", "Å": "ang", "angstrom": "ang", "Angstrom": "ang", "amu": "Da", "u": "Da",
            "sec": "s", "kelvin": "K", "ev": "eV", "e": "e-", "deg": "degree", "nanosecond": "ns",
            "picosecond": "ps", "femtosecond": "fs", "fs-1": "fs^-1", "ps-1": "ps^-1"}


class UnitError(ValueError):
    pass


def _dimension_name(dim):
    for name, d in DIMENSIONS.items():
        if all(abs(a - b) < 1e-12 for a, b in zip(d, dim)):
            return name
    return "(" + " ".join(f"{n}^{p:g}" for n, p in zip("LMTQKNJA", dim) if p) + ")"


def _parse_units(expr):
    """'ang^6/kcal' -> (factor, dimension); same splitting rules as make_quantity."""
    inverse = expr.startswith("1/")
    if inverse:
        expr = expr[2:]
    if not expr or expr[-1] in "/*." or expr[0] in "/*.":
        raise UnitError(f"malformed unit '{expr}'")
    factor, dim = 1.0, [0.0] * 8
    power = 1.0
    for token, sep in re.findall(r"([^/*.]+)([/*.]?)", expr):
        sym, p = token, 1.0
        if "^" in token:
            sym, ptxt = token.split("^", 1)
            try:
                p = float(ptxt)
            except ValueError:
                raise UnitError(f"invalid power '{ptxt}' in unit '{expr}'")
        if sym not in UNITS:
            hint = _SUGGEST.get(sym)
            raise UnitError(f"unknown unit '{sym}'" + (f" (did you mean '{hint}'?)" if hint else "")
                            + ". Valid units: " + " ".join(UNITS))
        f, d = UNITS[sym]
        e = power * p
        factor *= f ** e
        dim = [a + b * e for a, b in zip(dim, d)]
        power = -1.0 if sep == "/" else 1.0
    if inverse:
        factor, dim = 1.0 / factor, [-a for a in dim]
    return factor, tuple(dim)


def parse_quantity(text):
    """Parse '<number> [<units>]' -> (value_in_SI, dimension, has_units)."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text), DIMENSIONS["none"], False
    parts = str(text).split()
    if not parts:
        raise UnitError("empty quantity")
    try:
        value = float(parts[0])
    except ValueError:
        raise UnitError(f"'{parts[0]}' is not a number (in '{text}')")
    if len(parts) == 1:
        return value, DIMENSIONS["none"], False
    if len(parts) > 2:
        raise UnitError(f"unexpected text after the unit in '{text}' (write units without spaces, e.g. 'm/s')")
    factor, dim = _parse_units(parts[1])
    return value * factor, dim, True


def check_quantity(text, dimension=None):
    """Return None if ``text`` is a valid quantity of ``dimension`` (a DIMENSIONS
    key), else an error message.  Bare numbers are accepted: onika takes them
    in internal units (ang, Da, ps, e-, K)."""
    try:
        _, dim, has_units = parse_quantity(text)
    except UnitError as e:
        return str(e)
    if dimension and has_units:
        expected = DIMENSIONS[dimension]
        # molar forms are accepted too (g/mol for a mass, kcal/mol for an energy):
        # onika converts amounts to particles
        molar = tuple(e - (1.0 if i == _N else 0.0) for i, e in enumerate(expected))
        if not any(all(abs(a - b) < 1e-12 for a, b in zip(dim, ref)) for ref in (expected, molar)):
            return f"'{text}' is a {_dimension_name(dim)}, expected a {dimension}"
    return None
