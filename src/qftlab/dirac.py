"""Dirac algebra: explicit 4x4 matrices for exact traces, and n-dimensional contraction identities.

Conventions (as in the thesis): metric diag(+,-,-,-), Dirac representation, {γ^μ, γ^ν} = 2 g^{μν}, and in
dimensional regularization n = 4 - ε, so that γ^μ γ_μ = n and one-loop poles appear as 2/ε.

Two levels:

* numeric - `slash`, `trace`: 4x4 complex matrices. Spin sums of tree amplitudes are computed as explicit traces
  at given momenta (exact up to float rounding), which is how the closed forms in `tree.py` are checked.
* symbolic - `sandwich`: the n-dimensional identities for γ^μ X γ_μ with X a product of 0..3 slashed vectors.
  These are the only Dirac identities one-loop QED needs; each is verified against the explicit matrices at n = 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

METRIC = np.diag([1.0, -1.0, -1.0, -1.0])
_I2 = np.eye(2)
_Z2 = np.zeros((2, 2))
_SIGMA = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex),
          np.array([[1, 0], [0, -1]], complex)]

GAMMA = [np.block([[_I2, _Z2], [_Z2, -_I2]]).astype(complex)] + [
    np.block([[_Z2, s], [-s, _Z2]]) for s in _SIGMA]          # γ^0..γ^3 (upper index)
GAMMA_LOWER = [METRIC[m, m] * GAMMA[m] for m in range(4)]      # γ_μ = g_μν γ^ν
GAMMA5 = 1j * GAMMA[0] @ GAMMA[1] @ GAMMA[2] @ GAMMA[3]
ONE = np.eye(4, dtype=complex)


def dot(a, b) -> float:
    """Minkowski product a·b for contravariant 4-vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a[0] * b[0] - a[1:] @ b[1:])


def slash(p) -> np.ndarray:
    """p-slash = γ^μ p_μ = γ^0 p^0 - γ^i p^i."""
    p = np.asarray(p, float)
    return GAMMA[0] * p[0] - GAMMA[1] * p[1] - GAMMA[2] * p[2] - GAMMA[3] * p[3]


def trace(*mats) -> complex:
    m = ONE
    for x in mats:
        m = m @ x
    return complex(np.trace(m))


def contract(fn) -> np.ndarray:
    """Σ_μ fn(γ^μ, γ_μ): sums an expression with one contracted Lorentz index."""
    return sum(fn(GAMMA[m], GAMMA_LOWER[m]) for m in range(4))


# ------------------------------------------------------------------------------------------------ n dimensions
n = sp.Symbol("n")          # spacetime dimension; n = 4 - eps
eps = sp.Symbol("epsilon")


@dataclass(frozen=True)
class Term:
    """coef * (product of slashed vectors, in order). `coef` may contain dot products a·b as symbols."""
    coef: sp.Expr
    slashes: tuple[str, ...]


def dot_symbol(a: str, b: str) -> sp.Symbol:
    a, b = sorted((a, b))
    return sp.Symbol(f"{a}.{b}") if a != b else sp.Symbol(f"{a}^2")


def sandwich(slashes: tuple[str, ...]) -> list[Term]:
    """γ^μ (a̸ b̸ ...) γ_μ in n dimensions, for up to three slashed vectors:

        γ^μ γ_μ           = n
        γ^μ a̸ γ_μ         = (2 - n) a̸
        γ^μ a̸ b̸ γ_μ       = 4 a·b - (4 - n) a̸ b̸
        γ^μ a̸ b̸ c̸ γ_μ     = -2 c̸ b̸ a̸ + (4 - n) a̸ b̸ c̸
    """
    k = len(slashes)
    if k == 0:
        return [Term(n, ())]
    if k == 1:
        return [Term(2 - n, slashes)]
    if k == 2:
        a, b = slashes
        return [Term(4 * dot_symbol(a, b), ()), Term(-(4 - n), (a, b))]
    if k == 3:
        a, b, c = slashes
        return [Term(sp.Integer(-2), (c, b, a)), Term(4 - n, (a, b, c))]
    raise NotImplementedError("one-loop QED needs at most three slashed vectors inside a contraction")


def symmetric_integration(inner: tuple[str, ...], loop: str = "l") -> list[Term]:
    """l̸ X l̸ under ∫d^n l with an even integrand: l^α l^β -> g^{αβ} l²/n, so l̸ X l̸ -> (l²/n) γ_α X γ^α."""
    l2 = sp.Symbol(f"{loop}^2")
    return [Term(t.coef * l2 / n, t.slashes) for t in sandwich(inner)]


def evaluate_terms(terms: list[Term], vectors: dict[str, np.ndarray], n_value: int = 4) -> np.ndarray:
    """Numerically evaluate a list of Terms at n = n_value with explicit 4-vectors (for verification)."""
    out = np.zeros((4, 4), complex)
    for t in terms:
        subs = {n: n_value}
        for s in t.coef.free_symbols:
            name = str(s)
            if "." in name:
                a, b = name.split(".")
                subs[s] = dot(vectors[a], vectors[b])
            elif name.endswith("^2"):
                v = vectors[name[:-2]]
                subs[s] = dot(v, v)
        c = complex(t.coef.subs(subs))
        m = ONE
        for name in t.slashes:
            m = m @ slash(vectors[name])
        out += c * m
    return out
