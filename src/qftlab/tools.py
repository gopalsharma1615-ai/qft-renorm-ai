"""CAS tools an LLM can call (shared by the Experiment B harness and the MCP server).

Every tool takes and returns plain JSON. Expressions from a model are parsed through a strict whitelist: only
digits, operators, parentheses and known names reach SymPy, never arbitrary Python.
"""
from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from . import loops
from .dirac import eps, n

_ALLOWED_NAMES = {"pi", "E", "I", "sqrt", "log", "exp", "gamma", "sin", "cos", "tan", "Rational", "oo",
                  "eps", "epsilon", "n", "alpha", "e", "m", "M", "mu", "s", "t", "u", "x", "y", "q2", "p2", "Delta",
                  "nf", "EulerGamma"}
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/^().,\s]*$")


def safe_parse(text: str) -> sp.Expr:
    text = text.strip().replace("^", "**").replace("ε", "eps").replace("π", "pi")
    if not _CHARS.match(text) or "__" in text:
        raise ValueError("expression contains characters outside the allowed set")
    bad = {t for t in _TOKEN.findall(text) if t not in _ALLOWED_NAMES}
    if bad:
        raise ValueError(f"unknown names: {sorted(bad)}")
    local = {k: sp.Symbol(k) for k in _ALLOWED_NAMES if k not in ("pi", "E", "I", "sqrt", "log", "exp", "gamma",
                                                                    "sin", "cos", "tan", "Rational", "oo",
                                                                    "EulerGamma")}
    local.update({"eps": eps, "epsilon": eps, "n": n, "pi": sp.pi, "E": sp.E, "I": sp.I, "gamma": sp.gamma,
                  "EulerGamma": sp.EulerGamma})
    return parse_expr(text, local_dict=local, transformations=standard_transformations, evaluate=True)


def equivalent(a: str, b: str) -> bool:
    """Symbolic equivalence, with a numeric fallback at random points when simplify is inconclusive."""
    try:
        ea, eb = safe_parse(a), safe_parse(b)
    except Exception:  # noqa: BLE001 - an unparsable answer is simply wrong
        return False
    diff = sp.simplify(ea - eb)
    if diff == 0:
        return True
    syms = sorted(diff.free_symbols, key=str)
    import random

    rnd = random.Random(0)
    for _ in range(5):
        pt = {s: sp.Rational(rnd.randint(11, 97), rnd.randint(7, 13)) for s in syms}
        try:
            v = complex(diff.subs(pt).evalf())
        except (TypeError, ValueError):
            return False
        if abs(v) > 1e-9 * (1 + abs(complex(ea.subs(pt).evalf()))):
            return False
    return True


# ------------------------------------------------------------------------------------------------ tools
def tool_gamma_trace(indices: list[str]) -> str:
    """Trace of a product of 4D Dirac matrices. Upper index "mu", lower index "-mu"; a contracted pair is written
    once up and once down, e.g. ["mu","nu","-mu","-nu"]. Result in terms of metric(a,b)."""
    from sympy.physics.hep.gamma_matrices import GammaMatrix as G
    from sympy.physics.hep.gamma_matrices import LorentzIndex, gamma_trace
    from sympy.tensor.tensor import tensor_indices

    if not 0 < len(indices) <= 8:
        raise ValueError("1..8 indices")
    parsed = [(i.strip().startswith("-"), re.sub(r"\W", "", i) or "a") for i in indices]
    names = list(dict.fromkeys(nm for _, nm in parsed))
    idx = tensor_indices(" ".join(names), LorentzIndex)
    lookup = dict(zip(names, idx if isinstance(idx, (list, tuple)) else (idx,)))
    prod = sp.Integer(1)
    for lower, nm in parsed:
        prod = prod * G(-lookup[nm] if lower else lookup[nm])
    return str(gamma_trace(prod))


def tool_sandwich(k: int) -> str:
    """n-dimensional γ^μ (k slashed vectors a̸ b̸ c̸ ...) γ_μ, for k = 0..3."""
    from .dirac import sandwich

    names = ("a", "b", "c")[:k]
    return " + ".join(f"({t.coef})*{'*'.join(f'{s}slash' for s in t.slashes) or '1'}" for t in sandwich(names))


def tool_master_integral(a: int, numerator: str = "1") -> dict:
    """∫ d^n l/(2π)^n N/(l² - Δ)^a in n = 4 - eps for N = 1 or l2: full result and its 1/eps pole."""
    f = loops.I0 if numerator == "1" else loops.I2
    full = f(int(a))
    return {"result": str(sp.simplify(full)), "pole": str(loops.pole(full))}


def tool_series_eps(expr: str, order: int = 0) -> str:
    """Laurent expansion in eps around 0 up to eps**order."""
    return str(sp.series(safe_parse(expr), eps, 0, int(order) + 1).removeO())


def tool_simplify(expr: str) -> str:
    return str(sp.simplify(safe_parse(expr)))


def tool_qed_counterterms() -> dict:
    """One-loop QED MS counterterms derived by qftlab (n = 4 - eps, Feynman gauge) and the β-function."""
    ct = loops.counterterms()
    b = loops.beta_function()
    return {k: str(v) for k, v in {**ct, **b}.items()}


TOOLS = {
    "gamma_trace": (tool_gamma_trace, {"indices": {"type": "array", "items": {"type": "string"}}}),
    "sandwich": (tool_sandwich, {"k": {"type": "integer"}}),
    "master_integral": (tool_master_integral, {"a": {"type": "integer"}, "numerator": {"type": "string",
                                                                                         "enum": ["1", "l2"]}}),
    "series_eps": (tool_series_eps, {"expr": {"type": "string"}, "order": {"type": "integer"}}),
    "simplify": (tool_simplify, {"expr": {"type": "string"}}),
    "qed_counterterms": (tool_qed_counterterms, {}),
}


def openai_tool_specs() -> list[dict]:
    return [{"type": "function", "function": {"name": name, "description": (fn.__doc__ or "").strip(),
                                              "parameters": {"type": "object", "properties": props,
                                                             "required": list(props)}}}
            for name, (fn, props) in TOOLS.items()]


def call(name: str, args: dict):
    fn, _ = TOOLS[name]
    return fn(**args)
