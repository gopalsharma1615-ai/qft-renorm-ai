"""One-loop QED in dimensional regularization (thesis chapters 5-7), done symbolically with SymPy.

Convention n = 4 - ε (as in the thesis), coupling e -> e μ^{ε/2}, Feynman gauge, minimal subtraction.
Poles therefore appear as 2/ε. Everything below is DERIVED by the code from the Feynman rules plus the
n-dimensional identities in `dirac.py` - nothing is typed in from a textbook - and the tests compare with the
known results:

    vacuum polarization   Π^{μν}(q) = (q² g^{μν} - q^μ q^ν) Π(q²)   transversality emerges from dim-reg
                          Π(q²)_pole = -(α/3π)(2/ε)                  -> δZ3 = -(α/3π)(2/ε)
    electron self-energy  Σ(p)_pole  = (α/4π)(2/ε)(-p̸ + 4m)          -> δZ2 = -(α/4π)(2/ε), δm = (3α/4π)(2/ε) m
    vertex                Λ^μ_pole   = (α/4π)(2/ε) γ^μ               -> δZ1 = -(α/4π)(2/ε) = δZ2  (Ward identity)
    β-function            β(e) = e³/(12π²),  β(α) = 2α²/(3π)
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .dirac import eps, n, symmetric_integration

e, m, mu, alpha = sp.symbols("e m mu alpha", positive=True)
q2, p2, Delta, x = sp.symbols("q2 p2 Delta x", real=True)
PI = sp.pi


# ------------------------------------------------------------------------------------------------ master integrals
def I0(a: int, D=Delta, dim=n) -> sp.Expr:
    """∫ d^n l/(2π)^n 1/(l² - Δ)^a = i (-1)^a /(4π)^{n/2} Γ(a - n/2)/Γ(a) Δ^{n/2 - a}."""
    return sp.I * (-1) ** a / (4 * PI) ** (dim / 2) * sp.gamma(a - dim / 2) / sp.gamma(a) * D ** (dim / 2 - a)


def I2(a: int, D=Delta, dim=n) -> sp.Expr:
    """∫ d^n l/(2π)^n l²/(l² - Δ)^a = i (-1)^{a-1} /(4π)^{n/2} (n/2) Γ(a - n/2 - 1)/Γ(a) Δ^{n/2 - a + 1}."""
    return (sp.I * (-1) ** (a - 1) / (4 * PI) ** (dim / 2) * (dim / 2) * sp.gamma(a - dim / 2 - 1) / sp.gamma(a)
            * D ** (dim / 2 - a + 1))


def pole(expr: sp.Expr) -> sp.Expr:
    """Coefficient of the 1/ε pole of expr(n = 4 - ε), times 1/ε (i.e. the MS counterterm content)."""
    s = sp.series(sp.expand(expr.subs(n, 4 - eps)), eps, 0, 1).removeO()
    return sp.simplify(s.coeff(eps, -1)) / eps


# ------------------------------------------------------------------------------------------------ vacuum polarization
@dataclass
class VacuumPolarization:
    g_coefficient: sp.Expr      # coefficient of g^{μν} in iΠ^{μν}, integrated over l, as a function of x
    qq_coefficient: sp.Expr     # coefficient of q^μ q^ν
    dimreg_cancellation: sp.Expr   # (2/n - 1) I2 + Δ I0: zero in dim-reg -> gauge invariance
    transverse_residual: sp.Expr   # g-coefficient + q² * qq-coefficient: zero <=> Π^{μν} ∝ (q² g - q q)
    pi_pole: sp.Expr            # pole part of Π(q²)


def vacuum_polarization() -> VacuumPolarization:
    """iΠ^{μν} = -(-ie)² ∫ Tr[γ^μ i(l̸+m) γ^ν i(l̸+q̸+m)] / D1 D2   (the minus sign: fermion loop).

    The trace is 4[l^μ(l+q)^ν + l^ν(l+q)^μ - g^{μν}(l·(l+q) - m²)]. After Feynman parameters and the shift
    l -> l - xq (denominator (l² - Δ)², Δ = m² - x(1-x)q²), keeping even terms and l^μ l^ν -> g^{μν} l²/n:
        N^{μν} = 4[(2/n - 1) l² g^{μν} - 2x(1-x) q^μ q^ν + (m² + x(1-x) q²) g^{μν}]
    """
    D = m ** 2 - x * (1 - x) * q2
    pref = -(-sp.I * e * mu ** (eps / 2)) ** 2 * (sp.I ** 2)          # couplings and the two propagator i's
    g_l2 = 4 * (2 / n - 1)                                            # coefficient of l² g^{μν}
    g_const = 4 * (m ** 2 + x * (1 - x) * q2)                         # coefficient of g^{μν}
    qq = -8 * x * (1 - x)
    g_coef = pref * (g_l2 * I2(2, D) + g_const * I0(2, D))
    qq_coef = pref * qq * I0(2, D)
    cancel = sp.gammasimp(sp.powsimp((2 / n - 1) * I2(2, Delta) + Delta * I0(2, Delta), force=True))
    # the same coefficients with m² written through Δ: g + q² qq collapses to 4·pref·[(2/n-1) I2 + Δ I0]
    g_delta = pref * (g_l2 * I2(2, Delta) + 4 * (Delta + 2 * x * (1 - x) * q2) * I0(2, Delta))
    resid = sp.gammasimp(sp.powsimp(sp.expand(g_delta + q2 * pref * qq * I0(2, Delta)), force=True))
    # Transverse structure: iΠ^{μν} = (q² g - q q) iΠ(q²)  =>  iΠ(q²) = -qq_coef (integrated over x)
    ipi = -qq_coef
    pi = sp.integrate(sp.series((ipi / sp.I).subs(n, 4 - eps), eps, 0, 1).removeO().coeff(eps, -1), (x, 0, 1))
    return VacuumPolarization(sp.simplify(g_coef), sp.simplify(qq_coef), sp.simplify(cancel), sp.simplify(resid),
                              sp.simplify(pi.subs(e, sp.sqrt(4 * PI * alpha))) / eps)


# ------------------------------------------------------------------------------------------------ self-energy
def electron_self_energy_pole() -> dict:
    """-iΣ(p) = (-ie)² ∫ γ^μ i(l̸+m) γ_μ (-i)/((l²-m²)((p-l)²)).

    Numerator: γ^μ (l̸ + m) γ_μ = (2-n) l̸ + n m  (dirac.sandwich). Feynman parameter x on the electron line,
    shift l -> l + (1-x) p: the odd l̸ term vanishes; the numerator becomes (2-n)(1-x) p̸ + n m over (l² - Δ)².
    Returns the pole parts of the p̸ and m coefficients of Σ."""
    # -iΣ = (-ie)^2 * i * (-i) * ∫ N / (D1 D2) = -e^2 ∫ N/(D1 D2)   =>   Σ = -i e^2 ∫dx N' I0(2)
    # Feynman parameter x on the electron line: shift l -> l + (1-x)p, so N' = (2-n)(1-x) p̸ + n m
    integ = I0(2, Delta)
    pslash_coef = -sp.I * e ** 2 * sp.integrate((2 - n) * (1 - x), (x, 0, 1)) * integ
    m_coef = -sp.I * e ** 2 * n * m * integ
    a = sp.sqrt(4 * PI * alpha)
    return {"pslash": sp.simplify(pole(pslash_coef).subs(e, a)), "mass": sp.simplify(pole(m_coef).subs(e, a))}


def vertex_pole() -> sp.Expr:
    """UV pole of Λ^μ: only the l̸ γ^μ l̸ piece of γ^ν (l̸+..) γ^μ (l̸+..) γ_ν diverges.

    Symmetric integration turns l̸ γ^μ l̸ into (l²/n) γ_α γ^μ γ^α = (l²/n)(2-n) γ^μ, and the outer contraction
    gives another (2-n): numerator (2-n)² l²/n γ^μ over (l² - Δ)³, with ∫dF (Feynman measure over 3 parameters)
    contributing 2 x (1/2) = 1."""
    inner = symmetric_integration(("gmu",))                  # l̸ γ^μ l̸  -> (l²/n)(2-n) γ^μ
    coef_inner = inner[0].coef.subs(sp.Symbol("l^2"), 1)      # (2-n)/n, times l²
    outer = 2 - n                                             # γ^ν γ^μ γ_ν = (2-n) γ^μ
    numer = coef_inner * outer
    pref = (-sp.I * e) ** 2 * sp.I * sp.I * (-sp.I)           # three vertices' couplings and propagators
    lam = sp.I * pref * numer * I2(3, Delta) / sp.I           # Λ^μ coefficient of γ^μ (factor -i stripped)
    return sp.simplify(pole(lam * 2 * sp.Rational(1, 2)).subs(e, sp.sqrt(4 * PI * alpha)))


# ------------------------------------------------------------------------------------------------ Z factors and β
def counterterms() -> dict:
    vp = vacuum_polarization()
    se = electron_self_energy_pole()
    dZ3 = sp.simplify(vp.pi_pole)                    # Z3 = 1 + δZ3 cancels Π's pole
    dZ2 = sp.simplify(se["pslash"])                  # Z2: p̸ coefficient
    dm = sp.simplify(se["pslash"] * m + se["mass"])  # physical mass shift: Σ at p̸ = m
    dZ1 = sp.simplify(-vertex_pole())
    return {"dZ1": dZ1, "dZ2": dZ2, "dZ3": dZ3, "dm": dm, "ward_identity_Z1_eq_Z2": sp.simplify(dZ1 - dZ2) == 0}


def beta_function() -> dict:
    """e0 = μ^{ε/2} Z_e e with Z_e = Z1/(Z2 √Z3) = 1/√Z3 (Ward identity). μ de0/dμ = 0 gives
    β(e) = -(ε/2) e / (1 + e d ln Z_e/de); expand and take ε -> 0."""
    ct = counterterms()
    ee = sp.Symbol("e", positive=True)
    dZ3 = ct["dZ3"].subs(alpha, ee ** 2 / (4 * PI))
    lnZe = -sp.Rational(1, 2) * dZ3                  # ln Z_e at one loop
    beta = -(eps / 2) * ee / (1 + ee * sp.diff(lnZe, ee))
    b = sp.limit(sp.series(beta, ee, 0, 4).removeO(), eps, 0)
    b_alpha = sp.simplify((2 * ee * b / (4 * PI)).subs(ee, sp.sqrt(4 * PI * alpha)))   # dα/d ln μ
    return {"beta_e": sp.simplify(b), "beta_alpha": b_alpha}
