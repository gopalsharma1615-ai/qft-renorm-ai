"""Running couplings, and a sparse-regression method that recovers the β-function from coupling data.

QED: the on-shell-subtracted one-loop vacuum polarization of a fermion of mass m and charge Q,
    Δα_f(q²) = (2α Q² N_c/π) ∫₀¹ dx x(1-x) ln|1 - x(1-x) q²/m²|,      α(q²) = α / (1 - Σ_f Δα_f),
is the same Π(q²) that `loops.vacuum_polarization` derives, now finite and with its mass dependence.

Rediscovery: an RG equation dα/dL = β(α) (L = ln μ²) is written in "weak form" so that noisy data never has to
be differentiated:
    1/α(L) - 1/α(L₀) = Σ_k c_k ∫_{L₀}^{L} α^k dL'        (because d(1/α)/dL = -β(α)/α² = Σ c_k α^k)
The integrals are computed from the data by the trapezoid rule, and sequentially thresholded least squares
(STLSQ, as in SINDy) picks the few library terms that matter. For QCD at two loops the truth is
c₀ = b₀ = β₀/(4π), c₁ = b₁ = β₁/(16π²), c₂ = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp

ALPHA0 = 1 / 137.035999084
M_Z = 91.1876
LEPTONS = {"e": 0.51099895e-3, "mu": 0.1056583755, "tau": 1.77686}   # GeV


def delta_alpha_fermion(q2: float, m: float, charge: float = -1.0, colours: int = 1, alpha: float = ALPHA0) -> float:
    if q2 == 0:
        return 0.0
    r = q2 / m ** 2

    def f(x):
        return x * (1 - x) * np.log(abs(1 - x * (1 - x) * r))

    pts = None
    if r > 4:                                   # timelike above threshold: log singularities at the roots
        d = np.sqrt(1 - 4 / r)
        pts = [(1 - d) / 2, (1 + d) / 2]
    val, _ = quad(f, 0, 1, points=pts, limit=200)
    return 2 * alpha * charge ** 2 * colours / np.pi * val


def delta_alpha_leptons(q2: float = M_Z ** 2) -> dict[str, float]:
    return {k: delta_alpha_fermion(q2, m) for k, m in LEPTONS.items()}


def alpha_qed(q: float, extra_delta: float = 0.0) -> float:
    return ALPHA0 / (1 - sum(delta_alpha_leptons(q * q).values()) - extra_delta)


# ------------------------------------------------------------------------------------------------ QCD
def qcd_coefficients(nf: int) -> tuple[float, float]:
    """(b0, b1) for dα/dL = -b0 α² - b1 α³, L = ln μ² (MS-bar)."""
    beta0 = 11 - 2 * nf / 3
    beta1 = 102 - 38 * nf / 3
    return beta0 / (4 * np.pi), beta1 / (16 * np.pi ** 2)


def run_alpha_s(q: np.ndarray, nf: int, alpha_ref: float = 0.1180, q_ref: float = M_Z) -> np.ndarray:
    b0, b1 = qcd_coefficients(nf)
    L = np.log(np.asarray(q, float) ** 2)
    L0 = np.log(q_ref ** 2)

    def rhs(_, a):
        return -b0 * a ** 2 - b1 * a ** 3

    out = np.empty_like(L)
    for mask, end in ((L >= L0, L.max()), (L < L0, L.min())):
        if mask.any():
            sol = solve_ivp(rhs, (L0, end), [alpha_ref], dense_output=True, rtol=1e-10, atol=1e-12)
            out[mask] = sol.sol(L[mask])[0]
    return out


# ------------------------------------------------------------------------------------------------ rediscovery
def weak_form_library(L: np.ndarray, a: np.ndarray, powers=(0, 1, 2)) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix Θ (columns: ∫ α^k dL from the first point) and target y = 1/α - 1/α₀."""
    order = np.argsort(L)
    L, a = L[order], a[order]
    cols = []
    for k in powers:
        f = a ** k
        cols.append(np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(L))]))
    return np.stack(cols, axis=1)[1:], (1 / a - 1 / a[0])[1:]


def stlsq(theta: np.ndarray, y: np.ndarray, threshold: float, iters: int = 10) -> np.ndarray:
    """Sequentially thresholded least squares on column-normalised features."""
    scale = np.linalg.norm(theta, axis=0)
    X = theta / scale
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(c) < threshold * np.abs(c).max()
        c[small] = 0
        big = ~small
        if not big.any():
            break
        c[big] = np.linalg.lstsq(X[:, big], y, rcond=None)[0]
    return c / scale


def recover_beta(L: np.ndarray, a: np.ndarray, threshold: float = 0.02, powers=(0, 1, 2)) -> dict[int, float]:
    theta, y = weak_form_library(L, a, powers)
    c = stlsq(theta, y, threshold)
    return {k: float(v) for k, v in zip(powers, c)}
