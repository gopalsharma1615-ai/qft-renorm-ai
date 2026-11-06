"""Tree-level QED scattering (thesis chapter 4): spin-averaged |M|^2 from explicit Dirac traces.

Each process is computed the honest way - Feynman rules, spin sums as traces of (p̸ ± m), photon polarization
sums as -g_μν - and compared with its textbook closed form at random kinematics. Units: e = 1 unless given.

    pair production  e+ e- -> μ+ μ-   (s-channel photon)
    Compton          γ e- -> γ e-     (s- and u-channel electron)  -> Klein-Nishina in the lab frame
    Møller           e- e- -> e- e-   (t and u channels, relative minus sign: identical fermions)
    Bhabha           e+ e- -> e+ e-   (t and s channels)
"""
from __future__ import annotations

import numpy as np

from .dirac import GAMMA, GAMMA_LOWER, ONE, dot, slash, trace

MU = range(4)


# ------------------------------------------------------------------------------------------------ kinematics
def two_body(sqrt_s: float, m1: float, m2: float, m3: float, m4: float, theta: float, phi: float = 0.0):
    """CM-frame momenta for 1 + 2 -> 3 + 4, particle 1 along +z, 3 at polar angle theta."""
    s = sqrt_s ** 2

    def pcm(ma, mb):
        return np.sqrt((s - (ma + mb) ** 2) * (s - (ma - mb) ** 2)) / (2 * sqrt_s)

    pi, pf = pcm(m1, m2), pcm(m3, m4)
    e1, e2 = (s + m1 ** 2 - m2 ** 2) / (2 * sqrt_s), (s + m2 ** 2 - m1 ** 2) / (2 * sqrt_s)
    e3, e4 = (s + m3 ** 2 - m4 ** 2) / (2 * sqrt_s), (s + m4 ** 2 - m3 ** 2) / (2 * sqrt_s)
    d = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    p1 = np.array([e1, 0, 0, pi])
    p2 = np.array([e2, 0, 0, -pi])
    p3 = np.concatenate([[e3], pf * d])
    p4 = np.concatenate([[e4], -pf * d])
    return p1, p2, p3, p4


def mandelstam(p1, p2, p3, p4):
    s = dot(p1 + p2, p1 + p2)
    t = dot(p1 - p3, p1 - p3)
    u = dot(p1 - p4, p1 - p4)
    return s, t, u


def _u(p, m):        # Σ u ū = p̸ + m
    return slash(p) + m * ONE


def _v(p, m):        # Σ v v̄ = p̸ - m
    return slash(p) - m * ONE


# ------------------------------------------------------------------------------------------------ processes
def pair_production(p1, p2, p3, p4, m=0.0, M=0.0, e=1.0) -> float:
    """e-(p1) e+(p2) -> μ-(p3) μ+(p4): spin-averaged |M|^2 = (e^4/4s^2) L^{μν} M_{μν}."""
    s = dot(p1 + p2, p1 + p2)
    L = [[trace(_v(p2, m), GAMMA[a], _u(p1, m), GAMMA[b]) for b in MU] for a in MU]
    H = [[trace(_u(p3, M), GAMMA_LOWER[a], _v(p4, M), GAMMA_LOWER[b]) for b in MU] for a in MU]
    tot = sum(L[a][b] * H[a][b] for a in MU for b in MU)
    return float((e ** 4 / (4 * s ** 2) * tot).real)


def pair_production_closed(s, t, u, m=0.0, M=0.0, e=1.0) -> float:
    """Known result: 2e^4/s^2 [ (t - m^2 - M^2)^2 + (u - m^2 - M^2)^2 + 2 s (m^2 + M^2) ]."""
    mm = m ** 2 + M ** 2
    return 2 * e ** 4 / s ** 2 * ((t - mm) ** 2 + (u - mm) ** 2 + 2 * s * mm)


def moller(p1, p2, p3, p4, e=1.0) -> float:
    """Massless e-(p1) e-(p2) -> e-(p3) e-(p4); t- and u-channel with a relative minus sign."""
    _s, t, u = mandelstam(p1, p2, p3, p4)
    u1, u2, u3, u4 = (_u(p, 0.0) for p in (p1, p2, p3, p4))
    tt = sum(trace(u3, GAMMA[a], u1, GAMMA[b]) * trace(u4, GAMMA_LOWER[a], u2, GAMMA_LOWER[b])
             for a in MU for b in MU)
    uu = sum(trace(u4, GAMMA[a], u1, GAMMA[b]) * trace(u3, GAMMA_LOWER[a], u2, GAMMA_LOWER[b])
             for a in MU for b in MU)
    tu = sum(trace(u3, GAMMA[a], u1, GAMMA[b], u4, GAMMA_LOWER[a], u2, GAMMA_LOWER[b]) for a in MU for b in MU)
    tot = tt / t ** 2 + uu / u ** 2 - 2 * tu.real / (t * u)
    return float((e ** 4 / 4 * tot).real)


def moller_closed(s, t, u, e=1.0) -> float:
    return 2 * e ** 4 * ((s ** 2 + u ** 2) / t ** 2 + (s ** 2 + t ** 2) / u ** 2 + 2 * s ** 2 / (t * u))


def bhabha(p1, p2, p3, p4, e=1.0) -> float:
    """Massless e-(p1) e+(p2) -> e-(p3) e+(p4); t- and s-channel with a relative minus sign."""
    s, t, _ = mandelstam(p1, p2, p3, p4)
    u1, v2, u3, v4 = _u(p1, 0), _v(p2, 0), _u(p3, 0), _v(p4, 0)
    tt = sum(trace(u3, GAMMA[a], u1, GAMMA[b]) * trace(v2, GAMMA_LOWER[a], v4, GAMMA_LOWER[b])
             for a in MU for b in MU)
    ss = sum(trace(v2, GAMMA[a], u1, GAMMA[b]) * trace(u3, GAMMA_LOWER[a], v4, GAMMA_LOWER[b])
             for a in MU for b in MU)
    ts = sum(trace(u3, GAMMA[a], u1, GAMMA[b], v2, GAMMA_LOWER[a], v4, GAMMA_LOWER[b]) for a in MU for b in MU)
    tot = tt / t ** 2 + ss / s ** 2 - 2 * ts.real / (t * s)
    return float((e ** 4 / 4 * tot).real)


def bhabha_closed(s, t, u, e=1.0) -> float:
    return 2 * e ** 4 * ((s ** 2 + u ** 2) / t ** 2 + (u ** 2 + t ** 2) / s ** 2 + 2 * u ** 2 / (s * t))


def compton(k1, p1, k2, p2, m=1.0, e=1.0) -> float:
    """γ(k1) e-(p1) -> γ(k2) e-(p2); photon polarization sums -g_μν, spin-averaged over 2x2."""
    a = slash(p1 + k1) + m * ONE          # s-channel numerator
    b = slash(p1 - k2) + m * ONE          # u-channel numerator
    sd, ud = dot(p1 + k1, p1 + k1) - m ** 2, dot(p1 - k2, p1 - k2) - m ** 2
    tot = 0j
    for mu in MU:
        for nu in MU:
            g = GAMMA_LOWER
            amp = (GAMMA[nu] @ a @ GAMMA[mu]) / sd + (GAMMA[mu] @ b @ GAMMA[nu]) / ud
            ampbar = (g[mu] @ a @ g[nu]) / sd + (g[nu] @ b @ g[mu]) / ud
            tot += trace(_u(p2, m), amp, _u(p1, m), ampbar)
    return float((e ** 4 / 4 * tot).real)


def compton_closed(s, u, m=1.0, e=1.0) -> float:
    """Spin-averaged Compton |M|^2 (Peskin & Schroeder 5.87) in Mandelstam form, a = s - m², b = u - m²:
    2e^4 [ -b/a - a/b + 4m²(1/a + 1/b) + 4m⁴(1/a + 1/b)² ]."""
    a, b = s - m ** 2, u - m ** 2
    return 2 * e ** 4 * (-b / a - a / b + 4 * m ** 2 * (1 / a + 1 / b) + 4 * m ** 4 * (1 / a + 1 / b) ** 2)


def klein_nishina(omega: float, theta: float, m: float = 1.0, alpha: float = 1 / 137.035999) -> float:
    """dσ/dΩ in the electron rest frame (photon energy omega, scattering angle theta)."""
    w2 = omega / (1 + omega / m * (1 - np.cos(theta)))
    r = w2 / omega
    return alpha ** 2 / (2 * m ** 2) * r ** 2 * (r + 1 / r - np.sin(theta) ** 2)


def compton_lab_dsigma(omega: float, theta: float, m: float = 1.0, alpha: float = 1 / 137.035999) -> float:
    """dσ/dΩ in the lab frame from our trace-based |M|^2: dσ/dΩ = |M|^2 / (64 π^2 m^2) (ω'/ω)^2."""
    w2 = omega / (1 + omega / m * (1 - np.cos(theta)))
    k1 = np.array([omega, 0, 0, omega])
    p1 = np.array([m, 0, 0, 0.0])
    k2 = np.array([w2, w2 * np.sin(theta), 0, w2 * np.cos(theta)])
    p2 = k1 + p1 - k2
    e = np.sqrt(4 * np.pi * alpha)
    return compton(k1, p1, k2, p2, m, e) / (64 * np.pi ** 2 * m ** 2) * (w2 / omega) ** 2
