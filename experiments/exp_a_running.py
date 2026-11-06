"""Experiment A: from running-coupling data back to the β-function.

A1  QED: how far does one-loop lepton vacuum polarization run α from 0 to M_Z, and what is missing?
A2  QCD: from noisy α_s(Q) "measurements" (simulated from two-loop running, 0.5-2% noise), can sparse regression
    on the RG equation's weak form recover β0 and β1, reject a spurious α² term, and - across n_f - rediscover
    β0 = 11 - 2 n_f/3 and the loss of asymptotic freedom at n_f = 16.5?
A3  QED windows: the slope of 1/α between lepton thresholds counts the active charged leptons.

    python experiments/exp_a_running.py --out results/exp_a.json [--plots results/]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qftlab.running import (
    ALPHA0,
    LEPTONS,
    M_Z,
    alpha_qed,
    delta_alpha_leptons,
    qcd_coefficients,
    recover_beta,
    run_alpha_s,
)

# Data-driven five-flavour hadronic contribution from dispersive e+e- -> hadrons analyses (~0.0276; evaluations
# differ in the fourth digit). Not computable in perturbation theory at low q² - which is the point of A1.
DELTA_ALPHA_HAD5 = 0.0276


def a1() -> dict:
    lep = delta_alpha_leptons(M_Z ** 2)
    tot = sum(lep.values())
    return {"delta_alpha_leptons": {k: round(v, 6) for k, v in lep.items()}, "delta_alpha_lep_total": round(tot, 6),
            "inv_alpha_0": round(1 / ALPHA0, 3), "inv_alpha_MZ_leptons_only": round((1 - tot) / ALPHA0, 3),
            "delta_alpha_had5_input": DELTA_ALPHA_HAD5,
            "inv_alpha_MZ_with_hadrons": round((1 - tot - DELTA_ALPHA_HAD5) / ALPHA0, 3)}


def a2(noise: float, seeds: int = 50) -> dict:
    q = np.geomspace(2.0, 2000.0, 40)
    L = np.log(q ** 2)
    out = {}
    for nf in (3, 4, 5, 6):
        b0, b1 = qcd_coefficients(nf)
        truth = run_alpha_s(q, nf)
        fits, fits2 = [], []
        for s in range(seeds):
            rng = np.random.default_rng(1000 * nf + s)
            data = truth * (1 + noise * rng.standard_normal(truth.shape))
            fits.append(recover_beta(L, data, threshold=0.02))
            fits2.append(recover_beta(L, data, threshold=0.0, powers=(0, 1)))
        c0 = np.array([f[0] for f in fits])
        d1 = np.array([f[1] for f in fits2])
        c1 = np.array([f[1] for f in fits])
        c2 = np.array([f[2] for f in fits])
        out[nf] = {"beta0_true": round(4 * np.pi * b0, 4), "beta0_fit": round(4 * np.pi * c0.mean(), 4),
                   "beta0_std": round(4 * np.pi * c0.std(), 4),
                   "beta1_true": round(16 * np.pi ** 2 * b1, 2), "beta1_fit": round(16 * np.pi ** 2 * c1.mean(), 2),
                   "beta1_std": round(16 * np.pi ** 2 * c1.std(), 2),
                   "alpha2_term_rejected": float(np.mean(c2 == 0)),
                   "beta1_fit_two_term_library": round(16 * np.pi ** 2 * d1.mean(), 2),
                   "beta1_std_two_term_library": round(16 * np.pi ** 2 * d1.std(), 2)}
    nfs = np.array(sorted(out))
    b0s = np.array([out[k]["beta0_fit"] for k in nfs])
    slope, icpt = np.polyfit(nfs, b0s, 1)
    return {"noise": noise, "per_nf": out, "beta0_vs_nf": {"slope": round(slope, 4), "intercept": round(icpt, 3),
                                                           "nf_asymptotic_freedom_lost": round(-icpt / slope, 2)}}


def a3() -> dict:
    windows = {"e only (2-60 MeV)": (2e-3, 6e-2), "e, mu (0.5-1.5 GeV)": (0.5, 1.5), "e, mu, tau (10-90 GeV)": (10, 90)}
    out = {}
    for name, (lo, hi) in windows.items():
        q = np.geomspace(lo, hi, 30)
        inv = np.array([1 / alpha_qed(x) for x in q])
        slope = np.polyfit(np.log(q ** 2), inv, 1)[0]           # d(1/α)/d ln q² = -N/(3π) asymptotically
        out[name] = {"slope": round(slope, 5), "active_leptons_recovered": round(-3 * np.pi * slope, 2)}
    return out


def plots(res: dict, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = np.geomspace(1e-3, 200, 400)
    plt.figure(figsize=(6, 4))
    plt.semilogx(q, [1 / alpha_qed(x) for x in q], label="leptons (one loop, this code)")
    for k, m in LEPTONS.items():
        plt.axvline(2 * m, ls=":", c="grey")
        plt.text(2 * m * 1.1, 136.6, f"2m_{k}", fontsize=8)
    plt.scatter([M_Z], [res["A1"]["inv_alpha_MZ_with_hadrons"]], c="k", zorder=3, label="+ hadronic (data-driven)")
    plt.xlabel("Q [GeV]")
    plt.ylabel("1/α(Q)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "qed_running.png", dpi=150)

    plt.figure(figsize=(6, 4))
    per = res["A2"]["per_nf"]
    nfs = sorted(per, key=int)
    plt.errorbar([int(k) for k in nfs], [per[k]["beta0_fit"] for k in nfs], yerr=[per[k]["beta0_std"] for k in nfs],
                 fmt="o", label="recovered from noisy α_s(Q)")
    x = np.linspace(0, 18, 50)
    plt.plot(x, 11 - 2 * x / 3, label="β0 = 11 - 2nf/3")
    plt.axhline(0, c="grey", lw=0.5)
    plt.xlabel("n_f")
    plt.ylabel("β0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "beta0_vs_nf.png", dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/exp_a.json")
    ap.add_argument("--plots")
    a = ap.parse_args()
    res = {"A1": a1(), "A2": {f"noise_{n}": a2(n) for n in (0.005, 0.01, 0.02)}, "A3": a3()}
    res["A2"]["per_nf"] = res["A2"]["noise_0.005"]["per_nf"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if a.plots:
        plots(res, Path(a.plots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
