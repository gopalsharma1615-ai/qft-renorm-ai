import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from qftlab import loops, running, tools, tree
from qftlab.dirac import GAMMA, METRIC, ONE, contract, eps, evaluate_terms, sandwich, slash

RNG = np.random.default_rng(7)


def rvec():
    return RNG.normal(size=4)


# ------------------------------------------------------------------------------------------------ Dirac algebra
def test_clifford_algebra():
    for a in range(4):
        for b in range(4):
            assert np.allclose(GAMMA[a] @ GAMMA[b] + GAMMA[b] @ GAMMA[a], 2 * METRIC[a, b] * ONE)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_n_dim_sandwich_matches_explicit_matrices_at_n4(k):
    vecs = {name: rvec() for name in ("a", "b", "c")[:k]}
    inner = ONE
    for name in ("a", "b", "c")[:k]:
        inner = inner @ slash(vecs[name])
    explicit = contract(lambda up, low: up @ inner @ low)
    assert np.allclose(evaluate_terms(sandwich(tuple(vecs)), vecs), explicit)


# ------------------------------------------------------------------------------------------------ tree level
def test_pair_production_massive_and_massless():
    for m, M in ((0.0, 0.0), (0.3, 1.1)):
        for th in (0.4, 1.3, 2.7):
            p = tree.two_body(5.0, m, m, M, M, th, 0.7)
            s, t, u = tree.mandelstam(*p)
            assert np.isclose(tree.pair_production(*p, m=m, M=M), tree.pair_production_closed(s, t, u, m, M))


@pytest.mark.parametrize("th", [0.5, 1.2, 2.2])
def test_moller_and_bhabha(th):
    p = tree.two_body(3.0, 0, 0, 0, 0, th, 0.3)
    s, t, u = tree.mandelstam(*p)
    assert np.isclose(tree.moller(*p), tree.moller_closed(s, t, u))
    assert np.isclose(tree.bhabha(*p), tree.bhabha_closed(s, t, u))


def test_compton_closed_form_and_klein_nishina():
    m = 1.0
    for th in (0.3, 1.0, 2.5):
        k1, p1, k2, p2 = tree.two_body(2.4, 0, m, 0, m, th)
        s, _, u = tree.mandelstam(k1, p1, k2, p2)
        assert np.isclose(tree.compton(k1, p1, k2, p2, m), tree.compton_closed(s, u, m))
        for w in (0.01, 0.5, 3.0):
            assert np.isclose(tree.compton_lab_dsigma(w, th, m), tree.klein_nishina(w, th, m), rtol=1e-9)


def test_thomson_limit():
    th = np.linspace(1e-3, np.pi - 1e-3, 2001)
    dsig = np.array([tree.compton_lab_dsigma(1e-6, t) for t in th])
    sigma = np.trapezoid(dsig * 2 * np.pi * np.sin(th), th)
    assert np.isclose(sigma, 8 * np.pi * running.ALPHA0 ** 2 / 3, rtol=1e-3)


# ------------------------------------------------------------------------------------------------ one loop
def test_vacuum_polarization_gauge_invariant_and_pole():
    vp = loops.vacuum_polarization()
    assert vp.dimreg_cancellation == 0
    assert vp.transverse_residual == 0
    assert sp.simplify(vp.pi_pole + loops.alpha / (3 * sp.pi) * 2 / eps) == 0


def test_self_energy_vertex_and_ward_identity():
    se = loops.electron_self_energy_pole()
    a, m = loops.alpha, loops.m
    assert sp.simplify(se["pslash"] + a / (4 * sp.pi) * 2 / eps) == 0
    assert sp.simplify(se["mass"] - a / (4 * sp.pi) * 2 / eps * 4 * m) == 0
    ct = loops.counterterms()
    assert ct["ward_identity_Z1_eq_Z2"]
    assert sp.simplify(ct["dm"] - 3 * a / (4 * sp.pi) * 2 / eps * m) == 0


def test_beta_function():
    b = loops.beta_function()
    assert sp.simplify(b["beta_e"] - sp.Symbol("e", positive=True) ** 3 / (12 * sp.pi ** 2)) == 0
    assert sp.simplify(b["beta_alpha"] - 2 * loops.alpha ** 2 / (3 * sp.pi)) == 0


# ------------------------------------------------------------------------------------------------ running
def test_lepton_running_to_mz():
    tot = sum(running.delta_alpha_leptons().values())
    assert abs(tot - 0.03142) < 2e-4                    # one-loop leptonic Δα(M_Z²)


def test_large_q2_asymptotics():
    m, q2 = 0.1, 1e6
    asym = running.ALPHA0 / (3 * np.pi) * (np.log(q2 / m ** 2) - 5 / 3)
    assert np.isclose(running.delta_alpha_fermion(q2, m), asym, rtol=1e-4)


def test_recover_beta_noiseless():
    q = np.geomspace(2, 2000, 60)
    a = running.run_alpha_s(q, 5)
    c = running.recover_beta(np.log(q ** 2), a, threshold=0.0, powers=(0, 1))
    b0, b1 = running.qcd_coefficients(5)
    assert np.isclose(c[0], b0, rtol=1e-3) and np.isclose(c[1], b1, rtol=5e-2)


# ------------------------------------------------------------------------------------------------ tools / tasks
def test_safe_parse_rejects_code():
    for bad in ("__import__('os')", "x.__class__", "open('f')", "eval(1)", "e.func"):
        with pytest.raises(ValueError):
            tools.safe_parse(bad)
    assert tools.equivalent("2/(3*pi)", "4/(6*pi)") and not tools.equivalent("1/pi", "2/pi")


def test_task_answers_agree_with_derived_results():
    tasks = {t["id"]: t["answer"] for t in json.loads(
        (Path(__file__).parents[1] / "experiments" / "exp_b_tasks.json").read_text())}
    ct = loops.counterterms()
    a = loops.alpha
    for tid, key in (("dz1", "dZ1"), ("dz2", "dZ2"), ("dz3", "dZ3")):
        assert sp.simplify(ct[key] * eps / a - tools.safe_parse(tasks[tid])) == 0
    assert sp.simplify(ct["dm"] * eps / (a * loops.m) - tools.safe_parse(tasks["dm"])) == 0
    assert tools.equivalent(tools.tool_series_eps("gamma(eps/2)", 0), tasks["gammapole"])
    assert tools.tool_gamma_trace(["mu", "nu"]).replace(" ", "") in ("4*metric(mu,nu)", "4*metric(nu,mu)")


def test_gamma_trace_tool_four_indices():
    out = tools.tool_gamma_trace(["mu", "nu", "rho", "sigma"])
    assert "metric" in out and out.count("metric") >= 6
