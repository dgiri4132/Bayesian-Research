#!/usr/bin/env python3
"""
EXPERIMENT B i.e. the closed-form R diagnostic for the
Bayesian CVaR project .

Experiment A showed the pipeline is sound and isolated misspecification as the
cause of the ES miscentring.  It said nothing about WHY the failure has the
magnitude it does.  B answers that question analytically.
    The following are the basic cintents, if you will, of the file:
    B0   the reference series placed against the closed forms
    B1   the general closed form, and a third (mean-excess) form
    B2   specialisation to named families, checked against quadrature
    B2b  R = 1 at tau = 1/2 if and only if mean = median
    B4   the analytic refutation of the fat-tails diagnosis
    B5   R > 1: the diagnostic is two-sided, and R = 1 does not mean the data are AL
    B6   R across 492 real series. It is in experiment_b6.py, not here, because
         it needs downloaded prices. I cut it on 5 Sep and brought it back on
         27 Sep once I had downloaded the data myself.
    B7   the comparison with Chen, Gerlach and Lu (2012)
.

Conventions and the sample-splitting rule come from kkgibs.py( I made a spelling error 
and decided to stick with it), which Spy_run.py and experiment_a.py also use, so the 
three scripts cannot disagree about what the empirical VaR and CVaR of a sample are.  
Returns are SIMPLE throughout the project.

"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq, minimize

import kkgibs as KK
from rdiag import (
    R_general, R_original, R_mean_excess, R_sample,
    Normal, StudentT, Laplace, Uniform, Exponential, LogNormal,
    NegGPD, AsymmetricLaplace, BetaPower,
)

TAUS = np.array([0.01, 0.05, 0.10, 0.25, 0.50])

RESULTS: dict = {}
FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def one(x):
    # the families return arrays, this gives back a float
    return float(np.atleast_1d(x)[0])


def roster():
    return [
        Normal(), StudentT(5), StudentT(3), Laplace(), Uniform(),
        Exponential(), LogNormal(), NegGPD(0.30), NegGPD(0.15),
        BetaPower(2.0), BetaPower(0.5),
    ]


def B0(ref, ref_src):
    """Put the paper's own series on the same axis as the theory."""
    rule("B0  the reference series against the closed forms")
    print(f"\n  source: {ref_src}   n = {ref.size}")
    print(f"\n  {'tau':>6}{'R (reference)':>16}{'Normal':>11}{'t(5)':>11}{'t(3)':>11}{'k':>7}")
    rows = []
    for t in (0.01, 0.025, 0.05, 0.10, 0.25):
        s = R_sample(ref, t)
        cf = [one(f.R_closed(t)) for f in (Normal(), StudentT(5), StudentT(3))]
        rows.append({"tau": float(t), "R_reference": s["R_general"],
                     "k": s["k"], "tau_eff": s["tau_eff"],
                     "R_normal": cf[0], "R_t5": cf[1], "R_t3": cf[2]})
        print(f"  {t:>6g}{s['R_general']:>16.6f}{cf[0]:>11.6f}{cf[1]:>11.6f}"
              f"{cf[2]:>11.6f}{s['k']:>7d}")
    RESULTS["B0"] = {"source": ref_src, "n": int(ref.size), "rows": rows}



# B1


def B1():
    rule("B1  the general closed form")
    print("\n  three forms, analytic moments, max |difference| over the tau grid")
    worst, rows = 0.0, []
    for f in roster():
        mu, q = f.mean(), f.var_(TAUS)
        c, yu = f.cvar(TAUS), f.ybar_up(TAUS)
        r1 = R_general(mu, q, c, TAUS)
        r2 = R_original(yu, q, c, TAUS)
        r3 = R_mean_excess(q - c, yu - q, TAUS)
        d = max(np.max(np.abs(r1 - r2)), np.max(np.abs(r1 - r3)))
        rel = d / np.max(np.abs(r1))
        worst = max(worst, rel)
        rows.append({"family": f.label, "max_abs_diff": float(d), "max_rel_diff": float(rel)})
        print(f"      {f.label:24s}  abs {d:9.2e}   rel {rel:9.2e}")
    check("B1  three forms identical to < 1e-12 relative", worst < 1e-12,
          f"worst relative difference {worst:.2e}")
    RESULTS["B1"] = {"analytic": rows, "worst_rel_analytic": worst}


# B2 

def cvar_quad(f, tau):
    """CVaR(tau) = (1/tau) int_0^tau F^{-1}(u) du -- uses only the quantile function."""
    v, _ = quad(lambda u: one(f.var_(u)), 0.0, tau, limit=400, epsabs=1e-13, epsrel=1e-13)
    return v / tau


def mean_quad(f):
    v, _ = quad(lambda u: one(f.var_(u)), 0.0, 1.0, limit=400, epsabs=1e-13, epsrel=1e-13)
    return v


def B2():
    rule("B2  named families: closed form against quadrature")
    # the mean and CVaR are integrals of the quantile function, so this checks every
    # family's closed form with no Monte Carlo error. The AL rows are the R = 1 check.
    fams = roster() + [AsymmetricLaplace(p=float(t)) for t in TAUS]

    print("\n  max relative error over the tau grid")
    worst, rows = 0.0, []
    for f in fams:
        taus_here = np.array([f.p]) if isinstance(f, AsymmetricLaplace) else TAUS
        mq = mean_quad(f)
        dmu = abs(mq - f.mean()) / max(abs(f.mean()), 1e-3)
        dr = 0.0
        for t in taus_here:
            r_quad = R_general(mq, one(f.var_(t)), cvar_quad(f, float(t)), float(t))
            r_cf = one(f.R_closed(t))
            dr = max(dr, abs(r_quad - r_cf) / abs(r_cf))
        worst = max(worst, dmu, dr)
        rows.append({"family": f.label, "rel_err_mean": float(dmu), "rel_err_R": float(dr)})
        print(f"      {f.label:24s}  mean {dmu:9.2e}   R {dr:9.2e}")
    check("B2  closed form matches quadrature to < 1e-9 relative", worst < 1e-9,
          f"worst relative error {worst:.2e}")
    RESULTS["B2"] = {"quadrature": rows}



# B2b 


def B2b():
    rule("B2b  R = 1 at tau = 1/2 if and only if mean = median")
    # at tau = 1/2, R = (mu - c)/(q - c) and mu = (c + ybar_up)/2, so R = 1 exactly
    # when q, the median, equals mu
    rows = []
    for f in roster():
        r = one(f.R_closed(0.5))
        mu = f.mean()
        med = one(f.var_(0.5))
        sym = abs(mu - med) < 1e-10
        rows.append({"family": f.label, "R_half": r, "mean": float(mu),
                     "median": med, "mean_eq_median": bool(sym)})
        flag = "mean = median" if sym else f"mean - median = {mu - med:+.6f}"
        print(f"      {f.label:24s}  R(1/2) = {r:8.5f}    {flag}")
    ok = all((abs(r["R_half"] - 1.0) < 1e-4) == r["mean_eq_median"] for r in rows)
    check("B2b R(1/2) = 1 exactly on the mean = median families, and only those", ok)
    RESULTS["B2b"] = rows


def B4():
    rule("B4  the analytic refutation of the fat-tails diagnosis")
    print(f"\n      {'nu':>10s} {'R(tau=0.05)':>14s}")
    rows = []
    for nu in (2.5, 3, 5, 10, 100, 1000, 10000):
        r5 = one(StudentT(nu).R_closed(0.05))
        rows.append({"nu": float(nu), "R_05": r5})
        print(f"      {nu:10g} {r5:14.6f}")
    rn5 = one(Normal().R_closed(0.05))
    print(f"      {'Normal':>10s} {rn5:14.6f}   <- the limit")
    check("B4 R(Normal, tau=0.05) bounded well away from 1", rn5 < 0.30, f"R = {rn5:.6f}")
    check("B4 t(nu) -> Normal as nu -> infinity, not -> 1",
          (rows[-1]["R_05"] - rn5) < 1e-3 and rn5 < 0.30,
          f"R(t(10000)) = {rows[-1]['R_05']:.6f} vs Normal {rn5:.6f}")
    RESULTS["B4"] = {"t_sequence": rows, "normal": {"R_05": rn5}}


# B5: R > 1, and what R = 1 does and does not mean


def B5():
    rule("B5  R is two-sided, and R = 1 does not mean AL")
    print("\n  (a) R > 1: the AL overstates the gap on these")
    rows_a = []
    for f in (LogNormal(), Exponential(), BetaPower(0.5)):
        vals = [one(f.R_closed(t)) for t in TAUS]
        rows_a.append({"family": f.label, "taus": TAUS.tolist(), "R": vals})
        print(f"      {f.label:24s}" + "".join(f"{v:10.4f}" for v in vals))
    print(f"      {'(tau)':24s}" + "".join(f"{t:10g}" for t in TAUS))

    # the lognormal crossing
    ln = LogNormal()
    xr = brentq(lambda t: one(ln.R_closed(t)) - 1.0, 1e-6, 0.4)
    print(f"\n      the lognormal crosses R = 1 at tau = {xr:.5f}")

    # (b) R = 1 only says e_U/e_L = (1-tau)/tau, and the uniform has that at every tau
    print("\n  (b) the uniform")
    u = Uniform()
    tg = np.array([0.001, 0.01, 0.05, 0.1, 0.25, 0.5])
    ru = np.atleast_1d(u.R_closed(tg))
    print(f"      {'tau':>10s}" + "".join(f"{t:12g}" for t in tg))
    print(f"      {'R(uniform)':>10s}" + "".join(f"{v:12.9f}" for v in ru))
    check("B5(b) uniform gives R = 1 identically", np.max(np.abs(ru - 1.0)) < 1e-12,
          f"max |R-1| = {np.max(np.abs(ru - 1.0)):.2e}")
    RESULTS["B5"] = {"above_one": rows_a, "lognormal_crossing_tau": float(xr),
                     "uniform_R": ru.tolist(), "uniform_taus": tg.tolist()}


# B7  reconciling Chen, Gerlach & Lu


def B7(ref, ref_src):
    """
    CGL (2012) publish that AL tails are FATTER than Student-t.  This project says
    the AL's tail is far too thin.  Both are true, and the reconciliation is the
    single free shape parameter.  All of it is verified here rather than recalled.
    """
    rule("B7  reconciling Chen, Gerlach & Lu (2012)")

    def bp(pp):
        return np.sqrt(pp ** 2 + (1 - pp) ** 2)

    #  (a) their 2-6 sd claim, checked

    #  standardised ALD(0,1,p) density, their eq (2)
    def ald(x, pp):
        b = bp(pp)
        return b * np.exp(-b * np.abs(x) * np.where(x < 0, 1 / pp, 1 / (1 - pp)))
    #  standardised Student-t with kurtosis 6  ->  6/(nu-4) = 3  ->  nu = 6
    nu = 6.0
    sc = 1.0 / np.sqrt(nu / (nu - 2.0))
    #  Rather than score what fraction of [2,6] the ALD dominates, which needs an
    #  arbitrary threshold , we solve for where the two densities actually cross.  If
    #  CGL are right the crossings should be AT 2 and 6, and they are.
    gap = lambda x: ald(x, 0.5) - stats.t.pdf(x / sc, nu) / sc
    xs = -np.linspace(1.0, 8.0, 20001)
    idx = np.where(np.diff(np.sign(gap(xs))) != 0)[0]
    roots = [float(brentq(gap, xs[i], xs[i + 1])) for i in idx]
    print(f"\n  ALD(0,1,0.5) against the t with kurtosis 6 (nu = {nu:.0f}), "
          f"crossings on the left:")
    for rt in roots:
        print(f"        x = {rt:.4f} standard deviations")
    check("B7  CGL's 2-to-6-sd claim reproduces, as the exact crossing points",
          len(roots) == 2 and abs(abs(roots[0]) - 2.0) < 0.1
          and abs(abs(roots[-1]) - 6.0) < 0.1,
          f"crossings at {roots[0]:.3f} and {roots[-1]:.3f} sd")

    #  (d) the reconciliation on the reference series
    def nll(th, y):
        g, ls, lp = th
        s_, pp_ = np.exp(ls), 1 / (1 + np.exp(-lp))
        b, u = bp(pp_), y - g
        return -np.sum(np.log(b / s_) - np.abs(u) *
                       np.where(u < 0, b / (s_ * pp_), b / (s_ * (1 - pp_))))
    o = minimize(nll, [np.median(ref), np.log(ref.std()), 0.0], args=(ref,),
                 method="Nelder-Mead",
                 options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-12})
    g, s_, pp_ = o.x[0], np.exp(o.x[1]), 1 / (1 + np.exp(-o.x[2]))
    b = bp(pp_)
    v_free = g + (s_ * pp_ / b) * np.log(0.05 / pp_)
    e_free = v_free - s_ * pp_ / b
    emp = KK.empirical_var_cvar(ref, 0.05)
    fit = KK.al_mle(ref, 0.05)
    v_pin = fit["omega_hat"]
    e_pin = v_pin - fit["sigma_hat"] / (1 - fit["tau_eff"])
    gap_emp = emp["VaR"] - emp["CVaR"]
    r_pin = (v_pin - e_pin) / gap_emp
    r_free = (v_free - e_free) / gap_emp
    print(f"\n  {ref_src}, tau = 0.05, AL with a free shape: p_hat = {pp_:.4f}")
    print(f"      {'':26}{'VaR':>12}{'ES':>12}{'VaR - ES':>12}{'fraction of true gap':>22}")
    print(f"      {'empirical':26}{emp['VaR']:>12.6f}{emp['CVaR']:>12.6f}"
          f"{gap_emp:>12.6f}{1.0:>22.4f}")
    print(f"      {'AL, shape PINNED p = tau':26}{v_pin:>12.6f}{e_pin:>12.6f}"
          f"{v_pin-e_pin:>12.6f}{r_pin:>22.4f}")
    print(f"      {'AL, shape FREE':26}{v_free:>12.6f}{e_free:>12.6f}"
          f"{v_free-e_free:>12.6f}{r_free:>22.4f}")
    check("B7  freeing the shape recovers most of the gap; pinning it does not",
          r_free > 3 * r_pin,
          f"free {r_free:.4f} against pinned {r_pin:.4f}, factor {r_free/r_pin:.1f}")
    RESULTS["B7"] = {"density_crossings_sd": roots, "p_free": float(pp_),
                     "R_pinned": float(r_pin), "R_free": float(r_free)}



# main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", nargs="?", default="spy_returns_2015_2025.csv",
                    help="returns CSV for B0 and B7")
    ap.add_argument("--json", default="experiment_B_results.json")
    args = ap.parse_args()

    ref = np.loadtxt(args.reference, delimiter=",", skiprows=1, usecols=-1)
    ref = ref[np.isfinite(ref)]

    t0 = time.time()
    rule("EXPERIMENT B -- the closed-form R diagnostic")
    print(f"  python        {platform.python_version()}  ({platform.platform()})")
    print(f"  numpy         {np.__version__}")
    print(f"  scipy         {__import__('scipy').__version__}")

    B0(ref, args.reference)
    B1()
    B2()
    B2b()
    B4()
    B5()
    B7(ref, args.reference)

    rule("PASS CRITERIA")
    print("  all criteria passed" if not FAILURES else "\n".join(f"  FAIL  {f}" for f in FAILURES))
    print(f"\n  elapsed {time.time() - t0:.1f}s")

    RESULTS["_meta"] = {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "platform": platform.platform(),
        "failures": FAILURES, "elapsed_s": time.time() - t0,
    }
    with open(args.json, "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print(f"  wrote {args.json}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
