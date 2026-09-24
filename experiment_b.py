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
    B2   specialisation to named families, checked against Monte Carlo
    B2b  R = 1 at tau = 1/2 if and only if mean = median
    B3   the tau -> 0 theorem, all three max-domains of attraction
    B4   the analytic refutation of the fat-tails diagnosis
    B5   R > 1: the diagnostic is two-sided, and unbounded above
    (I cut B6, R across a panel of many assets,from the paper, it remains in the 
    repository but reports nothing the paper cites.  )
.

Conventions and the sample-splitting rule come from kkgibs.py( I made a spelling error 
and decided to stick with it), which spy_run1.p and experiment_A.py also use, so the 
three scripts cannot disagree about what the empirical VaR and CVaR of a sample are.  
Returns are SIMPLE throughout the project.

"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zlib

import numpy as np
from scipy import stats

from rdiag import (
    R_general, R_original, R_mean_excess, R_sample,
    Normal, StudentT, Laplace, Uniform, Exponential, LogNormal,
    NegGPD, AsymmetricLaplace, BetaPower,
)

SEED = 20260831
TAUS = np.array([0.01, 0.05, 0.10, 0.25, 0.50])

RESULTS: dict = {}
FAILURES: list[str] = []


def _tag(name: str) -> int:
    """Deterministic per-family seed component.

    Python's built-in hash() is salted per process unless PYTHONHASHSEED is set,
    so using it in a seed would make this script irreproducible across runs.
    crc32 is stable everywhere.
    """
    return zlib.crc32(name.encode()) % (2 ** 31)


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"    [{tag}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def B0(ref, ref_src, quick):
    """Put the paper's own series on the same axis as the theory."""
    rule("B0  the reference series against the closed forms")
    if ref is None:
        print("\n  no reference series supplied; pass a returns CSV to place the")
        print("  paper's own data in this table.  The closed forms below do not")
        print("  depend on it.")
        RESULTS["B0"] = None
        return
    import kkgibs as KK
    print(f"\n  source: {ref_src}   n = {ref.size}")
    print(f"\n  {'tau':>6}{'R (reference)':>16}{'Normal':>11}{'t(5)':>11}"
          f"{'t(3)':>11}{'AL':>8}{'k':>7}")
    rows = []
    for t in (0.01, 0.025, 0.05, 0.10, 0.25):
        s_ = R_sample(ref, t)
        cf = [float(np.atleast_1d(f.R_closed(t))[0])
              for f in (Normal(), StudentT(5), StudentT(3))]
        rows.append({"tau": float(t), "R_reference": s_["R_general"],
                     "k": s_["k"], "tau_eff": s_["tau_eff"],
                     "R_normal": cf[0], "R_t5": cf[1], "R_t3": cf[2]})
        print(f"  {t:>6g}{s_['R_general']:>16.6f}{cf[0]:>11.6f}{cf[1]:>11.6f}"
              f"{cf[2]:>11.6f}{1.0:>8.4f}{s_['k']:>7d}")

    #  the AL fitted to this very series must return R = 1 to sampling error
    fit = KK.al_mle(ref, 0.05)
    al_gap = fit["sigma_hat"] / (1 - fit["tau_eff"])
    emp_gap = fit["omega_hat"] - fit["cvar_emp"]
    print(f"\n  at tau = 0.05 the AL fitted to this series asserts a shortfall gap")
    print(f"  sigma_hat/(1-tau) = {al_gap:.6f}; the series' actual gap is "
          f"{emp_gap:.6f},")
    print(f"  a ratio of {al_gap/emp_gap:.4f} -- which is R, arrived at from the fit")
    print(f"  rather than from the definition.  spy_run1.py section 5g reports the")
    print(f"  same number from the posterior.")
    RESULTS["B0"] = {"source": ref_src, "n": int(ref.size), "rows": rows,
                     "al_gap": float(al_gap), "emp_gap": float(emp_gap),
                     "ratio": float(al_gap / emp_gap)}


def roster():
    return [
        Normal(), StudentT(5), StudentT(3), Laplace(), Uniform(),
        Exponential(), LogNormal(), NegGPD(0.30), NegGPD(0.15),
        BetaPower(2.0), BetaPower(0.5),
    ]


# =============================================================================
# B1 -- the general closed form
# =============================================================================

def B1(quick: bool) -> None:
    rule("B1  the general closed form")
    print("""
    definition   R = tau (ybar_up - CVaR) / (VaR - CVaR)
    general      R = tau (mu - CVaR) / [(1-tau)(VaR - CVaR)]                (B1)
    mean-excess  R = tau (1 + e_U/e_L),  e_L = VaR-CVaR,  e_U = ybar_up-VaR (B1')

    B1' makes the content explicit:  R = 1  iff  e_U/e_L = (1-tau)/tau, which is
    exactly the rate ratio lambda_L/lambda_U the uniqueness lemma pins.  R is the
    data's tail-asymmetry ratio measured against the one the AL is forced to assume.
""")

    # -- (a) the three forms agree analytically, on every family ---------------
    print("  (a) three forms, analytic moments, max |difference| over tau grid")
    worst = 0.0
    rows = []
    for f in roster():
        mu, q = f.mean(), f.var_(TAUS)
        c, yu = f.cvar(TAUS), f.ybar_up(TAUS)
        r1 = R_general(mu, q, c, TAUS)
        r2 = R_original(yu, q, c, TAUS)
        r3 = R_mean_excess(q - c, yu - q, TAUS)
        d = max(np.max(np.abs(r1 - r2)), np.max(np.abs(r1 - r3)))
        rel = d / np.max(np.abs(r1))
        worst = max(worst, rel)
        rows.append({"family": f.label, "max_abs_diff": float(d),
                     "max_rel_diff": float(rel)})
        print(f"      {f.label:24s}  abs {d:9.2e}   rel {rel:9.2e}")
    check("B1(a) three forms identical to < 1e-12 relative", worst < 1e-12,
          f"worst relative difference {worst:.2e}")

    # -- (b) on finite samples, at the effective tau ---------------------------
    print("\n  (b) finite samples.  A sample splits at an INDEX, not a probability.")
    print("      With k = round(tau*n) and tau_eff = k/n, mu = (1-tau_eff) ybar_up")
    print("      + tau_eff CVaR is an exact identity of the sample, so the two forms")
    print("      must agree to machine precision.  They do:")
    rng = np.random.default_rng(SEED)
    n = 2512                                  # the SPY sample size
    worst_b = 0.0
    for f in roster():
        y = f.rvs(n, rng)
        for t in (0.05, 0.25):
            s = R_sample(y, t, use_effective_tau=True)
            d = abs(s["R_general"] - s["R_original"])
            worst_b = max(worst_b, d / abs(s["R_general"]))
    check("B1(b) sample forms identical at tau_eff to < 1e-12 relative",
          worst_b < 1e-12, f"worst relative difference {worst_b:.2e}")

    # -- (c) the granularity residual, explained not ignored -------------------
    print("\n  (c) the granularity residual.  Evaluated at the NOMINAL tau instead,")
    print("      the identity no longer holds exactly and the two forms separate.")
    print("      n = 2512, tau = 0.05: tau*n = 125.60, so k = 126 and tau_eff =")
    print(f"      126/2512 = {126/2512:.8f}.  The residual is O(|tau_eff - tau|):\n")
    print(f"      {'family':24s} {'general':>12s} {'definition':>12s} {'residual':>11s}")
    res_rows = []
    rng = np.random.default_rng(SEED + 1)
    for f in roster():
        y = f.rvs(n, rng)
        s = R_sample(y, 0.05, use_effective_tau=False)
        d = s["R_general"] - s["R_original"]
        res_rows.append({"family": f.label, "R_general": s["R_general"],
                         "R_original": s["R_original"], "residual": d})
        print(f"      {f.label:24s} {s['R_general']:12.8f} {s['R_original']:12.8f} "
              f"{d:11.2e}")
    dtau = abs(126 / 2512 - 0.05)
    print(f"\n      |tau_eff - tau| = {dtau:.2e};  every residual is of that order.")
    print("      This is the 2.4e-05 seen on SPY at tau = 0.05.  It is a property of")
    print("      finite samples, not of the algebra: it vanishes at tau_eff, and it")
    print("      vanishes as n -> infinity.")

    # convergence of the residual in n
    print("\n      residual on Normal data at nominal tau = 0.05, by n:")
    conv = []
    for nn in (2512, 25120, 251200):
        rr = np.random.default_rng(SEED + 7)
        yy = Normal().rvs(nn, rr)
        s = R_sample(yy, 0.05, use_effective_tau=False)
        d = abs(s["R_general"] - s["R_original"])
        conv.append({"n": nn, "residual": d,
                     "dtau": abs(round(0.05 * nn) / nn - 0.05)})
        print(f"        n = {nn:7d}   |tau_eff - tau| = "
              f"{abs(round(0.05*nn)/nn - 0.05):.2e}   residual = {d:.2e}")

    RESULTS["B1"] = {"analytic": rows, "granularity": res_rows,
                     "granularity_convergence": conv,
                     "worst_rel_analytic": worst, "worst_rel_sample": worst_b}


# =============================================================================
# B2 -- named families, closed form vs Monte Carlo
# =============================================================================

def _cvar_quad(f, tau):
    """CVaR(tau) = (1/tau) int_0^tau F^{-1}(u) du -- uses only the quantile function."""
    from scipy.integrate import quad
    v, err = quad(lambda u: float(np.atleast_1d(f.var_(u))[0]), 0.0, tau,
                  limit=400, epsabs=1e-13, epsrel=1e-13)
    return v / tau, err / tau


def _mean_quad(f):
    from scipy.integrate import quad
    v, err = quad(lambda u: float(np.atleast_1d(f.var_(u))[0]), 0.0, 1.0,
                  limit=400, epsabs=1e-13, epsrel=1e-13)
    return v, err


def B2(quick: bool) -> None:
    rule("B2  named families: closed form, quadrature, and Monte Carlo")
    print("""
    Three arms, three different jobs.

      (a) closed form vs QUADRATURE.  mu and CVaR are integrals of the quantile
          function, so this checks every family's analytics against an independent
          computation with no Monte Carlo error.  The correctness check.

      (b) quantile function vs SAMPLER.  Guards what (a) cannot see: a family whose
          analytics and whose random draws disagree.

      (c) closed form vs MONTE CARLO.  Not a correctness check -- (a) did that --
          but whether the closed form describes a finite sample.  Reported with an
          MC standard error and the plug-in bias separately.
""")

    fams = roster() + [AsymmetricLaplace(p=float(t)) for t in TAUS]

    # ---- (a) quadrature ------------------------------------------------------
    print("  (a) closed form vs quadrature: max relative error over the tau grid")
    worst_a, rows_a = 0.0, []
    for f in fams:
        taus_here = np.array([f.p]) if isinstance(f, AsymmetricLaplace) else TAUS
        mq, _ = _mean_quad(f)
        dmu = abs(mq - f.mean()) / max(abs(f.mean()), 1e-3)
        dr = 0.0
        for t in taus_here:
            cq, _ = _cvar_quad(f, float(t))
            ca = float(np.atleast_1d(f.cvar(t))[0])
            r_quad = R_general(mq, float(np.atleast_1d(f.var_(t))[0]), cq, float(t))
            r_cf = float(np.atleast_1d(f.R_closed(t))[0])
            dr = max(dr, abs(r_quad - r_cf) / abs(r_cf))
        worst_a = max(worst_a, dmu, dr)
        rows_a.append({"family": f.label, "rel_err_mean": float(dmu),
                       "rel_err_R": float(dr)})
        print(f"      {f.label:24s}  mean {dmu:9.2e}   R {dr:9.2e}")
    check("B2(a) closed form matches quadrature to < 1e-9 relative", worst_a < 1e-9,
          f"worst relative error {worst_a:.2e}")

    # ---- (b) quantile function vs sampler ------------------------------------
    print("\n  (b) quantile function vs sampler, 2,000,000 draws, |z| of the")
    print("      empirical quantile against the analytic one")
    m = 200_000 if quick else 2_000_000
    worst_b, rows_b = 0.0, []
    for f in fams:
        rng = np.random.default_rng([SEED, 99, _tag(f.name)])
        y = np.sort(f.rvs(m, rng))
        taus_here = [float(f.p)] if isinstance(f, AsymmetricLaplace) else list(TAUS)
        worst_z = 0.0
        for t in taus_here:
            k = int(round(t * m))
            emp = y[k - 1]
            ana = float(np.atleast_1d(f.var_(t))[0])
            # asymptotic sd of a sample quantile: sqrt(t(1-t)/m) / f(q); estimate
            # the density by a local difference of the quantile function
            h = 1e-4
            dens = (2 * h) / (float(np.atleast_1d(f.var_(t + h))[0])
                              - float(np.atleast_1d(f.var_(t - h))[0]))
            sd = np.sqrt(t * (1 - t) / m) / dens
            worst_z = max(worst_z, abs(emp - ana) / sd)
        worst_b = max(worst_b, worst_z)
        rows_b.append({"family": f.label, "max_abs_z": float(worst_z)})
        print(f"      {f.label:24s}  max |z| = {worst_z:6.2f}")
    check("B2(b) sampler agrees with its own quantile function (|z| < 5)",
          worst_b < 5.0, f"worst |z| = {worst_b:.2f}")

    # ---- (c) Monte Carlo -----------------------------------------------------
    n_batch = 40 if quick else 60
    m_ = 100_000 if quick else 400_000
    print(f"\n  (c) Monte Carlo. {n_batch} independent batches of {m_:,} draws per")
    print(f"      cell (N = {n_batch*m_:,} per cell).  Each batch gives one plug-in")
    print("      Rhat; the MC standard error is the batch-to-batch sd / sqrt(B),")
    print("      which is honest about the dependence inside a batch.  Every tau on")
    print("      the grid divides m exactly, so this arm carries no granularity")
    print("      residual of its own.\n")
    print(f"      HOW MANY DRAWS?  With B = {n_batch}")
    print(f"      batches of {m_:,}, the MC standard error is a few times 1e-4 on an R")
    print("      of order 0.1-1, i.e. agreement to three decimals -- which is what")
    print("      6M draws bought before.  The plug-in Rhat also carries an O(1/m)")
    print("      finite-sample bias, but at this m it sits well inside the noise:")
    print("      see the m-doubling row at the end of this block.\n")

    for t in TAUS:
        assert abs(t * m_ - round(t * m_)) < 1e-9, f"tau {t} does not divide {m_}"

    def mc(f, t, mm, nb):
        sims = np.empty(nb)
        for b in range(nb):
            rng = np.random.default_rng([SEED, _tag(f.name), b, mm])
            y = np.sort(f.rvs(mm, rng))
            k = int(round(t * mm))
            sims[b] = R_general(y.mean(), y[k - 1], y[:k].mean(), t)
        return float(sims.mean()), float(sims.std(ddof=1) / np.sqrt(nb))

    print(f"      {'family':24s} {'tau':>6s} {'closed form':>12s} "
          f"{'simulated':>12s} {'MC se':>10s} {'|z|':>7s}")
    table = []
    for f in fams:
        taus_here = [float(f.p)] if isinstance(f, AsymmetricLaplace) else list(TAUS)
        for t in taus_here:
            cf = float(np.atleast_1d(f.R_closed(t))[0])
            s, e = mc(f, t, m_, n_batch)
            z = abs(cf - s) / e if e > 0 else 0.0
            table.append({"family": f.label, "tau": t, "closed_form": cf,
                          "sim": s, "mc_se": e, "z": float(z),
                          "rel_err": float(abs(s - cf) / abs(cf))})
            print(f"      {f.label:24s} {t:6.2f} {cf:12.6f} {s:12.6f} "
                  f"{e:10.2e} {z:7.2f}")

    ncell = len(table)
    #  many cells are tested at once, so the threshold has to be corrected for
    #  multiplicity or one cell in twenty will fail by construction.
    z_star = float(stats.norm.ppf(1.0 - 0.005 / (2.0 * ncell)))
    zmax = max(r["z"] for r in table)
    over3 = sum(1 for r in table if r["z"] > 3.0)
    rel = max(r["rel_err"] for r in table)
    print(f"\n      {ncell} cells tested at once.  A flat 3-sigma threshold would be")
    print(f"      expected to reject {ncell*0.0027:.2f} of them under the truth, so the")
    print("      criterion is applied family-wise: Bonferroni at 0.5% over all cells")
    print(f"      gives |z| < {z_star:.2f}.")
    check(f"B2(c) closed form matches simulation, family-wise |z| < {z_star:.2f}",
          zmax < z_star, f"max |z| = {zmax:.2f}; {over3} cell(s) above a flat 3")
    check("B2(c) closed form within 1% of simulation everywhere", rel < 1e-2,
          f"worst relative gap {rel:.2e}")

    #  bias check: does the discrepancy shrink when m grows?
    print("\n      finite-sample bias of the plug-in Rhat, four representative cells:")
    print(f"      {'family':16s} {'tau':>6s}" +
          "".join(f"{f'm={s//1000}k':>22s}" for s in (m_ // 4, m_)))
    bias_rows = []
    for f, t in ((Normal(), 0.05), (StudentT(3), 0.01),
                 (LogNormal(), 0.25), (Exponential(), 0.05)):
        cf = float(np.atleast_1d(f.R_closed(t))[0])
        cells = []
        for mm in (m_ // 4, m_):
            s, e = mc(f, t, mm, n_batch)
            cells.append((s - cf, e))
        bias_rows.append({"family": f.label, "tau": t, "closed_form": cf,
                          "d_small": cells[0][0], "se_small": cells[0][1],
                          "d_large": cells[1][0], "se_large": cells[1][1]})
        print(f"      {f.label:16s} {t:6.2f}" +
              "".join(f"{f'{d:+.2e} ({d/e:+.1f} se)':>22s}" for d, e in cells))
    ok_bias = all(abs(r["d_large"]) < 3.5 * r["se_large"] for r in bias_rows)
    check("B2(c) bias at the working m is inside the noise", ok_bias)

    print("""
      The Normal simplifies to  R = [phi(z)/(1-tau)] / [z + phi(z)/tau],
      z = Phi^{-1}(tau) -- free of both location and scale.
      The neg-GPD simplifies to  R = tau (1 - tau^xi) / (xi (1-tau)) -- free of the
      scale beta, so R depends on a Pareto left tail only through xi.
      R is invariant to location and scale for every family here, which is why a
      single curve per family describes it completely.
""")
    RESULTS["B2"] = {"quadrature": rows_a, "sampler": rows_b, "monte_carlo": table,
                     "mc_bias": bias_rows, "mc_m": m_, "mc_batches": n_batch,
                     "mc_z_threshold": z_star, "mc_max_z": zmax}


# =============================================================================
# B2b -- what happens at tau = 1/2
# =============================================================================

def B2b(quick: bool) -> None:
    rule("B2b  R = 1 at tau = 1/2 if and only if mean = median")
    #  An earlier draft of this project asserted R = 1 at tau = 1/2 for EVERY
    #  family and made it a pass criterion.  That is false; it held only because
    #  the grid at the time contained Normal, t(5) and t(3), all symmetric.  The
    #  correct statement and its proof are below.  (See CHANGES.md.)
    print("""
        at tau = 1/2,  R = 1  <=>  mean = median.

    Proof.  R = tau(mu - c)/[(1-tau)(q - c)].  At tau = 1/2 this is (mu - c)/(q - c).
    Also mu = (c + ybar_up)/2, so mu - c = (ybar_up - c)/2, and R = 1 requires
    q - c = (ybar_up - c)/2, i.e. q = (c + ybar_up)/2 = mu.  Since q is the median
    at tau = 1/2, R(1/2) = 1 iff median = mean.  []

    Every symmetric family therefore gives 1 -- which is what was observed, because
    only Normal, t(5) and t(3) were on the grid.  Skewed families do not:
""")
    rows = []
    for f in roster():
        r = float(np.atleast_1d(f.R_closed(0.5))[0])
        mu = f.mean()
        med = float(np.atleast_1d(f.var_(0.5))[0])
        sym = abs(mu - med) < 1e-10
        rows.append({"family": f.label, "R_half": r, "mean": float(mu),
                     "median": med, "mean_eq_median": bool(sym)})
        flag = "mean = median" if sym else f"mean - median = {mu - med:+.6f}"
        print(f"      {f.label:24s}  R(1/2) = {r:8.5f}    {flag}")

    ok = all((abs(r["R_half"] - 1.0) < 1e-4) == r["mean_eq_median"] for r in rows)
    check("B2b R(1/2) = 1 exactly on the mean = median families, and only those", ok)

    print("""
    The paper's claim survives in a better form.  What tau = 1/2 proves is not that
    the AL conversion is exact at the median for everything, but that for SYMMETRIC
    data the conversion is exact at the median and degrades monotonically into the
    tail.  That is still a clean analytic separation of 'tail phenomenon' from
    'general misspecification' -- it just needs the symmetry hypothesis stated.
""")
    RESULTS["B2b"] = rows


# =============================================================================
# B3 -- the tau -> 0 theorem
# =============================================================================

def B3(quick: bool) -> None:
    rule("B3  the tau -> 0 asymptotic, in general")
    #  The three regimes are derived as follows; each is confirmed numerically by
    #  the corresponding subsection below, so the derivations are not reprinted
    #  there.  (i) Frechet: Karamata gives e_L ~ |VaR|/(alpha-1) and e_U ~ |VaR|,
    #  hence R ~ alpha*tau.  (ii) Gumbel: for the Normal, Mills gives e_L ~ 1/|z|
    #  and e_U ~ |z|, hence R ~ tau z^2 ~ 2 tau log(1/tau).  (iii) Weibull: with
    #  F(x) ~ c (x - x_L)^gamma, e_L ~ (tau/c)^(1/gamma)/(gamma+1).
    #
    #  DRAFTING NOTE, not printed: B1', the R = 1 characterisation and the
    #  tau = 1/2 result are three readings of one line of algebra, and the (iii)
    #  trichotomy is three readings of one endpoint formula.  Write them in the
    #  paper as one lemma with corollaries; presenting them as separate results
    #  overstates the content.
    print("""
    THEOREM.  Let Y have finite mean mu and left endpoint x_L, and write
    e_L(tau) = VaR - CVaR,  e_U(tau) = ybar_up - VaR.  Then

        R(tau) = tau ( 1 + e_U/e_L )                                       (B1')

    and since e_U -> mu - x_L as tau -> 0,  R(tau) -> 0  iff  tau e_U/e_L -> 0.
    By max-domain of attraction of the LEFT tail: R ~ alpha*tau in the Frechet
    MDA (alpha the tail index), R ~ 2 tau log(1/tau) in the Gumbel MDA, and at a
    finite left endpoint R(0+) is 0, 2 f(x_L)(mu - x_L), or +infinity according as
    the density there vanishes, is positive and finite, or blows up.

    COROLLARY.  Any return distribution with a finite mean and an unbounded left
    tail lies in the Frechet or Gumbel MDA, so R(tau) -> 0.  The fraction of the
    true shortfall gap the AL conversion can represent VANISHES as the tail is
    entered -- a theorem about the working likelihood, not an observation on SPY.
""")

    out = {}

    # -- (i) Frechet: R/(alpha tau) -> 1 --------------------------------------
    print("  (i) Frechet MDA:  R(tau) / (alpha * tau)  ->  1.")
    print("      The neg-GPD is exact: R/(alpha tau) = (1 - tau^xi)/(1 - tau), so the")
    print("      relative error of the alpha*tau approximation is tau^(1/alpha).  A")
    print("      LIGHTER Pareto tail (bigger alpha) converges more slowly -- which is")
    print("      a reason to quote the exact form and not only the asymptote.\n")
    print(f"      {'family':22s} {'alpha':>7s}" +
          "".join(f"{f'1e-{k}':>11s}" for k in (2, 4, 6, 8, 10, 12)))
    frechet = [StudentT(3), StudentT(5), NegGPD(0.30), NegGPD(0.15)]
    rows_i = []
    for f in frechet:
        a = f.tail_index
        vals = []
        for k in (2, 4, 6, 8, 10, 12):
            t = 10.0 ** (-k)
            vals.append(float(np.atleast_1d(f.R_closed(t))[0]) / (a * t))
        # the tau at which the predicted relative error tau^(1/alpha) is below 1%
        t_ok = 10.0 ** (np.log10(0.01) * a)
        r_ok = float(np.atleast_1d(f.R_closed(t_ok))[0]) / (a * t_ok)
        rows_i.append({"family": f.label, "alpha": a, "ratios": vals,
                       "tau_for_1pct": t_ok, "ratio_there": r_ok})
        print(f"      {f.label:22s} {a:7.2f}" + "".join(f"{v:11.5f}" for v in vals))
    print(f"\n      {'family':22s} {'tau where tau^(1/a)=1%':>24s} {'R/(alpha tau) there':>21s}")
    for r in rows_i:
        print(f"      {r['family']:22s} {r['tau_for_1pct']:24.1e} {r['ratio_there']:21.5f}")
    ok_i = all(abs(r["ratio_there"] - 1.0) < 0.02 for r in rows_i)
    check("B3(i) R/(alpha tau) -> 1, at the tau the rate tau^(1/alpha) predicts",
          ok_i,
          "worst " + f"{max(abs(r['ratio_there'] - 1) for r in rows_i):.2e}")

    #  with xi < 0 the support is bounded and the family leaves the Frechet MDA
    #  entirely, so the regular-variation hypothesis is doing work here.
    fneg = NegGPD(-0.30)
    aneg = 1.0 / -0.30
    neg = [float(np.atleast_1d(fneg.R_closed(10.0 ** -k))[0]) / (aneg * 10.0 ** -k)
           for k in (2, 4, 6, 8, 10, 12)]
    print(f"      {'neg-GPD xi=-0.3':22s} {'--':>7s}" +
          "".join(f"{v:11.2f}" for v in neg))
    print("      ^ xi < 0 gives bounded support, so it is NOT in the Frechet MDA")
    print("        and R/(alpha tau) diverges.  alpha > 1 with a genuinely heavy")
    print("        tail is a hypothesis of the result, not a description of it.")
    check("B3(i) the regular-variation hypothesis is not vacuous (xi<0 diverges)",
          abs(neg[-1]) > 100.0, f"ratio at 1e-12 is {neg[-1]:.0f}")
    out["frechet"] = rows_i
    out["frechet_counterexample"] = {"family": "neg-GPD xi=-0.3", "ratios": neg}

    # -- (ii) Normal ----------------------------------------------------------
    print("\n  (ii) Gumbel MDA, the Normal.  Expanding the Mills ratio one order")
    print("       further than the sketch gives the sharp form")
    print("")
    print("           R(tau) = tau (z^2 + 3) / (1 - tau)  -  6 tau / ((1-tau) z^2)")
    print("                    + O(tau / z^4),")
    print("")
    print("       z = Phi^{-1}(tau).  The leading term alone, tau z^2, is off by")
    print("       3/z^2 -- still 7% at tau = 1e-10, too slow to quote on its own.")
    f = Normal()
    rows_ii = []
    print(f"\n      {'tau':>10s} {'R':>12s} {'R/(t z^2)':>12s} {'R/(t(z^2+3)/(1-t))':>20s}"
          f" {'R/(2t log(1/t))':>17s}")
    for k in range(2, 13):
        t = 10.0 ** (-k)
        r = float(np.atleast_1d(f.R_closed(t))[0])
        z = stats.norm.ppf(t)
        a1 = r / (t * z * z)
        a2 = r / (t * (z * z + 3.0) / (1.0 - t))
        a3 = r / (2.0 * t * np.log(1.0 / t))
        rows_ii.append({"tau": t, "R": r, "over_tau_z2": a1,
                        "over_sharp": a2, "over_2tlog": a3})
        print(f"      {t:10.0e} {r:12.4e} {a1:12.6f} {a2:20.6f} {a3:17.6f}")
    ok_ii = abs(rows_ii[-1]["over_sharp"] - 1.0) < 0.01
    check("B3(ii) R/[tau(z^2+3)/(1-tau)] -> 1 within 1% at tau = 1e-12", ok_ii,
          f"ratio {rows_ii[-1]['over_sharp']:.6f}")
    print("       R ~ 2 tau log(1/tau) is correct to leading order but converges")
    print("       only logarithmically; the (z^2+3) form is the sharp statement.")
    out["normal"] = rows_ii

    # -- (iii) Weibull trichotomy --------------------------------------------
    print("\n  (iii) Weibull MDA, the trichotomy, on Beta(gamma,1) where F(x) = x^gamma:")
    print(f"      {'gamma':>7s} {'predicted R(0+)':>18s}" +
          "".join(f"{f'tau=1e-{k}':>12s}" for k in (2, 4, 6, 8)))
    rows_iii = []
    for g in (2.0, 1.5, 1.0, 0.75, 0.5):
        fb = BetaPower(g)
        # predicted limit: 0 for g>1, 2 f(x_L)(mu-x_L) for g=1, inf for g<1
        if g > 1:
            pred = "0"
        elif g == 1:
            pred = f"{2.0 * 1.0 * fb.mean():.4f}"
        else:
            pred = "infinity"
        vals = [float(np.atleast_1d(fb.R_closed(10.0 ** -k))[0]) for k in (2, 4, 6, 8)]
        rows_iii.append({"gamma": g, "predicted": pred, "values": vals})
        print(f"      {g:7.2f} {pred:>18s}" + "".join(f"{v:12.4g}" for v in vals))

    print("\n      and the endpoint-density formula R(0+) = 2 f(x_L) (mu - x_L) on the")
    print("      two gamma = 1 cases in the roster:")
    rows_ep = []
    for fam, fxl, xl in ((Uniform(), 1.0, 0.0), (Exponential(), 1.0, 0.0)):
        pred = 2.0 * fxl * (fam.mean() - xl)
        got = float(np.atleast_1d(fam.R_closed(1e-9))[0])
        rows_ep.append({"family": fam.label, "predicted": pred, "R_at_1e-9": got})
        print(f"        {fam.label:16s} predicted {pred:8.5f}   R(1e-9) = {got:10.6f}")
    #  two families with a NON-flat density at the endpoint, so the formula is
    #  tested where it is not trivially the uniform.  CVaR is taken from the
    #  quantile integral, so this uses nothing from the family library.
    from scipy.integrate import quad
    for nm, dist, fxl, xl in (
            ("Beta(1,3)", stats.beta(1, 3), 3.0, 0.0),
            ("half-normal", stats.halfnorm(), float(2.0 / np.sqrt(2 * np.pi)), 0.0)):
        t = 1e-7
        mu_ = dist.mean()
        q_ = dist.ppf(t)
        c_ = quad(dist.ppf, 0.0, t, limit=400)[0] / t
        got = R_general(mu_, q_, c_, t)
        pred = 2.0 * fxl * (mu_ - xl)
        rows_ep.append({"family": nm, "predicted": float(pred),
                        "R_at_1e-9": float(got)})
        print(f"        {nm:16s} predicted {pred:8.5f}   R(1e-7) = {got:10.6f}")
    ok_iii = all(abs(r["predicted"] - r["R_at_1e-9"]) < 1e-5 for r in rows_ep)
    check("B3(iii) R(0+) = 2 f(x_L)(mu - x_L) at a positive endpoint density", ok_iii,
          "tested on uniform, exponential, Beta(1,3) and half-normal")
    print("""
      For the uniform, 2 * 1 * (1/2) = 1, and R == 1 at EVERY tau, not only in the
      limit.  See B5.
""")
    out["weibull"] = {"trichotomy": rows_iii, "endpoint_density": rows_ep}

    RESULTS["B3"] = out


# =============================================================================
# B4 -- the analytic refutation of the fat-tails diagnosis
# =============================================================================

def B4(quick: bool) -> None:
    rule("B4  the analytic refutation of the fat-tails diagnosis")
    print("""
    [CITATION VERIFIED 1 Sep 2026 from the published article.]
    Song Shi & Xinyu Wang, "Parametric Quantile Regression Using Mixture
    Distributions: Estimating Value at Risk and Expected Shortfall", Journal of
    Forecasting 2026; 45:1495-1510, DOI 10.1002/for.70103.  Their abstract:

        "However, the asymmetric Laplace distribution does not fit the
         distribution of asset returns, which is characterised by SHARP PEAKS AND
         HEAVY TAILS."

    and section 2.1: "The AL distribution (Yu and Moyeed 2001) is widely used for
    eps_t, but it is improper to describe the statistical feature of financial
    asset returns."

    The attribution is a SHAPE-FIT claim -- peaks and tails together -- not one
    about the tail being too thin for ES specifically.  That makes the refutation
    sharper: Gaussian data has neither a sharp peak nor a heavy tail, so on a
    shape-mismatch account the AL should do well there.  If fat tails were the
    mechanism, R would approach 1 as the tail is lightened.  Sending nu -> infinity
    in the Student-t converges to the Normal, and R does not approach 1.
""")
    print(f"      {'nu':>10s} {'R(tau=0.05)':>14s} {'R(tau=0.01)':>14s}")
    rows = []
    for nu in (2.5, 3, 4, 5, 10, 30, 100, 1000, 10000):
        f = StudentT(nu)
        r5 = float(np.atleast_1d(f.R_closed(0.05))[0])
        r1 = float(np.atleast_1d(f.R_closed(0.01))[0])
        rows.append({"nu": float(nu), "R_05": r5, "R_01": r1})
        print(f"      {nu:10g} {r5:14.6f} {r1:14.6f}")
    fn = Normal()
    rn5 = float(np.atleast_1d(fn.R_closed(0.05))[0])
    rn1 = float(np.atleast_1d(fn.R_closed(0.01))[0])
    print(f"      {'Normal':>10s} {rn5:14.6f} {rn1:14.6f}   <- the limit")
    print(f"      {'AL':>10s} {1.0:14.6f} {1.0:14.6f}   <- what 'no failure' looks like")

    ok = (rows[-1]["R_05"] - rn5) < 1e-3 and rn5 < 0.30
    check("B4 R(Normal, tau=0.05) bounded well away from 1", rn5 < 0.30,
          f"R = {rn5:.6f}")
    check("B4 t(nu) -> Normal as nu -> infinity, not -> 1", ok,
          f"R(t(10000)) = {rows[-1]['R_05']:.6f} vs Normal {rn5:.6f}")
    print(f"""
    The result does not depend on the attribution above.  If fat tails were the
    mechanism, R would approach 1 as the tail lightens; it approaches {rn5:.4f}.

    Monotone increasing in nu, converging to {rn5:.4f}, not to 1.  Fat tails make the
    failure worse -- t(3) at 0.05 gives {rows[1]['R_05']:.4f} against the Normal's {rn5:.4f} -- but
    they are demonstrably not its cause.  B3(i) says why: in the Frechet MDA
    R ~ alpha*tau, so the tail index sets the SLOPE, not whether R vanishes.
    Removing the fat tail changes alpha.  It does not change that R -> 0.
""")
    RESULTS["B4"] = {"t_sequence": rows, "normal": {"R_05": rn5, "R_01": rn1}}


# =============================================================================
# B5 -- R > 1, and what R = 1 does and does not mean
# =============================================================================

def B5(quick: bool) -> None:
    rule("B5  R is two-sided, unbounded above, and R = 1 does not mean AL")
    print("""
    (a) R > 1: the AL OVERSTATES the shortfall gap on some distributions.
""")
    rows_a = []
    for f in (LogNormal(), Exponential(), BetaPower(0.5)):
        vals = [float(np.atleast_1d(f.R_closed(t))[0]) for t in TAUS]
        rows_a.append({"family": f.label, "taus": TAUS.tolist(), "R": vals})
        print(f"      {f.label:24s}" + "".join(f"{v:10.4f}" for v in vals))
    print(f"      {'(tau)':24s}" + "".join(f"{t:10g}" for t in TAUS))

    # the lognormal crossing
    from scipy.optimize import brentq
    ln = LogNormal()
    xr = brentq(lambda t: float(np.atleast_1d(ln.R_closed(t))[0]) - 1.0, 1e-6, 0.4)
    print(f"""
    The lognormal crosses R = 1 at tau = {xr:.5f}: it OVERSTATES the gap above that
    level and UNDERSTATES it below, and by B3(iii) with gamma > 1 it still goes to 0
    as tau -> 0.  So R is a genuine two-sided diagnostic, and 'the AL is too thin'
    is not even the right one-line summary -- it is too thin WHERE IT MATTERS.

    (b) R is unbounded above.  Beta(gamma,1) with gamma < 1 has R(0+) = infinity:
""")
    rows_b = []
    for g in (0.75, 0.5, 0.25):
        fb = BetaPower(g)
        vals = [float(np.atleast_1d(fb.R_closed(10.0 ** -k))[0]) for k in (1, 2, 3, 4)]
        rows_b.append({"gamma": g, "R": vals})
        print(f"      gamma = {g:4.2f}" + "".join(f"{v:14.4g}" for v in vals))
    print(f"      {'(tau)':12s}" + "".join(f"{10.0**-k:14g}" for k in (1, 2, 3, 4)))

    print("""
    (c) R = 1 does NOT characterise the AL.  By B1', R = 1 is the single moment
    condition

        e_U/e_L = (1-tau)/tau,

    a one-dimensional constraint.  The AL satisfies it, but so does anything else
    with that mean-excess ratio.  The uniform satisfies it at EVERY tau:
""")
    u = Uniform()
    tg = np.array([0.001, 0.01, 0.05, 0.1, 0.25, 0.5])
    ru = np.atleast_1d(u.R_closed(tg))
    print(f"      {'tau':>10s}" + "".join(f"{t:12g}" for t in tg))
    print(f"      {'R(uniform)':>10s}" + "".join(f"{v:12.9f}" for v in ru))
    # exact statement
    eL = u.var_(tg) - u.cvar(tg)
    eU = u.ybar_up(tg) - u.var_(tg)
    print(f"      {'e_U/e_L':>10s}" + "".join(f"{v:12.6f}" for v in eU / eL))
    print(f"      {'(1-t)/t':>10s}" + "".join(f"{v:12.6f}" for v in (1 - tg) / tg))
    ok = np.max(np.abs(ru - 1.0)) < 1e-12
    check("B5(c) uniform gives R = 1 identically", ok,
          f"max |R-1| = {np.max(np.abs(ru - 1.0)):.2e}")

    print("""
    So R = 1 is necessary, not sufficient, for the AL's ES conversion to be right.
    The conversion needs one number right: the ratio of mean excess above the
    quantile to mean shortfall below it.  The uniqueness lemma says quantile
    regression under the AL has already spent that number on the quantile
    constraint.  A distribution that happens to agree gets the right ES by
    coincidence; nothing about the fit made it so.
""")
    RESULTS["B5"] = {"above_one": rows_a, "lognormal_crossing_tau": float(xr),
                     "unbounded": rows_b,
                     "uniform_R": ru.tolist(), "uniform_taus": tg.tolist()}


# =============================================================================
# B7  reconciling Chen, Gerlach & Lu
# =============================================================================

def B7(ref, ref_src, quick):
    """
    CGL (2012) publish that AL tails are FATTER than Student-t.  This project says
    the AL's tail is far too thin.  Both are true, and the reconciliation is the
    single free shape parameter.  All of it is verified here rather than recalled.
    """
    rule("B7  reconciling Chen, Gerlach & Lu (2012)")
    from scipy import integrate, optimize, stats  # noqa: F401

    print("""
    THE APPARENT CONTRADICTION.  Chen, Gerlach & Lu (2012, CSDA 56(11) 3498-3516,
    DOI 10.1016/j.csda.2010.06.018) write, section 2.2 p.3501, verbatim:

        "for p = 0.5, the ALD and standardized Student t distributions both have
         kurtosis of 6.  However, the standardized t density function is lower
         than that for an ALD(0, 1, 0.5), from 2 to 6 standard deviations away
         from the mean.  In this region, the symmetric Laplace density has fatter
         tails. ... For an ALD(0, 1, 0.55), the skewness and kurtosis are
         approximately -0.4 and 6.2 respectively.  A skewed t with the same
         skewness and kurtosis again has its left tail lighter than the ALD
         between 2 and 6 standard deviations from the mean."

    They are right, and this project says the AL's tail is much too thin.  Both
    hold because THEIR p IS FREE AND NEAR 0.5, WHILE QUANTILE REGRESSION PINS
    p = tau.  The shape parameter CGL spend on the tail is the one the quantile
    constraint has already taken.
""")

    def bp(pp):
        return np.sqrt(pp ** 2 + (1 - pp) ** 2)

    #  (a) their kurtosis formula and the 2-6 sd claim, checked
    def kurt(pp):
        return (9 * (1 - pp) ** 4 + 6 * (1 - pp) ** 2 * pp ** 2 + 9 * pp ** 4) / bp(pp) ** 4
    print(f"  (a) CGL's kurtosis formula at p = 0.5 gives {kurt(0.5):.4f} "
          f"(they state 6); at p = 0.55, {kurt(0.55):.4f} (they state 6.2)")

    #  standardised ALD(0,1,p) density, their eq (2)
    def ald(x, pp):
        b = bp(pp)
        return b * np.exp(-b * np.abs(x) * np.where(x < 0, 1 / pp, 1 / (1 - pp)))
    #  standardised Student-t with kurtosis 6  ->  6/(nu-4) = 3  ->  nu = 6
    nu = 6.0
    sc = 1.0 / np.sqrt(nu / (nu - 2.0))
    #  Rather than score what fraction of [2,6] the ALD dominates -- which needs an
    #  arbitrary threshold -- solve for where the two densities actually cross.  If
    #  CGL are right the crossings should be AT 2 and 6, and they are.
    gap = lambda x: ald(x, 0.5) - stats.t.pdf(x / sc, nu) / sc
    xs = -np.linspace(1.0, 8.0, 20001)
    sgn = np.sign(gap(xs))
    idx = np.where(np.diff(sgn) != 0)[0]
    roots = [float(optimize.brentq(gap, xs[i], xs[i + 1])) for i in idx]
    frac = float(np.mean(gap(-np.linspace(2, 6, 4001)) > 0))
    print(f"  (b) the standardised t with kurtosis 6 is nu = {nu:.0f}.  Solving for where")
    print(f"      the two densities cross on the left tail gives")
    for rt in roots:
        print(f"        x = {rt:.4f} standard deviations")
    print(f"      so the ALD(0,1,0.5) density dominates the t precisely on")
    print(f"      ({roots[0]:.2f}, {roots[-1]:.2f}) sd -- CGL's stated 2 to 6, to the nearest")
    print(f"      integer.  Over exactly [2,6] it dominates at {100*frac:.1f}% of points, the")
    print(f"      shortfall being the sliver beyond {abs(roots[-1]):.2f}.")
    ok_b = (len(roots) == 2 and abs(abs(roots[0]) - 2.0) < 0.1
            and abs(abs(roots[-1]) - 6.0) < 0.1)
    check("B7  CGL's 2-to-6-sd claim reproduces, as the exact crossing points",
          ok_b, f"crossings at {roots[0]:.3f} and {roots[-1]:.3f} sd")

    #  (c) their ES formula, and the fact that its RATIO form is singular at alpha=p
    print("""
  (c) CGL's eq (5) is  ES_a = (1 - 1/log(a/p)) VaR_a  for a < p.  Note what that
      says about the case this project studies.  Quantile regression sets a = p:
      the quantile IS the AL's own location.  There log(a/p) = 0, VaR_a = 0, and
      the ratio form is singular -- the multiplicative factor diverges while VaR
      vanishes.  Their whole apparatus is built for a < p with p free; the
      quantile-regression case is the one point their ratio cannot express.""")
    print()
    print(f"      {'alpha':>10}{'VaR':>14}{'ratio factor':>16}{'ES':>14}")
    pp = 0.05
    for a in (0.01, 0.03, 0.045, 0.049, 0.04999):
        v = (pp / bp(pp)) * np.log(a / pp)
        r_ = 1 - 1 / np.log(a / pp)
        print(f"      {a:>10g}{v:>14.8f}{r_:>16.2f}{r_*v:>14.8f}")
    print(f"      {pp:>10g}{0.0:>14.8f}{'inf':>16}{-pp/bp(pp):>14.8f}   <- alpha = p")
    #  and the limit agrees with our Proof 1
    ours = K_al_es_check(pp)
    print()
    print(f"      our Proof 1 at the same point: {ours:+.8f}  "
          f"(difference {abs(ours + pp/bp(pp)):.1e})")
    check("B7  CGL's ES at alpha = p is identical to Proof 1",
          abs(ours + pp / bp(pp)) < 1e-15,
          f"difference {abs(ours + pp/bp(pp)):.2e}")

    #  (d) the reconciliation on the reference series
    if ref is None:
        print("\n  (d) needs a reference series; pass a returns CSV.")
        RESULTS["B7"] = {"kurt_05": kurt(0.5), "frac_above": frac}
        return
    import kkgibs as KK
    def nll(th, y):
        g, ls, lp = th
        s_, pp_ = np.exp(ls), 1 / (1 + np.exp(-lp))
        b, u = bp(pp_), y - g
        return -np.sum(np.log(b / s_) - np.abs(u) *
                       np.where(u < 0, b / (s_ * pp_), b / (s_ * (1 - pp_))))
    o = optimize.minimize(nll, [np.median(ref), np.log(ref.std()), 0.0], args=(ref,),
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
    print()
    print(f"  (d) the reconciliation on {ref_src}, at tau = 0.05")
    print(f"      AL fitted with a FREE shape: p_hat = {pp_:.4f} "
          f"(CGL report ~0.55 on their data)")
    print()
    print(f"      {'':26}{'VaR':>12}{'ES':>12}{'VaR - ES':>12}"
          f"{'fraction of true gap':>22}")
    print(f"      {'empirical':26}{emp['VaR']:>12.6f}{emp['CVaR']:>12.6f}"
          f"{gap_emp:>12.6f}{1.0:>22.4f}")
    print(f"      {'AL, shape PINNED p = tau':26}{v_pin:>12.6f}{e_pin:>12.6f}"
          f"{v_pin-e_pin:>12.6f}{r_pin:>22.4f}")
    print(f"      {'AL, shape FREE':26}{v_free:>12.6f}{e_free:>12.6f}"
          f"{v_free-e_free:>12.6f}{r_free:>22.4f}")
    print(f"""
      Freeing the single shape parameter takes the AL from {r_pin:.2f} to {r_free:.2f} of the
      true shortfall gap -- a factor of {r_free/r_pin:.1f} -- without changing the family, the
      data or the estimation method.  CGL's AL has a tail because their p is
      free; the quantile-regression AL does not, because p is spent on the
      quantile constraint.

      With a free shape the AL still reaches only {100*r_free:.0f}% of the gap on this
      series.""")
    check("B7  freeing the shape recovers most of the gap; pinning it does not",
          r_free > 3 * r_pin,
          f"free {r_free:.4f} against pinned {r_pin:.4f}, factor {r_free/r_pin:.1f}")
    RESULTS["B7"] = {"kurt_05": kurt(0.5), "kurt_055": kurt(0.55),
                     "frac_ald_above_t_2to6sd": frac,
                     "density_crossings_sd": roots,
                     "p_free": float(pp_), "R_pinned": float(r_pin),
                     "R_free": float(r_free),
                     #  values recorded in an earlier draft that do not
                     #  reproduce; kept so the correction is traceable
                     "superseded_values": {"free": 0.83, "pinned": 0.06}}


def K_al_es_check(pp):
    """Our Proof 1 ES for the AL matched to CGL's ALD(0,1,p) at alpha = p."""
    import kkgibs as KK
    sig = pp * (1 - pp) / np.sqrt(pp ** 2 + (1 - pp) ** 2)
    return float(KK.al_es(0.0, sig, pp))


# =============================================================================
# main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", nargs="?", default=None,
                    help="returns CSV to place alongside the closed forms in B0")
    ap.add_argument("--quick", action="store_true",
                    help="fewer Monte Carlo draws per cell")
    ap.add_argument("--json", default="experiment_B_results.json")
    args = ap.parse_args()

    import os
    ref, ref_src = None, None
    for cand in ([args.reference] if args.reference else []) + \
                ["spy_returns_2015_2025.csv"]:
        if cand and os.path.exists(cand):
            ref = np.loadtxt(cand, delimiter=",", skiprows=1, usecols=-1)
            ref = ref[np.isfinite(ref)]
            ref_src = cand
            break

    t0 = time.time()
    print("=" * 78)
    print("EXPERIMENT B -- the closed-form R diagnostic")
    print("=" * 78)
    print(f"  seed          {SEED}")
    print(f"  python        {platform.python_version()}  ({platform.platform()})")
    print(f"  numpy         {np.__version__}")
    print(f"  scipy         {__import__('scipy').__version__}")
    print(f"  mode          {'quick' if args.quick else 'full'}")

    B0(ref, ref_src, args.quick)
    B1(args.quick)
    B2(args.quick)
    B2b(args.quick)
    B3(args.quick)
    B4(args.quick)
    B5(args.quick)
    B7(ref, ref_src, args.quick)

    rule("PASS CRITERIA")
    if FAILURES:
        for f in FAILURES:
            print(f"  FAIL  {f}")
    else:
        print("  all criteria passed")
    print(f"\n  elapsed {time.time() - t0:.1f}s")

    RESULTS["_meta"] = {
        "seed": SEED, "quick": args.quick,
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
