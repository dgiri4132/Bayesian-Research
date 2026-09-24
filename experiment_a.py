#!/usr/bin/env python3
"""This file is an extension experient side where I basically try to prove 
that there is nothing wrong with the code rather misspecifcation in the model chosen
it is gonna be complemented by other files that i have set out to upload in the next few days

EXPERIMENT A -- the positive control.

Bayesian CVaR project.  Darshan Giri / Prof. Songfeng Zheng.

THE QUESTION A ANSWERS.  spy_run1.py shows the ES posterior missing the empirical
CVaR by many posterior standard deviations.  On its own that is compatible with
several explanations, only one of which is interesting:

    (i)   the sampler is wrong
    (ii)  the prior is doing it
    (iii) n is too small / it is estimation error
    (iv)  the algebra of Proof 1 is wrong
    (v)   the AL working likelihood is misspecified for this data

A eliminates (i)-(iv) by holding the entire pipeline fixed -- same sampler, same
n, same prior, same code path -- and varying ONLY the data-generating process.  If
coverage is nominal when the data really is asymmetric-Laplace and zero otherwise,
then nothing about the machinery can be the cause, and (v) is what is left.

DESIGN NOTE, and a departure from the original plan.

    The coverage study runs on the EXACT posterior, not on the Gibbs sampler.
    For the intercept-only model sigma integrates out in closed form (see
    kkgibbs.exact_posterior), leaving a one-dimensional posterior that can be
    sampled i.i.d.  That is strictly better here for two reasons:

      - it removes Monte Carlo error from the control entirely, so a coverage of
        0.0% cannot be blamed on chains that had not mixed;
      - it is about 2000x faster, which is what makes A2 and A3 affordable at
        1000 replicates.

    Section A1 checks the exact posterior against the Gibbs sampler on real
    replicates, so the substitution is verified rather than assumed.

THE TWO SECTIONS THAT CARRY THE ARGUMENT.

    A3 (coverage against n).  Misspecification bias is O(1) while the posterior
    sd is O(n^-1/2), so the miscentring measured in posterior standard deviations
    grows like sqrt(n): more data makes the interval worse.  Estimation error is
    the thing that goes away with more data, so this rules out (iii) directly.

    A2 reports coverage for the QUANTILE as well as the ES.  Under
    misspecification the AL posterior is centred on the quantile but its spread
    is not calibrated (the sandwich problem).  Reporting both separates the two
    defects: the VaR interval is the wrong width, the ES interval is in the wrong
    place.

usage:  python3 experiment_A.py [reference.csv] [--quick] [--reps N] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

import kkgibs as K


def _tag(name: str) -> int:
    """This is the deterministic per-DGP seed component.  crc32 rather than hash(), which is
    salted  per  process  and  would  make  the  script  irreproducible  between runs."""
    import zlib
    return zlib.crc32(name.encode()) % 10_000


def coverage_band(reps, nominal=95.0, n_se=2.0):
    """
    The width is the binomial standard error of the coverage estimate itself, not
    a fixed tolerance: at 60 replicates a calibrated interval lands on 100% about
    one run in 22, while at 3000 a fixed +/- 5 points would pass a broken one.

    n_se = 2, not 3 because I found out that 3-se band at 1000 replicates is 
    [92.9, 97.1] and would pass a true coverage of 93%; at 2 se it is [93.6, 96.4]
     and 93% is rejected.

    Returns (lo, hi, se); `lo` is the lower edge of the band, any true coverage
    below it gets rejected.

    Calibration of the exact posterior was measured directly, not assumed i.e. at
    n = 2512, tau = 0.05, 3000 replicates gave 94.80% +/- 0.41 (z = -0.49), and an
    independent audit at 16 000 replicates gave 95.01% +/- 0.17 (z = +0.04).
    """
    p = nominal / 100.0
    se = 100.0 * np.sqrt(p * (1 - p) / reps)
    lo, hi = nominal - n_se * se, nominal + n_se * se
    return lo, hi, se

TAU = 0.05
SEED = 20260901
N_REF = 2512
RHAT_TARGET = 1.01

RESULTS: dict = {}
FAILURES: list[str] = []


def check(name, ok, detail="", hard=True):
    tag = "PASS" if ok else ("FAIL" if hard else "note")
    print(f"    [{tag}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok and hard:
        FAILURES.append(f"{name}: {detail}")
    return ok


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

# the data-generating processes
def discrete_var_es(y, tau):
    """
    This is the exact VaR and ES of the EMPIRICAL distribution of y.

        VaR = y_(ceil(tau n))
        ES  = (1/tau) [ (1/n) sum_{y < VaR} y  +  VaR (tau - P(Y < VaR)) ]

    The second term is the partial atom at VaR, dropping it is the usual way this
    is gotten slightly wrong.  Checked against an 8-million-draw resample: agreement
    to Monte Carlo error.
    """
    s = np.sort(np.asarray(y, dtype=float))
    n = s.size
    #  Float-safe ceiling: 0.07*100 evaluates to 7.000000000000001, so a bare
    # ceil() returns the wrong order statistic.  VaR only, the ES is immune
    j = min(max(int(np.ceil(tau * n - 1e-9)), 1), n)
    var_ = s[j - 1]
    m = int(np.searchsorted(s, var_, side="left"))
    es = (s[:m].sum() / n + var_ * (tau - m / n)) / tau
    return float(var_), float(es)


class DGP:
    """A data-generating process with an EXACTLY known VaR and ES at tau."""

    def __init__(self, name, sampler, var_, es_, note=""):
        self.name, self._s, self.var_, self.es_ = name, sampler, var_, es_
        self.note = note

    def rvs(self, n, rng):
        return self._s(n, rng)


def build_dgps(ref, tau):
    """
    Five processes, all calibrated to the SAME reference return series, so the
    comparison is  kind of like one dataset, five ways of describing it rather 
    than five unrelated simulations.  Only the AL row is correctly specified.
    """
    fit = K.al_mle(ref, tau)
    om, sg = fit["omega_hat"], fit["sigma_hat"]

    mu, sd = float(ref.mean()), float(ref.std(ddof=1))
    z = stats.norm.ppf(tau)

    def t_dgp(nu):
        #  match location and scale to the reference by moments of the scaled t
        sc = sd / np.sqrt(nu / (nu - 2.0))
        q = stats.t.ppf(tau, nu)
        f = stats.t.pdf(q, nu)
        es_std = -(f / tau) * (nu + q * q) / (nu - 1.0)
        return DGP(f"Student-t({nu})",
                   lambda n, rng, nu=nu, sc=sc: mu + sc * rng.standard_t(nu, n),
                   mu + sc * q, mu + sc * es_std,
                   f"loc {mu:+.6f}, scale {sc:.6f}")

    bs_var, bs_es = discrete_var_es(ref, tau)

    return [
        DGP("AL (correct)",
            lambda n, rng: K.al_rvs(n, om, sg, tau, rng),
            om, K.al_es(om, sg, tau),
            f"omega {om:+.6f}, sigma {sg:.6f} (AL MLE of the reference)"),
        DGP("Normal",
            lambda n, rng: rng.normal(mu, sd, n),
            mu + sd * z, mu - sd * stats.norm.pdf(z) / tau,
            f"mean {mu:+.6f}, sd {sd:.6f}"),
        t_dgp(5),
        t_dgp(3),
        DGP("bootstrap (reference)",
            lambda n, rng: rng.choice(ref, size=n, replace=True),
            bs_var, bs_es, f"resampled from the reference series, n = {ref.size}"),
    ]

# one replicate

def posterior_summary(y, tau, seed, grid_points, n_draws, refine=False):
    ex = K.exact_posterior(y, tau, n_draws, seed=seed,
                           grid_points=grid_points, verify_refinement=refine)
    es = ex.omega - ex.sigma / (1 - tau)
    lo, hi = np.percentile(es, [2.5, 97.5])
    olo, ohi = np.percentile(ex.omega, [2.5, 97.5])
    return {"es_mean": float(es.mean()), "es_sd": float(es.std(ddof=1)),
            "es_lo": float(lo), "es_hi": float(hi),
            "om_mean": float(ex.omega.mean()),
            "om_lo": float(olo), "om_hi": float(ohi)}


# A1  the exact posterior is the Gibbs posterior

def A1(dgps, tau, n, quick):
    rule("A1  the exact posterior agrees with the Gibbs sampler")
    print("""
    The coverage study below uses the exact posterior.  That substitution is only
    legitimate if the two really are the same posterior, so it is checked here on
    a fresh replicate from every DGP, with the full four-chain Gibbs run and its
    convergence diagnostics.
""")
    #  quick mode uses the shortest chains that still converge here: at 3000/1000
    #  R-hat reaches 1.013 and this section fails honestly.
    n_iter, burn = (8000, 3000) if quick else (15000, 5000)
    rows, worst, worst_sd, worst_ratio = [], 0.0, 0.0, 0.0
    print(f"  {'DGP':<22}{'quantity':<10}{'Gibbs':>13}{'exact':>13}{'diff':>11}"
          f"{'R-hat':>9}")
    for d in dgps:
        rng = np.random.default_rng([SEED, 1, _tag(d.name)])
        y = d.rvs(n, rng)
        X = np.ones((n, 1))
        th, sg = [], []
        for c in range(4):
            r = K.gibbs_qr_sampler(X, y, tau, n_iter, burn, seed=1000 + 111 * c,
                                   omega_init=[float(np.quantile(y, tau))],
                                   sigma_init=0.01)
            th.append(r.omega[:, 0])
            sg.append(r.sigma)
        rhat = max(K.split_rhat(th), K.split_rhat(sg))
        tha, sga = np.concatenate(th), np.concatenate(sg)
        g_es = tha - sga / (1 - tau)
        e = posterior_summary(y, tau, seed=7, grid_points=400_001, n_draws=400_000)
        pairs = [("ES mean", float(g_es.mean()), e["es_mean"]),
                 ("ES 2.5%", float(np.percentile(g_es, 2.5)), e["es_lo"]),
                 ("ES 97.5%", float(np.percentile(g_es, 97.5)), e["es_hi"])]
        es_sd = float(g_es.std(ddof=1))
        #  Tolerance is set by the Gibbs run's own Monte Carlo error. Requiring
        #  two estimates to agree more closely than either one is determined is
        #  not a test.
        es_chains = [t - sc / (1 - tau) for t, sc in zip(th, sg)]
        es_mcse = K.mcse(es_chains, sum(K.ess(c) for c in es_chains))
        tol = max(0.05, 4.0 * es_mcse / es_sd)
        for i, (nm, a, b) in enumerate(pairs):
            worst = max(worst, abs(a - b))
            worst_sd = max(worst_sd, abs(a - b) / es_sd)
            worst_ratio = max(worst_ratio, (abs(a - b) / es_sd) / tol)
            print(f"  {d.name if i == 0 else '':<22}{nm:<10}{a:>13.7f}{b:>13.7f}"
                  f"{a-b:>11.2e}" + (f"{rhat:>9.5f}" if i == 0 else ""))
        rows.append({"dgp": d.name, "rhat": float(rhat), "es_sd": es_sd,
                     "gibbs_es_mean": float(g_es.mean()), "exact_es_mean": e["es_mean"]})
    #  The gate is in posterior-sd units.  I found out that an absolute 1e-4 threshold would not
    #  bind: a 5%  bias  injected  into  the  Gibbs  sigma  block  moves  the  worst  ES
    #  summary by 7.4e-5, inside 1e-4 but 0.6 posterior sd, which is the defect
    #  this section exists to exclude.
    print(f"\n  worst disagreement: {worst:.2e} absolute = {worst_sd:.3f} posterior sd")
    #  The coverage loops disable the refinement diagnostic for speed, so it runs
    #  here instead, and grid error is measured rather than assumed.
    print(f"\n  grid diagnostics, one replicate per process")
    print(f"  {'DGP':<22}{'points / posterior sd':>22}{'refinement rel change':>24}")
    grid_rows, worst_ref, worst_res = [], 0.0, np.inf
    for d in dgps:
        rng = np.random.default_rng([SEED, 2, _tag(d.name)])
        y = d.rvs(n, rng)
        ex = K.exact_posterior(y, tau, 40_000, seed=11, grid_points=40_001,
                               verify_refinement=True)
        worst_ref = max(worst_ref, ex.refinement_max_rel_change)
        worst_res = min(worst_res, ex.points_per_posterior_sd)
        grid_rows.append({"dgp": d.name,
                          "points_per_sd": ex.points_per_posterior_sd,
                          "refinement": ex.refinement_max_rel_change})
        print(f"  {d.name:<22}{ex.points_per_posterior_sd:>22.0f}"
              f"{ex.refinement_max_rel_change:>24.2e}")
    check("A1  grid error is negligible at the settings the coverage loops use",
          worst_ref < 1e-6, f"worst relative change on doubling the grid "
                            f"{worst_ref:.2e}")
    check("A1  the grid resolves the posterior everywhere", worst_res > 100.0,
          f"fewest points per posterior sd: {worst_res:.0f}")

    print(f"\n  tolerance is max(0.05 sd, 4 x the Gibbs run's own MCSE/sd);"
          f" the worst cell used {worst_ratio:.2f} of it")
    check("A1  Gibbs and exact posteriors agree within the Gibbs run's own MC error",
          worst_ratio < 1.0,
          f"worst cell at {worst_ratio:.2f} of tolerance "
          f"({worst_sd:.3f} posterior sd, {worst:.2e} absolute)")
    check("A1  every Gibbs run converged", all(r["rhat"] < RHAT_TARGET for r in rows),
          f"worst R-hat {max(r['rhat'] for r in rows):.5f}")
    RESULTS["A1"] = {"rows": rows, "worst_diff": float(worst),
                     "worst_diff_post_sd": float(worst_sd),
                     "worst_tolerance_ratio": float(worst_ratio),
                     "grid": grid_rows,
                     "worst_refinement": float(worst_ref),
                     "worst_points_per_sd": float(worst_res)}


# A2  the coverage table

def coverage(d, tau, n, reps, seed, grid_points, n_draws):
    cov_es = cov_om = 0
    bias_sd, es_means, om_means = [], [], []
    for i in range(reps):
        rng = np.random.default_rng([seed, _tag(d.name), i])
        y = d.rvs(n, rng)
        s = posterior_summary(y, tau, seed=int(1_000_003 + i),
                              grid_points=grid_points, n_draws=n_draws)
        if s["es_lo"] <= d.es_ <= s["es_hi"]:
            cov_es += 1
        if s["om_lo"] <= d.var_ <= s["om_hi"]:
            cov_om += 1
        bias_sd.append((s["es_mean"] - d.es_) / s["es_sd"])
        es_means.append(s["es_mean"])
        om_means.append(s["om_mean"])
    return {"cov_es": 100.0 * cov_es / reps, "cov_om": 100.0 * cov_om / reps,
            "bias_sd_mean": float(np.mean(bias_sd)),
            "es_mean_avg": float(np.mean(es_means)),
            "om_mean_avg": float(np.mean(om_means)),
            "reps": reps}


def A2(dgps, tau, n, reps, grid_points, n_draws):
    rule("A2  coverage of the 95% credible interval")
    print(f"""
    {reps} replicates per process, n = {n}, tau = {tau}.  Identical pipeline
    throughout; only the data-generating process changes.

    Read the ES column first, then the VaR column.  The VaR column is what makes
    the ES column mean something: the same fit, on the same draw, gets the
    quantile right and the shortfall wrong.
""")
    print("""  A note on independence.  Replicate i uses the same posterior-sampling
  seed for every process, so the five rows share random numbers.  That is
  deliberate because common random numbers reduce the variance of the DIFFERENCES
  between rows, which is what the table is for but it does mean the five
  columns are correlated and are not five independent pieces of evidence.  The
  AL row against the rest is the comparison this design is powered for.
""")
    print(f"  {'DGP':<22}{'true VaR':>11}{'true ES':>11}{'VaR cov':>9}"
          f"{'ES cov':>9}{'ES bias':>10}")
    print(f"  {'':<22}{'':>11}{'':>11}{'%':>9}{'%':>9}{'post sd':>10}")
    print("  " + "-" * 72)
    rows = []
    for d in dgps:
        c = coverage(d, tau, n, reps, SEED, grid_points, n_draws)
        rows.append({"dgp": d.name, "note": d.note, "true_var": d.var_,
                     "true_es": d.es_, **c})
        print(f"  {d.name:<22}{d.var_:>11.6f}{d.es_:>11.6f}"
              f"{c['cov_om']:>9.1f}{c['cov_es']:>9.1f}{c['bias_sd_mean']:>10.1f}")
    print("\n  calibration of each process:")
    for d in dgps:
        print(f"    {d.name:<22}{d.note}")

    al = rows[0]
    others = rows[1:]
    lo_b, hi_b, se_b = coverage_band(reps)
    print(f"\n  a calibrated 95% interval, estimated from {reps} replicates, should")
    print(f"  land in [{lo_b:.1f}, {hi_b:.1f}] (2 binomial standard errors, se = {se_b:.2f}).")
    print(f"  POWER of that band: it rejects any true coverage below {lo_b:.1f}% or")
    print(f"  above {hi_b:.1f}%, so it is a real test and not a formality.  The AL row")
    print(f"  is the only one it is applied to; the misspecified rows are at 0.0%,")
    print(f"  which no band question arises about.")
    check("A2  the correctly specified process attains nominal ES coverage",
          lo_b <= al["cov_es"] <= hi_b,
          f"AL ES coverage {al['cov_es']:.1f}%, band [{lo_b:.1f},{hi_b:.1f}]")
    check("A2  every misspecified process has ES coverage below 10%",
          all(r["cov_es"] < 10.0 for r in others),
          "max " + f"{max(r['cov_es'] for r in others):.1f}%")
    check("A2  the correctly specified process attains nominal VaR coverage",
          lo_b <= al["cov_om"] <= hi_b,
          f"AL VaR coverage {al['cov_om']:.1f}%, band [{lo_b:.1f},{hi_b:.1f}]")

    print(f"""
  WHAT THE VaR COLUMN SAYS.  Under correct specification VaR coverage is
  {al['cov_om']:.1f}%.  Under misspecification it is {min(r['cov_om'] for r in others):.1f}-{max(r['cov_om'] for r in others):.1f}% -- the posterior is centred on
  the right quantile but its WIDTH is not calibrated, which is the sandwich
  problem for the AL working likelihood and is already known.  Report it rather
  than let a referee find it.  It is also a different defect from the ES failure:
  the VaR interval is the wrong width, the ES interval is in the wrong place.""")
    RESULTS["A2"] = rows
    return rows


# A3  coverage against n

def A3(dgps, tau, ns, reps, grid_points, n_draws):
    rule("A3  more data makes the ES interval WORSE")
    print("""
    Misspecification bias does not shrink with n; the posterior standard deviation
    shrinks like n^-1/2.  So the miscentring measured in posterior standard
    deviations grows like sqrt(n), and coverage falls towards zero as the sample
    grows.  This is the cleanest available proof that the failure is not
    estimation error: estimation error is the thing that goes away with more data.
""")
    print(f"  {'DGP':<22}" + "".join(f"{f'n={n}':>22}" for n in ns))
    print(f"  {'':<22}" + "".join(f"{'cov%   bias(sd)':>22}" for _ in ns))
    print("  " + "-" * (22 + 22 * len(ns)))
    rows = []
    for d in dgps:
        cells, rec = [], {"dgp": d.name, "by_n": {}}
        for n in ns:
            c = coverage(d, tau, n, reps, SEED + 5, grid_points, n_draws)
            rec["by_n"][str(n)] = c
            cells.append(f"{c['cov_es']:6.1f}   {c['bias_sd_mean']:+8.1f}")
        rows.append(rec)
        print(f"  {d.name:<22}" + "".join(f"{c:>22}" for c in cells))

    #  the sqrt(n) prediction, checked on the misspecified rows
    print(f"\n  the sqrt(n) prediction: bias in posterior sd should scale as sqrt(n).")
    print(f"  {'DGP':<22}{'bias ratio':>14}{'sqrt(n) ratio':>16}{'agreement':>12}")
    ratio_pred = np.sqrt(ns[-1] / ns[0])
    #  The band is asymmetric because the observed quantity is a ratio of two
    #  noisy estimates and is skewed upward.  Both edges and the worst observed
    #  value are reported, so the margin is visible rather than implied.
    AGREE_LO, AGREE_HI = 0.75, 1.35
    ok, agrees = True, []
    for r in rows[1:]:
        b0 = abs(r["by_n"][str(ns[0])]["bias_sd_mean"])
        b1 = abs(r["by_n"][str(ns[-1])]["bias_sd_mean"])
        got = b1 / b0 if b0 > 0 else np.nan
        agree = got / ratio_pred
        agrees.append((r["dgp"], float(agree)))
        ok &= AGREE_LO < agree < AGREE_HI
        print(f"  {r['dgp']:<22}{got:>14.2f}{ratio_pred:>16.2f}{agree:>12.2f}")
    worst_dgp, worst_agree = min(agrees, key=lambda t: min(abs(t[1] - AGREE_LO),
                                                           abs(t[1] - AGREE_HI)))
    check("A3  the miscentring grows like sqrt(n), as fixed bias must", ok,
          f"observed/predicted in [{min(a for _, a in agrees):.2f}, "
          f"{max(a for _, a in agrees):.2f}] against a band of "
          f"[{AGREE_LO:.2f}, {AGREE_HI:.2f}]; closest to an edge is "
          f"{worst_dgp} at {worst_agree:.2f}")
    #  Monotone at every step, not just the endpoints.  The falsifiability check
    #  is required because a row already pinned at 0.0% satisfies ">=" anyways,
    #  so without it the monotonicity test could not fail.
    monotone = all(
        all(r["by_n"][str(a)]["cov_es"] >= r["by_n"][str(b)]["cov_es"] - 1e-9
            for a, b in zip(ns, ns[1:]))
        for r in rows[1:])
    falsifiable = any(r["by_n"][str(ns[0])]["cov_es"] > 0.5 and
                      r["by_n"][str(ns[-1])]["cov_es"] <
                      r["by_n"][str(ns[0])]["cov_es"] - 0.5
                      for r in rows[1:])
    check("A3  ES coverage is non-increasing in n at every step, all processes",
          monotone)
    check("A3  the monotonicity test is falsifiable (some process starts above 0)",
          falsifiable,
          "at least one misspecified process has non-zero coverage at the "
          f"smallest n ({ns[0]}) and strictly lower coverage at the largest")
    RESULTS["A3"] = {"ns": list(ns), "rows": rows}
    return rows


# A4  scale invariance

def A4(ref, tau, reps, grid_points, n_draws):
    rule("A4  scale equivariance, and prior sensitivity -- two separate claims")
    print("""
    Daily returns are order 1e-2 and the prior sigma ~ IG(2, 0.012) is tuned to
    that scale, so a reader may reasonably ask whether the failure is an artefact
    of a badly scaled prior.  That question has two halves and they need two
    tests, because a single test cannot answer both:

      (a) EQUIVARIANCE.  Rescale the data by c AND scale the prior with it.  Every
          posterior summary must scale by exactly c.  This is an exact algebraic
          property, so the tolerance is machine precision, not 1e-5.  (A loose
          tolerance here is worse than none: with the same seeds the agreement is
          1e-17, twelve orders of slack, so a gate at 1e-5 would wave through a
          prior mis-scaled by 50% and still print PASS.)

      (b) SENSITIVITY.  Hold the data fixed and vary the prior on purpose.  This
          is the claim the section actually wants -- that the prior is not doing
          the work -- and (a) cannot test it, because in (a) the prior moves in
          lockstep with the data by construction.
""")
    fit = K.al_mle(ref, tau)
    om, sg = fit["omega_hat"], fit["sigma_hat"]

    #  ---- (a) equivariance ------------------------------------------------
    print("  (a) equivariance: summaries after dividing out c")
    print(f"  {'scale c':>10}{'s0':>12}{'B_p0':>12}{'ES mean/c':>16}"
          f"{'ES lo/c':>16}{'ES hi/c':>16}{'coverage %':>12}")
    rows_a, base = [], None
    for c in (1.0, 100.0, 0.01):
        s0c, bpc = 0.024 * c, 1.0 * c ** 2
        cov, acc = 0, []
        for i in range(reps):
            rng = np.random.default_rng([SEED, 4, i])
            y = K.al_rvs(ref.size, om * c, sg * c, tau, rng)
            ex = K.exact_posterior(y, tau, n_draws, seed=int(2_000_003 + i),
                                   n0=4.0, s0=s0c, B_p0=bpc,
                                   grid_points=grid_points, verify_refinement=False)
            es = ex.omega - ex.sigma / (1 - tau)
            lo, hi = np.percentile(es, [2.5, 97.5])
            if lo <= K.al_es(om * c, sg * c, tau) <= hi:
                cov += 1
            acc.append((es.mean() / c, lo / c, hi / c))
        a = np.mean(acc, axis=0)
        rows_a.append({"c": c, "s0": s0c, "B_p0": bpc,
                       "es_mean_over_c": float(a[0]), "es_lo_over_c": float(a[1]),
                       "es_hi_over_c": float(a[2]), "coverage": 100.0 * cov / reps})
        if base is None:
            base = a
        print(f"  {c:>10g}{s0c:>12.5g}{bpc:>12.5g}{a[0]:>16.10f}{a[1]:>16.10f}"
              f"{a[2]:>16.10f}{100.0*cov/reps:>12.1f}")
    worst = max(float(np.max(np.abs(np.array(
        [r["es_mean_over_c"], r["es_lo_over_c"], r["es_hi_over_c"]]) - base)))
        for r in rows_a)
    check("A4a  scale equivariance holds to machine precision", worst < 1e-12,
          f"worst deviation after dividing out c: {worst:.2e}")

    #  ---- (b) prior sensitivity, data held fixed --------------------------
    print("\n  (b) prior sensitivity on the reference series itself, c = 1")
    print(f"  {'s0':>10}{'B_p0':>10}{'omega mean':>14}{'sigma mean':>14}"
          f"{'ES mean':>14}{'ES 2.5%':>14}{'ES 97.5%':>14}")
    rows_b, ref_row = [], None
    for s0v, bpv, lab in ((0.024, 1.0, "ours"), (0.024, 100.0, "K-K B_p0"),
                          (0.024, 0.01, "tight B_p0"), (0.012, 1.0, "s0 / 2"),
                          (0.048, 1.0, "s0 x 2"), (0.0024, 1.0, "s0 / 10"),
                          (0.24, 1.0, "s0 x 10")):
        ex = K.exact_posterior(ref, tau, 200_000, seed=SEED + 77,
                               n0=4.0, s0=s0v, B_p0=bpv,
                               grid_points=200_001, verify_refinement=False)
        es = ex.omega - ex.sigma / (1 - tau)
        lo, hi = np.percentile(es, [2.5, 97.5])
        r = {"label": lab, "s0": s0v, "B_p0": bpv,
             "omega": float(ex.omega.mean()), "sigma": float(ex.sigma.mean()),
             "es": float(es.mean()), "es_lo": float(lo), "es_hi": float(hi)}
        rows_b.append(r)
        if ref_row is None:
            ref_row = r
        print(f"  {s0v:>10.4g}{bpv:>10.4g}{r['omega']:>14.7f}{r['sigma']:>14.8f}"
              f"{r['es']:>14.7f}{lo:>14.7f}{hi:>14.7f}   {lab}")
    #  "Defensible" is the range a referee might propose, including K-K's own
    #  N(0,100).  s0 x 10 is deliberately absurd -- prior mean 0.12 against a true
    #  sigma of 0.00144 -- and SHOULD move the answer.  The claim is not that no
    #  prior matters, but that none comes close to explaining the miscentring.
    defensible = [r for r in rows_b if r["label"] in
                  ("ours", "K-K B_p0", "tight B_p0", "s0 / 2", "s0 x 2")]
    shift_def = max(abs(r["es"] - ref_row["es"]) for r in defensible)
    shift = max(abs(r["es"] - ref_row["es"]) for r in rows_b)
    #  the scale against which "does the prior matter" has to be measured
    ex_ref = K.exact_posterior(ref, tau, 200_000, seed=SEED + 78,
                               grid_points=200_001, verify_refinement=False)
    sd_ref = float((ex_ref.omega - ex_ref.sigma / (1 - tau)).std(ddof=1))
    emp = K.empirical_var_cvar(ref, tau)
    gap_sd = abs(ref_row["es"] - emp["CVaR"]) / sd_ref
    print(f"\n  posterior sd of ES                        {sd_ref:.3e}")
    print(f"  shift across DEFENSIBLE priors            {shift_def:.3e}"
          f"   = {shift_def/sd_ref:.3f} posterior sd")
    print(f"  shift across ALL priors incl. s0 x 10     {shift:.3e}"
          f"   = {shift/sd_ref:.3f} posterior sd")
    print(f"  the miscentring the prior would have to explain"
          f"           = {gap_sd:.1f} posterior sd")
    print(f"  so the most extreme prior tested accounts for "
          f"{100*shift/(gap_sd*sd_ref):.2f}% of it.")
    print("""
    The s0 x 10 row is not a failure, it is the control working.  A prior with
    mean 0.12 against a true sigma of 0.00144 is wrong by a factor of 83 and
    SHOULD move the answer; that it moves it by 0.14 posterior sd is the measure
    of how little room the prior has.  Across the priors anyone would actually
    defend -- including Kozumi and Kobayashi's own N(0, 100) -- the shift is
    smaller still.  Neither is remotely enough to explain the finding.""")
    check("A4b  defensible priors move the ES by less than 0.05 posterior sd",
          shift_def / sd_ref < 0.05,
          f"{shift_def/sd_ref:.3f} posterior sd (K-K's own prior included)")
    check("A4b  even an absurd prior explains under 2% of the miscentring",
          shift / (gap_sd * sd_ref) < 0.02,
          f"{100*shift/(gap_sd*sd_ref):.2f}% of a {gap_sd:.1f} posterior sd gap")

    RESULTS["A4"] = {"equivariance": rows_a, "prior_sensitivity": rows_b,
                     "prior_max_shift": float(shift),
                     "prior_max_shift_defensible": float(shift_def),
                     "prior_max_shift_post_sd": float(shift / sd_ref),
                     "es_posterior_sd": float(sd_ref),
                     "miscentring_post_sd": float(gap_sd)}


# A5  where the miscentring comes from

def A5(dgps, tau, n, reps, grid_points, n_draws):
    rule("A5  the miscentring is the model's error in the mean")
    print("""
    Section 5f of spy_run1.py shows on one dataset that substituting the SAMPLE
    mean into ES = mu - sigma/tau recovers the empirical CVaR exactly.  Here that
    is checked across replicates and processes, so it is a property of the
    estimator and not of one sample.

    Columns:
      ES(AL)       posterior mean of omega - sigma/(1-tau)
      ES(mean id)  ybar - sigma_hat/tau_eff, the same formula with the SAMPLE mean
      truth        the DGP's exact ES
""")
    print(f"  {'DGP':<22}{'ES(AL)':>12}{'ES(mean id)':>13}{'truth':>12}"
          f"{'|id - emp CVaR|':>17}")
    rows = []
    worst_id = 0.0
    for d in dgps:
        acc_al, acc_id, acc_err = [], [], []
        for i in range(reps):
            rng = np.random.default_rng([SEED, 9, _tag(d.name), i])
            y = d.rvs(n, rng)
            s = posterior_summary(y, tau, seed=int(3_000_003 + i),
                                  grid_points=grid_points, n_draws=n_draws)
            f = K.al_mle(y, tau)
            rec = f["ybar"] - f["sigma_hat"] / f["tau_eff"]
            acc_al.append(s["es_mean"])
            acc_id.append(rec)
            acc_err.append(abs(rec - f["cvar_emp"]))
        worst_id = max(worst_id, max(acc_err))
        rows.append({"dgp": d.name, "es_al": float(np.mean(acc_al)),
                     "es_identity": float(np.mean(acc_id)), "truth": d.es_,
                     "max_identity_err": float(max(acc_err))})
        print(f"  {d.name:<22}{np.mean(acc_al):>12.6f}{np.mean(acc_id):>13.6f}"
              f"{d.es_:>12.6f}{max(acc_err):>17.2e}")
    check("A5  the mean identity recovers the empirical CVaR exactly, every replicate",
          worst_id < 1e-12, f"worst error {worst_id:.2e}")
    print("""
    On the misspecified rows the middle column tracks the truth and the left
    column does not.  The two formulas differ in one place: which mean is substituted.
    So the ES error is the model's error in the mean, and nothing else.""")
    RESULTS["A5"] = rows


# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", nargs="?", default=None,
                    help="reference returns CSV; the DGPs are calibrated to it")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--json", default="experiment_A_results.json")
    args = ap.parse_args()

    K.set_numeric_guards()
    t0 = time.time()
    reps = args.reps if args.reps else (60 if args.quick else 1000)
    grid_points = 40_001
    n_draws = 40_000
    ns = (250, 1000, 2512) if not args.quick else (250, 1000)

    rule("EXPERIMENT A -- the positive control")
    env = K.print_env_stamp("  ")
    print(f"  seed {SEED} | tau {TAU} | {reps} replicates per cell")
    print(f"  coverage computed from the EXACT posterior "
          f"(grid {grid_points:,}, {n_draws:,} i.i.d. draws)")

    #  reference series
    if args.reference and os.path.exists(args.reference):
        ref = np.loadtxt(args.reference, delimiter=",", skiprows=1, usecols=-1)
        ref = ref[np.isfinite(ref)]
        src = f"file: {args.reference}"
    elif os.path.exists("spy_returns_2015_2025.csv"):
        ref = np.loadtxt("spy_returns_2015_2025.csv", delimiter=",",
                         skiprows=1, usecols=-1)
        src = "cache: spy_returns_2015_2025.csv"
    else:
        rng = np.random.default_rng(SEED)
        ref = 0.0005 + 0.008 * rng.standard_t(4, N_REF) / np.sqrt(4 / 2)
        src = (f"SYNTHETIC t(4) reference, n = {N_REF} -- no returns file found. "
               "Pass one to calibrate the DGPs to real data.")
    n = int(ref.size)
    print(f"  reference: {src}  (n = {n})")

    dgps = build_dgps(ref, TAU)

    A1(dgps, TAU, n, args.quick)
    A2(dgps, TAU, n, reps, grid_points, n_draws)
    A3(dgps, TAU, ns, max(reps // 2, 40), grid_points, n_draws)
    A4(ref, TAU, max(reps // 5, 30), grid_points, n_draws)
    A5(dgps, TAU, n, max(reps // 10, 20), grid_points, n_draws)

    rule("PASS CRITERIA")
    if FAILURES:
        for f in FAILURES:
            print(f"  FAIL  {f}")
    else:
        print("  all criteria passed")
    print(f"\n  elapsed {time.time()-t0:.1f}s")

    RESULTS["_meta"] = {"env": env, "seed": SEED, "tau": TAU, "reps": reps,
                        "reference": src, "n_reference": n,
                        "grid_points": grid_points, "n_draws": n_draws,
                        "failures": FAILURES,
                        "elapsed_s": time.time() - t0}
    with open(args.json, "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print(f"  wrote {args.json}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())