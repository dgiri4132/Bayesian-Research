#!/usr/bin/env python3
"""This file is an extension experient side where I basically try to prove 
that there is nothing wrong with the code rather misspecifcation in the model chosen.
it is gonna be complemented by other files that i have set out to upload in the next few days.
I will try to push it in a single time for all the changes unless I have to back and fix something.

EXPERIMENT A i.e. the positive control.

Bayesian CVaR project.  Darshan Giri 

THE QUESTION A ANSWERS.  Spy_run.py shows the ES posterior missing the empirical
CVaR by many posterior standard deviations.  On its own that is compatible with
several explanations, only one of which I believe is interesting/important for my case:

    (i)   the sampler is wrong
    (ii)  the prior is doing it
    (iii) n is too small / it is estimation error
    (iv)  the algebra of Proof 1 is wrong
    (v)   the AL working likelihood is misspecified for this data

A eliminates (i)-(iv) by holding the entire pipeline fixed that means same sampler, same
n, same prior, same code path and then varying ONLY the data-generating process.  If
coverage is nominal when the data really is asymmetric-Laplace and zero otherwise,
then nothing about the machinery can be the cause, and (v) is what is left.

DESIGN NOTE, and a departure from the original plan.

    The coverage study runs on the EXACT posterior, not on the Gibbs sampler.
    For the intercept-only model sigma integrates out in closed form (see
    kkgibs.exact_posterior), leaving a one-dimensional posterior that can be
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
    is not calibrated.  Reporting both separates the two defects: the VaR interval
    is the wrong width, the ES interval is in the wrong place.

usage:  python3 experiment_a.py [returns.csv] [--quick] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib

import numpy as np
from scipy import stats

import kkgibs as K

TAU = 0.05
SEED = 20260901
RHAT_TARGET = 1.01

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


def tag(name):
    """This is the deterministic per-DGP seed component. It uses crc32 rather than hash(),
    which would  make  the  script  irreproducible  between runs."""
    return zlib.crc32(name.encode()) % 10_000


def coverage_band(reps, nominal=95.0):
    """
    The width is the binomial standard error of the coverage estimate itself, not
    a fixed tolerance: at 60 replicates a calibrated interval lands on 100% about
    one run in 22, while at 3000 a fixed +/- 5 points would pass a broken one.

    Here,the band is 2 se, not 3 because I found out that 3-se band at 1000 replicates is 
    [92.9, 97.1] and would pass a true coverage of 93%.At 2 se it is [93.6, 96.4]
     and 93% is rejected.

    It returns (lo, hi) `lo` is the lower edge of the band, any true coverage
    below it gets rejected.

    I measured the calibration of the exact posterior  and 3000 replicates gave 
    94.80% +/- 0.41 (z = -0.49), and an independent audit at 16 000 replicates 
    gave 95.01% +/- 0.17 (z = +0.04).
    """
    p = nominal / 100.0
    se = 100.0 * np.sqrt(p * (1 - p) / reps)
    return nominal - 2 * se, nominal + 2 * se


# the data-generating processes
def discrete_var_es(y, tau):
    """
    This is the exact VaR and ES of the EMPIRICAL distribution of y.

        VaR = y_(ceil(tau n))
        ES  = (1/tau) [ (1/n) sum_{y < VaR} y  +  VaR (tau - P(Y < VaR)) ]

    The second term is the partial atom at VaR, dropping it is the usual way this
    is gotten slightly wrong.  Checked it against an 8-million-draw resample and it agrees
    to Monte Carlo error.
    """
    s = np.sort(np.asarray(y, dtype=float))
    n = s.size
    j = min(max(int(np.ceil(tau * n)), 1), n)
    var_ = s[j - 1]
    m = int(np.searchsorted(s, var_, side="left"))
    es = (s[:m].sum() / n + var_ * (tau - m / n)) / tau
    return float(var_), float(es)


class DGP:
    """A data-generating process with an EXACTLY known VaR and ES at tau."""

    def __init__(self, name, sampler, var_, es_):
        self.name, self.rvs, self.var_, self.es_ = name, sampler, var_, es_


def build_dgps(ref, tau):
    """
    Here are five dgps, all calibrated to the SAME reference return series, so the
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
                   mu + sc * q, mu + sc * es_std)

    bs_var, bs_es = discrete_var_es(ref, tau)
    return [
        DGP("AL (correct)", lambda n, rng: K.al_rvs(n, om, sg, tau, rng),
            om, K.al_es(om, sg, tau)),
        DGP("Normal", lambda n, rng: rng.normal(mu, sd, n),
            mu + sd * z, mu - sd * stats.norm.pdf(z) / tau),
        t_dgp(5),
        t_dgp(3),
        DGP("bootstrap (reference)", lambda n, rng: rng.choice(ref, size=n, replace=True),
            bs_var, bs_es),
    ]


# one replicate

def posterior_summary(y, tau, seed, grid_points, n_draws):
    ex = K.exact_posterior(y, tau, n_draws, seed=seed, grid_points=grid_points)
    es = ex.omega - ex.sigma / (1 - tau)
    lo, hi = np.percentile(es, [2.5, 97.5])
    olo, ohi = np.percentile(ex.omega, [2.5, 97.5])
    return {"es_mean": float(es.mean()), "es_sd": float(es.std(ddof=1)),
            "es_lo": float(lo), "es_hi": float(hi),
            "om_lo": float(olo), "om_hi": float(ohi),
            "om_mean": float(ex.omega.mean()), "om_sd": float(ex.omega.std(ddof=1))}


# A1  the exact posterior is the Gibbs posterior

def A1(dgps, tau, n, quick):
    rule("A1  the exact posterior agrees with the Gibbs sampler")
    #  quick mode uses the shortest chains that still converge here: at 3000/1000
    #  R-hat reaches 1.013 and this section fails honestly.
    n_iter, burn = (8000, 3000) if quick else (15000, 5000)
    rows, worst, worst_ratio = [], 0.0, 0.0
    print(f"  {'design':<22}{'quantity':<10}{'Gibbs':>13}{'exact':>13}{'diff':>11}{'R-hat':>9}")
    for d in dgps:
        rng = np.random.default_rng([SEED, 1, tag(d.name)])
        y = d.rvs(n, rng)
        X = np.ones((n, 1))
        th, sg = [], []
        for c in range(4):
            r = K.gibbs_qr_sampler(X, y, tau, n_iter, burn, seed=1000 + 111 * c,
                                   omega_init=[float(np.quantile(y, tau))], sigma_init=0.01)
            th.append(r.omega[:, 0])
            sg.append(r.sigma)
        rhat = max(K.split_rhat(th), K.split_rhat(sg))
        g_es = np.concatenate(th) - np.concatenate(sg) / (1 - tau)
        es_sd = float(g_es.std(ddof=1))
        e = posterior_summary(y, tau, seed=7, grid_points=400_001, n_draws=400_000)
        es_chains = [a - b / (1 - tau) for a, b in zip(th, sg)]
        tol = max(0.05, 4.0 * K.mcse(es_chains, sum(K.ess(c) for c in es_chains)) / es_sd)
        pairs = [("ES mean", float(g_es.mean()), e["es_mean"]),
                 ("ES 2.5%", float(np.percentile(g_es, 2.5)), e["es_lo"]),
                 ("ES 97.5%", float(np.percentile(g_es, 97.5)), e["es_hi"])]
        for i, (name, a, b) in enumerate(pairs):
            worst = max(worst, abs(a - b))
            worst_ratio = max(worst_ratio, abs(a - b) / es_sd / tol)
            print(f"  {d.name if i == 0 else '':<22}{name:<10}{a:>13.7f}{b:>13.7f}{a-b:>11.2e}"
                  + (f"{rhat:>9.5f}" if i == 0 else ""))
        rows.append({"dgp": d.name, "rhat": float(rhat), "es_sd": es_sd,
                     "gibbs_es_mean": float(g_es.mean()), "exact_es_mean": e["es_mean"]})
    #  The gate is in posterior-sd units.  I found out that an absolute 1e-4 threshold would not
    #  bind that is  a 5%  bias  injected  into  the  Gibbs  sigma  block  moves  the  worst  ES
    #  summary by 7.4e-5, inside 1e-4 but 0.6 posterior sd, which is the defect
    #  this section exists to exclude.
    print(f"\n  largest difference: {worst:.2e}")
    check("A1  every Gibbs run converged", all(r["rhat"] < RHAT_TARGET for r in rows),
          f"largest R-hat {max(r['rhat'] for r in rows):.5f}")
    check("A1  Gibbs and exact agree within the Gibbs Monte Carlo error", worst_ratio < 1.0,
          f"largest difference is {worst_ratio:.2f} of the tolerance")
    RESULTS["A1"] = {"rows": rows, "worst_diff": float(worst),
                     "worst_tolerance_ratio": float(worst_ratio)}


# A2  the coverage table

def coverage(d, tau, n, reps, seed, grid_points, n_draws):
    cov_es = cov_om = 0
    bias_sd, om_means, om_sds = [], [], []
    for i in range(reps):
        rng = np.random.default_rng([seed, tag(d.name), i])
        y = d.rvs(n, rng)
        s = posterior_summary(y, tau, seed=int(1_000_003 + i),
                              grid_points=grid_points, n_draws=n_draws)
        if s["es_lo"] <= d.es_ <= s["es_hi"]:
            cov_es += 1
        if s["om_lo"] <= d.var_ <= s["om_hi"]:
            cov_om += 1
        bias_sd.append((s["es_mean"] - d.es_) / s["es_sd"])
        om_means.append(s["om_mean"])
        om_sds.append(s["om_sd"])
    om_means, om_sds = np.array(om_means), np.array(om_sds)
    # for the VaR intervals: are they off-centre, or too narrow? the second number is
    # the posterior sd against how much the posterior mean really moves between replicates
    return {"cov_es": 100.0 * cov_es / reps, "cov_om": 100.0 * cov_om / reps,
            "bias_sd_mean": float(np.mean(bias_sd)),
            "var_bias_sd_mean": float(np.mean((om_means - d.var_) / om_sds)),
            "var_sd_ratio": float(om_sds.mean() / om_means.std(ddof=1)),
            "reps": reps}


def A2(dgps, tau, n, reps, grid_points, n_draws):
    rule("A2  coverage of the 95% credible intervals")
    # replicate i uses the same posterior seed in every design, so the rows are
    # correlated. that helps the comparison between rows, which is what this is for
    print(f"  {reps} replicates per design, n = {n}, tau = {tau}\n")
    print(f"  {'design':<22}{'true VaR':>11}{'true ES':>11}{'VaR cov %':>11}"
          f"{'ES cov %':>10}{'ES miscentring (sd)':>21}{'VaR miscentring (sd)':>22}"
          f"{'VaR sd ratio':>14}")
    rows = []
    for d in dgps:
        c = coverage(d, tau, n, reps, SEED, grid_points, n_draws)
        rows.append({"dgp": d.name, "true_var": d.var_, "true_es": d.es_, **c})
        print(f"  {d.name:<22}{d.var_:>11.6f}{d.es_:>11.6f}{c['cov_om']:>11.1f}"
              f"{c['cov_es']:>10.1f}{c['bias_sd_mean']:>21.1f}"
              f"{c['var_bias_sd_mean']:>22.2f}{c['var_sd_ratio']:>14.3f}")
    al, others = rows[0], rows[1:]
    lo, hi = coverage_band(reps)
    print(f"\n  a calibrated interval should land in [{lo:.1f}, {hi:.1f}]")
    check("A2  AL design: ES coverage is nominal", lo <= al["cov_es"] <= hi,
          f"{al['cov_es']:.1f}%")
    check("A2  AL design: VaR coverage is nominal", lo <= al["cov_om"] <= hi,
          f"{al['cov_om']:.1f}%")
    check("A2  other designs: ES coverage below 10%", all(r["cov_es"] < 10.0 for r in others),
          f"highest {max(r['cov_es'] for r in others):.1f}%")
    check("A2  other designs: VaR coverage below nominal", all(r["cov_om"] < lo for r in others),
          f"{min(r['cov_om'] for r in others):.1f} to {max(r['cov_om'] for r in others):.1f}%")
    check("A2  other designs: the VaR intervals are too narrow, not off-centre",
          all(abs(r["var_bias_sd_mean"]) < 1.0 and r["var_sd_ratio"] < 0.75 for r in others),
          f"miscentring at most {max(abs(r['var_bias_sd_mean']) for r in others):.2f} sd, "
          f"sd ratio {min(r['var_sd_ratio'] for r in others):.3f} to "
          f"{max(r['var_sd_ratio'] for r in others):.3f}")
    RESULTS["A2"] = rows


# A3  coverage against n

def A3(dgps, tau, ns, reps, grid_points, n_draws):
    rule("A3  ES coverage as n grows")
    # Here the bias does not shrink with n but the posterior sd shrinks like n^-1/2, so the
    # miscentring in posterior sd should grow like sqrt(n) and the coverage should fall I think
    print(f"  {reps} replicates per cell\n")
    print(f"  {'design':<22}" + "".join(f"{f'n={n}':>20}" for n in ns))
    print(f"  {'':<22}" + "".join(f"{'cov %  bias (sd)':>20}" for _ in ns))
    rows = []
    for d in dgps:
        rec, cells = {"dgp": d.name, "by_n": {}}, []
        for n in ns:
            c = coverage(d, tau, n, reps, SEED + 5, grid_points, n_draws)
            rec["by_n"][str(n)] = c
            cells.append(f"{c['cov_es']:6.1f}  {c['bias_sd_mean']:+8.1f}")
        rows.append(rec)
        print(f"  {d.name:<22}" + "".join(f"{c:>20}" for c in cells))

    # This secotion showshow much the miscentring grew from the smallest n to the largest, against sqrt(n)
    pred = float(np.sqrt(ns[-1] / ns[0]))
    growth = {}
    print(f"\n  growth of the miscentring from n={ns[0]} to n={ns[-1]} (sqrt(n) gives {pred:.2f})")
    for r in rows[1:]:
        g = abs(r["by_n"][str(ns[-1])]["bias_sd_mean"]) / abs(r["by_n"][str(ns[0])]["bias_sd_mean"])
        growth[r["dgp"]] = float(g)
        print(f"  {r['dgp']:<22}{g:>8.2f}")
    never_rises = all(r["by_n"][str(a)]["cov_es"] >= r["by_n"][str(b)]["cov_es"]
                      for r in rows[1:] for a, b in zip(ns, ns[1:]))
    check("A3  other designs: ES coverage never rises with n", never_rises)
    RESULTS["A3"] = {"ns": list(ns), "rows": rows, "growth": growth, "sqrt_n_ratio": pred}


# A4  prior sensitivity

def A4(ref, tau):
    rule("A4  prior sensitivity on the SPY returns")
    priors = ((0.024, 1.0, "ours"), (0.024, 100.0, "K-K B_p0"), (0.024, 0.01, "tight B_p0"),
              (0.012, 1.0, "s0 / 2"), (0.048, 1.0, "s0 x 2"), (0.0024, 1.0, "s0 / 10"),
              (0.24, 1.0, "s0 x 10"))
    print(f"  {'s0':>8}{'B_p0':>8}{'omega mean':>14}{'sigma mean':>14}"
          f"{'ES mean':>14}{'ES 2.5%':>14}{'ES 97.5%':>14}")
    rows = []
    for s0, bp, label in priors:
        ex = K.exact_posterior(ref, tau, 200_000, seed=SEED + 77, n0=4.0, s0=s0, B_p0=bp,
                               grid_points=200_001)
        es = ex.omega - ex.sigma / (1 - tau)
        lo, hi = np.percentile(es, [2.5, 97.5])
        rows.append({"label": label, "s0": s0, "B_p0": bp, "omega": float(ex.omega.mean()),
                     "sigma": float(ex.sigma.mean()), "es": float(es.mean()),
                     "es_lo": float(lo), "es_hi": float(hi)})
        print(f"  {s0:>8.4g}{bp:>8.4g}{ex.omega.mean():>14.7f}{ex.sigma.mean():>14.8f}"
              f"{es.mean():>14.7f}{lo:>14.7f}{hi:>14.7f}   {label}")
    #  "K-K B_p0" is K-K's own
    #  N(0,100).  s0 x 10 is deliberately absurd with prior mean 0.12 against a true
    #  sigma of 0.00144 ,and SHOULD move the answer.  The claim here is not that no
    #  prior matters, but that none comes close to explaining the miscentring.
    shift = max(abs(r["es"] - rows[0]["es"]) for r in rows)

    #  the scale against which "does the prior matter" has to be measured
    ex = K.exact_posterior(ref, tau, 200_000, seed=SEED + 78, grid_points=200_001)
    sd = float((ex.omega - ex.sigma / (1 - tau)).std(ddof=1))
    gap = abs(rows[0]["es"] - K.empirical_var_cvar(ref, tau)["CVaR"]) / sd
    print(f"\n  largest shift in the ES mean across the priors: {shift/sd:.3f} posterior sd")
    print(f"  distance of the ES mean from the historical CVaR: {gap:.1f} posterior sd")
    check("A4  the largest prior shift is under 2% of the miscentring", shift / (gap * sd) < 0.02,
          f"{100 * shift / (gap * sd):.2f}%")
    RESULTS["A4"] = {"prior_sensitivity": rows, "prior_max_shift": float(shift),
                     "prior_max_shift_post_sd": float(shift / sd),
                     "es_posterior_sd": sd, "miscentring_post_sd": float(gap)}



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("returns", nargs="?", default="spy_returns_2015_2025.csv",
                    help="returns CSV, the designs take their parameters from it")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="experiment_A_results.json")
    args = ap.parse_args()

    K.set_numeric_guards()
    t0 = time.time()
    reps = 60 if args.quick else 1000
    grid_points, n_draws = 40_001, 40_000
    ns = (250, 1000) if args.quick else (250, 1000, 2512)

    rule("EXPERIMENT A, the coverage study")
    env = K.print_env_stamp("  ")
    #  reference series
    ref = np.loadtxt(args.returns, delimiter=",", skiprows=1, usecols=-1)
    ref = ref[np.isfinite(ref)]
    n = int(ref.size)
    print(f"  returns {args.returns} (n = {n}) | seed {SEED} | tau {TAU} | {reps} replicates")

    dgps = build_dgps(ref, TAU)
    A1(dgps, TAU, n, args.quick)
    A2(dgps, TAU, n, reps, grid_points, n_draws)
    A3(dgps, TAU, ns, max(reps // 2, 40), grid_points, n_draws)
    A4(ref, TAU)

    rule("RESULT")
    print("  all checks passed" if not FAILURES else "\n".join(f"  FAIL  {f}" for f in FAILURES))
    print(f"\n  elapsed {time.time()-t0:.1f}s")
    RESULTS["_meta"] = {"env": env, "seed": SEED, "tau": TAU, "reps": reps,
                        "returns": args.returns, "n": n, "grid_points": grid_points,
                        "n_draws": n_draws, "failures": FAILURES,
                        "elapsed_s": time.time() - t0}
    with open(args.json, "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print(f"  wrote {args.json}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
