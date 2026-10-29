#!/usr/bin/env python3
"""
EXPERIMENT C the regression case.

Bayesian CVaR project.  Darshan Giri 

I made this file because Spy_run.py, experiment_a.pyand experiment_b_overall.py
all fit an intercept-only model, i.e. one VaR for every day, but the paper is
about quantile REGRESSION. So someone could basically say I showed the failure
in the wrong model. This file fits an actual regression to see if the failure
is still there.

The main idea is that, under the AL, the overshoot past the quantile is

    VaR_t - ES_t = sigma / (1 - tau)

and sigma is just one number, it has no t in it. So even if VaR_t moves around
with volatility, the model still says the average breach is the same size on
every day, a calm day and March 2020 get the same number. This comes straight
from how sigma is drawn in kkgibs.gibbs_qr_sampler.

What I found:
    - C2: the overshoot the AL claims is way too small in every volatility
      quartile. Every quartile is outside the AL's own 95% interval, by 4.0x
      to 9.0x.

The model:
    VaR_t = b0 + b1 |r_{t-1}| + b2 sd(r_{t-20..t-1})
    Both covariates only use data up to t-1 so there's no look-ahead. I
    standardise them so the N(0,1) prior is wide compared to the coefficients,
    which means the coefficients are per sd of each covariate.

Note: this uses 2492 rows since the 20 day window eats the first 20, not the
2512 in Spy_run.py, so  please don't put a number from one next to a number from
the other.

usage:  python3 experiment_c.py [returns.csv] [--quick] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import kkgibs as K

TAU = 0.05
WINDOW = 20
N_ITER = 15_000
BURN_IN = 5_000
N_CHAINS = 4
BASE_SEED = 1000
N0, S0 = 4.0, 0.024
RHAT_TARGET = 1.01
N_BOOT = 4000
SEED = 20260901

RESULTS: dict = {}
FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def boot_mean_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float("nan"), float("nan")
    bs = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    return tuple(float(v) for v in np.percentile(bs, [2.5, 97.5]))


#C0  the design matrix 

def build_design(path):
    rule("C0  the conditional specification")
    raw = np.genfromtxt(path, delimiter=",", skip_header=1,
                        dtype=[("date", "U10"), ("r", "f8")])
    dates, r = raw["date"], raw["r"]
    r = np.asarray(r, dtype=float)

    # both covariates only use returns from before day t
    abs_lag = np.abs(r[:-1])[WINDOW - 1:]
    vol = np.array([r[i - WINDOW:i].std(ddof=1) for i in range(WINDOW, r.size)])
    y = r[WINDOW:]
    d = dates[WINDOW:]
    assert abs_lag.size == vol.size == y.size == d.size

    m1, s1 = abs_lag.mean(), abs_lag.std(ddof=1)
    m2, s2 = vol.mean(), vol.std(ddof=1)
    X = np.column_stack([np.ones(y.size), (abs_lag - m1) / s1, (vol - m2) / s2])

    print(f"  observations      {y.size} (from {r.size}, window {WINDOW})")
    print(f"  period            {d[0]} .. {d[-1]}")

    RESULTS["C0"] = {"n": int(y.size), "window": WINDOW,
                     "first": str(d[0]), "last": str(d[-1])}
    return y, X, vol, d


# C1  the fit 

def C1(y, X, quick):
    rule("C1  the conditional fit")
    n_iter = 6000 if quick else N_ITER
    burn = 2000 if quick else BURN_IN
    q0 = float(np.quantile(y, TAU))
    sigma_inits = [0.01, 0.05, 0.002, 0.10]

    chains, sgs = [], []
    for c in range(N_CHAINS):
        res = K.gibbs_qr_sampler(
            X, y, TAU, n_iter, burn, seed=BASE_SEED + 111 * c,
            omega_init=[q0, 0.0, 0.0], sigma_init=sigma_inits[c], n0=N0, s0=S0)
        chains.append(res.omega)
        sgs.append(res.sigma)

    names = ["b0 (intercept)", "b1 |r_(t-1)|", f"b2 vol{WINDOW}"]
    rhats = [K.split_rhat([c[:, j] for c in chains]) for j in range(X.shape[1])]
    rhat_s = K.split_rhat(sgs)
    om = np.vstack(chains)
    sa = np.concatenate(sgs)

    print(f"  {N_CHAINS} chains x {n_iter} ({burn} burn-in), over-dispersed starts\n")
    print(f"  {'parameter':<18}{'post mean':>12}{'2.5%':>12}{'97.5%':>12}"
          f"{'R-hat':>9}{'ESS':>9}")
    coefs = []
    for j, nm in enumerate(names):
        lo, hi = np.percentile(om[:, j], [2.5, 97.5])
        e = sum(K.ess(c[:, j]) for c in chains)
        coefs.append({"name": nm, "mean": float(om[:, j].mean()),
                      "lo": float(lo), "hi": float(hi),
                      "rhat": float(rhats[j]), "ess": float(e)})
        print(f"  {nm:<18}{om[:,j].mean():>12.6f}{lo:>12.6f}{hi:>12.6f}"
              f"{rhats[j]:>9.5f}{e:>9.0f}")
    lo, hi = np.percentile(sa, [2.5, 97.5])
    print(f"  {'sigma':<18}{sa.mean():>12.7f}{lo:>12.7f}{hi:>12.7f}"
          f"{rhat_s:>9.5f}{sum(K.ess(c) for c in sgs):>9.0f}")

    converged = max(max(rhats), rhat_s) < RHAT_TARGET
    check("C1  every parameter converged", converged,
          f"worst R-hat {max(max(rhats), rhat_s):.5f}")
    if not converged:
        return None

    var_t = X @ om.mean(axis=0)
    hit = float((y < var_t).mean())
    check("C1  the fitted plane cuts tau of the data (within 1 point)",
          abs(hit - TAU) < 0.01, f"{100*hit:.2f}% against {100*TAU:.2f}%")

    print(f"\n  fitted VaR_t ranges {var_t.min():.5f} .. {var_t.max():.5f}"
          f"  -- a factor of {var_t.min()/var_t.max():.1f}")

    RESULTS["C1"] = {"coefficients": coefs, "sigma_mean": float(sa.mean()),
                     "sigma_lo": float(lo), "sigma_hi": float(hi),
                     "rhat_sigma": float(rhat_s), "hit_rate": hit,
                     "var_min": float(var_t.min()), "var_max": float(var_t.max()),
                     "n_iter": n_iter, "burn_in": burn}
    return om, sa, var_t


# C2  the main result 

def C2(y, var_t, om, sa, vol):
    rule("C2  one overshoot for every day, against what actually happened")
    rng = np.random.default_rng(SEED)
    br = y < var_t
    al = float(sa.mean() / (1 - TAU))
    al_lo, al_hi = (float(v) for v in np.percentile(sa / (1 - TAU), [2.5, 97.5]))

    print(f"  AL overshoot {al:.6f}, 95% CI [{al_lo:.6f}, {al_hi:.6f}], on every day")
    print(f"  breaches: {int(br.sum())} of {y.size} = {100*br.mean():.2f}%\n")
    all_rows, worst_ratio, all_exclude = [], np.inf, True
    for nb in (4,):
        qs = np.quantile(vol, np.linspace(0, 1, nb + 1))
        print(f"  {nb} volatility buckets")
        print(f"    {'bucket':<8}{'breaches':>9}{'mean VaR_t':>12}"
              f"{'realised overshoot':>20}{'95% CI':>26}{'x AL':>7}{'excl':>6}")
        for i in range(nb):
            m = (vol >= qs[i]) & (vol <= qs[i + 1] if i == nb - 1 else vol < qs[i + 1])
            b = m & br
            o = var_t[b] - y[b]
            lo_b, hi_b = boot_mean_ci(o, rng)
            ratio = float(o.mean() / al)
            excl = bool(lo_b > al_hi)
            all_exclude &= excl
            worst_ratio = min(worst_ratio, ratio)
            all_rows.append({"n_buckets": nb, "bucket": i + 1,
                             "breaches": int(b.sum()),
                             "mean_var": float(var_t[m].mean()),
                             "realised": float(o.mean()),
                             "ci": [lo_b, hi_b], "ratio": ratio,
                             "excludes_AL": excl})
            print(f"    B{i+1:<7}{int(b.sum()):>9}{var_t[m].mean():>12.5f}"
                  f"{o.mean():>20.6f}   [{lo_b:.6f}, {hi_b:.6f}]{ratio:>7.2f}"
                  f"{'yes' if excl else 'NO':>6}")
        print()

    check("C2  the realised overshoot exceeds the AL's credible interval in "
          "every volatility quartile", all_exclude)
    check("C2  the AL understates the overshoot by at least 3x everywhere",
          worst_ratio > 3.0, f"smallest ratio {worst_ratio:.2f}x")

    RESULTS["C2"] = {"al_overshoot": al, "al_ci": [al_lo, al_hi],
                     "breaches": int(br.sum()), "rows": all_rows,
                     "worst_ratio": float(worst_ratio)}
    return br


# C4  does adding covariates fix anything? The real question

def C4(y, var_t, sa, om):
    rule("C4  does adding covariates fix the shortfall?")
    al = float(sa.mean() / (1 - TAU))

    # refit the intercept-only model on these same rows so it's a fair
    # comparison (not the 2512 day Spy_run fit)
    ex = K.exact_posterior(y, TAU, n_draws=40_000, seed=SEED)
    io_om = float(ex.omega.mean())
    io_sa = float(ex.sigma.mean())
    io_al = io_sa / (1 - TAU)

    print(f"""
    {'':<26}{'intercept-only':>18}{'with covariates':>18}
    {'VaR':<26}{f'{io_om:.6f} (fixed)':>18}{f'{var_t.min():.5f}..{var_t.max():.5f}':>18}
    {'sigma':<26}{f'{io_sa:.7f}':>18}{f'{sa.mean():.7f}':>18}
    {'asserted overshoot':<26}{f'{io_al:.6f}':>18}{f'{al:.6f}':>18}
    {'varies with t?':<26}{'no':>18}{'no':>18}""")
    RESULTS["C4"] = {"intercept_only_sigma": io_sa,
                     "intercept_only_omega": io_om,
                     "intercept_only_overshoot": io_al,
                     "regression_sigma": float(sa.mean()),
                     "regression_overshoot": al,
                     "var_min": float(var_t.min()), "var_max": float(var_t.max())}
    return io_om, io_sa, io_al, ex


# C5  the two models compared and every direction checked 

def C5(y, var_t, sa, io_om, io_sa, io_al, ex):
    rule("C5  intercept-only vs conditional, every direction checked")

    al = float(sa.mean() / (1 - TAU))

    # each model measured against its own breaches, on the same rows
    br_c = y < var_t
    over_c = float(np.mean(var_t[br_c] - y[br_c]))
    br_i = y < io_om
    over_i = float(np.mean(io_om - y[br_i]))

    # sigma-hat is the minimised check loss, so evaluate each at its own fit
    cl_i = float(np.mean(K.rho_tau(y - io_om, TAU)))
    cl_c = float(np.mean(K.rho_tau(y - var_t, TAU)))

    rows = [("minimised check loss (= sigma-hat)", cl_i, cl_c),
            ("posterior mean sigma",               io_sa, float(sa.mean())),
            ("asserted overshoot",                 io_al, al),
            ("realised overshoot, own breaches",   over_i, over_c),
            ("ratio realised / asserted",          over_i / io_al, over_c / al),
            ("breaches",                           int(br_i.sum()), int(br_c.sum()))]
    print(f"\n    {'':<36}{'intercept-only':>17}{'conditional':>15}")
    for nm, a, b in rows:
        if isinstance(a, int):
            print(f"    {nm:<36}{a:>17d}{b:>15d}")
        else:
            print(f"    {nm:<36}{a:>17.6f}{b:>15.6f}")

    # (1) the mechanical part. the intercept-only model is nested in the
    # conditional one (set the slopes to 0), so the minimised check loss can't
    # go up, and sigma-hat IS that loss.
    check("C5  the conditional fit lowers the check loss (nesting => it must)",
          cl_c <= cl_i,
          f"{cl_i:.9f} -> {cl_c:.9f}, {(1-cl_c/cl_i)*100:.2f}% lower")
    check("C5  and the asserted shortfall contracts with it",
          al < io_al,
          f"{io_al:.6f} -> {al:.6f}, {(1-al/io_al)*100:.2f}% thinner")

    check("C5  the conditional model is CLOSER, not further (ratio)",
          over_c / al < over_i / io_al,
          f"{over_i/io_al:.3f}x -> {over_c/al:.3f}x")

    # (3) what the paper can say: conditioning doesn't make ES worse, it just
    # doesn't make it right, which I think is proofable enough for the paper.
    check("C5  conditioning does not make the shortfall right (still >3x)",
          over_c / al > 3.0, f"{over_c/al:.2f}x")

    RESULTS["C5"] = {
        "check_loss": {"intercept_only": cl_i, "conditional": cl_c,
                       "reduction_pct": (1 - cl_c / cl_i) * 100},
        "asserted_overshoot": {"intercept_only": io_al, "conditional": al,
                               "contraction_pct": (1 - al / io_al) * 100},
        "realised_overshoot": {"intercept_only": over_i, "conditional": over_c},
        "ratio": {"intercept_only": over_i / io_al, "conditional": over_c / al},
        "breaches": {"intercept_only": int(br_i.sum()),
                     "conditional": int(br_c.sum())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="spy_returns_2015_2025.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="experiment_C_results.json")
    args = ap.parse_args()

    K.set_numeric_guards()
    t0 = time.time()
    rule("EXPERIMENT C -- the regression case")
    env = K.print_env_stamp("  ")
    print(f"  tau {TAU} | window {WINDOW} | seed {BASE_SEED}")

    y, X, vol, dates = build_design(args.csv)
    fit = C1(y, X, args.quick)
    if fit is None:
        rule("HALTED -- chains did not converge")
        return 1
    om, sa, var_t = fit
    C2(y, var_t, om, sa, vol)
    io_om, io_sa, io_al, ex_io = C4(y, var_t, sa, om)
    C5(y, var_t, sa, io_om, io_sa, io_al, ex_io)

    written = []

    RESULTS["_meta"] = {"env": env, "tau": TAU, "window": WINDOW,
                        "seed": BASE_SEED, "failures": FAILURES,
                        "elapsed_s": time.time() - t0}
    with open(args.json, "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    written.append(args.json)

    rule("PASS CRITERIA")
    if FAILURES:
        for f in FAILURES:
            print(f"  FAIL  {f}")
    else:
        print("  all criteria passed")
    for w in written:
        print(f"  wrote {w}")
    print(f"\n  elapsed {time.time()-t0:.1f}s")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())