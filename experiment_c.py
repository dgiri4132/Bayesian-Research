#!/usr/bin/env python3
"""
EXPERIMENT C -- the regression case.

Bayesian CVaR project.  Darshan Giri / Prof. Songfeng Zheng.

I made this file because Spy_run.py, experiment_A.py and experiment_B.py all fit
an intercept-only model, i.e. one VaR for every day, but the paper is about
quantile REGRESSION. So someone could basically say I showed the failure in the
wrong model. This file fits an actual regression to see if the failure is still
there.

The main idea is that, under the AL, the overshoot past the quantile is

    VaR_t - ES_t = sigma / (1 - tau)

and sigma is just one number, it has no t in it. So even if VaR_t moves around
with volatility, the model still says the average breach is the same size on
every day, a calm day and March 2020 get the same number. This comes straight
from how sigma is drawn in kkgibs.gibbs_qr_sampler.

What I found:
    - C2: the overshoot the AL claims is way too small in every volatility
      bucket. I tried 2, 3 and 4 buckets and every single one is outside the
      AL's own 95% interval, by 3.9x to 9.0x.
    - C3: whether the real overshoot itself goes up with volatility. This one
      is NOT established. There are only ~123 breaches, and the one test that
      looks significant (OLS, p = 0.0006) is carried by 3 days in the 2020
      crash. If I drop those 3 days p = 0.96. So I report it as a negative and
      I don't claim it, C2 is enough on its own.

The model:
    VaR_t = b0 + b1 |r_{t-1}| + b2 sd(r_{t-20..t-1})
    Both covariates only use data up to t-1 so there's no look-ahead. I
    standardise them so the N(0,1) prior is wide compared to the coefficients,
    which means the coefficients are per sd of each covariate (C1 converts b2
    back to raw units).

Note: this uses 2492 rows since the 20 day window eats the first 20, not the
2512 in Spy_run.py, so  please don't put a number from one next to a number from
the other.

usage:  python3 experiment_C.py [returns.csv] [--quick] [--json OUT] [--no-figures]
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

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
       "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
       "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"

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

    print(f"""
    VaR_t = b0 + b1 |r_(t-1)| + b2 sd(r_(t-20..t-1))

    Both covariates are known at t-1. |r_(t-1)| is yesterday's absolute return
    and the other one is the sd of the {WINDOW} returns before t, so no day gets
    used to predict itself. The first {WINDOW} rows are lost to the window.
""")
    print(f"  observations      {y.size} (from {r.size}, window {WINDOW})")
    print(f"  period            {d[0]} .. {d[-1]}")
    print(f"  |r_(t-1)|         mean {m1:.6f}  sd {s1:.6f}")
    print(f"  vol{WINDOW}             mean {m2:.6f}  sd {s2:.6f}"
          f"   range {vol.min():.6f}..{vol.max():.6f}")
    print(f"  corr(|r_(t-1)|, vol{WINDOW}) = {np.corrcoef(abs_lag, vol)[0,1]:.4f}")
    print("  covariates are standardised so the N(0,1) prior is wide enough,")
    print("  so the coefficients are per sd (C1 converts b2 to raw units).")

    RESULTS["C0"] = {"n": int(y.size), "window": WINDOW,
                     "first": str(d[0]), "last": str(d[-1]),
                     "abs_lag_mean": float(m1), "abs_lag_sd": float(s1),
                     "vol_mean": float(m2), "vol_sd": float(s2),
                     "corr": float(np.corrcoef(abs_lag, vol)[0, 1])}
    return y, X, vol, abs_lag, d, (m1, s1, m2, s2)


# C1  the fit 

def C1(y, X, scal, quick):
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

    m1, s1, m2, s2 = scal
    print(f"\n  in raw units: a one sd rise in vol{WINDOW}"
          f"\n  ({s2:.6f}) moves the fitted 5% quantile by {om[:,2].mean():+.6f}.")

    converged = max(max(rhats), rhat_s) < RHAT_TARGET
    check("C1  every parameter converged", converged,
          f"worst R-hat {max(max(rhats), rhat_s):.5f}")
    if not converged:
        return None

    var_t = X @ om.mean(axis=0)
    hit = float((y < var_t).mean())
    print(f"\n  validity check: the fitted quantile plane cuts "
          f"{100*hit:.2f}% of the data")
    print(f"  (target {100*TAU:.2f}%). If this was off there wouldn't be a real")
    print("  quantile model to talk about.")
    check("C1  the fitted plane cuts tau of the data (within 1 point)",
          abs(hit - TAU) < 0.01, f"{100*hit:.2f}% against {100*TAU:.2f}%")

    print(f"\n  fitted VaR_t ranges {var_t.min():.5f} .. {var_t.max():.5f}"
          f"  -- a factor of {var_t.min()/var_t.max():.1f}")
    print(f"  the intercept-only fit in Spy_run.py (all 2512 days) gave one VaR,")
    print(f"  -0.017353, for every day. C4 refits it on these same {y.size} rows.")

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
    over = var_t[br] - y[br]
    al = float(sa.mean() / (1 - TAU))
    al_lo, al_hi = (float(v) for v in np.percentile(sa / (1 - TAU), [2.5, 97.5]))

    print(f"""
    Under the AL, VaR_t - ES_t = sigma/(1-tau) and sigma is one number. The
    fitted value is {al:.6f}, 95% CI [{al_lo:.6f}, {al_hi:.6f}], and the model uses
    it on all {y.size} days no matter what x_t is.

    Below is the actual average overshoot on the days the quantile was breached,
    split up by volatility. Bootstrap CIs with {N_BOOT} resamples.
""")
    print(f"  breaches: {int(br.sum())} of {y.size} = {100*br.mean():.2f}%\n")
    all_rows, worst_ratio, all_exclude = [], np.inf, True
    for nb in (2, 3, 4):
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
          "every bucket, at every granularity", all_exclude)
    check("C2  the AL understates the overshoot by at least 3x everywhere",
          worst_ratio > 3.0, f"smallest ratio {worst_ratio:.2f}x")

    print(f"""    So the AL is too thin by {worst_ratio:.1f}x to {max(r['ratio'] for r in all_rows):.1f}x depending on the
    regime, and none of the buckets even get close to the AL's interval.
    Adding covariates did move the quantile (VaR_t now ranges over a factor of
    {var_t.min()/var_t.max():.1f}) but the overshoot the model claims is still one number for
    every day, since sigma doesn't depend on x. (C5 shows conditioning does
    make the miss smaller overall, it just doesn't make it right.)""")
    RESULTS["C2"] = {"al_overshoot": al, "al_ci": [al_lo, al_hi],
                     "breaches": int(br.sum()), "rows": all_rows,
                     "worst_ratio": float(worst_ratio)}
    return br, over


# ---- C3  the negative result

def C3(y, var_t, vol, br):
    rule("C3  does the realised overshoot VARY with volatility?  NOT ESTABLISHED.")
    over = var_t[br] - y[br]
    v = vol[br]
    print(f"""
    It's tempting to also say the real overshoot grows with volatility while
    the model keeps it fixed. But the data here doesn't really support that, so
    I  think it should be reported as a negative.
""")
    rho, p_s = stats.spearmanr(v, over)
    sl, ic, r_, p_o, se = stats.linregress(v, over)
    print(f"  Spearman rank correlation   rho = {rho:+.3f},  p = {p_s:.3f}")
    print(f"  OLS slope                   {sl:+.4f} (se {se:.4f}),  p = {p_o:.3f}")
    print(f"\n  The two tests don't agree, so I check if a few days are driving it")
    print(f"  by dropping the highest-volatility breach days one at a time:\n")
    print(f"    {'dropped':<10}{'n':>5}{'OLS slope':>12}{'p':>9}{'Spearman p':>13}")
    order = np.argsort(v)[::-1]
    drops = []
    for k in (0, 1, 2, 3, 5, 10):
        m = np.ones(v.size, bool)
        m[order[:k]] = False
        s2, _, _, p2, e2 = stats.linregress(v[m], over[m])
        _, ps2 = stats.spearmanr(v[m], over[m])
        drops.append({"k": k, "n": int(m.sum()), "slope": float(s2),
                      "p_ols": float(p2), "p_spearman": float(ps2)})
        print(f"    {k:<10}{int(m.sum()):>5}{s2:>12.4f}{p2:>9.3f}{ps2:>13.3f}")

    p3 = drops[3]["p_ols"]
    print(f"""
    So the OLS result is basically carried by 3 days. If you take them out then
    slope is {drops[3]['slope']:+.4f} with p = {p3:.3f}, and the rank test never finds
    anything. With {int(br.sum())} breaches spread over four regimes this experiment can't
    really answer the question, so I'm reporting it as a negative.

    This doesn't hurt C2 though. C2 says the model's overshoot is off by 3.9x
    to 9x in every regime, and that doesn't need C3 at all.""")
    check("C3  reported as not established (this check documents, it does not gate)",
          True, f"Spearman p = {p_s:.3f}; OLS p = {p_o:.3f} but {p3:.3f} "
                f"after dropping 3 days", hard=False)
    RESULTS["C3"] = {"spearman_rho": float(rho), "spearman_p": float(p_s),
                     "ols_slope": float(sl), "ols_p": float(p_o),
                     "leverage": drops, "established": False}


# C4  does adding covariates fix anything? 

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
    {'varies with t?':<26}{'no':>18}{'no':>18}

    The covariates did what they should for the quantile, VaR_t now spans a
    factor of {var_t.min()/var_t.max():.0f}. But the overshoot the model claims is still one number,
    and it's actually SMALLER than in the intercept-only fit ({al:.6f} vs
    {io_al:.6f}). C5 shows why that has to happen.

    This is basically the uniqueness lemma showing up in a regression: the AL
    only has one scale parameter, the quantile constraint already used it, and
    adding covariates to the quantile doesn't give it a second one.""")
    RESULTS["C4"] = {"intercept_only_sigma": io_sa,
                     "intercept_only_omega": io_om,
                     "intercept_only_overshoot": io_al,
                     "regression_sigma": float(sa.mean()),
                     "regression_overshoot": al,
                     "var_min": float(var_t.min()), "var_max": float(var_t.max())}
    return io_om, io_sa, io_al, ex


# ---- C5  the two models compared, every direction checked ----
#
# I added this section because I once wrote "the conditional model is more
# confident in a shortfall FURTHER from the truth".So  every comparison
# about C is computed and checked here.

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

    # width of the 95% interval on the claimed overshoot
    w_i = float(np.diff(np.percentile(ex.sigma / (1 - TAU), [2.5, 97.5]))[0])
    w_c = float(np.diff(np.percentile(sa / (1 - TAU), [2.5, 97.5]))[0])

    rows = [("minimised check loss (= sigma-hat)", cl_i, cl_c),
            ("posterior mean sigma",               io_sa, float(sa.mean())),
            ("asserted overshoot",                 io_al, al),
            ("realised overshoot, own breaches",   over_i, over_c),
            ("ratio realised / asserted",          over_i / io_al, over_c / al),
            ("absolute miss",                      over_i - io_al, over_c - al),
            ("95% CI width on the overshoot",      w_i, w_c),
            ("breaches",                           int(br_i.sum()), int(br_c.sum()))]
    print(f"\n    {'':<36}{'intercept-only':>17}{'conditional':>15}")
    for nm, a, b in rows:
        if isinstance(a, int):
            print(f"    {nm:<36}{a:>17d}{b:>15d}")
        else:
            print(f"    {nm:<36}{a:>17.6f}{b:>15.6f}")

    # (1) the mechanical part. the intercept-only model is nested in the
    # conditional one (set the slopes to 0), so the minimised check loss can't
    # go up, and sigma-hat IS that loss. so the claimed overshoot sigma/(1-tau)
    # shrinks whenever the quantile fit gets better
    check("C5  the conditional fit lowers the check loss (nesting => it must)",
          cl_c <= cl_i,
          f"{cl_i:.9f} -> {cl_c:.9f}, {(1-cl_c/cl_i)*100:.2f}% lower")
    check("C5  and the asserted shortfall contracts with it",
          al < io_al,
          f"{io_al:.6f} -> {al:.6f}, {(1-al/io_al)*100:.2f}% thinner")
    check("C5  the conditional model is MORE confident (narrower interval)",
          w_c < w_i, f"CI width {w_i:.7f} -> {w_c:.7f}, "
                     f"{(1-w_c/w_i)*100:.1f}% narrower")

    check("C5  the conditional model is CLOSER, not further (ratio)",
          over_c / al < over_i / io_al,
          f"{over_i/io_al:.3f}x -> {over_c/al:.3f}x")
    check("C5  the conditional model is CLOSER, not further (absolute)",
          (over_c - al) < (over_i - io_al),
          f"{over_i-io_al:.6f} -> {over_c-al:.6f}")

    # (3) what the paper can say: conditioning doesn't make ES worse, it just
    # doesn't make it right
    check("C5  conditioning does not make the shortfall right (still >3x)",
          over_c / al > 3.0, f"{over_c/al:.2f}x")

    print(f"""
    So what the paper can say is that conditioning does NOT make the shortfall
    worse: {over_i/io_al:.2f}x becomes {over_c/al:.2f}x and the absolute miss drops from {over_i-io_al:.6f}
    to {over_c-al:.6f}, because a quantile line that adapts doesn't get breached as
    badly. What it doesn't do is make the shortfall RIGHT, the miss is still a
    factor of {over_c/al:.1f} overall and 3.9x to 9x in every volatility bucket (C2), and
    the model gets {(1-w_c/w_i)*100:.0f}% MORE confident about it.

    The reason is mechanical. sigma-hat is the minimised check loss, so it's
    both the fit statistic for the quantile and the only thing that sets the
    claimed overshoot sigma/(1-tau). Since the models are nested the check loss
    can't go up when I add covariates (here it drops {(1-cl_c/cl_i)*100:.1f}%), so the claimed
    overshoot has to shrink too (here {(1-al/io_al)*100:.1f}%).""")

    RESULTS["C5"] = {
        "check_loss": {"intercept_only": cl_i, "conditional": cl_c,
                       "reduction_pct": (1 - cl_c / cl_i) * 100},
        "asserted_overshoot": {"intercept_only": io_al, "conditional": al,
                               "contraction_pct": (1 - al / io_al) * 100},
        "realised_overshoot": {"intercept_only": over_i, "conditional": over_c},
        "ratio": {"intercept_only": over_i / io_al, "conditional": over_c / al},
        "absolute_miss": {"intercept_only": over_i - io_al,
                          "conditional": over_c - al},
        "ci_width": {"intercept_only": w_i, "conditional": w_c,
                     "narrowing_pct": (1 - w_c / w_i) * 100},
        "breaches": {"intercept_only": int(br_i.sum()),
                     "conditional": int(br_c.sum())},
        "conditional_is_closer": bool(over_c / al < over_i / io_al),
        "conditional_is_more_confident": bool(w_c < w_i)}


# figures 

def make_figure(y, var_t, vol, sa, dates, prefix="spy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    br = y < var_t
    al = float(sa.mean() / (1 - TAU))
    rng = np.random.default_rng(SEED)
    qs = np.quantile(vol, np.linspace(0, 1, 5))
    cent, los, his, xs = [], [], [], []
    for i in range(4):
        m = (vol >= qs[i]) & (vol <= qs[i + 1] if i == 3 else vol < qs[i + 1])
        o = var_t[m & br] - y[m & br]
        lo, hi = boot_mean_ci(o, rng)
        cent.append(o.mean()); los.append(lo); his.append(hi)
        xs.append(vol[m].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=3)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    # (a) the fitted quantile moves and returns breach it about 5% of the time
    t = np.arange(y.size)
    ax1.plot(t, y, lw=0.4, color=GRID, label="daily return")
    ax1.plot(t, var_t, lw=1.4, color=PAL["blue"], label=r"fitted VaR$_t$ (5%)")
    ax1.plot(t, var_t - al, lw=1.4, color=PAL["red"], ls=(0, (6, 3)),
             label=r"AL's ES$_t$ = VaR$_t-\sigma/(1-\tau)$")
    ax1.scatter(t[br], y[br], s=7, color=PAL["violet"], zorder=5,
                label=f"breaches ({int(br.sum())})")
    step = max(y.size // 5, 1)
    ax1.set_xticks(t[::step])
    ax1.set_xticklabels([d[:7] for d in dates[::step]], fontsize=8)
    ax1.set_ylabel("daily return", fontsize=10, color=INK)
    ax1.set_title("(a)  the quantile is conditional; the shortfall gap is not\n"
                  r"       VaR$_t$ moves with volatility, VaR$_t-$ES$_t$ never does",
                  fontsize=10.5, color=INK, loc="left", pad=8)
    ax1.legend(fontsize=8, frameon=True, loc="lower left")

    # (b) realised vs claimed overshoot by volatility bucket
    ax2.axhline(al, color=PAL["red"], lw=2.0, ls=(0, (6, 3)),
                label=r"AL asserts $\sigma/(1-\tau)$, every day")
    ax2.fill_between([0, 5], *np.percentile(sa / (1 - TAU), [2.5, 97.5]),
                     color=PAL["red"], alpha=0.15, lw=0)
    ax2.errorbar(range(1, 5), cent,
                 yerr=[np.array(cent) - np.array(los), np.array(his) - np.array(cent)],
                 fmt="o", color=PAL["blue"], capsize=4, ms=7, lw=1.6,
                 label="realised mean overshoot (95% bootstrap)")
    for i, (c, r_) in enumerate(zip(cent, [c / al for c in cent])):
        ax2.annotate(f"{r_:.1f}x", (i + 1, c), textcoords="offset points",
                     xytext=(12, 0), fontsize=9, color=INK, va="center")
    ax2.set_xlim(0.5, 4.7)
    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels([f"Q{i}\n{x:.3f}" for i, x in enumerate(xs, 1)], fontsize=9)
    ax2.set_xlabel(f"volatility quartile (mean vol{WINDOW})", fontsize=10, color=INK)
    ax2.set_ylabel(r"VaR$_t-$ES$_t$", fontsize=10, color=INK)
    ax2.set_ylim(0, max(his) * 1.15)
    ax2.set_title("(b)  the overshoot the model asserts, against the one\n"
                  "       that happened, in every volatility regime",
                  fontsize=10.5, color=INK, loc="left", pad=8)
    ax2.legend(fontsize=8.5, frameon=True, loc="upper left")

    fig.tight_layout()
    f = f"{prefix}_fig_regression.png"
    fig.savefig(f, dpi=200, facecolor="white")
    plt.close(fig)
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="spy_returns_2015_2025.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--json", default="experiment_C_results.json")
    args = ap.parse_args()

    K.set_numeric_guards()
    t0 = time.time()
    rule("EXPERIMENT C -- the regression case")
    env = K.print_env_stamp("  ")
    print(f"  tau {TAU} | window {WINDOW} | seed {BASE_SEED}")
    if not os.path.exists(args.csv):
        print(f"  returns file {args.csv} not found.")
        return 2

    y, X, vol, abs_lag, dates, scal = build_design(args.csv)
    fit = C1(y, X, scal, args.quick)
    if fit is None:
        rule("HALTED -- chains did not converge")
        return 1
    om, sa, var_t = fit
    br, over = C2(y, var_t, om, sa, vol)
    C3(y, var_t, vol, br)
    io_om, io_sa, io_al, ex_io = C4(y, var_t, sa, om)
    C5(y, var_t, sa, io_om, io_sa, io_al, ex_io)

    written = []
    if not args.no_figures:
        written.append(make_figure(y, var_t, vol, sa, dates))

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