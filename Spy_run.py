"""I am creatng this file to produce every number and figure that will be used
    for the paper. The following list contains the things that it does:-
    - Loads the data via yfinance, which then in caches and reuses the cache on
    later runs.
    - It computes frequentist baseline which includes Historical, Normal, and S
    Student t-VaR/CVaR .
    - Computes the Bayesian estimation with Kozumi-Kobayashi sigma- extension
    Gibbs sampler: intercept-only model at tau = 0.05 including four independent
    chains, each with its own seed, 15,000 iterations each and 5,000 burn-in period.
    - Diagnostics using plit-chain Gelman-Rubin R-hat for theta and sigma against the
    1.01 bar which prints PASS/FAIL
    - It pools the 40000 retained draws and also reports posterior means and 95%
    credible intervals.
    - Outputs comparision table and three figures as well.

    This is probably draft1 let's see if I need to make any more changes otherwise this 
    can be the whole file """

# added from spy_run1.py:
# - sampler, R-hat and ESS come from kkgibs now
# - stops if R-hat fails
# - exact posterior check, evidence section, prior and guard checks
# - saves spy_results.json

import sys
import os
import json
import time
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t

import kkgibs as K

TAU = 0.05
START = "2015-08-17"
END = "2025-08-14"
N_ITER = 15000
BURN_IN = 5000
N_CHAINS = 4
BASE_SEED = 1000
N0, S0 = 4.0, 0.024
CSV_CACHE = "spy_returns_2015_2025.csv"
RHAT_TARGET = 1.01
EXACT_DRAWS = 400_000

FAILURES = []
RESULTS = {}


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


def load_data(path=None):
    if path:
        r = np.loadtxt(path, delimiter =",", skiprows = 1, usecols = -1 )
        return r
    if os.path.exists(CSV_CACHE):
        r = np.loadtxt(CSV_CACHE, delimiter = ",", skiprows = 1, usecols = -1)
        return r
    import yfinance as yf
    df = yf.download("SPY", start=START, end=END, auto_adjust=False)["Adj Close"].dropna()
    r = df.pct_change().dropna()
    r.to_csv(CSV_CACHE, header=["return"])
    r = r.values.ravel()
    print(f"Downloaded {len(r)} SPY daily returns "
            f"({START} .. {END}, exclusive end), cached to {CSV_CACHE}")
    return r


# records which data was used (hash, n, min, max etc)
def provenance(r, source):
    return {"returns": "simple (pct_change)", "source": source, "n": int(r.size),
            "sha256": hashlib.sha256(r.tobytes()).hexdigest()[:16],
            "min": float(r.min()), "max": float(r.max()),
            "mean": float(r.mean()), "sd": float(r.std(ddof=1))}


# fake AL data with known omega and sigma, for --self-test
def synthetic_data(n, rng):
    truth = {"omega": -0.01735, "sigma": 0.0014397, "tau": TAU}
    truth["ES"] = K.al_es(truth["omega"], truth["sigma"], TAU)
    truth["mean"] = K.al_mean(truth["omega"], truth["sigma"], TAU)
    y = K.al_rvs(n, truth["omega"], truth["sigma"], TAU, rng)
    return y, truth

# gibbs_qr_sampler, split_rhat and ess were here before, now using the ones in kkgibs


def frequentist_baseline(r, alpha):
    # historical VaR/CVaR with the order statistic (k = 126) same as the other files
    emp = K.empirical_var_cvar(r, alpha)
    var_hist = emp["VaR"]
    cvar_hist = emp["CVaR"]

    mu_p,std_p = r.mean(), r.std(ddof = 1)
    var_norm = norm.ppf(alpha, mu_p, std_p)
    cvar_norm = mu_p - std_p*norm.pdf(norm.ppf(alpha))/alpha

    df_t, loc_t, sc_t = t.fit(r)
    var_t = t.ppf(alpha, df_t, loc_t, sc_t)
    q = t.ppf(alpha, df_t)
    cvar_t_std = -(t.pdf(q, df_t)/alpha) * (df_t + q**2)/(df_t-1)
    cvar_t = loc_t+ sc_t*cvar_t_std
    return {"Historical":(var_hist,cvar_hist),
            "Normal":(float(var_norm),float(cvar_norm)),
            "Student-t":(float(var_t),float(cvar_t))}, emp, \
           {"df": float(df_t), "loc": float(loc_t), "scale": float(sc_t)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None, help="returns CSV")
    ap.add_argument("--b-p0", type=float, default=None,
                    help="prior variance for omega, 1.0 (mine) or 100.0 (K-K)")
    ap.add_argument("--self-test", action="store_true",
                    help="run on fake AL data where the answer is known")
    ap.add_argument("--self-test-n", type=int, default=2512)
    ap.add_argument("--quick", action="store_true", help="short chains")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out-prefix", default="spy")
    args = ap.parse_args()

    n_iter = 8000 if args.quick else N_ITER
    burn_in = 3000 if args.quick else BURN_IN
    exact_draws = 100_000 if args.quick else EXACT_DRAWS
    b_p0 = args.b_p0
    if b_p0 is None:
        b_p0 = float(os.environ.get("SPY_B_P0", 1.0))

    K.set_numeric_guards()
    t0 = time.time()
    env = K.print_env_stamp("  ")

    truth = None
    if args.self_test:
        r, truth = synthetic_data(args.self_test_n, np.random.default_rng(BASE_SEED))
        prov = provenance(r, f"SYNTHETIC AL, n={args.self_test_n}")
        print("  self-test: fake AL data")
    else:
        try:
            r = load_data(args.csv)
        except Exception as e:
            print(f"  no data: {e}")
            return 2
        r = np.asarray(r, dtype=float).ravel()
        r = r[np.isfinite(r)]
        prov = provenance(r, args.csv or CSV_CACHE)
    n = len(r)
    y = r
    X = np.ones((n, 1))
    if n < 100:
        print("  too few observations")
        return 2

    print(f"\nn = {n} observations | tau = {TAU} | "
          f"{N_CHAINS} chains x {n_iter} iter ({burn_in} burn-in)")
    print(f"  sha256 {prov['sha256']}")

    freq, emp, t_fit = frequentist_baseline(r, TAU)
    print(f"  k = {emp['k']}, tau_eff = {emp['tau_eff']:.8f}, "
          f"VaR {emp['VaR']:.6f} (np.quantile gives {emp['VaR_interp']:.6f})")


    q_emp = np.quantile(r, TAU)
    sig_rough = 0.01
    omega_inits = [q_emp, q_emp + 0.02, q_emp - 0.02, 0.0]
    sigma_inits = [sig_rough, 5*sig_rough, 0.2*sig_rough, 10*sig_rough]

    theta_chains, sigma_chains = [], []
    guard = {"delta": 0, "v": 0}
    for c in range(N_CHAINS):
        res = K.gibbs_qr_sampler(
            X, y, TAU, n_iter, burn_in, seed=BASE_SEED + 111*c,
            omega_init=[omega_inits[c]], sigma_init=sigma_inits[c],
            B_p0=np.array([[b_p0]]), n0=N0, s0=S0)
        om, sg = res.omega, res.sigma
        guard["delta"] += res.delta_floor_hits
        guard["v"] += res.v_clip_hits
        theta_chains.append(om[:, 0])
        sigma_chains.append(sg)
        assert all(c.ndim == 1 for c in theta_chains + sigma_chains), "chains must be 1-D"
        print(f"  chain {c+1}: start(theta={omega_inits[c]:+.4f}, "
              f"sigma={sigma_inits[c]:.4f}) -> "
              f"post mean theta={om[:,0].mean():+.5f}, sigma={sg.mean():.5f}")

    # ---- diagnostics ----
    rhat_theta = K.split_rhat(theta_chains)
    rhat_sigma = K.split_rhat(sigma_chains)
    ess_theta = sum(K.ess(c) for c in theta_chains)
    ess_sigma = sum(K.ess(c) for c in sigma_chains)
    mcse_theta = K.mcse(theta_chains, ess_theta)
    mcse_sigma = K.mcse(sigma_chains, ess_sigma)
    kept = N_CHAINS*(n_iter-burn_in)
    print(f"\nR-hat: theta={rhat_theta:.5f}, sigma={rhat_sigma:.5f} "
          f"(target < {RHAT_TARGET})")
    print(f"ESS  : theta={ess_theta:,.0f}, sigma={ess_sigma:,.0f} "
          f"(of {kept:,} retained)")
    print(f"MCSE : theta={mcse_theta:.2e}, sigma={mcse_sigma:.2e}")
    # kkgibs split_rhat gives inf (not nan) for a stuck chain so this fails properly
    converged = max(rhat_theta, rhat_sigma) < RHAT_TARGET
    print("Convergence:", "PASS" if converged else "FAIL -- do not use results")
    # stop if it didn't converge, don't save any results
    if not converged:
        rule("stopped, chains did not converge")
        with open(f"{args.out_prefix}_CONVERGENCE_FAILED.txt", "w") as fh:
            fh.write(f"R-hat theta={rhat_theta} sigma={rhat_sigma} target={RHAT_TARGET}\n")
            fh.write(f"env={env}\n")
        return 1

    # ---- pooled posterior ----
    theta_all = np.concatenate(theta_chains)
    sigma_all = np.concatenate(sigma_chains)
    var_draws = theta_all
    es_draws  = theta_all - sigma_all / (1 - TAU)
    # ES the second way (AL mean - sigma/tau) to check it gives the same thing
    es_alt = K.al_es_via_mean(theta_all, sigma_all, TAU)

    def ci(x):
        return np.percentile(x, [2.5, 97.5])

    var_ci, es_ci = ci(var_draws), ci(es_draws)

    print(f"\n{'Method':<22}{'VaR(5%)':>10}{'CVaR/ES(5%)':>14}")
    print("-" * 46)
    for name, (v, c_) in freq.items():
        print(f"{name:<22}{v:>10.4f}{c_:>14.4f}")
    print(f"{'Bayesian (post mean)':<22}{var_draws.mean():>10.4f}{es_draws.mean():>14.4f}")
    print(f"{'  95% CI VaR':<22}[{var_ci[0]:.4f}, {var_ci[1]:.4f}]")
    print(f"{'  95% CI ES':<22}{'':>10}[{es_ci[0]:.4f}, {es_ci[1]:.4f}]")

    # ---- sanity check ----
    gap = abs(var_draws.mean() - freq["Historical"][0])
    print(f"\nSanity: |post-mean VaR - historical VaR| = {gap:.5f} "
          f"({'OK' if gap < 0.002 else 'INVESTIGATE'})")

    # ---- exact posterior check ----
    # sigma integrates out so omega can be sampled on a grid without gibbs,
    # if both give the same answer the ES result isn't from the sampler
    rule("4  exact posterior vs gibbs")
    ex = K.exact_posterior(y, TAU, exact_draws, seed=BASE_SEED + 7,
                           omega_p0=0.0, B_p0=b_p0, n0=N0, s0=S0)
    ex_es = ex.omega - ex.sigma / (1 - TAU)
    print(f"\n  grid {ex.grid_points:,} points, edge log-density drop "
          f"{ex.edge_log_drop:.0f}, refinement max rel change "
          f"{ex.refinement_max_rel_change:.1e}")
    rows = [("omega mean", var_draws.mean(), ex.omega.mean()),
            ("omega 2.5%", var_ci[0], float(np.percentile(ex.omega, 2.5))),
            ("omega 97.5%", var_ci[1], float(np.percentile(ex.omega, 97.5))),
            ("sigma mean", sigma_all.mean(), ex.sigma.mean()),
            ("ES mean", es_draws.mean(), ex_es.mean()),
            ("ES 2.5%", es_ci[0], float(np.percentile(ex_es, 2.5))),
            ("ES 97.5%", es_ci[1], float(np.percentile(ex_es, 97.5)))]
    print(f"\n  {'quantity':<14}{'Gibbs':>14}{'exact iid':>14}{'difference':>14}")
    worst = 0.0
    for nm, g, e in rows:
        worst = max(worst, abs(g - e))
        print(f"  {nm:<14}{g:>14.7f}{e:>14.7f}{g-e:>14.2e}")
    check("4  Gibbs agrees with the exact posterior to 1e-4",
          worst < 1e-4, f"worst absolute difference {worst:.2e}")
    check("4  grid error is negligible", ex.refinement_max_rel_change < 1e-6,
          f"refinement change {ex.refinement_max_rel_change:.1e}")

    # ---- evidence ----
    rule("5  evidence")
    post_sd_es = float(es_draws.std(ddof=1))
    post_sd_var = float(var_draws.std(ddof=1))
    cvar_emp = emp["CVaR"]
    var_emp = emp["VaR"]

    # 5a VaR should match the historical VaR
    print("\n  5a  VaR")
    d_var = var_draws.mean() - var_emp
    print(f"      posterior mean VaR {var_draws.mean():+.6f}, empirical {var_emp:+.6f}, "
          f"difference {abs(d_var)/post_sd_var:.2f} posterior sd")
    check("5a  VaR within 3 posterior sd of historical",
          abs(d_var) / post_sd_var < 3.0, f"{abs(d_var)/post_sd_var:.2f} sd")

    # 5b ES vs historical CVaR, this is the main result
    print("\n  5b  ES")
    es_gap = es_draws.mean() - cvar_emp
    n_reach = int(np.count_nonzero(es_draws <= cvar_emp))
    print(f"      posterior mean ES {es_draws.mean():+.6f}, empirical CVaR {cvar_emp:+.6f}")
    print(f"      gap {es_gap:+.6f} = {abs(es_gap)/post_sd_es:.1f} posterior sd, "
          f"{n_reach:,} of {es_draws.size:,} draws reach the empirical CVaR")
    check("5b  ES off by more than 5 posterior sd",
          abs(es_gap) / post_sd_es > 5.0, f"{abs(es_gap)/post_sd_es:.1f} sd", hard=False)
    check("5b  no draw reaches the empirical CVaR", n_reach == 0,
          f"{n_reach} draws reach it", hard=False)

    # 5c the AL says the average loss past VaR is sigma/(1-tau)
    print("\n  5c  mean excess below VaR")
    al_excess = float(sigma_all.mean() / (1 - TAU))
    emp_excess = emp["mean_excess_lower"]
    print(f"      AL sigma/(1-tau) {al_excess:.6f}, empirical VaR - CVaR "
          f"{emp_excess:.6f}, ratio {emp_excess/al_excess:.3f}x")

    # 5d AL mean vs sample mean
    print("\n  5d  AL mean vs sample mean")
    mu_al = float(K.al_mean(var_draws, sigma_all, TAU).mean())
    print(f"      AL mean {mu_al:+.6f}, sample mean {emp['mean']:+.6f}, "
          f"ratio {mu_al/emp['mean']:.2f}x")

    # 5e both ES formulas should give the same number on every draw
    print("\n  5e  ES identity")
    id_err = float(np.max(np.abs(es_draws - es_alt)))
    check("5e  omega - sigma/(1-tau) equals mu_AL - sigma/tau",
          id_err < 1e-15, f"max difference {id_err:.2e}")

    # 5f using the sample mean instead of the AL mean gives back the empirical CVaR
    print("\n  5f  mean identity")
    fit = K.al_mle(y, TAU)
    recovered = fit["ybar"] - fit["sigma_hat"] / fit["tau_eff"]
    check("5f  ybar - sigma_hat/tau_eff equals the empirical CVaR",
          abs(recovered - fit["cvar_emp"]) < 1e-14,
          f"difference {abs(recovered-fit['cvar_emp']):.2e}")
    # this identity is from Taylor (2019, JBES 37(1)) p.6, who takes it from
    # Bassett, Koenker & Kordas (2004), checking it with his eq (5)
    a_t = fit["tau_eff"]
    Q_t = fit["omega_hat"]
    tay_sig = float(np.mean((y - Q_t) * (a_t - (y <= Q_t))))
    tay_es = float(y.mean() - tay_sig / a_t)
    check("5f  same as Taylor's expression",
          abs(tay_sig - fit["sigma_hat"]) < 1e-15
          and abs(tay_es - fit["cvar_emp"]) < 1e-14,
          f"sigma diff {abs(tay_sig-fit['sigma_hat']):.2e}, "
          f"ES diff {abs(tay_es-fit['cvar_emp']):.2e}")

    # 5g R diagnostic
    print("\n  5g  R")
    R = TAU * (emp["mean"] - cvar_emp) / ((1 - TAU) * (var_emp - cvar_emp))
    R_eff = (emp["tau_eff"] * (emp["mean"] - cvar_emp)
             / ((1 - emp["tau_eff"]) * (var_emp - cvar_emp)))
    print(f"      R at nominal tau {R:.6f}, at tau_eff {R_eff:.6f}")

    # 5h mean excess at deeper tail levels. the growth with depth goes away
    # without 2020 so I'm not using it in the paper, only kept for the json
    print("\n  5h  mean excess at deeper tail levels")
    ys = np.sort(y)
    n_y = ys.size
    al_flat = fit["sigma_hat"] / (1 - fit["tau_eff"])
    depths = [0.05, 0.025, 0.01, 0.005, 0.002]
    depth_rows = []
    print(f"      {'depth':>8}{'threshold':>13}{'n below':>9}"
          f"{'AL claims':>12}{'realised':>12}{'ratio':>9}")
    for p in depths:
        k_p = int(np.floor(p * n_y + 0.5))
        if k_p < 2:
            continue
        u = float(ys[k_p - 1])
        realised = float(np.mean(u - ys[:k_p]))
        depth_rows.append({"depth": p, "k": k_p, "threshold": u,
                           "al_mean_excess": float(al_flat),
                           "emp_mean_excess": realised,
                           "ratio": realised / al_flat})
        print(f"      {p:>8.3%}{u:>13.6f}{k_p:>9d}"
              f"{al_flat:>12.6f}{realised:>12.6f}{realised/al_flat:>8.2f}x")
    al_vals = [row["al_mean_excess"] for row in depth_rows]
    emp_vals = [row["emp_mean_excess"] for row in depth_rows]
    growth = emp_vals[-1] / emp_vals[0]
    check("5h  AL mean excess is the same at every depth",
          max(al_vals) - min(al_vals) == 0.0,
          f"spread {max(al_vals) - min(al_vals):.2e} over {len(depth_rows)} depths")
    if args.self_test:
        check("5h  fake AL data shows no growth with depth",
              growth < 1.5, hard=False, detail=f"factor {growth:.2f}")
    else:
        check("5h  empirical mean excess grows with depth",
              growth > 1.5, f"factor {growth:.2f}  (P = 0.0074 under the AL null)")

    # ---- robustness ----
    rule("6  robustness")
    # 6a rerun with K-K's prior N(0,100) instead of N(0,1)
    print("\n  6a  prior N(0,1) vs N(0,100)")
    alt_b = 100.0 if b_p0 == 1.0 else 1.0
    th_alt, sg_alt = [], []
    for c in range(N_CHAINS):
        res = K.gibbs_qr_sampler(
            X, y, TAU, n_iter, burn_in, seed=BASE_SEED + 111 * c,
            omega_init=[omega_inits[c]], sigma_init=sigma_inits[c],
            B_p0=np.array([[alt_b]]), n0=N0, s0=S0)
        th_alt.append(res.omega[:, 0])
        sg_alt.append(res.sigma)
    th_a = np.concatenate(th_alt)
    es_a = th_a - np.concatenate(sg_alt) / (1 - TAU)
    prior_shift = max(abs(th_a.mean() - var_draws.mean()),
                      abs(es_a.mean() - es_draws.mean()))
    check("6a  prior doesn't change the result", prior_shift < 1e-4,
          f"largest shift in a posterior mean {prior_shift:.2e}")

    # 6b rerun with much smaller guard floors to see if the guards matter
    print("\n  6b  numerical guards")
    print(f"      delta^2 floor hit {guard['delta']:,} times, v floor "
          f"{guard['v']:,} times, out of {N_CHAINS*n_iter*n:,} latent draws")
    th_g, sg_g = [], []
    for c in range(N_CHAINS):
        res = K.gibbs_qr_sampler(
            X, y, TAU, n_iter, burn_in, seed=BASE_SEED + 111 * c,
            omega_init=[omega_inits[c]], sigma_init=sigma_inits[c],
            B_p0=np.array([[b_p0]]), n0=N0, s0=S0,
            delta2_floor=1e-24, v_floor=1e-20)
        th_g.append(res.omega[:, 0])
        sg_g.append(res.sigma)
    th_gg = np.concatenate(th_g)
    es_gg = th_gg - np.concatenate(sg_g) / (1 - TAU)
    guard_shift = max(abs(th_gg.mean() - var_draws.mean()),
                      abs(es_gg.mean() - es_draws.mean()))
    check("6b  guards don't change any reported digit", guard_shift < 1e-5,
          f"largest shift {guard_shift:.2e}")

    # 6c only for --self-test, the 95% intervals should cover the true values
    if truth is not None:
        print("\n  6c  self-test coverage")
        ok_all = True
        for nm, tv, dr in (("omega", truth["omega"], var_draws),
                           ("sigma", truth["sigma"], sigma_all),
                           ("ES", truth["ES"], es_draws)):
            lo, hi = ci(dr)
            cov = lo <= tv <= hi
            ok_all &= cov
            print(f"      {nm:<8} truth {tv:+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
                  f"{'covers' if cov else 'MISSES'}")
        check("6c  every 95% interval covers the true value", ok_all)

    # ---- save numerical results ----
    rule("7  outputs")
    with open(f"{args.out_prefix}_results.txt", "w") as f:
        f.write(f"n={n}, tau={TAU}, chains={N_CHAINS}, iter={n_iter}, burn={burn_in}\n")
        f.write(f"priors: sigma~IG({N0/2:g},{S0/2:g}), omega~N(0,{b_p0:g})\n")
        f.write(f"R-hat theta={rhat_theta:.5f} sigma={rhat_sigma:.5f}  "
                f"ESS theta={ess_theta:.0f} sigma={ess_sigma:.0f}\n\n")
        for name, (v, c_) in freq.items():
            f.write(f"{name:<22}VaR={v:.4f}  ES={c_:.4f}\n")
        f.write(f"{'Bayesian':<22}VaR={var_draws.mean():.4f} "
                f"[{var_ci[0]:.4f},{var_ci[1]:.4f}]  "
                f"ES={es_draws.mean():.4f} [{es_ci[0]:.4f},{es_ci[1]:.4f}]\n")
        f.write(f"\nES - empirical CVaR = {es_gap:+.6f} "
                f"({abs(es_gap)/post_sd_es:.1f} posterior sd); "
                f"{n_reach} of {es_draws.size} draws reach it\n")
        f.write(f"mean excess: AL {al_excess:.6f} vs empirical {emp_excess:.6f} "
                f"({emp_excess/al_excess:.2f}x)\n")
        f.write(f"R = {R:.6f} (nominal tau), {R_eff:.6f} (tau_eff)\n")

    # same keys as spy_run1's json so the verify scripts can read it
    RESULTS.update({
        "env": env, "provenance": prov,
        "config": {"tau": TAU, "n_iter": n_iter, "burn_in": burn_in,
                   "n_chains": N_CHAINS, "base_seed": BASE_SEED,
                   "B_p0": b_p0, "n0": N0, "s0": S0, "self_test": args.self_test},
        "frequentist": freq, "empirical": emp, "t_fit": t_fit,
        "diagnostics": {"rhat_omega": rhat_theta, "rhat_sigma": rhat_sigma,
                        "ess_omega": ess_theta, "ess_sigma": ess_sigma,
                        "mcse_omega": mcse_theta, "mcse_sigma": mcse_sigma,
                        "kept": kept,
                        "guard_delta_hits": guard["delta"],
                        "guard_v_hits": guard["v"]},
        "posterior": {"VaR_mean": float(var_draws.mean()), "VaR_ci": list(var_ci),
                      "sigma_mean": float(sigma_all.mean()),
                      "ES_mean": float(es_draws.mean()), "ES_ci": list(es_ci),
                      "ES_sd": post_sd_es, "VaR_sd": post_sd_var},
        "exact": {"omega_mean": float(ex.omega.mean()),
                  "sigma_mean": float(ex.sigma.mean()),
                  "ES_mean": float(ex_es.mean()),
                  "ES_ci": [float(v) for v in np.percentile(ex_es, [2.5, 97.5])],
                  "worst_gibbs_diff": float(worst),
                  "grid_refinement_rel": ex.refinement_max_rel_change},
        "evidence": {"gap_ES_minus_CVaR": float(es_gap),
                     "gap_in_posterior_sd": float(abs(es_gap) / post_sd_es),
                     "draws_reaching_CVaR": n_reach,
                     "al_mean_excess": al_excess,
                     "emp_mean_excess": emp_excess,
                     "excess_ratio": float(emp_excess / al_excess),
                     "al_implied_mean": mu_al,
                     "sample_mean": emp["mean"],
                     "es_identity_max_err": id_err,
                     "mean_identity_err": float(abs(recovered - fit["cvar_emp"])),
                     "R_nominal": float(R), "R_tau_eff": float(R_eff),
                     "tail_depth": depth_rows,
                     "tail_depth_growth": float(growth),
                     "al_mle": fit},
        "robustness": {"prior_alt_B_p0": alt_b,
                       "prior_max_shift": float(prior_shift),
                       "guard_max_shift": float(guard_shift)},
        "truth": truth,
        "failures": FAILURES,
    })
    with open(f"{args.out_prefix}_results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)

    if not args.no_figures:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(es_draws, bins=80, density=True, color="steelblue", alpha=0.65,
                edgecolor="white", label="ES posterior (Bayesian)")
        ax.hist(ex_es, bins=80, density=True, histtype="step", color="black",
                lw=1.0, label="ES posterior (exact, i.i.d.)")
        for (name, (v, c_)), col in zip(freq.items(), ["red", "green", "orange"]):
            ax.axvline(c_, color=col, ls="--", lw=1.6, label=f"{name} CVaR ({c_:.4f})")
        ax.axvline(es_draws.mean(), color="navy", lw=1.6,
                   label=f"Posterior mean ({es_draws.mean():.4f})")
        for edge in es_ci:
            ax.axvline(edge, color="navy", lw=1.0, ls=":")
        ax.set_xlabel("Daily return"); ax.set_ylabel("Posterior density")
        ax.set_title(f"Posterior distribution of 5% Expected Shortfall, SPY "
                     f"({n} obs)\nwith frequentist point estimates")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{args.out_prefix}_fig_es_posterior.png", dpi=150); plt.close(fig)


        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for c in range(N_CHAINS):
            axes[0].plot(theta_chains[c][::10], lw=0.4, alpha=0.7, label=f"chain {c+1}")
            axes[1].plot(sigma_chains[c][::10], lw=0.4, alpha=0.7)
        axes[0].set_ylabel(r"$\theta$ (VaR)"); axes[1].set_ylabel(r"$\sigma$")
        axes[1].set_xlabel("retained iteration / 10")
        axes[0].set_title(f"Trace plots, {N_CHAINS} chains "
                          f"(R-hat: {rhat_theta:.4f} / {rhat_sigma:.4f})")
        axes[0].legend(fontsize=7, ncol=4)
        fig.tight_layout(); fig.savefig(f"{args.out_prefix}_fig_traces.png", dpi=150); plt.close(fig)


        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(var_draws, bins=80, density=True, color="slategray", alpha=0.65,
                edgecolor="white", label=r"VaR posterior ($\theta$)")
        for (name, (v, c_)), col in zip(freq.items(), ["red", "green", "orange"]):
            ax.axvline(v, color=col, ls="--", lw=1.6, label=f"{name} VaR ({v:.4f})")
        ax.set_xlabel("Daily return"); ax.set_ylabel("Posterior density")
        ax.set_title("Posterior distribution of 5% VaR, SPY, with frequentist points")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{args.out_prefix}_fig_var_posterior.png", dpi=150); plt.close(fig)

    print(f"\nSaved: {args.out_prefix}_results.txt, {args.out_prefix}_results.json"
          + ("" if args.no_figures else
             f", {args.out_prefix}_fig_es_posterior.png, {args.out_prefix}_fig_traces.png, "
             f"{args.out_prefix}_fig_var_posterior.png"))

    rule("summary")
    if FAILURES:
        for f in FAILURES:
            print(f"  FAIL  {f}")
    else:
        print("  all hard checks passed")
    print(f"\n  elapsed {time.time()-t0:.1f}s")
    return 1 if FAILURES else 0

if __name__ == "__main__":
    sys.exit(main())