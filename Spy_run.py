"""I am creatng this file to produce every number that will be used
    for the paper. The following list contains the things that it does:-
    - Loads the data via yfinance, which then in caches and reuses the cache on
    later runs.
    - It computes the historical VaR/CVaR.
    - Computes the Bayesian estimation with Kozumi-Kobayashi sigma- extension
    Gibbs sampler: intercept-only model at tau = 0.05 including four independent
    chains, each with its own seed, 15,000 iterations each and 5,000 burn-in period.
    - Diagnostics using plit-chain Gelman-Rubin R-hat for theta and sigma against the
    1.01 bar which prints PASS/FAIL
    - It pools the 40000 retained draws and also reports posterior means and 95%
    credible intervals.
    - Outputs comparision table as well.

    This is probably draft1 let's see if I need to make any more changes otherwise this
    can be the whole file """

# added from spy_run1.py:
# - sampler, R-hat and ESS come from kkgibs now
# - stops if R-hat fails
# - exact posterior check, evidence section, guard counts.
# - saves spy_results.json
# - 29 Oct 2026: sigma_hat split (Lemma 2), mean_gap and the VaR distance are saved too,
#   so every SPY number in the paper is in spy_results.json

import sys
import os
import json
import time
import numpy as np

import kkgibs as K

TAU = 0.05
START = "2015-08-17"
END = "2025-08-14"
N_ITER = 15000
BURN_IN = 5000
N_CHAINS = 4
BASE_SEED = 1000
N0, S0 = 4.0, 0.024
B_P0 = 1.0
CSV_CACHE = "spy_returns_2015_2025.csv"
RHAT_TARGET = 1.01
EXACT_DRAWS = 400_000

FAILURES = []
RESULTS = {}


def check(name, ok, detail="", hard=True):
    tag = "PASS" if ok else ("FAIL" if hard else "note")
    print(f"    [{tag}] {name}" + (f": {detail}" if detail else ""))
    if not ok and hard:
        FAILURES.append(f"{name}: {detail}")
    return ok


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_data():
    if len(sys.argv)>1:
        r = np.loadtxt(sys.argv[1], delimiter =",", skiprows = 1, usecols = -1 )
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

# gibbs_qr_sampler, split_rhat and ess were here before, now using the ones in kkgibs


def main():
    K.set_numeric_guards()
    t0 = time.time()
    env = K.print_env_stamp("  ")

    r = load_data()
    n = len(r)
    y = r
    X = np.ones((n, 1))

    print(f"\nn = {n} observations | tau = {TAU} | "
          f"{N_CHAINS} chains x {N_ITER} iter ({BURN_IN} burn-in)")

    # historical VaR/CVaR with the order statistic (k = 126) same as the other files
    emp = K.empirical_var_cvar(r, TAU)
    print(f"  k = {emp['k']}, tau_eff = {emp['tau_eff']:.8f}, VaR {emp['VaR']:.6f}")


    q_emp = np.quantile(r, TAU)
    sig_rough = 0.01
    omega_inits = [q_emp, q_emp + 0.02, q_emp - 0.02, 0.0]
    sigma_inits = [sig_rough, 5*sig_rough, 0.2*sig_rough, 10*sig_rough]

    theta_chains, sigma_chains = [], []
    guard = {"delta": 0, "v": 0}
    for c in range(N_CHAINS):
        res = K.gibbs_qr_sampler(
            X, y, TAU, N_ITER, BURN_IN, seed=BASE_SEED + 111*c,
            omega_init=[omega_inits[c]], sigma_init=sigma_inits[c],
            B_p0=np.array([[B_P0]]), n0=N0, s0=S0)
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
    kept = N_CHAINS*(N_ITER-BURN_IN)
    print(f"\nR-hat: theta={rhat_theta:.5f}, sigma={rhat_sigma:.5f} "
          f"(target < {RHAT_TARGET})")
    print(f"ESS  : theta={ess_theta:,.0f}, sigma={ess_sigma:,.0f} "
          f"(of {kept:,} retained)")
    # kkgibs split_rhat gives inf (not nan) for a stuck chain so this fails properly
    converged = max(rhat_theta, rhat_sigma) < RHAT_TARGET
    print("Convergence:", "PASS" if converged else "FAIL, do not use results")
    # stop if it didn't converge, don't save any results
    if not converged:
        rule("stopped, chains did not converge")
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
    print(f"{'Historical':<22}{emp['VaR']:>10.4f}{emp['CVaR']:>14.4f}")
    print(f"{'Bayesian (post mean)':<22}{var_draws.mean():>10.4f}{es_draws.mean():>14.4f}")
    print(f"{'  95% CI VaR':<22}[{var_ci[0]:.4f}, {var_ci[1]:.4f}]")
    print(f"{'  95% CI ES':<22}{'':>10}[{es_ci[0]:.4f}, {es_ci[1]:.4f}]")

    # ---- exact posterior check ----
    # sigma integrates out so omega can be sampled on a grid without gibbs,
    # if both give the same answer the ES result isn't from the sampler
    rule("4  exact posterior vs gibbs")
    ex = K.exact_posterior(y, TAU, EXACT_DRAWS, seed=BASE_SEED + 7,
                           omega_p0=0.0, B_p0=B_P0, n0=N0, s0=S0)
    ex_es = ex.omega - ex.sigma / (1 - TAU)
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

    # 5g sigma_hat split at VaR (Lemma 2 in the paper), at tau_eff
    print("\n  5g  sigma_hat split")
    te = fit["tau_eff"]
    e_L = var_emp - cvar_emp
    e_U = emp["ybar_up"] - var_emp
    part_lower = te * (1 - te) * e_L
    part_upper = te * (1 - te) * e_U
    split_err = abs(fit["sigma_hat"] - (part_lower + part_upper))
    print(f"      e_L {e_L:.6f}, e_U {e_U:.6f}, share from below VaR "
          f"{part_lower/fit['sigma_hat']:.1%}, from above {part_upper/fit['sigma_hat']:.1%}")
    check("5g  sigma_hat equals tau(1-tau)(e_L + e_U) at tau_eff",
          split_err < 1e-15, f"difference {split_err:.2e}")

    # ---- save numerical results ----
    rule("6  outputs")
    with open("spy_results.txt", "w") as f:
        f.write(f"n={n}, tau={TAU}, chains={N_CHAINS}, iter={N_ITER}, burn={BURN_IN}\n")
        f.write(f"priors: sigma~IG({N0/2:g},{S0/2:g}), omega~N(0,{B_P0:g})\n")
        f.write(f"R-hat theta={rhat_theta:.5f} sigma={rhat_sigma:.5f}  "
                f"ESS theta={ess_theta:.0f} sigma={ess_sigma:.0f}\n\n")
        f.write(f"{'Historical':<22}VaR={emp['VaR']:.4f}  ES={emp['CVaR']:.4f}\n")
        f.write(f"{'Bayesian':<22}VaR={var_draws.mean():.4f} "
                f"[{var_ci[0]:.4f},{var_ci[1]:.4f}]  "
                f"ES={es_draws.mean():.4f} [{es_ci[0]:.4f},{es_ci[1]:.4f}]\n")

    # the numbers in the paper are read from this file
    RESULTS.update({
        "env": env, "provenance": {"source": CSV_CACHE, "n": int(n)},
        "config": {"tau": TAU, "n_iter": N_ITER, "burn_in": BURN_IN,
                   "n_chains": N_CHAINS, "base_seed": BASE_SEED,
                   "B_p0": B_P0, "n0": N0, "s0": S0},
        "empirical": emp,
        "diagnostics": {"rhat_omega": rhat_theta, "rhat_sigma": rhat_sigma,
                        "ess_omega": ess_theta, "ess_sigma": ess_sigma,
                        "kept": kept,
                        "guard_delta_hits": guard["delta"],
                        "guard_v_hits": guard["v"]},
        "posterior": {"VaR_mean": float(var_draws.mean()), "VaR_ci": list(var_ci),
                      "sigma_mean": float(sigma_all.mean()),
                      "ES_mean": float(es_draws.mean()), "ES_ci": list(es_ci),
                      "ES_sd": post_sd_es, "VaR_sd": post_sd_var},
        "exact": {"omega_mean": float(ex.omega.mean()),
                  "omega_ci": [float(v) for v in np.percentile(ex.omega, [2.5, 97.5])],
                  "sigma_mean": float(ex.sigma.mean()),
                  "ES_mean": float(ex_es.mean()),
                  "ES_ci": [float(v) for v in np.percentile(ex_es, [2.5, 97.5])],
                  "worst_gibbs_diff": float(worst)},
        "evidence": {"gap_ES_minus_CVaR": float(es_gap),
                     "gap_in_posterior_sd": float(abs(es_gap) / post_sd_es),
                     "draws_reaching_CVaR": n_reach,
                     "al_mean_excess": al_excess,
                     "emp_mean_excess": emp_excess,
                     "excess_ratio": float(emp_excess / al_excess),
                     "al_implied_mean": mu_al,
                     "mean_gap": float(mu_al - emp["mean"]),
                     "var_gap_in_posterior_sd": float(abs(d_var) / post_sd_var),
                     "es_identity_max_err": id_err,
                     "mean_identity_err": float(abs(recovered - fit["cvar_emp"])),
                     "al_mle": fit},
        "sigma_split": {"tau_eff": te, "e_L": float(e_L), "e_U": float(e_U),
                        "n_below": int(emp["k"]), "n_above": int(n - emp["k"]),
                        "sigma_hat": fit["sigma_hat"],
                        "share_lower": float(part_lower / fit["sigma_hat"]),
                        "share_upper": float(part_upper / fit["sigma_hat"]),
                        "al_overshoot": float(fit["sigma_hat"] / (1 - te)),
                        "identity_err": float(split_err)},
        "failures": FAILURES,
    })
    with open("spy_results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)

    print("\nSaved: spy_results.txt, spy_results.json")

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
