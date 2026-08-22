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
    
    These are the things I plan to code in this file which hopefully I can finish over
    the weekend and """

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, t

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

def gibbs_qr_sampler(X,y,p,n_iter,burn_in,seed,
                     omega_init, sigma_init,
                     omega_p0=None, B_p0 = None, n0=4.0, s0=2.0):
    rng = np.random.default_rng(seed)
    n, k =X.shape
    theta_c = (1-2*p)/(p*(1-p))
    tau2_c = 2/(p*(1-p))
    if omega_p0 is None:
        omega_p0 = np.zeros(k)
    if B_p0 is None:
        B_p0 = np.eye(k)
    B_p0_inv = np.linalg.inv(B_p0)
    omega = np.array(omega_init, dtype = float).reshape(k)
    sigma = float(sigma_init)
    omega_draws = np.zeros((n_iter, k))
    sigma_draws = np.zeros(n_iter)

    for it in range(n_iter):
        resid = y-X@omega
        delta2 = np.maximum(resid **2/(tau2_c * sigma), 1e-12)
        delta = np.sqrt(delta2)
        gamma2 = 2/sigma + theta_c**2/(tau2_c*sigma)
        gamma = np.sqrt(gamma2)
       
        w = rng.wald(gamma / delta, gamma2)
        v = np.clip(1.0 / w, 1e-8, None)

        Xz = X / (tau2_c * sigma * v)[:, None]
        B_p_inv = X.T @ Xz + B_p0_inv
        rhs = X.T @ ((y - theta_c * v) / (tau2_c * sigma * v)) + B_p0_inv @ omega_p0
        B_p = np.linalg.inv(B_p_inv)
        omega_hat = B_p @ rhs
        omega = rng.multivariate_normal(omega_hat, B_p)

        resid2 = y - X @ omega - theta_c * v
        n_tilde = n0 + 3*n
        s_tilde = s0 + 2*np.sum(v) + np.sum(resid2**2 / (tau2_c * v))
        g = rng.gamma(shape=n_tilde/2, scale=1.0/(s_tilde/2))
        sigma = 1.0 / g

        omega_draws[it] = omega
        sigma_draws[it] = sigma

    return omega_draws[burn_in:], sigma_draws[burn_in:]



def split_rhat(chains):
    """Gelman-Rubin R-hat on split chains (each chain halved -> 2m sequences).
    chains: list of 1-D arrays, equal length."""
    halves = []
    for c in chains:
        n = len(c) // 2
        halves.append(c[:n]); halves.append(c[n:2*n])
    seqs = np.array(halves)              # (2m, n)
    m, n = seqs.shape
    means = seqs.mean(axis=1)
    W = seqs.var(axis=1, ddof=1).mean()
    B = n * means.var(ddof=1)
    V_hat = (n - 1)/n * W + B/n
    return np.sqrt(V_hat / W)


def ess(x, max_lag=1000):
    """Effective sample size via truncated autocorrelation (first-negative rule),
    consistent with kk_gibbs.inefficiency_factor."""
    x = x - x.mean()
    n = len(x)
    var0 = np.dot(x, x) / n
    if var0 == 0:
        return np.nan
    IF = 1.0
    for k in range(1, max_lag):
        c = np.dot(x[:-k], x[k:]) / n / var0
        if c < 0:
            break
        IF += 2 * c
    return n / IF

def frequentist_baseline(r, alpha):
    var_hist = np.quantile(r, alpha)
    cvar_hist = r[r<=var_hist].mean()

    mu_p,std_p = r.mean(), r.std(ddof = 1)
    var_norm = norm.ppf(alpha, mu_p, std_p)
    cvar_norm = mu_p - std_p*norm.pdf(norm.ppf(alpha))/alpha

    df_t, loc_t, sc_t = t.fit(r)
    var_t = t.ppf(alpha, df_t, loc_t, sc_t)
    q = t.ppf(alpha, df_t)
    cvar_t_std = -(t.pdf(q, df_t)/alpha) * (df_t + q**2)/(df_t-1)
    cvar_t = loc_t+ sc_t*cvar_t_std
    return {"Historical":(var_hist,cvar_hist),
            "Normal":(var_norm,cvar_norm),
            "Student-t":(var_t,cvar_t)}

def main():
    r = load_data()
    n = len(r)
    y = r
    X = np.ones((n, 1))                      

    print(f"\nn = {n} observations | tau = {TAU} | "
          f"{N_CHAINS} chains x {N_ITER} iter ({BURN_IN} burn-in)")

    freq = frequentist_baseline(r, TAU)

    
    q_emp = np.quantile(r, TAU)              
    sig_rough = 0.01                          
    omega_inits = [q_emp, q_emp + 0.02, q_emp - 0.02, 0.0]
    sigma_inits = [sig_rough, 5*sig_rough, 0.2*sig_rough, 10*sig_rough]

    theta_chains, sigma_chains = [], []
    for c in range(N_CHAINS):
        om, sg = gibbs_qr_sampler(
            X, y, TAU, N_ITER, BURN_IN, seed=BASE_SEED + 111*c,
            omega_init=[omega_inits[c]], sigma_init=sigma_inits[c],
            n0=N0, s0=S0)
        theta_chains.append(om[:, 0])
        sigma_chains.append(sg)
        print(f"  chain {c+1}: start(theta={omega_inits[c]:+.4f}, "
              f"sigma={sigma_inits[c]:.4f}) -> "
              f"post mean theta={om[:,0].mean():+.5f}, sigma={sg.mean():.5f}")

    # ---- diagnostics ----
    rhat_theta = split_rhat(theta_chains)
    rhat_sigma = split_rhat(sigma_chains)
    ess_theta = sum(ess(c) for c in theta_chains)
    ess_sigma = sum(ess(c) for c in sigma_chains)
    print(f"\nR-hat: theta={rhat_theta:.5f}, sigma={rhat_sigma:.5f} "
          f"(target < {RHAT_TARGET})")
    print(f"ESS  : theta={ess_theta:,.0f}, sigma={ess_sigma:,.0f} "
          f"(of {N_CHAINS*(N_ITER-BURN_IN):,} retained)")
    converged = max(rhat_theta, rhat_sigma) < RHAT_TARGET
    print("Convergence:", "PASS" if converged else "FAIL -- do not use results")

    # ---- pooled posterior ----
    theta_all = np.concatenate(theta_chains)
    sigma_all = np.concatenate(sigma_chains)
    var_draws = theta_all                          
    es_draws  = theta_all - sigma_all / (1 - TAU)  

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

    # ---- save numerical results ----
    with open("spy_results.txt", "w") as f:
        f.write(f"n={n}, tau={TAU}, chains={N_CHAINS}, iter={N_ITER}, burn={BURN_IN}\n")
        f.write(f"priors: sigma~IG(2,0.012), omega~N(0,1)\n")
        f.write(f"R-hat theta={rhat_theta:.5f} sigma={rhat_sigma:.5f}  "
                f"ESS theta={ess_theta:.0f} sigma={ess_sigma:.0f}\n\n")
        for name, (v, c_) in freq.items():
            f.write(f"{name:<22}VaR={v:.4f}  ES={c_:.4f}\n")
        f.write(f"{'Bayesian':<22}VaR={var_draws.mean():.4f} "
                f"[{var_ci[0]:.4f},{var_ci[1]:.4f}]  "
                f"ES={es_draws.mean():.4f} [{es_ci[0]:.4f},{es_ci[1]:.4f}]\n")

   
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(es_draws, bins=80, density=True, color="steelblue", alpha=0.65,
            edgecolor="white", label="ES posterior (Bayesian)")
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
    fig.tight_layout(); fig.savefig("fig_es_posterior.png", dpi=150); plt.close(fig)


    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for c in range(N_CHAINS):
        axes[0].plot(theta_chains[c][::10], lw=0.4, alpha=0.7, label=f"chain {c+1}")
        axes[1].plot(sigma_chains[c][::10], lw=0.4, alpha=0.7)
    axes[0].set_ylabel(r"$\theta$ (VaR)"); axes[1].set_ylabel(r"$\sigma$")
    axes[1].set_xlabel("retained iteration / 10")
    axes[0].set_title(f"Trace plots, {N_CHAINS} chains "
                      f"(R-hat: {rhat_theta:.4f} / {rhat_sigma:.4f})")
    axes[0].legend(fontsize=7, ncol=4)
    fig.tight_layout(); fig.savefig("fig_traces.png", dpi=150); plt.close(fig)


    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(var_draws, bins=80, density=True, color="slategray", alpha=0.65,
            edgecolor="white", label=r"VaR posterior ($\theta$)")
    for (name, (v, c_)), col in zip(freq.items(), ["red", "green", "orange"]):
        ax.axvline(v, color=col, ls="--", lw=1.6, label=f"{name} VaR ({v:.4f})")
    ax.set_xlabel("Daily return"); ax.set_ylabel("Posterior density")
    ax.set_title("Posterior distribution of 5% VaR, SPY, with frequentist points")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("fig_var_posterior.png", dpi=150); plt.close(fig)

    print("\nSaved: spy_results.txt, fig_es_posterior.png, fig_traces.png, "
          "fig_var_posterior.png")

if __name__ == "__main__":
    main()

