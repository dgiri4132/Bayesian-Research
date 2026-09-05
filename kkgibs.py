"""
This is the shared core for the Bayesian CVaR project since the revision of my paper.

It is a single module so that spy_run1.py, experiment_A.py, experiment_B.py,
experiment_C.py and rdiag.py cannot drift apart: there is exactly one sampler,
one set of diagnostics, one set of AL formulas, and one definition of the
empirical estimators in the whole project.

Contents
    AL distribution          al_pdf / al_cdf / al_ppf / al_mean / al_cvar / al_es
                             al_es_via_mean   (the second route, for the check)
    AL fit                   al_mle           (the exact maximum-likelihood fit)
    empirical estimators     empirical_var_cvar  (index convention made explicit here)
    the sampler              gibbs_qr_sampler    (Kozumi-Kobayashi, sigma extension)
    diagnostics              split_rhat / ess / mcse
    the exact posterior      exact_posterior     (sigma integrated out analytically)
    environment              env_stamp / set_numeric_guards

    Of the AL block, al_pdf / al_cdf / al_ppf / al_cvar currently have no call
    site in this project; they are kept because they are the canonical interface
    a replicator will reach for, and because al_cvar is the only route to CVaR at
    a level other than tau.  All four are verified against numerical integration
    (al_cvar to a relative error of 5e-10 across levels 0.01 to 0.90, both
    branches).  They are correct but unused -- do not treat them as tested by the
    paper's own runs, because they are not.

CONVENTIONS.

    tau         the quantile level, chosen not estimated.  This module is general
                in tau; the paper's runs use tau = 0.05, and experiment_B sweeps
                0.01 to 0.25.  Kozumi-Kobayashi call this p.  It is different
                from the mixture constant, which is tau2_c below.
    omega       the location = the tau-quantile = VaR.  K-K call this beta_p.
    sigma       the AL scale.
    theta_c,
    tau2_c      the K-K mixture constants (18.947, 42.105 at tau = 0.05).  These
                are the symbols K-K write as theta and tau^2.  They have nothing
                to do with tau or with omega.

    Lower tail throughout.  For a return series VaR and CVaR are negative and
    CVaR < VaR.

THE AL's ES CONVERSION, which is what the paper is about.

    Under AL(omega, sigma, tau) the exact expected shortfall is

        ES = omega - sigma/(1-tau)                                        (Proof 1)

    and this is algebraically identical to

        ES = mu_AL - sigma/tau                                            (identity)

    where mu_AL is the AL's own mean.  Both routes are implemented -- `al_es` and
    `al_es_via_mean` -- and spy_run1.py section 5 CALLS BOTH on the same posterior
    draws and asserts they agree, so the identity is verified at run time rather
    than asserted.  That claim is only true while spy_run1.py calls
    `al_es_via_mean` rather than re-typing the formula inline; an inline copy is
    not a test of this function.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import numpy as np

#  numpy renamed trapz -> trapezoid in 2.0.  Support both so the module runs on
#  the numpy 1.26 build the earlier results were produced on as well as on 2.x.
_trapz = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "al_pdf", "al_cdf", "al_ppf", "al_mean", "al_cvar", "al_es",
    "al_es_via_mean", "al_rvs", "rho_tau", "split_index",
    "al_mle", "empirical_var_cvar", "gibbs_qr_sampler", "GibbsResult",
    "split_rhat", "ess", "mcse", "exact_posterior", "ExactPosterior",
    "env_stamp", "set_numeric_guards", "mixture_constants",
]


def set_numeric_guards() -> None:
    """Turn silent numerical failures into exceptions.

    divide / invalid / overflow raise; underflow is left alone, because
    underflow to zero in the AL likelihood tails is benign and expected.
    """
    np.seterr(divide="raise", invalid="raise", over="raise", under="ignore")


def env_stamp() -> dict:
    import scipy
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def print_env_stamp(prefix: str = "  ") -> dict:
    e = env_stamp()
    print(f"{prefix}python {e['python']} | numpy {e['numpy']} | scipy {e['scipy']}")
    print(f"{prefix}{e['platform']} ({e['machine']})")
    return e

# the asymmetric Laplace

def split_index(tau: float, n: int) -> int:
    
    k = int(np.floor(tau * n + 0.5))
    return min(max(k, 1), n - 1)


def mixture_constants(tau: float) -> tuple[float, float]:
    """K-K's theta and tau^2 -- the location-scale mixture constants."""
    return (1 - 2 * tau) / (tau * (1 - tau)), 2.0 / (tau * (1 - tau))


def al_pdf(y, omega, sigma, tau):
    y = np.asarray(y, dtype=float)
    u = (y - omega) / sigma
    rho = np.where(u >= 0, tau * u, (tau - 1.0) * u)
    return tau * (1 - tau) / sigma * np.exp(-rho)


def al_cdf(y, omega, sigma, tau):
    y = np.asarray(y, dtype=float)
    d = (y - omega) / sigma
    lo = tau * np.exp((1 - tau) * np.minimum(d, 0.0))
    hi = 1.0 - (1 - tau) * np.exp(-tau * np.maximum(d, 0.0))
    return np.where(y <= omega, lo, hi)


def al_ppf(q, omega, sigma, tau):
    """
    Quantile function.  q = 0 and q = 1 return -inf and +inf rather than the
    finite garbage a clipped logarithm would give (-724 and +13815 at the old
    1e-300 clip), which is silently wrong and hard to notice downstream.
    """
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = omega + sigma * np.log(np.where(q > 0, q, np.nan) / tau) / (1 - tau)
        hi = omega - sigma * np.log(np.where(q < 1, (1 - q) / (1 - tau), np.nan)) / tau
    out = np.where(q <= tau, lo, hi)
    out = np.where(q <= 0.0, -np.inf, out)
    out = np.where(q >= 1.0, np.inf, out)
    return out[()] if out.ndim == 0 else out


def al_mean(omega, sigma, tau):
    """E[Y] under AL(omega, sigma, tau)."""
    return omega - tau * sigma / (1 - tau) + (1 - tau) * sigma / tau


def al_cvar(omega, sigma, tau, level=None):
    
    if level is None:
        return omega - sigma / (1 - tau)
    level = np.asarray(level, dtype=float)
    q = al_ppf(level, omega, sigma, tau)
    return np.where(level <= tau, q - sigma / (1 - tau),
                    _al_cvar_upper(q, omega, sigma, tau, level))


def _al_cvar_upper(q, omega, sigma, tau, level):
    d = np.clip(q - omega, 0.0, None)
    return omega + (-tau * sigma / (1 - tau)
                    + (level - tau) * sigma / tau
                    - d * (1 - level)) / level


def al_es(omega, sigma, tau):
    """Proof 1: ES = omega - sigma/(1-tau).  Identical to al_mean - sigma/tau."""
    return omega - sigma / (1 - tau)


def al_es_via_mean(omega, sigma, tau):
    
    return al_mean(omega, sigma, tau) - sigma / tau


def al_rvs(n, omega, sigma, tau, rng):
    u = rng.uniform(size=n)
    return np.where(u <= tau,
                    omega + sigma * np.log(u / tau) / (1 - tau),
                    omega - sigma * np.log((1 - u) / (1 - tau)) / tau)


def rho_tau(u, tau):
    """The check (pinball) loss."""
    u = np.asarray(u, dtype=float)
    return np.where(u >= 0, tau * u, (tau - 1.0) * u)


def al_mle(y, tau, index_convention=True):
    """
    Maximum-likelihood fit of AL(omega, sigma, .) to a sample, AT tau_eff = k/n.

    Read that carefully: with `index_convention=True` (the default) this is the
    exact MLE at the sample's own split level tau_eff, not at the nominal tau.
    The distinction is small -- on SPY at tau = 0.05, sigma_hat = 0.0014383 here
    against 0.0014354 for the true nominal-tau MLE, a difference of 0.2% -- but a
    table headed "AL MLE sigma_hat" carries that 0.2%, so the label has to say
    which.  Both are returned: `sigma_hat` (at tau_eff) and `sigma_hat_nominal`.

    The identities below hold at tau_eff and only at tau_eff, which is why it is
    the default.

    omega_hat is the tau-quantile (any minimiser of the check loss); sigma_hat is
    the mean check loss at that point.  Returns the value at the sample's own
    split index, which is where the algebra below is exact.

    THE MEAN IDENTITY, which the paper uses.  With k = round(tau*n),
    tau_eff = k/n, omega_hat = y_(k) and sigma_hat = mean(rho_tau(y - omega_hat)):

        sigma_hat = tau_eff (1 - tau_eff) (ybar_up - CVaR_emp)
        ybar - sigma_hat/tau_eff = CVaR_emp        exactly, to machine precision

    That is: substituting the SAMPLE mean for the model's mean in ES = mu - sigma/tau
    recovers the empirical CVaR exactly.  The ES error is therefore exactly the
    model's error in the mean.  `al_mle` returns the pieces so a caller can check it.
    """
    y = np.sort(np.asarray(y, dtype=float))
    n = y.size
    k = split_index(tau, n)
    tau_eff = k / n if index_convention else tau
    omega_hat = y[k - 1]
    sigma_hat = float(np.mean(rho_tau(y - omega_hat, tau_eff)))
    return {
        "omega_hat": float(omega_hat), "sigma_hat": sigma_hat,
        "n": n, "k": k, "tau_eff": tau_eff, "tau_nominal": float(tau),
        "ybar": float(y.mean()),
        "cvar_emp": float(y[:k].mean()),
        "ybar_up": float(y[k:].mean()),
        #  the MLE at the NOMINAL tau, for callers who want that label instead
        "sigma_hat_nominal": float(np.mean(rho_tau(y - omega_hat, tau))),
    }

# empirical estimators for the research project


def empirical_var_cvar(y, tau, index_convention=True):
    
    y = np.sort(np.asarray(y, dtype=float))
    n = y.size
    k = split_index(tau, n)
    return {
        "n": n, "k": k, "tau_eff": k / n, "tau_nominal": float(tau),
        "VaR": float(y[k - 1]),
        "CVaR": float(y[:k].mean()),
        "ybar_up": float(y[k:].mean()),
        "mean": float(y.mean()),
        "VaR_interp": float(np.quantile(y, tau)),
        "CVaR_interp": float(y[y <= np.quantile(y, tau)].mean()),
        "mean_excess_lower": float(y[k - 1] - y[:k].mean()),
    }


# the sampler

@dataclass
class GibbsResult:
    omega: np.ndarray                 # (n_kept, k)
    sigma: np.ndarray                 # (n_kept,)
    seed: int
    n_iter: int
    burn_in: int
    
    delta_floor_hits: int = 0
    v_clip_hits: int = 0
    n_obs: int = 0
    n_par: int = 0
    diagnostics: dict = field(default_factory=dict)



DELTA2_FLOOR = 1e-12
V_FLOOR = 1e-8


def gibbs_qr_sampler(X, y, p, n_iter, burn_in, seed,
                     omega_init, sigma_init,
                     omega_p0=None, B_p0=None, n0=4.0, s0=0.024,
                     count_guards=True,
                     delta2_floor=DELTA2_FLOOR, v_floor=V_FLOOR):
    """
    Kozumi-Kobayashi Gibbs sampler for Bayesian quantile regression, with the
    sigma extension.  General in the number of columns of X.

    Blocks, in this order (the order fixes the random stream -- do not reorder):
        v      | omega, sigma      GIG(1/2, .), drawn as the reciprocal of a Wald
        omega  | v, sigma          multivariate normal, K-K eq. (11)
        sigma  | omega, v          inverse gamma, K-K eq. (13)

    Priors:  omega ~ N(omega_p0, B_p0),  sigma ~ IG(n0/2, s0/2).

    The default s0 is 0.024, matching the module-level constant used for the
    paper's runs.  (The original file defaulted to 2.0 while passing 0.024
    explicitly -- harmless there, a trap for any new caller, so the default is
    now the value that is actually used.)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, k = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries")
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")
    if burn_in >= n_iter:
        raise ValueError("burn_in must be < n_iter")

    rng = np.random.default_rng(seed)
    theta_c, tau2_c = mixture_constants(p)

    if omega_p0 is None:
        omega_p0 = np.zeros(k)
    omega_p0 = np.asarray(omega_p0, dtype=float).reshape(k)
    if B_p0 is None:
        B_p0 = np.eye(k)
    B_p0 = np.atleast_2d(np.asarray(B_p0, dtype=float))
    if B_p0.shape != (k, k):
        raise ValueError(f"B_p0 must be {k}x{k}, got {B_p0.shape}")
    B_p0_inv = np.linalg.inv(B_p0)

    omega = np.array(omega_init, dtype=float).reshape(k)
    sigma = float(sigma_init)
    if sigma <= 0:
        raise ValueError("sigma_init must be positive")

    omega_draws = np.zeros((n_iter, k))
    sigma_draws = np.zeros(n_iter)
    delta_hits = v_hits = 0

    for it in range(n_iter):
        resid = y - X @ omega
        raw = resid ** 2 / (tau2_c * sigma)
        delta2 = np.maximum(raw, delta2_floor)
        if count_guards:
            delta_hits += int(np.count_nonzero(raw < delta2_floor))
        delta = np.sqrt(delta2)
        gamma2 = 2 / sigma + theta_c ** 2 / (tau2_c * sigma)
        gamma = np.sqrt(gamma2)
        wald_mean = gamma / delta
        if not np.all(np.isfinite(wald_mean)) or np.any(wald_mean <= 0):
            raise FloatingPointError(
                f"non-positive or non-finite Wald mean at iteration {it}")
        w = rng.wald(wald_mean, gamma2)
        v_raw = 1.0 / w
        v = np.clip(v_raw, v_floor, None)
        if count_guards:
            v_hits += int(np.count_nonzero(v_raw < v_floor))

        Xz = X / (tau2_c * sigma * v)[:, None]
        B_p_inv = X.T @ Xz + B_p0_inv
        rhs = X.T @ ((y - theta_c * v) / (tau2_c * sigma * v)) + B_p0_inv @ omega_p0
        B_p = np.linalg.inv(B_p_inv)
        omega_hat = B_p @ rhs
        omega = rng.multivariate_normal(omega_hat, B_p)

        resid2 = y - X @ omega - theta_c * v
        n_tilde = n0 + 3 * n
        s_tilde = s0 + 2 * np.sum(v) + np.sum(resid2 ** 2 / (tau2_c * v))
        g = rng.gamma(shape=n_tilde / 2, scale=1.0 / (s_tilde / 2))
        sigma = 1.0 / g

        omega_draws[it] = omega
        sigma_draws[it] = sigma

    return GibbsResult(
        omega=omega_draws[burn_in:], sigma=sigma_draws[burn_in:],
        seed=int(seed), n_iter=int(n_iter), burn_in=int(burn_in),
        delta_floor_hits=delta_hits, v_clip_hits=v_hits, n_obs=n, n_par=k,
    )



# diagnostics

def split_rhat(chains) -> float:
    """
    Split-chain Gelman-Rubin R-hat.  Each chain is halved, giving 2m sequences.

    Returns +inf rather than nan when the within-chain variance is zero (a stuck
    chain).  nan would silently PASS a `rhat< target` gate, because every
    comparison with nan is False; +inf fails it, which is the safedirection.
    """
    halves = []
    for c in chains:
        c = np.asarray(c, dtype=float).ravel()
        h = len(c) // 2
        halves.append(c[:h])
        halves.append(c[h:2 * h])
    seqs = np.array(halves)
    n_seq, n = seqs.shape
    if n < 2:
        return float("inf")
    W = seqs.var(axis=1, ddof=1).mean()
    B = n * seqs.mean(axis=1).var(ddof=1)
    if not np.isfinite(W) or W <= 0:
        return float("inf")
    V_hat = (n - 1) / n * W + B / n
    r = float(np.sqrt(V_hat / W))
    return r if np.isfinite(r) else float("inf")


def ess(x, max_lag=1000) -> float:
    """
    Effective sample size for one chain, truncated autocorrelation with the
    first-negative rule.  Consistent with the inefficiency factor K-K report,
    so that IF = n/ESS is directly comparable to your respectiveTable 1.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    n = len(x)
    var0 = np.dot(x, x) / n
    if var0 <= 0:
        return float("nan")
    IF = 1.0
    for k in range(1, min(max_lag, n - 1)):
        c = np.dot(x[:-k], x[k:]) / n / var0
        if c < 0:
            break
        IF += 2 * c
    return n / IF


def mcse(chains, ess_total=None) -> float:
    """
    Monte Carlo standard error of the pooled posterior mean.

    This is the number that says how many reported digits are real.  Without it,
    a table of five-decimal posterior means is an unsupported claim.
    """
    pooled = np.concatenate([np.asarray(c).ravel() for c in chains])
    if ess_total is None:
        ess_total = sum(ess(c) for c in chains)
    if not np.isfinite(ess_total) or ess_total <= 0:
        return float("nan")
    return float(pooled.std(ddof=1) / np.sqrt(ess_total))



# the exact posterior -- sigma integrated out analytically

@dataclass
class ExactPosterior:
    omega: np.ndarray
    sigma: np.ndarray
    grid: np.ndarray
    log_post: np.ndarray
    edge_log_drop: float
    grid_points: int
    refinement_max_rel_change: float
    points_per_posterior_sd: float = float("nan")

    @property
    def es(self):
        return None  # computed by the caller, which owns tau


def _check_loss_total(grid, y_sorted, cum, tau):
    """
    S(omega) = sum_i rho_tau(y_i - omega), vectorised over a grid of omega.

    With y sorted and C the cumulative sum, for j = #{y_i < omega}:
        sum_{y<omega} (omega - y) = j*omega - C[j]
        sum_{y>=omega} (y - omega) = (C[n] - C[j]) - (n-j)*omega
    so the whole grid costs one searchsorted, not one pass per grid point.
    """
    n = y_sorted.size
    j = np.searchsorted(y_sorted, grid, side="left")
    lower = j * grid - cum[j]
    upper = (cum[n] - cum[j]) - (n - j) * grid
    return tau * upper + (1.0 - tau) * lower


def exact_posterior(y, tau, n_draws, seed, omega_p0=0.0, B_p0=1.0,
                    n0=4.0, s0=0.024, grid_points=400_001,
                    target_log_drop=80.0, verify_refinement=True):
    """
    Exact i.i.d. draws from the posterior of the INTERCEPT-ONLY model, with zero
    Monte Carlo error in the MCMC sense.

        L(omega, sigma) = (tau(1-tau))^n sigma^-n exp(-S(omega)/sigma),
            S(omega) = sum_i rho_tau(y_i - omega)

        prior:  omega ~ N(omega_p0, B_p0),  sigma ~ IG(n0/2, s0/2)

        joint posterior
            propto sigma^-(n + n0/2 + 1) exp(-(S(omega) + s0/2)/sigma) * N(omega)

        hence  sigma | omega, y ~ IG(n + n0/2,  S(omega) + s0/2)          [exact]

        and integrating sigma out with int sigma^-(a+1) e^{-b/sigma} dsigma
        = Gamma(a) b^-a,

            log p(omega | y) = -(omega - omega_p0)^2/(2 B_p0)
                               - (n + n0/2) log(S(omega) + s0/2) + const

    So the posterior is ONE dimensional and can be evaluated to machine precision
    on a grid.  Draws: inverse-CDF on omega, then the exact inverse gamma for
    sigma | omega.  The result is an independent sample, so it isolates the
    question "is the miscentring an artefact of the MCMC?" completely, there is
    no MCMC.

    The grid is expanded until the log-density at both edges has dropped
    `target_log_drop` below the mode (e^-80 ~ 1e-35 of the peak), and, when
    `verify_refinement`, the whole computation is repeated at double resolution
    and the summaries compared, so grid error is measured rather than assumed.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    y_sorted = np.sort(y)
    cum = np.concatenate([[0.0], np.cumsum(y_sorted)])
    a_post = n + n0 / 2.0

    def log_post_of(grid):
        S = _check_loss_total(grid, y_sorted, cum, tau)
        return (-(grid - omega_p0) ** 2 / (2.0 * B_p0)
                - a_post * np.log(S + s0 / 2.0))

    centre = float(np.quantile(y, tau))
    spread = float(np.std(y)) / np.sqrt(n)
    half = max(50.0 * spread, 1e-9)

    def drop_at(c, h):
        g = np.linspace(c - h, c + h, 4001)
        lp = log_post_of(g)
        return lp.max() - max(lp[0], lp[-1]), float(g[int(np.argmax(lp))])

    for _ in range(200):
        d, centre = drop_at(centre, half)
        if d >= target_log_drop:
            break
        half *= 1.8
    else:
        raise RuntimeError("could not bracket the exact posterior")

    #  contract while the criterion still holds, so the bracket is set by the
    #  posterior and never by the data scale
    for _ in range(200):
        d, c_new = drop_at(centre, half / 1.5)
        if d < target_log_drop:
            break
        half /= 1.5
        centre = c_new

    def build(m):
        grid = np.linspace(centre - half, centre + half, m)
        lp = log_post_of(grid)
        lp -= lp.max()
        dens = np.exp(lp)
        cdf = np.concatenate([[0.0], np.cumsum(
            0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
        cdf /= cdf[-1]
        return grid, lp, dens, cdf

    #  Resolution is escalated, not merely asserted.  At very small n the marginal
    #  posterior has POLYNOMIAL tails, ~|omega|^-(n + n0/2), so the log-drop
    #  criterion legitimately pushes the bracket far out and the default grid can
    #  fall below 20 points per posterior sd (n = 10 gives 18.7).  That is a
    #  resolution problem with an obvious answer: use more points.
    used_points = int(grid_points)
    for _ in range(6):
        grid, lp, dens, cdf = build(used_points)
        step = float(grid[1] - grid[0])
        m1 = _trapz(grid * dens, grid) / _trapz(dens, grid)
        m2 = _trapz((grid - m1) ** 2 * dens, grid) / _trapz(dens, grid)
        post_sd = float(np.sqrt(max(m2, 0.0)))
        pts_per_sd = post_sd / step if step > 0 else np.inf
        if pts_per_sd >= 20.0:
            break
        used_points = (used_points - 1) * 4 + 1
    else:
        raise RuntimeError(
            f"exact_posterior: grid resolves only {pts_per_sd:.1f} points per "
            f"posterior sd after escalating to {used_points:,} points.  Check "
            f"that the prior and data scales are compatible.")
    grid_points = used_points

    #  resolution guard.  The only way to know the grid resolves the posterior is
    #  to measure it against the posterior's own scale.  This is the check that
    #  would have caught the bracketing failure above without needing the
    #  refinement pass, so it always runs.
    rel = 0.0
    if verify_refinement:
        g2, lp2, d2, c2 = build(2 * grid_points - 1)
        for q in (0.025, 0.5, 0.975):
            a = np.interp(q, cdf, grid)
            b = np.interp(q, c2, g2)
            rel = max(rel, abs(a - b) / max(abs(a), 1e-12))
        m1 = _trapz(grid * dens, grid) / _trapz(dens, grid)
        m2 = _trapz(g2 * d2, g2) / _trapz(d2, g2)
        rel = max(rel, abs(m1 - m2) / max(abs(m1), 1e-12))

    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n_draws)
    #  strictly increasing cdf is required by np.interp's inverse use
    keep = np.concatenate([[True], np.diff(cdf) > 0])
    om = np.interp(u, cdf[keep], grid[keep])

    S_at = _check_loss_total(om, y_sorted, cum, tau)
    b_post = S_at + s0 / 2.0
    sg = b_post / rng.gamma(shape=a_post, scale=1.0, size=n_draws)

    return ExactPosterior(
        omega=om, sigma=sg, grid=grid, log_post=lp,
        edge_log_drop=float(lp.max() - max(lp[0], lp[-1])),
        grid_points=int(grid_points),
        refinement_max_rel_change=float(rel),
        points_per_posterior_sd=float(pts_per_sd),
    )
