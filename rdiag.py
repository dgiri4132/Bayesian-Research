"""
rdiag.py -- the R diagnostic, its closed forms, and the distribution families
I test it on.

R is how much of the real shortfall gap (VaR - CVaR) the AL working likelihood
can represent.

    Original definition:

        R = tau * (ybar_up - CVaR) / (VaR - CVaR),    ybar_up = E[Y | Y > VaR]

    General form (B1). Using mu = (1-tau)*ybar_up + tau*CVaR to get rid of
    ybar_up:

                tau * (mu - CVaR)
        R = ---------------------------
             (1 - tau) * (VaR - CVaR)

    Mean-excess form (B1'). With

        e_L = VaR - CVaR          average distance below VaR
        e_U = ybar_up - VaR       average distance above VaR

    it becomes

        R = tau * (1 + e_U / e_L)

    so R = 1 only when e_U / e_L = (1 - tau) / tau, which is the ratio of the
    AL's two exponential rates. That's the same ratio the uniqueness lemma pins,
    so R compares the data's tail asymmetry with the one the AL assumes.

Signs: Y is a return and I look at the LOWER tail, tau < 1/2,
VaR = F^{-1}(tau), CVaR = E[Y | Y <= VaR], so CVaR < VaR and e_L > 0.

Every family gives mean / var_ / cvar in closed form and R_closed plugs them
into B1.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "R_general", "R_original", "R_mean_excess", "R_sample", "Family",
    "Normal", "StudentT", "Laplace", "Uniform", "Exponential", "LogNormal",
    "NegGPD", "AsymmetricLaplace", "BetaPower",
]


# ---- the three ways of writing R ----

def R_general(mu, var_, cvar, tau):
    """B1: R = tau (mu - CVaR) / [(1-tau)(VaR - CVaR)]."""
    return tau * (mu - cvar) / ((1.0 - tau) * (var_ - cvar))


def R_original(ybar_up, var_, cvar, tau):
    """The original definition: tau (ybar_up - CVaR) / (VaR - CVaR)."""
    return tau * (ybar_up - cvar) / (var_ - cvar)


def R_mean_excess(e_lower, e_upper, tau):
    """B1': R = tau (1 + e_U / e_L)."""
    return tau * (1.0 + e_upper / e_lower)


def R_sample(y, tau, use_effective_tau=True):
    """
    R from a sample.

    A sample splits at an index, not a probability. With n = 2512 and
    tau = 0.05, tau*n = 125.6, so no split gives exactly 0.05. I take
    k = floor(tau*n + 1/2) and, with use_effective_tau, use tau_eff = k/n. At tau_eff
    all three forms agree to machine precision because
    mu = (1-tau_eff) ybar_up + tau_eff CVaR holds exactly in the sample. At the
    nominal tau the general form differs from the other two by a small
    granularity residual (about 2e-5 on SPY).

    Returns all three forms so that residual can be seen.
    """
    from kkgibs import empirical_var_cvar
    e = empirical_var_cvar(y, tau)
    n, k, tau_eff = e["n"], e["k"], e["tau_eff"]
    t = tau_eff if use_effective_tau else tau
    var_, cvar, ybar_up, mu = e["VaR"], e["CVaR"], e["ybar_up"], e["mean"]

    return {
        "n": n, "k": k, "tau_nominal": tau, "tau_eff": tau_eff, "tau_used": t,
        "mu": mu, "VaR": var_, "CVaR": cvar, "ybar_up": ybar_up,
        "e_lower": var_ - cvar, "e_upper": ybar_up - var_,
        "R_general": R_general(mu, var_, cvar, t),
        "R_original": R_original(ybar_up, var_, cvar, t),
        "R_mean_excess": R_mean_excess(var_ - cvar, ybar_up - var_, t),
    }


# ---- families ----

class Family:
    """Base class. Each family gives mean(), var_(tau), cvar(tau)."""

    name = "family"
    label = "family"
    # domain of attraction of the lower tail: 'frechet' (power tail),
    # 'gumbel', or 'weibull' (bounded on the left)
    mda = None
    tail_index = None      # alpha, for frechet

    def mean(self):
        raise NotImplementedError

    def var_(self, tau):
        raise NotImplementedError

    def cvar(self, tau):
        raise NotImplementedError

    # ---- worked out from the three above ----

    def ybar_up(self, tau):
        """E[Y | Y > VaR_tau] = (mu - tau CVaR)/(1-tau)."""
        return (self.mean() - tau * self.cvar(tau)) / (1.0 - tau)

    def R_closed(self, tau):
        tau = np.asarray(tau, dtype=float)
        return R_general(self.mean(), self.var_(tau), self.cvar(tau), tau)

    def __repr__(self):
        return f"<{self.name}>"


class Normal(Family):
    name = "normal"
    label = "Normal"
    mda = "gumbel"

    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = float(loc), float(scale)

    def mean(self):
        return self.loc

    def var_(self, tau):
        return self.loc + self.scale * stats.norm.ppf(tau)

    def cvar(self, tau):
        z = stats.norm.ppf(tau)
        return self.loc - self.scale * stats.norm.pdf(z) / tau


class StudentT(Family):
    mda = "frechet"

    def __init__(self, nu, loc=0.0, scale=1.0):
        assert nu > 1, "mean must exist"
        self.nu, self.loc, self.scale = float(nu), float(loc), float(scale)
        self.name = f"t{int(nu)}"
        self.label = f"Student-t({int(nu)})"
        self.tail_index = float(nu)

    def mean(self):
        return self.loc

    def var_(self, tau):
        return self.loc + self.scale * stats.t.ppf(tau, self.nu)

    def cvar(self, tau):
        q = stats.t.ppf(tau, self.nu)
        f = stats.t.pdf(q, self.nu)
        # standard t result: ES = -( (nu + q^2) / (nu - 1) ) f(q) / tau
        return self.loc - self.scale * (self.nu + q * q) / (self.nu - 1.0) * f / tau


class Laplace(Family):
    name = "laplace"
    label = "Laplace"
    mda = "gumbel"

    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = float(loc), float(scale)

    def mean(self):
        return self.loc

    def var_(self, tau):
        b = self.scale
        return np.where(
            tau < 0.5,
            self.loc + b * np.log(2.0 * tau),
            self.loc - b * np.log(2.0 * (1.0 - tau)),
        )

    def cvar(self, tau):
        q = self.var_(tau)
        b = self.scale
        # tau < 1/2: the tail below q is exponential, so memoryless, CVaR = q - b
        lower = q - b
        # tau >= 1/2: whole lower half plus the part of the upper half below q
        # E[Y 1{Y<q}] = loc*tau - b/2 + (upper-half part)
        u = 2.0 * (1.0 - tau)                      # = exp(-(q-loc)/b) for tau>=1/2
        d = np.where(tau >= 0.5, -b * np.log(np.clip(u, 1e-300, None)), 0.0)
        # mass between loc and q is tau - 1/2, its mean uses a truncated exponential on (0, d)
        m = np.where(tau >= 0.5, b - d * u / np.clip(1.0 - u, 1e-300, None), 0.0)
        upper = np.where(
            tau >= 0.5,
            (0.5 * (self.loc - b) + (tau - 0.5) * (self.loc + m)) / tau,
            0.0,
        )
        return np.where(tau < 0.5, lower, upper)


class Uniform(Family):
    name = "uniform"
    label = "Uniform"
    mda = "weibull"

    def __init__(self, a=0.0, b=1.0):
        self.a, self.b = float(a), float(b)

    def mean(self):
        return 0.5 * (self.a + self.b)

    def var_(self, tau):
        return self.a + tau * (self.b - self.a)

    def cvar(self, tau):
        return self.a + 0.5 * tau * (self.b - self.a)


class Exponential(Family):
    name = "exponential"
    label = "Exponential"
    mda = "weibull"          # lower tail is bounded at 0

    def __init__(self, scale=1.0):
        self.scale = float(scale)

    def mean(self):
        return self.scale

    def var_(self, tau):
        return -self.scale * np.log1p(-tau)

    def cvar(self, tau):
        L = -np.log1p(-tau)
        return self.scale * (tau - (1.0 - tau) * L) / tau


class LogNormal(Family):
    name = "lognormal"
    label = "Lognormal"
    # lower tail is bounded at 0 but F(x) goes to 0 faster than any power of x
    # there, so it is NOT in the Weibull domain (that needs F(x) ~ c x^gamma).
    # it's in the Gumbel domain, same as the normal it comes from
    mda = "gumbel"

    def __init__(self, s=1.0, scale=1.0):
        self.s, self.scale = float(s), float(scale)

    def mean(self):
        return self.scale * np.exp(0.5 * self.s ** 2)

    def var_(self, tau):
        return self.scale * np.exp(self.s * stats.norm.ppf(tau))

    def cvar(self, tau):
        z = stats.norm.ppf(tau)
        return self.mean() * stats.norm.cdf(z - self.s) / tau


class BetaPower(Family):
    """
    Beta(gamma, 1) on [0,1]:  F(x) = x^gamma,  f(x) = gamma x^(gamma-1).

    One parameter that covers the whole Weibull domain and shows the B3
    three-way split at a bounded left endpoint:

        R(tau) = tau * gamma * (1 - tau^(1/gamma)) / ( (1-tau) * tau^(1/gamma) )

    so as tau -> 0,  R -> 0 for gamma > 1 (density goes to 0 at the endpoint),
                     R -> 1 for gamma = 1 (the uniform),
                     R -> inf for gamma < 1 (density blows up at the endpoint).
    """
    mda = "weibull"

    def __init__(self, gamma):
        self.gamma = float(gamma)
        self.name = f"beta{self.gamma:g}".replace(".", "")
        self.label = f"Beta($\\gamma$={self.gamma:g},1)"

    def mean(self):
        return self.gamma / (self.gamma + 1.0)

    def var_(self, tau):
        return np.asarray(tau, dtype=float) ** (1.0 / self.gamma)

    def cvar(self, tau):
        return self.gamma / (self.gamma + 1.0) * self.var_(tau)


class NegGPD(Family):
    """
    Y = -X with X ~ GPD(xi, beta) on [0, inf). The lower tail of Y is the
    Pareto upper tail of X, so this is the family with a real Pareto left tail,
    the one used for the Frechet case in B3(i) and Figure 2. Needs xi < 1 for a finite mean.

        R(tau) = tau (1 - tau^xi) / ( xi (1 - tau) )        (no beta in it)

    As xi -> 0 this becomes tau log(1/tau) / (1 - tau) (an exponential tail),
    and as tau -> 0, R -> tau/xi = alpha*tau, same as the regular-variation result.
    """
    mda = "frechet"          # only for xi > 0. xi = 0 is gumbel, xi < 0 is weibull

    def __init__(self, xi, beta=1.0):
        assert xi < 1.0, "mean must exist"
        self.xi, self.beta = float(xi), float(beta)
        self.name = f"gpd{self.xi:g}".replace(".", "")
        self.label = f"neg-GPD($\\xi$={self.xi:g})"
        self.tail_index = (1.0 / self.xi) if self.xi > 0 else np.inf

    # ---- X-side helpers ----
    def _u(self, tau):
        """upper (1-tau) quantile of X"""
        if abs(self.xi) < 1e-12:
            return -self.beta * np.log(tau)
        return self.beta * (tau ** (-self.xi) - 1.0) / self.xi

    def mean(self):
        return -self.beta / (1.0 - self.xi)

    def var_(self, tau):
        return -self._u(tau)

    def cvar(self, tau):
        u = self._u(tau)
        return -(u + (self.beta + self.xi * u) / (1.0 - self.xi))


class AsymmetricLaplace(Family):
    """
    AL(theta, sigma, p) with P(Y < theta) = p: lower rate (1-p)/sigma, upper
    rate p/sigma. This is the working likelihood itself. At tau = p it gives
    R = 1 by construction; at any other tau it doesn't.
    """
    mda = "gumbel"

    def __init__(self, p, theta=0.0, sigma=1.0):
        self.p, self.theta, self.sigma = float(p), float(theta), float(sigma)
        self.name = f"al{self.p:g}".replace(".", "")
        self.label = f"AL($p$={self.p:g})"

    def mean(self):
        p, s = self.p, self.sigma
        return self.theta - p * s / (1.0 - p) + (1.0 - p) * s / p

    def var_(self, tau):
        p, s, th = self.p, self.sigma, self.theta
        tau = np.asarray(tau, dtype=float)
        lower = th + s * np.log(tau / p) / (1.0 - p)
        upper = th - s * np.log((1.0 - tau) / (1.0 - p)) / p
        return np.where(tau <= p, lower, upper)

    def cvar(self, tau):
        p, s, th = self.p, self.sigma, self.theta
        tau = np.asarray(tau, dtype=float)
        q = self.var_(tau)
        lower = q - s / (1.0 - p)                       # memoryless, tau <= p
        d = np.clip(q - th, 0.0, None)                  # tau > p
        upper = th + (-p * s / (1.0 - p)
                      + (tau - p) * s / p
                      - d * (1.0 - tau)) / np.where(tau > 0, tau, 1.0)
        return np.where(tau <= p, lower, upper)
