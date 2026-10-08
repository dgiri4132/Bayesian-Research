#!/usr/bin/env python3
"""
/fig_eta_tau.py/ -- the R(tau) figure for Experiment B.

This makes the one figure that shows the whole R story at a glance.

Panel (a)  R against tau over the range people actually use for risk, with R on
           a normal scale. The AL sits flat at 1 by construction, and every
           family with an unbounded tail drops away from it as tau goes into
           the tail.

Panel (b)  the same thing on log-log axes over eight decades of tau, so you can
           see where R ends up as tau -> 0. Frechet and Gumbel families go to 0,
           a bounded endpoint with a positive density levels off at a constant,
           and a density that blows up at the endpoint goes to infinity. The
           lognormal is bounded at 0 too, but its tail there is Gumbel-type, so
           it goes to 0 (it is NOT the gamma > 1 Weibull case).

Every line has its own linestyle as well as its own colour, so you can tell
them apart without colour.

/writes fig_eta_tau.png, fig_eta_tau.pdf, eta_tau_table.csv/
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # real fonts in the pdf, not Type 3
import matplotlib.pyplot as plt
import numpy as np

from rdiag import (Normal, StudentT, Laplace, Uniform, Exponential,
                   LogNormal, NegGPD, BetaPower)

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
     "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"

# family, colour, linestyle, label
PANEL_A = [
    (Normal(),      C["blue"],    "-",             "Normal"),
    (StudentT(5),   C["orange"],  (0, (5, 2)),     r"$t(5)$"),
    (StudentT(3),   C["aqua"],    (0, (1, 1.6)),   r"$t(3)$"),
    (Laplace(),     C["violet"],  (0, (7, 2, 1, 2)), "Laplace"),
    (NegGPD(0.30),  C["red"],     "-",             r"neg-GPD $\xi=0.3$"),
    (LogNormal(),   C["green"],   (0, (6, 2)),     "Lognormal"),
    (Uniform(),     C["magenta"], (0, (2, 2)),     "Uniform"),
]
# families for the log-log panel: three that go to 0, two that level off,
# one that blows up
PANEL_B = [
    (StudentT(3),      C["aqua"],    (0, (1, 1.6)), "$t(3)$  (Fréchet, " + r"$\eta\sim\alpha\tau$)"),
    (Normal(),         C["blue"],    "-",           r"Normal  (Gumbel, $\eta\sim\tau z_\tau^2$)"),
    (LogNormal(),      C["green"],   (0, (6, 2)),   r"Lognormal  (endpoint, Gumbel-type)"),
    (Uniform(),        C["magenta"], (0, (2, 2)),   r"Uniform  (endpoint, $\gamma=1$)"),
    (Exponential(),    C["yellow"],  (0, (7, 2, 1, 2)), r"Exponential  (endpoint, $\gamma=1$)"),
    (BetaPower(0.5),   C["orange"],  (0, (4, 1, 1, 1)), r"Beta$(0.5,1)$  (endpoint, $\gamma<1$)"),
]


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _legend(ax, loc, ncol=1, title=None, bbox=None, fs=8.5):
    lg = ax.legend(loc=loc, ncol=ncol, frameon=True, fontsize=fs,
                   bbox_to_anchor=bbox,
                   title=title, borderpad=0.6, handlelength=2.6,
                   labelspacing=0.45, framealpha=1.0)
    lg.get_frame().set_edgecolor(GRID)
    lg.get_frame().set_linewidth(0.6)
    for t in lg.get_texts():
        t.set_color(INK)
    if lg.get_title():
        lg.get_title().set_color(INK2)
        lg.get_title().set_fontsize(8.5)
    return lg


def main() -> None:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.4, 6.2))
    fig.patch.set_facecolor("white")

    # ---- panel (a) ----
    ta = np.logspace(-3, np.log10(0.5), 500)
    _style(axa)
    axa.axvspan(0.01, 0.05, color="#f4f3ee", zorder=0)
    axa.text(0.0224, 0.42, "$\\tau$ used\nin risk work", ha="center", va="center",
             fontsize=8, color=INK2, zorder=5)

    axa.axhline(1.0, color=INK, lw=1.8, ls=(0, (6, 3)), zorder=3,
                label="AL working likelihood")
    for fam, col, ls, lab in PANEL_A:
        axa.plot(ta, np.atleast_1d(fam.R_closed(ta)), color=col, ls=ls, lw=2.0,
                 zorder=4, label=lab)

    axa.annotate(r"$\eta\equiv 1$", xy=(1.3e-3, 1.0), xytext=(1.3e-3, 1.12),
                 fontsize=8.5, color=INK, zorder=6)
    axa.set_xscale("log")
    axa.set_xlim(1e-3, 0.5)
    axa.set_ylim(0, 2.5)
    axa.set_xlabel(r"quantile level  $\tau$", fontsize=10, color=INK)
    axa.set_ylabel(r"$\eta(\tau)$", fontsize=10.5, color=INK)
    axa.set_title("(a)  the fraction of the true shortfall gap\n"
                  "       the AL is able to represent",
                  fontsize=10.5, color=INK, loc="left", pad=8)
    _legend(axa, "upper center", ncol=2, bbox=(0.5, -0.17), fs=8)

    # ---- panel (b) ----
    tb = np.logspace(-8, np.log10(0.5), 700)
    _style(axb)
    axb.axhline(1.0, color=INK, lw=1.8, ls=(0, (6, 3)), zorder=3,
                label="AL working likelihood")
    for fam, col, ls, lab in PANEL_B:
        axb.plot(tb, np.atleast_1d(fam.R_closed(tb)), color=col, ls=ls, lw=2.0,
                 zorder=4, label=lab)

    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(1e-8, 0.5)
    axb.set_ylim(1e-9, 1e9)
    axb.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
    axb.set_xlabel(r"quantile level  $\tau$", fontsize=10, color=INK)
    axb.set_ylabel(r"$\eta(\tau)$", fontsize=10.5, color=INK)
    axb.set_title("(b)  the three limits of $\\eta$ as $\\tau\\to 0$,\n"
                  "       over eight decades",
                  fontsize=10.5, color=INK, loc="left", pad=8)

    axb.text(2e-4, 3e6, r"$\eta\to\infty$", fontsize=9.5, color=INK2)
    axb.text(1e-7, 25.0, r"$\eta\to$ const", fontsize=9.5, color=INK2)
    axb.text(2e-7, 4e-9, r"$\eta\to 0$", fontsize=9.5, color=INK2)
    # legends go under the panels, inside panel (b) it covered the R -> 0 curves
    _legend(axb, "upper center", ncol=2, bbox=(0.5, -0.17), fs=8)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.33,
                        wspace=0.24)
    fig.text(0.065, 0.028,
             r"$\eta(\tau)=\dfrac{\tau\,(\mu-\mathrm{CVaR}_\tau)}"
             r"{(1-\tau)\,(\mathrm{VaR}_\tau-\mathrm{CVaR}_\tau)}$"
             r"$\;=\;\tau\left(1+e_U/e_L\right)$,"
             r"  invariant to location and scale.  $\eta=1$ at every $\tau$ for the"
             r" AL whose parameter equals $\tau$, and $\eta=1$ exactly when"
             r" $e_U/e_L=(1-\tau)/\tau$.",
             fontsize=8, color=INK2)

    for ext in ("png", "pdf"):
        fig.savefig(f"fig_eta_tau.{ext}", dpi=200, facecolor="white")
    print("wrote fig_eta_tau.png, fig_eta_tau.pdf")

    # ---- the table behind the figure ----
    taus = [0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50]
    rows = [("AL (working likelihood)", None)]
    rows += [(l.split("  (")[0].replace("$", "").replace("\\", ""), f)
             for f, _, _, l in PANEL_A]
    rows += [("Exponential", Exponential()), ("Beta(0.5,1)", BetaPower(0.5))]
    with open("eta_tau_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family"] + [f"tau={t:g}" for t in taus])
        for lab, fam in rows:
            if fam is None:
                w.writerow([lab] + ["1.000000"] * len(taus))
            else:
                w.writerow([lab] + [f"{float(np.atleast_1d(fam.R_closed(t))[0]):.6f}"
                                    for t in taus])
    print("wrote eta_tau_table.csv")


if __name__ == "__main__":
    main()