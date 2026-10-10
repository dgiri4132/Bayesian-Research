#!/usr/bin/env python3
"""
fig_eta_tau.py -- the R(tau) figure for Experiment B.

This makes the one figure that shows the whole R story at a glance.

Panel (a)  R against tau over the range people actually use for risk, with R on
           a normal scale. The AL sits flat at 1 by construction, and every
           family with an unbounded tail drops away from it as tau goes into
           the tail.

Every line is black or grey and has its own linestyle or shade, so they can be
told apart in black and white.

writes fig_eta_tau.png, fig_eta_tau.pdf, eta_tau_table.csv
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

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
GREY = "#8c8c8c"

# family, colour, linestyle, label
PANEL_A = [
    (Normal(),      INK,  "-",                     "Normal"),
    (StudentT(5),   INK,  (0, (5, 2)),             r"$t(5)$"),
    (StudentT(3),   INK,  (0, (1, 1.6)),           r"$t(3)$"),
    (Laplace(),     GREY, (0, (7, 2, 1, 2)),       "Laplace"),
    (NegGPD(0.30),  INK,  (0, (7, 2, 1, 2, 1, 2)), r"neg-GPD $\xi=0.3$"),
    (LogNormal(),   GREY, "-",                     "Lognormal"),
    (Uniform(),     GREY, (0, (2, 2)),             "Uniform"),
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
                   title=title, borderpad=0.6, handlelength=3.4,
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
    fig, axa = plt.subplots(figsize=(7.2, 6.0))
    fig.patch.set_facecolor("white")

    # ---- panel (a) ----
    ta = np.logspace(-3, np.log10(0.5), 500)
    _style(axa)
    axa.axvspan(0.01, 0.05, color="#ececec", zorder=0)
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
    axa.set_title("the fraction of the true shortfall gap\n"
                  "the AL is able to represent",
                  fontsize=10.5, color=INK, loc="left", pad=8)
    _legend(axa, "upper center", ncol=2, bbox=(0.5, -0.17), fs=8)

    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.36)
    fig.text(0.1, 0.02,
             r"$\eta(\tau)=\dfrac{\tau\,(\mu-\mathrm{CVaR}_\tau)}"
             r"{(1-\tau)\,(\mathrm{VaR}_\tau-\mathrm{CVaR}_\tau)}$"
             r"$\;=\;\tau\left(1+e_U/e_L\right)$,"
             r"  invariant to location and scale."
             "\n"
             r"$\eta=1$ at every $\tau$ for the"
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