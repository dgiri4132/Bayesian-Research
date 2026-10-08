#!/usr/bin/env python3
"""
EXPERIMENT B6  is R across many real stocks.

Bayesian CVaR project.  Darshan Giri.

Everything else in the paper is on SPY, so I thought someone could say the R
result is just an SPY thing. This file works out R for a lot of stocks at once.
R only needs three numbers per asset, the mean, the tau-quantile and the mean
below the quantile, so a few hundred stocks is cheap once I have the prices.

My download:
    - SPY plus the 503 current S&P 500 stocks from the Wikipedia list, fetched
      27 Sep 2026, saved in sp500_tickers_20260927.txt (BRK.B and BF.B are
      written BRK-B and BF-B for Yahoo)
    - yfinance, auto-adjusted close, 2015-08-17 to 2025-08-14, the same window
      as the SPY sample (2512 returns)
    - 6 tickers failed to download and 6 had fewer than 750 returns, so 492
      series are used (SPY and 491 stocks)
    - prices saved in B6_mine_panel.csv, results in B6_mine.json and
      B6_mine_by_asset.csv

    to make the same files:
       python3 experiment_b6.py --tickers "$(cat sp500_tickers_20260927.txt)" --save-panel B6_mine_panel.csv
        

    This downloads again and Yahoo revises prices, so the last digits can move
    a bit. The saved panel is what the paper's numbers come from.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np
import pandas as pd

from rdiag import R_sample, Normal, StudentT

TAUS = (0.01, 0.05, 0.10)


#loaders

def from_panel(path):
    # the wide file from_yfinance saves, a Date column then one column of prices
    # per ticker. round_trip reads every price back exactly as it was saved
    px = pd.read_csv(path, index_col=0, float_precision="round_trip")
    return {c: px[c].dropna().reset_index(drop=True) for c in px.columns}


def from_yfinance(tickers, start, end, save_to="panel_downloaded.csv"
                  ) -> dict[str, pd.Series]:
    """
    Downloads the prices and SAVES them. I keep the file because Yahoo revises
    prices and drops delisted tickers, so without it nobody could check the
    numbers later, me included. The saved file goes in the commit.
    """
    import yfinance as yf
    px = yf.download(list(tickers), start=start, end=end, progress=False,
                     auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(axis=1, how="all")
    if save_to:
        px.to_csv(save_to)
        print(f"  saved the downloaded panel to {save_to} "
              f"({px.shape[1]} tickers x {px.shape[0]} days) -- COMMIT THIS FILE, "
              f"it is what makes the run reproducible")
    return {c: px[c].dropna().reset_index(drop=True) for c in px.columns}


# returns 

def simple_returns(px: pd.Series) -> np.ndarray:
    """
    SIMPLE returns (pct_change), same as Spy_run.py.

    This used to use log returns, which gives a slightly different R. The paper
    quotes R to four decimals, so every file uses simple returns.
    """
    v = px.to_numpy(dtype=float)
    v = v[v > 0]
    return v[1:] / v[:-1] - 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="B6_mine_panel.csv")
    ap.add_argument("--tickers", default=None,
                    help="comma-separated; uses yfinance")
    ap.add_argument("--start", default="2015-08-17")
    ap.add_argument("--end", default="2025-08-14")
    ap.add_argument("--min-obs", type=int, default=750)
    ap.add_argument("--json", default="B6_mine.json")
    ap.add_argument("--csv", default="B6_mine_by_asset.csv")
    ap.add_argument("--save-panel", default="panel_downloaded.csv",
                    help="where to save a downloaded panel so the run is "
                         "reproducible; pass '' to skip")
    args = ap.parse_args()

    print("=" * 78)
    print("EXPERIMENT B6: R across many real assets")
    print("=" * 78)

    if args.tickers:
        src = f"yfinance {args.start}..{args.end}"
        tick = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        print(f"  downloading {len(tick)} tickers, {args.start} .. {args.end}")
        data = from_yfinance(tick, args.start, args.end, save_to=args.save_panel)
    else:
        src = f"panel {args.panel}"
        data = from_panel(args.panel)

    print(f"  source        {src}")
    print(f"  python        {platform.python_version()}  numpy {np.__version__}")
    print(f"  assets loaded {len(data)}")

    rows = []
    for name, px in sorted(data.items()):
        r = simple_returns(px)
        if r.size < args.min_obs:
            continue
        rec = {"asset": name, "n": int(r.size),
               "mean": float(r.mean()), "sd": float(r.std(ddof=1))}
        for t in TAUS:
            s = R_sample(r, t, use_effective_tau=True)
            rec[f"R_{t:g}"] = float(s["R_general"])
            rec[f"VaR_{t:g}"] = float(s["VaR"])
            rec[f"CVaR_{t:g}"] = float(s["CVaR"])
            rec[f"tail_n_{t:g}"] = int(s["k"])
        rows.append(rec)

    if not rows:
        print(f"  no asset had at least {args.min_obs} returns.")
        return 2

    df = pd.DataFrame(rows)
    print(f"  assets used   {len(df)}  (>= {args.min_obs} returns each)")
    print(f"  return counts {int(df['n'].min())} to {int(df['n'].max())}"
          f", median {int(df['n'].median())}")

    print("\n  R by quantile level, across assets")
    print(f"  {'tau':>6s} {'tail obs':>9s} {'min':>9s} {'q05':>9s} {'median':>9s}"
          f" {'q95':>9s} {'max':>9s} {'% > 1':>8s}")
    summary = {}
    for t in TAUS:
        c = df[f"R_{t:g}"]
        q = np.percentile(c, [0, 5, 50, 95, 100])
        frac = float((c > 1).mean() * 100)
        summary[f"{t:g}"] = {
            "min": float(q[0]), "q05": float(q[1]), "median": float(q[2]),
            "q95": float(q[3]), "max": float(q[4]), "pct_above_1": frac,
            "median_tail_obs": float(df[f"tail_n_{t:g}"].median()),
        }
        print(f"  {t:6.2f} {df[f'tail_n_{t:g}'].median():9.0f} "
              f"{q[0]:9.4f} {q[1]:9.4f} {q[2]:9.4f} {q[3]:9.4f} {q[4]:9.4f}"
              f" {frac:7.1f}%")

    # reference values from the closed forms, same tau grid
    print("\n  for comparison, the closed forms (Experiment B2):")
    print(f"  {'':>6s} {'Normal':>9s} {'t(3)':>9s}")
    refs = {}
    for t in TAUS:
        vals = [float(np.atleast_1d(f.R_closed(t))[0])
                for f in (Normal(), StudentT(3))]
        refs[f"{t:g}"] = vals
        print(f"  {t:6.2f} " + "".join(f"{v:9.4f}" for v in vals))

    df.to_csv(args.csv, index=False)
    out = {"source": src, "n_assets": len(df), "min_obs": args.min_obs,
           "taus": list(TAUS), "summary": summary, "closed_form_reference": refs,
           "python": platform.python_version(), "numpy": np.__version__}
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {args.csv} and {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
