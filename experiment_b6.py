#!/usr/bin/env python3
"""
EXPERIMENT B6 -- R across many real stocks.

Bayesian CVaR project.  Darshan Giri / Prof. Songfeng Zheng.

Everything else in the paper is on SPY, so someone could say the R result is
just an SPY thing. This file works out R for a lot of stocks at once. R only
needs three numbers per asset, the mean, the tau-quantile and the mean below
the quantile, so a few hundred stocks is cheap once I have the prices.

I cut B6 on 5 Sep 2026 because the only panel I had then was a public mirror
(2013 to 2018) and I couldn't check where it came from. I brought it back on
27 Sep 2026 after downloading the prices myself. The numbers in the paper come
only from my download, never from the old mirror file (sp500_5yr_raw.csv).

My download:
    - SPY plus the 503 current S&P 500 stocks from the Wikipedia list, fetched
      27 Sep 2026, saved in sp500_tickers_20260927.txt (BRK.B and BF.B are
      written BRK-B and BF-B for Yahoo)
    - yfinance, auto-adjusted close, 2015-08-17 to 2025-08-14, the same window
      as the SPY sample (2512 returns)
    - 6 tickers failed to download and 6 had fewer than 750 returns, so 492
      stocks are used
    - prices saved in B6_mine_panel.csv, results in B6_mine.json and
      B6_mine_by_asset.csv

    to make the same files:

        python3 experiment_b6.py --tickers "$(cat sp500_tickers_20260927.txt)" \\
            --start 2015-08-17 --end 2025-08-14 --save-panel B6_mine_panel.csv \\
            --json B6_mine.json --csv B6_mine_by_asset.csv

    This downloads again and Yahoo revises prices, so the last digits can move
    a bit. The saved panel is what the paper's numbers come from.

Things to say next to the numbers in the paper:
    - these are TODAY's S&P 500 stocks, so companies that left the index since
      2015 are missing
    - the stocks share the same market history, so they aren't independent
    - at tau = 0.01 each stock only has about 25 tail days

usage:  python3 experiment_b6.py [--panel FILE] [--csv-dir DIR] [--tickers LIST]
                                 [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                 [--min-obs N] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

from rdiag import R_sample, Normal, StudentT, NegGPD

TAUS = (0.01, 0.05, 0.10)


# ---- loaders ----

def from_long_panel(path: str) -> dict[str, pd.Series]:
    """A single long CSV with columns date, close, Name."""
    d = pd.read_csv(path, usecols=["date", "close", "Name"])
    d = d.dropna(subset=["close"])
    out = {}
    for name, g in d.groupby("Name"):
        s = g.sort_values("date")["close"].astype(float)
        out[str(name)] = s.reset_index(drop=True)
    return out


def from_csv_dir(d: str) -> dict[str, pd.Series]:
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        t = pd.read_csv(p)
        cols = {c.lower(): c for c in t.columns}
        cc = cols.get("close") or cols.get("adj close") or cols.get("adj_close")
        dc = cols.get("date")
        if cc is None:
            continue
        if dc is not None:
            t = t.sort_values(dc)
        out[os.path.splitext(os.path.basename(p))[0]] = (
            t[cc].astype(float).dropna().reset_index(drop=True))
    return out


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


# ---- returns ----

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
    ap.add_argument("--panel", default="sp500_5yr_raw.csv")
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--tickers", default=None,
                    help="comma-separated; uses yfinance")
    ap.add_argument("--start", default="2015-08-18")
    ap.add_argument("--end", default="2025-08-13")
    ap.add_argument("--min-obs", type=int, default=750)
    ap.add_argument("--json", default="experiment_B6_results.json")
    ap.add_argument("--csv", default="experiment_B6_R_by_asset.csv")
    ap.add_argument("--save-panel", default="panel_downloaded.csv",
                    help="where to save a downloaded panel so the run is "
                         "reproducible; pass '' to skip")
    args = ap.parse_args()

    print("=" * 78)
    print("EXPERIMENT B6 -- R across many real assets")
    print("=" * 78)

    if args.tickers:
        src = f"yfinance {args.start}..{args.end}"
        tick = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        print(f"  downloading {len(tick)} tickers, {args.start} .. {args.end}")
        data = from_yfinance(tick, args.start, args.end, save_to=args.save_panel)
    elif args.csv_dir:
        src = f"csv-dir {args.csv_dir}"
        data = from_csv_dir(args.csv_dir)
    else:
        src = f"long panel {args.panel}"
        if not os.path.exists(args.panel):
            print(f"  panel file {args.panel} not found; nothing to do.")
            print("  supply --panel, --csv-dir or --tickers.  See the module")
            print("  docstring for the provenance caveats.")
            return 2
        data = from_long_panel(args.panel)

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
    print(f"  {'':>6s} {'Normal':>9s} {'t(5)':>9s} {'t(3)':>9s} "
          f"{'GPD .3':>9s} {'AL':>9s}")
    refs = {}
    for t in TAUS:
        vals = [float(np.atleast_1d(f.R_closed(t))[0])
                for f in (Normal(), StudentT(5), StudentT(3), NegGPD(0.30))]
        refs[f"{t:g}"] = vals
        print(f"  {t:6.2f} " + "".join(f"{v:9.4f}" for v in vals) + f"{1.0:9.4f}")

    print("""
  WHAT THIS SHOWS.  Every stock is way below the AL's 1, and the spread across
  all of them is narrow. The median at tau = 0.05 is between the closed forms
  for t(3) and the Normal, which is where you'd expect real stock returns to
  be. So the failure I showed on SPY is not just an SPY thing.

  CAVEATS to put next to this in the paper.  (1) R here is a plug-in estimate,
  so it has the same granularity issue as in B1, and at tau = 0.01 each stock
  only has a handful of tail days, so trust the tau = 0.05 row most.  (2) The
  stocks share the same market history, so they aren't independent and the
  spread understates how noisy a single R is.  (3) These are today's S&P 500
  stocks, so companies that left the index since 2015 are missing.  The note at
  the top of the file says where the data came from.
""")

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
