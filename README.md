# Expected Shortfall under the Asymmetric Laplace Working Likelihood in Bayesian Quantile Regression

Darshan Giri and Songfeng Zheng, Department of Mathematics, Missouri State University.

This is the code for our paper on expected shortfall (ES) under the asymmetric
Laplace (AL) working likelihood. Bayesian quantile regression with an AL
likelihood gets VaR right, but the ES you get from the same posterior,
ES = ω − σ/(1−τ), is structurally wrong. On SPY the VaR is 0.29 posterior sd
from the historical value and the ES is 29.7 posterior sd off. The files below
show that, check it isn't the sampler, the prior or the sample, and measure how
big it is for any distribution.

Every number in the paper comes from a results file in this repo.

The diagnostic is called η in the paper and R in the code and results files
(`rdiag.py` and the JSON keys).

## Where each result in the paper comes from

| in the paper | script | results file |
|---|---|---|
| Section 3.3, after Corollary 1 (the AL's mean against the sample mean) | `Spy_run.py` | `spy_results.json` |
| Section 4, Table 1 (the SPY fit) | `Spy_run.py` | `spy_results.json` |
| Section 5, Tables 2, 3 and 4 (coverage, sample size, priors) | `experiment_a.py` | `experiment_A_results.json` |
| Sections 6.1 and 6.2, Table 5 (η in closed form, tail weight) | `experiment_b_overall.py` | `experiment_B_results.json` |
| Section 6.3, Table 6 (S&P 500 stocks) | `experiment_b6.py` | `B6_mine.json`, `B6_mine_by_asset.csv` |
| Section 7, Table 7 (quantile regression) | `experiment_c.py` | `experiment_C_results.json` |

The numbers in the abstract, Sections 1–3 and the conclusion come also from the same
files.

## Files

| file | what it does |
|---|---|
| `kkgibs.py` | shared core: the Kozumi–Kobayashi Gibbs sampler, split R-hat, ESS, the exact posterior, the AL formulas and the sample split rule. Everything else imports it |
| `Spy_run.py` | the SPY fit. Every headline number. Writes `spy_results.json` and `spy_results.txt` |
| `experiment_a.py` | coverage study over 5 data-generating processes. Is it anything other than the model? Writes `experiment_A_results.json` |
| `rdiag.py` | the R diagnostic and the library of distribution families it's evaluated on |
| `experiment_b_overall.py` | R in closed form, heavy tails vs the Normal, and the Chen–Gerlach–Lu comparison. Writes `experiment_B_results.json` |
| `experiment_b6.py` | R across the S&P 500 (SPY plus 491 current stocks, my own download). Writes `B6_mine.json` and `B6_mine_by_asset.csv` |
| `experiment_c.py` | the only real quantile regression (lagged \|r\| and 20-day volatility). Writes `experiment_C_results.json` |
| `spy_returns_2015_2025.csv` | SPY daily simple returns, 18 Aug 2015 to 13 Aug 2025, n = 2512 |
| `B6_mine_panel.csv`, `sp500_tickers_20260927.txt` | the B6 prices I downloaded and the ticker list I used |

## Running it

```
python3 -m venv ~/.venvs/research
source ~/.venvs/research/bin/activate
python3 -m pip install -r requirements.txt

python3 Spy_run.py               # about 30 s
python3 experiment_a.py          # about 4.5 min
python3 experiment_b_overall.py  # a few seconds
python3 experiment_b6.py         # a few seconds
python3 experiment_c.py          # about 30 s
```

The experiment scripts print `[PASS]` / `[FAIL]` lines and exit non-zero if a
check fails. Running them rewrites the committed JSONs, so `git diff` afterwards shows
whether any number moved or not for verifcation.

`experiment_b6.py` reads the saved `B6_mine_panel.csv`, which is where the
numbers in the paper come from. With `--tickers` it downloads the prices again
from Yahoo instead (the command is at the top of the script btw), so it needs a
network connection, and Yahoo revises prices so the last digits can move.

## Things to know

- There are two samples. Spy_run, A and B use all 2512 days. C uses 2492,
  because the 20-day volatility window uses up the first 20 days. Please do not put a
  number from one next to a number from the other.
- `Generator.wald` differs by about one unit in the last place across numpy builds,
  so R-hat and ESS can move in the last digits. With numpy 2.5 the draws in
  `experiment_c.py` (three coefficients) change as well, so its numbers move in the
  last reported digit. Use the versions in `requirements.txt` to get the paper's
  numbers exactly.
- The sampler's numerical floors do get hit (465 and 223 times on the SPY run in my case).
