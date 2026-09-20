# News-based market timing strategy

A news-driven trading pipeline: CNBC headlines are pulled from GDELT, passed to
an LLM acting as a portfolio manager, turned into a next-day long/short call,
and backtested against realised S&P 500 returns with realistic ETF trading
costs.

Over 243 trading days (2025-09-23 … 2026-09-14) the LLM Direction strategy
returned **+32.08% net** against **+12.90%** for buy-and-hold, at an excess
Sharpe of **2.01**. Read [`output/performance.md`](output/performance.md) for
the full metric set, the figures, and the caveats — several of which matter a
great deal, and are summarised at the bottom of this file.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1. Headlines | `scripts/fetch_gdelt_headlines.py` | `data/cnbc_year_*.csv` |
| 2. Market returns | `scripts/fetch_market_returns.py` | `data/market_returns.csv` |
| 2b. Fama/French factor | `scripts/fetch_ff_mktrf.py` | `data/mkt_rf_*.csv` |
| 3. Predictions + backtest | `scripts/predict_positions.py` | `data/predictions_*.csv`, `data/predictions_strategy.csv` |
| 4. Performance report | `scripts/performance_analysis.py` | `output/performance.md`, `output/charts/*.png` |

Every script runs standalone — no package, no cross-script imports — and the
project is standard-library only apart from `matplotlib`, which is used solely
for the figures. Without it the performance report is still written, just
without charts.

## Reproducing

```sh
py scripts/predict_positions.py --headlines data/cnbc_year_20250919-20260919.csv --resume
py scripts/performance_analysis.py
```

The first command needs an Anthropic API key and costs roughly $3.60 for a full
year of predictions; `--resume` skips days already present. `--backtest-only`
re-runs the backtest against the existing predictions at no cost. The second
command is free and offline.

## Strategies

| Strategy | Position |
|---|---|
| Benchmark | Always +1 (buy and hold) |
| LLM Direction | +1 long / −1 short on the model's call |
| LLM Direction (Contrarian) | The opposite of the above |
| LLM Sentiment-Weighted | `sentiment / 5`, so −1.0 … +1.0 |
| LLM Sentiment-Weighted (Contrarian) | The opposite of the above |

Costs are modelled as a 2.0 bps round-turn spread charged on `|Δposition|`,
30 bps/yr borrow on short exposure, and SPY's 9.45 bps/yr expense ratio.

## Point-in-time discipline

A prediction for a given trading day may only use headlines published **before
that day's 4:00 PM ET close** (1:00 PM on early-close days). Anything later
could not have informed a trade placed that day, so letting it in would
manufacture a backtest that looks excellent for the wrong reason.

`predict_positions.py` enforces this in `load_headlines` and re-verifies it in
an audit pass that runs on every backtest, checking four things: no headline at
or after the cutoff was used, every prediction targets the *next* trading day,
no duplicate predictions, and the direction and sentiment fields are not
collinear. The current run passes 243 / 243.

GDELT's `seendate` is when GDELT first *indexed* an article, which is at or
after actual publication, so filtering on it is conservative in the safe
direction.

## Caveats

Read these before treating the headline number as an edge.

- **Training-data contamination.** 172 of the 243 days fall on or before the
  model's training cutoff, so for most of the sample the model may be recalling
  outcomes rather than forecasting them. Splitting at the cutoff is mildly
  reassuring — the strategy earns +8.80% in the 71 post-cutoff days against a
  flat market (the Benchmark returns −0.22%), at an excess Sharpe of 2.09
  against 1.98 before the cutoff — but 71 days carries a Sharpe standard error
  of about ±1.88, so that is not significant on its own.
- **Long tilt does part of the work.** Mean signed position is +0.374 in a year
  the market rose. The short side is genuinely selective (the market fell 8.77%
  across the 76 short days), but the tilt is not nothing.
- **Moderate statistical strength.** The net daily mean has t = 2.26 over 243
  days, before any correction for having picked the best of five strategies.
- **Turnover is high.** 197.9×/yr annualized, which is what makes the strategy
  sensitive to the cost assumptions above.
- **Returns exclude dividends.** FRED's `SP500` is a price index.
