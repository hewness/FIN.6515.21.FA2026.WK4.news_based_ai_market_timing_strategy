# Strategy performance

**2025-09-23 .. 2026-09-14 — 243 trading days — 5 strategies**

Source: `C:\Users\Gordon Hew\workspace\news_based_ai\data\predictions_strategy.csv` (1215 rows).
Risk-free: `C:\Users\Gordon Hew\workspace\news_based_ai\data\market_returns.csv`, `rf_daily`, mean 0.00014849/day = 3.81%/yr over 243 day(s), 2 gap(s) filled.

> ### Read this before the tables
>
> - **VaR and CVaR are historical (empirical) quantiles**, no interpolation: the k-th worst day with k = ceil((1-conf)·n), here k = 13 at 95% and k = 3 at 99%.
> - **Sharpe is computed on excess returns** (return − `rf_daily`). `scripts/predict_positions.py`'s console summary does not subtract `rf`, so its Sharpe column is overstated. Where the two disagree, this file is correct.
> - **VaR and CVaR are signed returns: negative is a loss**, on a 1-day horizon. Not annualized, not scaled.

**Legend** — `(C)` = Contrarian.

| Short | Full name |
|---|---|
| Benchmark | Benchmark |
| Direction | LLM Direction |
| Direction (C) | LLM Direction (Contrarian) |
| Sentiment | LLM Sentiment-Weighted |
| Sentiment (C) | LLM Sentiment-Weighted (Contrarian) |

## Gross returns (before trading costs)

| Metric | Benchmark | Direction | Direction (C) | Sentiment | Sentiment (C) |
|---|---:|---:|---:|---:|---:|
| Observations (days) | 243 | 243 | 243 | 243 | 243 |
| Total return | +13.02% | +34.79% | -27.01% | +4.51% | -4.52% |
| Annualized return (geometric) | +13.53% | +36.29% | -27.86% | +4.68% | -4.68% |
| Annualized volatility | 13.03% | 12.90% | 12.90% | 4.69% | 4.69% |
| Sharpe (excess, annualized) | 0.75 | 2.18 | -2.76 | 0.20 | -1.80 |
| Sortino (annualized) | 1.51 | 4.01 | -3.08 | 1.52 | -1.32 |
| Maximum drawdown | -9.10% | -7.89% | -28.87% | -4.65% | -4.98% |
| VaR 95%, historical † | -1.436% | -1.159% | -1.547% | -0.434% | -0.490% |
| VaR 99%, historical † | -2.063% | -1.741% | -2.508% | -0.707% | -0.983% |
| CVaR 95% (expected shortfall) † | -1.786% | -1.548% | -1.898% | -0.658% | -0.739% |
| CVaR 99% (expected shortfall) † | -2.473% | -2.239% | -2.621% | -1.037% | -1.187% |
| Win rate, days in position | 53.9% (131/243) | 58.0% (141/243) | 42.0% (102/243) | 55.4% (123/222) | 44.6% (99/222) |
| Win rate, all days | 53.9% (131/243) | 58.0% (141/243) | 42.0% (102/243) | 50.6% (123/243) | 40.7% (99/243) |
| Flat days (position = 0) | 0 | 0 | 0 | 21 | 21 |

## Net returns (after trading costs)

| Metric | Benchmark | Direction | Direction (C) | Sentiment | Sentiment (C) |
|---|---:|---:|---:|---:|---:|
| Observations (days) | 243 | 243 | 243 | 243 | 243 |
| Total return | +12.90% | +32.08% | -28.52% | +3.70% | -5.26% |
| Annualized return (geometric) | +13.41% | +33.45% | -29.40% | +3.84% | -5.45% |
| Annualized volatility | 13.03% | 12.89% | 12.92% | 4.69% | 4.69% |
| Sharpe (excess, annualized) | 0.74 | 2.01 | -2.92 | 0.03 | -1.97 |
| Sortino (annualized) | 1.50 | 3.72 | -3.25 | 1.25 | -1.54 |
| Maximum drawdown | -9.11% | -8.26% | -30.29% | -4.93% | -5.65% |
| VaR 95%, historical † | -1.437% | -1.174% | -1.548% | -0.440% | -0.496% |
| VaR 99%, historical † | -2.063% | -1.741% | -2.509% | -0.713% | -0.990% |
| CVaR 95% (expected shortfall) † | -1.786% | -1.557% | -1.914% | -0.662% | -0.745% |
| CVaR 99% (expected shortfall) † | -2.473% | -2.246% | -2.635% | -1.042% | -1.195% |
| Win rate, days in position | 53.9% (131/243) | 56.8% (138/243) | 42.0% (102/243) | 54.5% (121/222) | 43.2% (96/222) |
| Win rate, all days | 53.9% (131/243) | 56.8% (138/243) | 42.0% (102/243) | 49.8% (121/243) | 39.5% (96/243) |
| Flat days (position = 0) | 0 | 0 | 0 | 21 | 21 |

![Net cumulative return and drawdown](charts/equity_drawdown.png)

*Growth of $1 after trading costs (top) and drawdown from the running peak (bottom), both on the net series tabulated above. The shaded band covers days after the model's training cutoff (2026-05-31) — the only stretch of this sample the model could not have memorised. Read the slope inside the band against the slope outside it.*

## Trading activity and cost

| Metric | Benchmark | Direction | Direction (C) | Sentiment | Sentiment (C) |
|---|---:|---:|---:|---:|---:|
| Total cost paid (sum of daily) | 0.1011% | 2.0344% | 2.0744% | 0.7806% | 0.7841% |
| Gross − net (total return) | 0.11 pp | 2.71 pp | 1.50 pp | 0.81 pp | 0.75 pp |
| Turnover (total, one-way units) | 1.00 | 191.00 | 191.00 | 74.00 | 74.00 |
| of which day-1 entry | 1.00 | 1.00 | 1.00 | 0.60 | 0.60 |
| Turnover (annualized, ex-entry) ‡ | 0.0x | 197.9x | 197.9x | 76.4x | 76.4x |
| Days with a trade | 1 | 96 | 96 | 173 | 173 |
| Mean \|position\| | 1.000 | 1.000 | 1.000 | 0.293 | 0.293 |
| Mean signed position | +1.000 | +0.374 | -0.374 | +0.033 | -0.033 |
| Sharpe drag from costs | -0.01 | -0.16 | -0.16 | -0.17 | -0.17 |

‡ Turnover is the sum of `|position_t − position_{t-1}|` with `position_0 = 0`. The day-1 entry is a one-off cost of establishing the book, not a rate, so the annualized figure excludes it and divides the remainder over the 242 days on which rebalancing could occur. Benchmark is 0.0x because buy-and-hold trades once and never again. A book flipping fully long to fully short every day would show 504x.

![Timing skill versus directional tilt](charts/exposure_attribution.png)

*Left: what the market did on the days the strategy chose each side. Holding long through a rising year earns money without any skill at all, so the short-day pair is where security selection has to show up. Right: an additive decomposition of the net total. Daily returns are summed rather than compounded so the parts add up; the gap against the true compounded total is carried explicitly as the `Compounding` bar instead of being absorbed into rounding.*

## Signal quality

The model states a `confidence` between 0 and 1 on every call, and **no strategy in this report uses it**. The figure below tests it against what actually happened, so it is the one result here that is not already priced into the tables above.

![Hit rate and realised return by stated confidence](charts/confidence_calibration.png)

*Left: directional hit rate by equal-count confidence bin, with 95% Wilson intervals and a 50% reference line. Right: the mean net return actually earned in each bin. A profile that rises left to right would justify sizing by confidence, or standing aside below a threshold; a flat one says the self-assessment carries no information and should be ignored. Bins are equal-count rather than equal-width because confidence clusters tightly around its mean.*

## Notes

- **Total return is compounded**, not summed.
- **Annualized return is geometric**, `(1 + total) ** (252/n) - 1`, so it reconciles with the total-return row directly.
- **Annualized volatility** is the sample standard deviation (n−1) of the raw series × √252. Sharpe uses the standard deviation of the *excess* series, which is the textbook definition.
- **Sortino** uses a minimum acceptable return of **0%**. Downside deviation is `sqrt(Σ min(r−MAR, 0)² / n)` — divided by **n**, the full sample count, not by the number of losing days. The loser-count form discards loss-frequency information and is non-monotone: turning a losing day into a winning one could *raise* the denominator and *lower* the ratio.
  Because MAR = 0, Sortino's numerator is the mean **raw** return while Sharpe's is the mean **excess** return, so the two ratios are measured against different benchmarks and are not directly comparable. `--sortino-mar rf` makes them consistent.
- A **negative Sharpe or Sortino ranks nothing**: once the numerator is negative, a larger denominator moves the ratio toward zero.
- **Win rate, days in position** excludes days a strategy deliberately held no position. Standing aside on a neutral read is not a loss, but a strict `return > 0` test scores it as one. Flat days are counted from `Position == 0`, not `return == 0`, so the gross and net tables share a denominator.
- **Maximum drawdown** is close-to-close on the strategy's own equity curve and understates intraday drawdown.
- Contrarian variants are **not** the exact negative of their parent. Only the daily *gross* return negates; compounded totals do not, and costs are strictly positive on both sides.
- A high Sharpe on a low-exposure strategy is low volatility, not necessarily skill — check the `Mean |position|` row.
- **Figures are drawn on net returns** and are regenerated on every run; they are written to `C:\Users\Gordon Hew\workspace\news_based_ai\output\charts`. Pass `--no-charts` to skip them, which is also what happens automatically if matplotlib is not installed.
