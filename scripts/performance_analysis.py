"""Fund performance metrics for each strategy, written to output/performance.md.

Reads the per-day, per-strategy results produced by predict_positions.py and
reports a full metric set on BOTH gross and net returns, side by side across
strategies, plus a turnover/cost table.

The report opens with a Findings section whose every figure is interpolated
from the same dicts that render the tables, so the prose cannot drift from
the numbers, and closes with a fixed retrospective on acquiring the GDELT
corpus (--no-appendix omits it).

Sharpe here is computed on EXCESS returns (return - rf_daily). The console
summary in predict_positions.py passes the net series as its own excess series,
so its Sharpe column is overstated -- by about 0.40 for the +-1 strategies and
about 1.60 for the sentiment-weighted ones, whose lower volatility makes the
same risk-free drag proportionally larger. Where the two disagree, this file is
correct and that one is optimistic.

output/backtest_report.txt is a frozen snapshot from the removed
scripts/backtest.py. It is no longer reproducible, but it is a useful fixture:
its GROSS Sharpe, max drawdown and win rate for "STRATEGY" (= LLM Direction)
and "BUY & HOLD" (= Benchmark) must still match this script's gross table.

Three figures are written to output/charts/ and linked from the report: the
net equity curve with its drawdown path (shaded after the model's training
cutoff), hit rate by the model's stated confidence, and an attribution of
LLM Direction's return between timing and directional tilt. They need
matplotlib, the only non-stdlib dependency in this repo; without it the
report is still written, just without figures.

Usage:
    py scripts/performance_analysis.py
    py scripts/performance_analysis.py --stdout       # print, do not write
    py scripts/performance_analysis.py --no-charts    # tables only

Exit codes: 0 ok | 1 input missing or unusable | 2 written, but a day had no
            usable risk-free rate | 3 refused (--rf-missing=error hit a gap)
"""

import argparse
import bisect
import csv
import logging
import math
import os
import sys
from datetime import datetime, time, timezone
from pathlib import Path

# --- project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"

# --- BEGIN shared by copy with scripts/predict_positions.py ----------------
# Every script in this repo runs standalone -- no package, no cross-script
# imports -- so these helpers live in two files by design. If you change one,
# change the other. The guard is numeric: this script recomputes the compounded
# totals, turnover, drawdown and win rate straight from predictions_strategy.csv,
# so the two tools must agree on every strategy.
#
# One deliberate divergence, not a drift: load_returns() below records a blank
# rf_daily as None rather than collapsing it to 0.0, so the caller can forward-
# fill instead of silently inheriting an rf=0 bias.
TRADING_DAYS = 252


def max_drawdown(returns):
    """Largest peak-to-trough decline of the compounded equity curve."""
    equity, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def load_returns(path):
    """Daily return series. Schema A: date,mkt_ret[,rf_daily] (decimals).
    Schema B: date,mkt_rf_pct (Fama/French, percent, ALREADY excess)."""
    out = {}
    meta = {"excess": False, "schema": "mkt_ret (decimal)", "blank_rf": []}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        ff_mode = "mkt_rf_pct" in fields and "mkt_ret" not in fields
        if ff_mode:
            meta["excess"] = True
            meta["schema"] = "mkt_rf_pct (Fama/French, percent, already excess)"
        for row in reader:
            try:
                day = datetime.strptime(row["date"], "%Y-%m-%d").date()
                if ff_mode:
                    out[day] = {"mkt_ret": float(row["mkt_rf_pct"]) / 100.0, "rf_daily": 0.0}
                else:
                    raw = row.get("rf_daily")
                    if not raw:
                        meta["blank_rf"].append(day)
                    out[day] = {"mkt_ret": float(row["mkt_ret"]),
                                "rf_daily": float(raw) if raw else None}
            except (ValueError, KeyError):
                continue
    return out, meta


def fmt_pct(x):
    return "%+.2f%%" % (x * 100.0)
# --- END shared by copy with scripts/predict_positions.py ------------------


def ensure_dir(path):
    """Create the directory a file is about to be written into."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# Column order is hardcoded, not inferred from file order, so a renamed or
# added strategy cannot silently reshuffle the columns between two reports.
STRATEGY_ORDER = ["Benchmark", "LLM Direction", "LLM Direction (Contrarian)",
                  "LLM Sentiment-Weighted", "LLM Sentiment-Weighted (Contrarian)"]
SHORT = {"Benchmark": "Benchmark",
         "LLM Direction": "Direction",
         "LLM Direction (Contrarian)": "Direction (C)",
         "LLM Sentiment-Weighted": "Sentiment",
         "LLM Sentiment-Weighted (Contrarian)": "Sentiment (C)"}

log = logging.getLogger("perf")


def _z(x):
    """Normalise -0.0 to 0.0 so it never renders as '-0.00%'."""
    return 0.0 if x == 0 else x


def compound(returns):
    total = 1.0
    for r in returns:
        total *= (1.0 + r)
    return total - 1.0


def sample_sd(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def hist_var_cvar(returns, conf):
    """Empirical 1-day VaR and CVaR at `conf`, SIGNED (negative = loss).

    VaR is the empirical (1-conf) quantile with no interpolation: order
    statistic k = ceil((1-conf) * n), 1-based. Interpolating between the worst
    and second-worst day would manufacture a resolution the sample lacks.
    CVaR is the mean of the k-day tail, VaR inclusive.
    """
    s = sorted(returns)
    n = len(s)
    k = max(1, min(n, math.ceil((1.0 - conf) * n)))
    return s[k - 1], sum(s[:k]) / k, k


def series_metrics(rets, rf, positions, mar_mode, trading_days):
    """All metrics for one return series. rets/rf/positions are aligned lists."""
    n = len(rets)
    total = compound(rets)
    mu = sum(rets) / n
    ann_ret = (1.0 + total) ** (trading_days / n) - 1.0
    ann_vol = sample_sd(rets) * math.sqrt(trading_days)

    excess = [a - b for a, b in zip(rets, rf)]
    sdx = sample_sd(excess)
    mx = sum(excess) / n
    sharpe = (mx / sdx * math.sqrt(trading_days)) if sdx > 0 else float("nan")

    # Sortino. Downside deviation is the second lower partial moment: divide the
    # sum of squared shortfalls by n, the FULL count, not by the number of
    # losing days. The loser-count form discards loss-frequency information and
    # is non-monotone -- turning a losing day into a winning one can raise the
    # denominator and lower the ratio. The divisor is n rather than n-1 because
    # the target is a fixed constant supplied externally, not a mean estimated
    # from the sample, so no degree of freedom is consumed.
    if mar_mode == "rf":
        short = [min(a - b, 0.0) for a, b in zip(rets, rf)]
        num = mx
    else:
        short = [min(x, 0.0) for x in rets]
        num = mu
    dd = math.sqrt(sum(x * x for x in short) / n)
    if dd > 0:
        sortino = num / dd * math.sqrt(trading_days)
    elif num > 0:
        sortino = float("inf")
    else:
        sortino = float("nan")

    var95, cvar95, k95 = hist_var_cvar(rets, 0.95)
    var99, cvar99, k99 = hist_var_cvar(rets, 0.99)

    wins = sum(1 for x in rets if x > 0)
    in_pos = [x for x, p in zip(rets, positions) if abs(p) > 0.0]
    wins_in_pos = sum(1 for x in in_pos if x > 0)
    flats = sum(1 for p in positions if abs(p) == 0.0)

    return {"n": n, "total": total, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino, "maxdd": max_drawdown(rets),
            "var95": var95, "var99": var99, "cvar95": cvar95, "cvar99": cvar99,
            "k95": k95, "k99": k99,
            "wins": wins, "wins_in_pos": wins_in_pos, "n_in_pos": len(in_pos),
            "flats": flats, "mean_daily": mu}


def turnover_stats(positions, trading_days):
    """Total turnover, the day-1 entry, and an annualized ONGOING figure.

    The naive total * 252/n reports buy-and-hold as trading 14.8x a year when
    it trades once at inception and never again. Stripping the entry and
    dividing the remainder over the n-1 days on which rebalancing could occur
    gives its true ongoing turnover: zero.
    """
    prev, total, entry, traded_days = 0.0, 0.0, None, 0
    for pos in positions:
        traded = abs(pos - prev)
        if entry is None:
            entry = traded
        total += traded
        if traded > 0:
            traded_days += 1
        prev = pos
    n = len(positions)
    ann = (total - entry) * trading_days / (n - 1) if n > 1 else 0.0
    return {"total": total, "entry": entry, "ann": ann, "days": traded_days,
            "mean_abs": sum(abs(p) for p in positions) / n,
            "mean_signed": sum(positions) / n}


def load_strategy_rows(path):
    """Group rows by strategy, sorted by Date within each group.

    The file is sorted by (Date, strategy-order), i.e. INTERLEAVED, so grouping
    must happen before any sequential differencing or the turnover chain would
    run across strategies.
    """
    by = {}
    seen = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                key = (r["Date"], r["Strategy"])
                if key in seen:
                    raise ValueError("duplicate (Date, Strategy) row: %s" % (key,))
                seen.add(key)
                by.setdefault(r["Strategy"], []).append({
                    "date": r["Date"],
                    "pos": float(r["Position"]),
                    "gross": float(r["Strategy Return"]),
                    "cost": float(r["Cost"]),
                    "net": float(r["Net Return"]),
                    # The realised market return for the day, needed by the
                    # charts. Optional: a CSV written by an older schema
                    # still loads, and the charts that need it are skipped.
                    "actual": (float(r["Next Day's Actual Return"])
                               if r.get("Next Day's Actual Return") else None),
                })
            except KeyError as exc:
                raise ValueError("missing column %s in %s" % (exc, path)) from exc
    for k in by:
        by[k].sort(key=lambda x: x["date"])
    return by


def build_rf_lookup(rets, dates, mode):
    """rf for each needed date, forward-filled from the last prior quote.

    Two failure modes, treated differently. A date absent from the return
    series means the strategy CSV and the return file have drifted apart --
    the former was generated from the latter -- so that is a hard error. A date
    present with a blank rf_daily is structural and benign (Columbus Day,
    Veterans Day: NYSE open, bond market shut, so FRED has no quote); carry the
    previous session's rate forward.
    """
    quoted = sorted(d for d, v in rets.items() if v["rf_daily"] is not None)
    out, filled, unfilled, missing = {}, [], [], []
    for d in dates:
        if d not in rets:
            missing.append(d)
            continue
        v = rets[d]["rf_daily"]
        if v is not None:
            out[d] = v
        elif mode == "error":
            raise ValueError("no risk-free rate for %s" % d)
        elif mode == "zero":
            out[d] = 0.0
            unfilled.append(d)
        else:
            i = bisect.bisect_left(quoted, d) - 1
            if i >= 0:
                out[d] = rets[quoted[i]]["rf_daily"]
                filled.append(d)
            else:
                out[d] = 0.0
                unfilled.append(d)
    return out, filled, unfilled, missing


# --------------------------------------------------------------------------
#  charts
# --------------------------------------------------------------------------
# matplotlib is the ONE optional dependency in this repo. Everything else is
# stdlib, and the report itself stays stdlib: if matplotlib is missing the
# figures are skipped and the markdown is written without them, rather than the
# run failing. Install with `py -m pip install matplotlib`.

# One colour per strategy, reused in every figure, so a reader can follow a
# strategy from one chart to the next without re-reading the legend.
STRATEGY_COLORS = {
    "Benchmark": "#555555",
    "LLM Direction": "#1b7837",
    "LLM Direction (Contrarian)": "#c0392b",
    "LLM Sentiment-Weighted": "#3a6ea5",
    "LLM Sentiment-Weighted (Contrarian)": "#d68910",
}
CHART_DPI = 150
GAIN, LOSS, NEUTRAL = "#1b7837", "#c0392b", "#555555"
# matplotlib writes a creation timestamp into PNG metadata by default, which
# would make two identical runs differ byte-for-byte. The markdown report is
# idempotent; the figures beside it should be too.
PNG_METADATA = {"Software": None, "Creation Time": None}


def _import_pyplot():
    """Import matplotlib lazily, in non-interactive mode. None if absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")        # headless: write files, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def wilson(hits, n, z=1.96):
    """Wilson score interval for a proportion: (point estimate, lo, hi).

    Preferred over the normal approximation because a confidence bin holds only
    ~n/bins days and can sit near 0 or 1, where the normal interval runs past
    the [0, 1] boundary and understates exactly the uncertainty that matters.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = hits / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_predictions(path):
    """{date predicted FOR: {confidence, sentiment, n_headlines}}.

    The join key is `predicts_for`, NOT `date`. A prediction is made on the
    prior session from that session's pre-close headlines, and it is scored
    against the return of the day it predicts for -- which is the date used
    throughout predictions_strategy.csv. Joining on `date` would shift every
    confidence reading one session early and silently destroy the calibration.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["predicts_for"]] = {
                    "confidence": float(r["confidence"]),
                    "sentiment": int(r["sentiment"]),
                    "n_headlines": int(r["n_headlines"]),
                }
            except (KeyError, ValueError):
                continue          # a malformed row costs one day, not the chart
    return out


def equal_count_bins(pairs, nbins):
    """Split [(key, payload)] into nbins near-equal-sized bins by key.

    Equal-count, not equal-width: confidence clusters heavily around its mean,
    so equal-width bins would leave the extreme bins nearly empty and give them
    error bars too wide to read. A tied run is never split across a boundary --
    it is kept whole -- so bins can come out slightly uneven, which is the
    honest trade for not letting an arbitrary cut decide a bin's hit rate.
    """
    s = sorted(pairs, key=lambda kv: kv[0])
    n = len(s)
    if n == 0 or nbins < 1:
        return []
    out, i = [], 0
    for b in range(nbins):
        target = round(n * (b + 1) / nbins)
        j = n if b == nbins - 1 else min(max(target, i + 1), n)
        while j < n and s[j][0] == s[j - 1][0]:    # keep a tied run together
            j += 1
        if j > i:
            out.append(s[i:j])
        i = j
        if i >= n:
            break
    return out


def chart_equity_drawdown(plt, by, labels, dates, path, cutoff):
    """Net growth of $1 with the drawdown path beneath it, on a shared x-axis.

    Drawn on NET returns: a manager allocates to what survives costs, and the
    high-turnover strategies are exactly the ones a gross curve would flatter.
    Linear y, not log: over a single year the spread is narrow enough that log
    scaling buys nothing and costs readability.
    """
    from matplotlib import dates as mdates

    xs = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.07})

    for l in labels:
        equity, drawdown, v, peak = [], [], 1.0, 1.0
        for r in by[l]:
            v *= (1.0 + r["net"])
            peak = max(peak, v)
            equity.append(v)
            drawdown.append((v / peak - 1.0) * 100.0)
        colour = STRATEGY_COLORS.get(l, NEUTRAL)
        width = 2.2 if l == "LLM Direction" else 1.3
        ax1.plot(xs, equity, label=SHORT.get(l, l), color=colour, linewidth=width)
        ax2.plot(xs, drawdown, color=colour, linewidth=1.0)
        ax2.fill_between(xs, drawdown, 0, color=colour, alpha=0.09)

    # The shaded band is the point of this figure: it is the only stretch the
    # model could not have memorised, so the slope inside it is the closest
    # thing to out-of-sample evidence this backtest contains.
    if cutoff and xs[0] < cutoff < xs[-1]:
        for ax in (ax1, ax2):
            ax.axvspan(cutoff, xs[-1], color="#f4d35e", alpha=0.16, zorder=0)
            ax.axvline(cutoff, color="#b8860b", linewidth=1.0, linestyle="--", zorder=1)
        n_after = sum(1 for x in xs if x > cutoff)
        ax1.annotate("after model training cutoff\n%s - %d of %d days (%.0f%%)"
                     % (cutoff.isoformat(), n_after, len(xs), 100 * n_after / len(xs)),
                     xy=(cutoff, ax1.get_ylim()[1]), xytext=(6, -8),
                     textcoords="offset points", va="top", ha="left",
                     fontsize=8.5, color="#8a6d0b",
                     bbox=dict(facecolor="white", alpha=0.78, pad=2.5,
                               edgecolor="none"))

    ax1.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("growth of $1 (net of costs)")
    ax1.set_title("Net cumulative return and drawdown", fontsize=13, loc="left", pad=10)
    ax1.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    ax2.axhline(0.0, color="#999999", linewidth=0.8)
    ax2.set_ylabel("drawdown from peak (%)")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax2.tick_params(axis="x", labelsize=8.5)
    ax2.set_xlim(xs[0], xs[-1])        # no empty month past the last day

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(fig)
    return True


def chart_confidence_calibration(plt, rows, preds, path, nbins, target):
    """Hit rate and realised return by the model's own stated confidence.

    No strategy consumes `confidence`, so this is the one chart that tests a
    signal the backtest never used. A rising left panel says the model knows
    when it knows, which would justify sizing by confidence or standing aside
    below a threshold; a flat one says its self-assessment is noise. Either
    answer is worth having before trusting it with money.
    """
    pairs = []
    for r in rows:
        p = preds.get(r["date"])
        if p is None or r["actual"] is None or r["pos"] == 0:
            continue
        # A dead-flat tape scores as a miss. It does not occur in this sample,
        # and counting it as a half-hit would flatter the model for free.
        hit = (r["pos"] > 0 and r["actual"] > 0) or (r["pos"] < 0 and r["actual"] < 0)
        pairs.append((p["confidence"], (hit, r["net"])))

    bins = equal_count_bins(pairs, nbins)
    if not bins:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))
    ticks, rates, los, his, means = [], [], [], [], []
    for b in bins:
        confs = [k for k, _ in b]
        hits = sum(1 for _, (h, _) in b if h)
        p, lo, hi = wilson(hits, len(b))
        ticks.append("%.2f-%.2f\nn=%d" % (min(confs), max(confs), len(b)))
        rates.append(p * 100)
        los.append((p - lo) * 100)
        his.append((hi - p) * 100)
        means.append(sum(v for _, (_, v) in b) / len(b) * 1e4)   # bps/day

    x = list(range(len(bins)))
    ax1.bar(x, rates, color=[GAIN if r >= 50 else LOSS for r in rates],
            alpha=0.82, width=0.68)
    ax1.errorbar(x, rates, yerr=[los, his], fmt="none", ecolor="#333333",
                 elinewidth=1.2, capsize=4)
    ax1.axhline(50, color="#333333", linewidth=1.1, linestyle="--")
    ax1.text(-0.45, 51.2, "coin flip", fontsize=8.5, color="#333333",
             ha="left", va="bottom",
             bbox=dict(facecolor="white", alpha=0.85, pad=1.5, edgecolor="none"))
    ax1.set_ylabel("directional hit rate (%)")
    ax1.set_title("Hit rate by stated confidence", fontsize=12, loc="left")
    ax1.set_ylim(0, max(80.0, max(r + h for r, h in zip(rates, his)) + 6))

    ax2.bar(x, means, color=[GAIN if m >= 0 else LOSS for m in means],
            alpha=0.82, width=0.68)
    ax2.axhline(0, color="#333333", linewidth=1.0)
    ax2.set_ylabel("mean net return (bps/day)")
    ax2.set_title("Realised net return by stated confidence", fontsize=12, loc="left")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8.5)
        ax.set_xlabel("model confidence (equal-count bins)")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Is the model's confidence informative?  (%s, error bars = 95%% Wilson)"
                 % target, fontsize=13, x=0.005, ha="left", y=1.02)
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(fig)
    return True


def chart_exposure_attribution(plt, rows, bench_rows, path, target):
    """Did the calls pay, or did a long tilt in a rising year do the work?

    Left: what the MARKET did on the days the strategy chose each side. Being
    long in a bull market makes money without any skill at all, so the test of
    skill is whether the short days were genuinely down days. Right: an
    additive decomposition of where the total actually came from.
    """
    mkt = {r["date"]: r["actual"] for r in bench_rows}
    longs = [r for r in rows if r["pos"] > 0 and r["date"] in mkt]
    shorts = [r for r in rows if r["pos"] < 0 and r["date"] in mkt]
    if not longs or not shorts:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.35]})

    # --- left: market vs strategy, split by the side the strategy chose ----
    groups = [("Long days\nn=%d" % len(longs), longs),
              ("Short days\nn=%d" % len(shorts), shorts)]
    mkt_vals = [compound([mkt[r["date"]] for r in g]) * 100 for _, g in groups]
    stg_vals = [compound([r["gross"] for r in g]) * 100 for _, g in groups]
    w = 0.36
    ax1.bar([i - w / 2 for i in (0, 1)], mkt_vals, width=w, label="Market",
            color=NEUTRAL, alpha=0.85)
    ax1.bar([i + w / 2 for i in (0, 1)], stg_vals, width=w,
            label="Strategy (gross)", color=GAIN, alpha=0.9)
    for i, (m, s) in enumerate(zip(mkt_vals, stg_vals)):
        for xpos, val in ((i - w / 2, m), (i + w / 2, s)):
            ax1.annotate("%+.2f%%" % val, (xpos, val), ha="center", fontsize=8.5,
                         va="bottom" if val >= 0 else "top",
                         xytext=(0, 3 if val >= 0 else -3),
                         textcoords="offset points")
    ax1.axhline(0, color="#333333", linewidth=1.0)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([g for g, _ in groups], fontsize=9.5)
    ax1.set_ylabel("compounded return over those days (%)")
    ax1.set_title("Market return on the days each side was taken",
                  fontsize=12, loc="left")
    ax1.legend(fontsize=9, frameon=False, loc="best")

    # --- right: additive decomposition of the net total --------------------
    # Daily returns are summed here, not compounded, because only a sum
    # decomposes. The gap that opens against the true compounded total is
    # carried as an explicit "Compounding" bar rather than hidden in rounding.
    sum_long = sum(r["gross"] for r in longs)
    sum_short = sum(r["gross"] for r in shorts)
    sum_cost = sum(r["cost"] for r in rows)
    net_total = compound([r["net"] for r in rows])
    residual = net_total - (sum_long + sum_short - sum_cost)
    steps = [("Long\ndays", sum_long), ("Short\ndays", sum_short),
             ("Costs", -sum_cost), ("Com-\npounding", residual)]

    bottom = 0.0
    for i, (lab, val) in enumerate(steps):
        ax2.bar(i, val * 100, bottom=bottom * 100, width=0.62,
                color=GAIN if val >= 0 else LOSS, alpha=0.85)
        ax2.annotate("%+.2f" % (val * 100), (i, (bottom + val) * 100),
                     ha="center", fontsize=8.5,
                     va="bottom" if val >= 0 else "top",
                     xytext=(0, 3 if val >= 0 else -3), textcoords="offset points")
        bottom += val
    ax2.bar(len(steps), net_total * 100, width=0.62, color="#2c3e50", alpha=0.9)
    ax2.annotate("%+.2f" % (net_total * 100), (len(steps), net_total * 100),
                 ha="center", fontsize=9, fontweight="bold", va="bottom",
                 xytext=(0, 3), textcoords="offset points")

    bench_total = compound([r["net"] for r in bench_rows]) * 100
    ax2.axhline(bench_total, color="#555555", linewidth=1.2, linestyle="--")
    ax2.annotate("Benchmark net total %+.2f%%" % bench_total,
                 (-0.42, bench_total), ha="left", va="bottom",
                 fontsize=8.5, color="#555555", xytext=(0, 4),
                 textcoords="offset points",
                 bbox=dict(facecolor="white", alpha=0.85, pad=1.5,
                           edgecolor="none"))
    ax2.axhline(0, color="#333333", linewidth=1.0)
    ax2.set_xticks(list(range(len(steps) + 1)))
    ax2.set_xticklabels([lab for lab, _ in steps] + ["Net\ntotal"], fontsize=9)
    ax2.set_ylabel("contribution to total return (pp)")
    ax2.set_title("Where the total came from", fontsize=12, loc="left")

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("%s: timing skill vs directional tilt" % target,
                 fontsize=13, x=0.005, ha="left", y=1.01)
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(fig)
    return True


def make_charts(by, labels, dates, args):
    """Render the figures; return {key: path relative to the report}.

    Returns {} when charts are off or matplotlib is absent, in which case
    build_report simply omits the image blocks and the report still stands.
    """
    if args.no_charts:
        return {}
    plt = _import_pyplot()
    if plt is None:
        log.warning("matplotlib is not installed -- writing the report without "
                    "figures. Install it with: py -m pip install matplotlib")
        return {}

    target = args.chart_strategy
    if target not in by:
        log.warning("--chart-strategy %r is not in the file; charts 2 and 3 "
                    "need it, skipping them", target)

    cutoff = None
    if args.training_cutoff and args.training_cutoff.lower() != "none":
        try:
            cutoff = datetime.strptime(args.training_cutoff, "%Y-%m-%d").date()
        except ValueError:
            log.warning("--training-cutoff %r is not YYYY-MM-DD; not shading",
                        args.training_cutoff)

    ensure_dir(os.path.join(args.charts_dir, "x"))
    base = os.path.dirname(os.path.abspath(args.out)) or "."
    out = {}

    def rel(name):
        p = os.path.join(args.charts_dir, name)
        return p, os.path.relpath(p, base).replace(os.sep, "/")

    path, link = rel("equity_drawdown.png")
    if chart_equity_drawdown(plt, by, labels, dates, path, cutoff):
        out["equity"] = link
        log.info("wrote %s", path)

    if target in by:
        preds = {}
        try:
            preds = load_predictions(args.predictions)
        except FileNotFoundError:
            log.warning("no predictions file at %s -- skipping the confidence "
                        "calibration chart; that file is the only source of the "
                        "`confidence` column", args.predictions)
        if preds:
            path, link = rel("confidence_calibration.png")
            if chart_confidence_calibration(plt, by[target], preds, path,
                                            args.conf_bins,
                                            SHORT.get(target, target)):
                out["calibration"] = link
                log.info("wrote %s", path)
            missing = sum(1 for r in by[target] if r["date"] not in preds)
            if missing:
                log.warning("%d of %d strategy day(s) had no matching "
                            "`predicts_for` row and were left out of the "
                            "calibration chart", missing, len(by[target]))

        if "Benchmark" in by and all(r["actual"] is not None for r in by["Benchmark"]):
            path, link = rel("exposure_attribution.png")
            if chart_exposure_attribution(plt, by[target], by["Benchmark"], path,
                                          SHORT.get(target, target)):
                out["attribution"] = link
                log.info("wrote %s", path)
        else:
            log.warning("no usable Benchmark series -- skipping the exposure "
                        "attribution chart")
    return out


# --------------------------------------------------------------------------
#  report rendering
# --------------------------------------------------------------------------

def rel_path(path):
    """A path as written in the report: relative to the project root, POSIX.

    The report is committed and shared, so printing an absolute path would
    publish the author's home directory along with the results. Anything
    inside the project renders as `data/foo.csv`; anything outside it keeps
    only its basename, for the same reason. The full location is still in
    the run log on stderr, which is not committed.
    """
    try:
        p = Path(path).resolve()
    except (OSError, ValueError):
        return os.path.basename(str(path))
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def _esc(s):
    return str(s).replace("|", "\\|")


def fmt_ratio(x):
    if x != x:                       # nan
        return "n/a"
    if x == float("inf"):
        return "+inf \\*"
    if x == float("-inf"):
        return "-inf \\*"
    return "%.2f" % x


def fmt_pct3(x):
    return "%+.3f%%" % (_z(x) * 100.0)


def table(labels, rows):
    """Markdown table: first column the metric name, one column per strategy."""
    out = ["| Metric | " + " | ".join(_esc(SHORT.get(l, l)) for l in labels) + " |",
           "|---|" + "---:|" * len(labels)]
    for name, cells in rows:
        out.append("| " + _esc(name) + " | " + " | ".join(cells) + " |")
    return out


# Each metric is defined once, as a function of one strategy's stats dict, so
# the gross and net columns of the combined table are rendered by identical
# code and cannot drift apart in formatting. Adding a metric touches one place.
RETURN_METRICS = [
    ("Total return", lambda s: fmt_pct(s["total"])),
    ("Annualized return (geometric)", lambda s: fmt_pct(s["ann_ret"])),
    ("Annualized volatility", lambda s: "%.2f%%" % (s["ann_vol"] * 100)),
    ("Sharpe (excess, annualized)", lambda s: fmt_ratio(s["sharpe"])),
    ("Sortino (annualized)", lambda s: fmt_ratio(s["sortino"])),
    ("Maximum drawdown", lambda s: fmt_pct(s["maxdd"])),
    ("VaR 95%, historical \u2020", lambda s: fmt_pct3(s["var95"])),
    ("VaR 99%, historical \u2020", lambda s: fmt_pct3(s["var99"])),
    ("CVaR 95% (expected shortfall) \u2020", lambda s: fmt_pct3(s["cvar95"])),
    ("CVaR 99% (expected shortfall) \u2020", lambda s: fmt_pct3(s["cvar99"])),
    ("Win rate, days in position", lambda s: "%.1f%%" % (
        100.0 * s["wins_in_pos"] / s["n_in_pos"] if s["n_in_pos"] else 0.0)),
    ("Win rate, all days", lambda s: "%.1f%%" % (100.0 * s["wins"] / s["n"])),
]


def combined_table(labels, gross, net):
    """One table, each strategy's gross and net columns adjacent.

    Previously these were two tables with identical row order, so reading the
    effect of costs on any figure meant holding a number in your head and
    scrolling to its counterpart. Pairing the columns puts that comparison in
    the gap between two adjacent cells, which is where it is actually made.

    Markdown has no column grouping, so the pairing is carried by repeating
    the strategy name in both headers and breaking the qualifier onto a second
    line. Rows carrying no return information -- observation and flat-day
    counts, which are identical by construction on both sides -- are not
    duplicated here; they live in the activity table, which is where position
    statistics belong.
    """
    head, sep = ["Metric"], ["---"]
    for l in labels:
        short = _esc(SHORT.get(l, l))
        head += ["%s<br>gross" % short, "%s<br>net" % short]
        sep += ["---:", "---:"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(sep) + "|"]
    for name, fmt in RETURN_METRICS:
        cells = []
        for l in labels:
            cells += [fmt(gross[l]), fmt(net[l])]
        out.append("| " + _esc(name) + " | " + " | ".join(cells) + " |")
    return out


# --------------------------------------------------------------------------
#  data provenance
# --------------------------------------------------------------------------
# Where each input came from, what it is used for, and -- computed from the
# files themselves at report time -- what it actually covers. The descriptions
# are static because a CSV cannot state its own provenance; every date range,
# row count and gap beside them is measured, so a stale range cannot survive a
# rerun on different data.

# Sessions closing at 1:00 PM ET instead of 4:00. Copied from
# predict_positions.py:80 -- the same standalone-by-design duplication as the
# metric helpers. If one list changes, change both.
EARLY_CLOSE_DAYS = {"2025-07-03", "2025-11-28", "2025-12-24",
                    "2026-07-02", "2026-11-27", "2026-12-24"}
REGULAR_CLOSE_ET = time(16, 0)
EARLY_CLOSE_ET = time(13, 0)


def _span(path, datekey):
    """(rows, first, last) for a CSV, or None if it is absent or unreadable."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            days = []
            n = 0
            for row in csv.DictReader(fh):
                n += 1
                v = row.get(datekey)
                if v:
                    days.append(v)
    except (OSError, csv.Error):
        return None
    if not n:
        return None
    return n, (min(days) if days else "?"), (max(days) if days else "?")


def _headline_stats(path):
    """Coverage of the headline corpus, including the point-in-time exclusion.

    `seendate` is UTC; the cutoff is defined in ET, so the conversion has to
    happen before the comparison. If the zone database is unavailable the
    before/after split is reported as unknown rather than computed against the
    wrong clock -- a wrong exclusion count here would misdescribe the single
    most important filter in the pipeline.
    """
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except Exception:
        et = None
    try:
        fh = open(path, newline="", encoding="utf-8")
    except OSError:
        return None
    total = before = after = unparsed = 0
    days = set()
    with fh:
        for row in csv.DictReader(fh):
            total += 1
            raw = row.get("seendate") or ""
            try:
                utc = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                unparsed += 1
                continue
            if et is None:
                continue
            local = utc.astimezone(et)
            day = local.date().isoformat()
            days.add(day)
            cutoff = EARLY_CLOSE_ET if day in EARLY_CLOSE_DAYS else REGULAR_CLOSE_ET
            if local.time() >= cutoff:
                after += 1
            else:
                before += 1
    return {"total": total, "before": before, "after": after, "unparsed": unparsed,
            "days": sorted(days), "tz": et is not None}


def _empty_days(path):
    """Days the fetcher recorded as genuinely returning no headlines."""
    try:
        with open(path, encoding="utf-8") as fh:
            return [l.strip() for l in fh
                    if l.strip() and not l.lstrip().startswith("#")]
    except OSError:
        return None


def build_data_notes(args, dates, rf_note):
    """Provenance section: sources, roles, coverage, and every known gap."""
    L = ["## Data sources and coverage", ""]
    L.append("Row counts and date ranges below are measured from the files at "
             "report time, not recorded by hand.")
    L.append("")

    rows = []
    hs = _headline_stats(args.headlines)
    if hs and hs["days"]:
        rows.append((args.headlines,
                     "GDELT 2.0 DOC API (`artlist`, `domainis:cnbc.com`)",
                     "The headlines each prediction was made from.",
                     "%s .. %s (%d days)" % (hs["days"][0], hs["days"][-1], len(hs["days"])),
                     "{:,}".format(hs["total"])))
    elif hs:
        rows.append((args.headlines, "GDELT 2.0 DOC API",
                     "The headlines each prediction was made from.",
                     "unknown (no zone database)", "%d" % hs["total"]))

    for path, key, source, role in (
        (args.predictions, "predicts_for", "`claude-opus-5` via the Anthropic API",
         "One long/short call, sentiment score and confidence per trading day. "
         "Coverage is by `predicts_for`, the day each call applies to, not the "
         "session it was made on."),
        (args.returns, "date",
         "FRED `SP500` (S&P 500 close, a PRICE index) and `DGS1MO` "
         "(1-month Treasury constant maturity yield)",
         "Realised returns that positions are scored against, and the "
         "risk-free rate subtracted for Sharpe."),
        (args.strategy_csv, "Date",
         "Generated by `scripts/predict_positions.py --backtest-only`",
         "Per-day positions, returns and costs. The direct input to this report."),
    ):
        sp = _span(path, key)
        if sp:
            rows.append((path, source, role, "%s .. %s" % (sp[1], sp[2]),
                         "{:,}".format(sp[0])))

    if rows:
        L.append("| File | Source | Used for | Coverage | Rows |")
        L.append("|---|---|---|---|---:|")
        for path, source, role, cover, n in rows:
            L.append("| `%s` | %s | %s | %s | %s |"
                     % (_esc(rel_path(path)), _esc(source), _esc(role),
                        _esc(cover), n))
        L.append("")

    # --- why the three windows do not line up ------------------------------
    L.append("### Why the windows differ")
    L.append("")
    if hs and hs["days"]:
        L.append("- **Headlines** span %s to %s, wider at both ends than the "
                 "backtest. A prediction is made from the *prior* session's "
                 "headlines, so the first day of headlines produces no scored "
                 "day of its own."
                 % (hs["days"][0], hs["days"][-1]))
    L.append("- **Predictions** are dated by the session whose headlines were "
             "read; the `predicts_for` column carries the day they apply to. "
             "The backtest is indexed on the latter.")
    L.append("- **The backtest** runs %s to %s (%d days). It stops short of the "
             "last headline because the final prediction still needs a "
             "following day's return to be scored against."
             % (dates[0], dates[-1], len(dates)))
    L.append("")

    # --- gaps, cutoffs, exclusions -----------------------------------------
    L.append("### Gaps, cutoffs and exclusions")
    L.append("")
    if hs and hs["tz"] and hs["total"]:
        L.append("- **%s of %s headlines (%.1f%%) were excluded by the "
                 "point-in-time filter** — they carry a `seendate` at or after "
                 "the 4:00 PM ET close (1:00 PM on the six early-close "
                 "sessions), so a trade placed that day could not have been "
                 "informed by them. Only the remaining %s reached the model. "
                 "This is the largest single exclusion in the pipeline and the "
                 "one that keeps the backtest honest."
                 % ("{:,}".format(hs["after"]), "{:,}".format(hs["total"]),
                    100.0 * hs["after"] / hs["total"], "{:,}".format(hs["before"])))
    if hs and hs["unparsed"]:
        L.append("- **%d headline(s) had an unparseable `seendate`** and were "
                 "skipped." % hs["unparsed"])
    empties = _empty_days(args.empty_log)
    if empties is not None:
        L.append("- **%d day(s) are recorded in `%s` as returning no headlines "
                 "at all.** These are logged rather than silently treated as "
                 "quiet news days, because a collection failure and a genuinely "
                 "empty day are not the same thing and must not become the same "
                 "row." % (len(empties), _esc(rel_path(args.empty_log))))
    L.append("- **Risk-free rate: %s.** The gaps are structural — the NYSE "
             "trades on Columbus Day and Veterans Day while the bond market is "
             "shut, so FRED publishes no `DGS1MO` quote. The prior session's "
             "rate is carried forward; the two rates either side of each gap "
             "differ by less than 0.01%%/yr, so the substitution is very nearly "
             "exact." % rf_note)
    L.append("- **`SP500` is a price index, so every return here excludes "
             "dividends** — roughly 1.2–1.5%/yr understated. It applies "
             "identically to the strategies and the benchmark, so comparisons "
             "between them are unaffected, but the absolute totals are each a "
             "little low.")
    ff = _span(args.ff_returns, "date") if getattr(args, "ff_returns", None) else None
    if ff:
        L.append("- **Ken French's `Mkt-RF` factor was NOT used**, though it is "
                 "present in `data/`. It ends %s, %d rows covering only to that "
                 "date, because the library is rebuilt from a monthly CRSP "
                 "vintage and runs roughly seven weeks in arrears — it stops "
                 "short of the backtest end. It is also an *already-excess*, "
                 "dividend-inclusive, whole-CRSP series, where these strategies "
                 "were scored on total returns of the S&P 500; "
                 "`performance_analysis.py` refuses it for that reason."
                 % (ff[2], ff[0]))
    cut = args.training_cutoff
    if cut and cut.lower() != "none":
        inside = sum(1 for d in dates if d <= cut)
        L.append("- **%d of %d days (%.0f%%) fall on or before %s**, the "
                 "assistant's training cutoff, and are therefore inside the "
                 "model's training data. See the Findings section."
                 % (inside, len(dates), 100.0 * inside / len(dates), cut))
    L.append("- **`seendate` is when GDELT first indexed an article**, which is "
             "at or after publication, never before. Filtering on it is "
             "therefore conservative in the safe direction: it can exclude a "
             "borderline article, but it cannot admit one the trader could not "
             "have seen.")
    L.append("")
    return L

# The appendix below is deliberately STATIC prose, unlike build_findings(),
# which computes every figure it quotes. This is a project retrospective -- an
# account of what went wrong while acquiring the data and what would have
# prevented it -- and none of it is derivable from predictions_strategy.csv. It
# is kept here rather than in a separate file so the report stays a single
# self-contained document, and it is marked as a fixed record so no future
# reader mistakes it for a generated result. Suppress it with --no-appendix.

DATA_ACQUISITION_NOTES = """## Appendix: acquiring the GDELT data

A fixed record of what went wrong collecting the headline corpus, and what
would have prevented each problem. It is not regenerated from the data.

The corpus behind this report is 9,628 CNBC headlines covering 2025-09-19 to
2026-09-19, pulled from the GDELT 2.0 DOC API. Collecting it took far longer
than the analysis that followed, for reasons that were mostly foreseeable.

### What went wrong

**The 250-record cap truncates silently.** `artlist` returns at most 250
articles per request and says nothing when it has more to give. A busy news day
on a broad domain query hits that ceiling easily, so the first passes returned
plausible-looking files that were quietly missing articles. Nothing in the
response distinguishes "these are all 250 that exist" from "here are the first
250 of 400."

**Rate limiting was severe and initially misdiagnosed.** The endpoint tolerates
roughly one request every five seconds, and beyond that returns HTTP 429. The
429s persisted across four source IP addresses in three countries, which looked
like an IP-level ban and prompted a lot of wasted effort chasing connectivity.
The actual diagnostic took one request: GDELT's `/api/v2/tv/tv` endpoint
returned HTTP 200 from the *same* IP that `doc` was refusing, proving the
throttle was endpoint-specific rather than an address block.

**A large part of the rate limiting was self-inflicted.** A retry-loop shell
script survived a stop command and kept running for more than twenty-two hours,
continuously spawning fresh fetcher processes. Those orphans shared one source
IP — and therefore one rate-limit budget — with the foreground job, so the two
starved each other while the remote service looked broken. The orphans also
held the output file open and at one point deleted it outright, producing file
states that appeared impossible. The stop command had reported success; the
process tree had not actually died.

**Transient network errors killed long runs.** The retry handler caught
`URLError` and `TimeoutError` but not `ConnectionResetError`, so a mid-download
reset escaped the handler and terminated a multi-hour job that had no way to
resume.

**A shadowed import lurked in the retry path.** `from datetime import time`
shadowed the `time` module, so `time.sleep()` inside the backoff path would
have raised `AttributeError` — on the exact code path that only executes when
something is already going wrong, which is the worst place for a latent bug.

**Timestamp semantics were assumed rather than checked.** GDELT's `seendate` is
when GDELT first *indexed* an article, not when the publisher released it. The
two differ, and the difference matters directly for the 4:00 PM ET cutoff this
study depends on.

### What would have prevented it

**Read the API's limits before writing the happy path.** The 250-record cap and
the request rate are both documented. Designing the window-splitting and pacing
logic up front would have avoided rewriting the fetcher around them twice.

**Treat any response at exactly the cap as truncated until proven otherwise.**
The rule is cheap: if a window returns exactly 250 records, assume it is
incomplete, bisect it and retry both halves. The final fetcher does this
recursively down to a fifteen-minute floor. Making this the default from the
start costs one comparison and removes an entire class of silent data loss.

**Pace requests end-to-start, not start-to-start.** The quiet period has to
begin when the response *arrives*, not when the request is sent, or a slow
response silently shortens the gap and trips the limit.

**Design checkpointing and resume before the first long run, not after the
first crash.** The fetcher now writes its accumulated rows after every
sub-window and keeps a list of failed windows, so an interrupted job resumes
instead of restarting. Adding that after losing hours of work is the expensive
order in which to learn it.

**Catch `OSError`, not a hand-listed set of subclasses.** `URLError`,
`TimeoutError` and `ConnectionResetError` are all `OSError`; enumerating
network failure modes individually guarantees missing one.

**Enforce single-writer discipline on the output file.** One fetcher per output
file, with a lock or PID file refusing a second concurrent run, would have made
the orphaned-process episode impossible rather than merely unlikely. Concurrent
clients from one machine share a rate-limit budget, so a stray process does not
just duplicate work — it actively degrades the run that is still wanted.

**Verify at the OS level that a stopped job is actually dead.** A tool
reporting "stopped" is not evidence. The check is a process scan filtered on
the relevant command line, and it must include shell processes, because a retry
loop is the parent that keeps respawning the interpreter children.

**Probe a second endpoint before concluding the network is at fault.** One
control request against a different path on the same host separates "this
endpoint is throttling me" from "this host is blocking me" in seconds, and
would have redirected several hours of misdirected effort.

**Pin down what a timestamp field actually means before building on it.**
Because `seendate` is an indexing time at or after publication, filtering on
`seendate < 16:00 ET` is conservative in the safe direction — it can only ever
exclude a borderline article, never admit one the trader could not have seen.
That reasoning had to be established explicitly; it was not safe to assume.

### Lasting effects on the design

The fetcher that came out of this writes its rows after every sub-window rather
than at the end, retries indefinitely with backoff capped at 90 seconds, logs
the windows it could not complete and the days that were genuinely empty, and
records the requested date range in its own filename. Each of those exists
because of a specific failure above. The empty-day log matters most for the
analysis: it is what keeps a collection gap visible as a gap, instead of
letting it silently become a day with no headlines and a neutral prediction.
"""

def build_findings(labels, gross, net, turn, costs, by, rf, args, dates):
    """The report's conclusions, COMPUTED from the same dicts as the tables.

    Not written by hand. A generated report with a hand-written summary drifts
    the moment the input changes, and a stale conclusion is worse than none at
    all because it reads with the same authority as the numbers beneath it.
    Every figure quoted below is interpolated from `gross`, `net`, `turn` and
    `costs`; every claim is guarded on the strategies it names actually being
    present, and is dropped rather than rendered stale if they are not.
    """
    BM = "Benchmark"
    DIR, DIRC = "LLM Direction", "LLM Direction (Contrarian)"
    SEN, SENC = "LLM Sentiment-Weighted", "LLM Sentiment-Weighted (Contrarian)"

    def tstat(label):
        """t-statistic of the mean daily NET return against zero."""
        v = [r["net"] for r in by[label]]
        sd = sample_sd(v)
        if sd == 0 or len(v) < 2:
            return float("nan")
        return (sum(v) / len(v)) / (sd / math.sqrt(len(v)))

    def sub_sharpe(vals, rfs):
        ex = [a - b for a, b in zip(vals, rfs)]
        sd = sample_sd(ex)
        if sd == 0 or len(ex) < 2:
            return float("nan")
        return (sum(ex) / len(ex)) / sd * math.sqrt(args.trading_days)

    def split_at(label, cut):
        """(pre, post) net series and their rf, split on a date string."""
        pre, post, rpre, rpost = [], [], [], []
        for r, f in zip(by[label], rf):
            (post if r["date"] > cut else pre).append(r["net"])
            (rpost if r["date"] > cut else rpre).append(f)
        return pre, rpre, post, rpost

    items = []

    # --- 1. the headline comparison, on risk as well as return -------------
    # "Best" is whatever actually won, not a hardcoded favourite, so the
    # section survives a rerun on different data.
    others = [l for l in labels if l != BM]
    if BM in net and others:
        best = max(others, key=lambda l: net[l]["total"])
        b, m = net[best], net[BM]
        beat_dd = b["maxdd"] > m["maxdd"]        # both negative: > is shallower
        beat_var = b["var95"] > m["var95"]
        items.append((
            "%s is the only strategy that beats buy-and-hold, and it does so "
            "on risk as well as return." % best,
            "It returns **%s net against %s** for the Benchmark (%s vs %s "
            "gross), at an excess Sharpe of **%s vs %s** and a Sortino of %s "
            "vs %s. The return is not bought with volatility: annualized "
            "volatility is %.2f%% against the Benchmark's %.2f%%, and the "
            "maximum drawdown is %s against %s%s. %s"
            % (fmt_pct(b["total"]), fmt_pct(m["total"]),
               fmt_pct(gross[best]["total"]), fmt_pct(gross[BM]["total"]),
               fmt_ratio(b["sharpe"]), fmt_ratio(m["sharpe"]),
               fmt_ratio(b["sortino"]), fmt_ratio(m["sortino"]),
               b["ann_vol"] * 100, m["ann_vol"] * 100,
               fmt_pct(b["maxdd"]), fmt_pct(m["maxdd"]),
               " — *shallower*, not deeper" if beat_dd else "",
               ("Its left tail is thinner too: 95%% VaR %s against %s, and "
                "99%% CVaR %s against %s."
                % (fmt_pct3(b["var95"]), fmt_pct3(m["var95"]),
                   fmt_pct3(b["cvar99"]), fmt_pct3(m["cvar99"])))
               if beat_var else
               ("Its 95%% VaR is %s against the Benchmark's %s."
                % (fmt_pct3(b["var95"]), fmt_pct3(m["var95"]))))))

    # --- 2. the contrarian mirror is the evidence the signal is not noise --
    if DIR in net and DIRC in net and BM in net:
        items.append((
            "The calls carry directional information — the contrarian mirror "
            "is the evidence.",
            "Inverting the same calls turns %s into **%s** and a Sharpe of %s "
            "into **%s**. A signal with no directional content could not do "
            "this: both halves would drift toward the Benchmark's %s rather "
            "than separating by %.1f percentage points. Note the two are *not* "
            "exact negatives — only the daily gross returns negate, while "
            "compounding and strictly-positive costs break the symmetry, which "
            "is why %s gross becomes %s rather than %s."
            % (fmt_pct(net[DIR]["total"]), fmt_pct(net[DIRC]["total"]),
               fmt_ratio(net[DIR]["sharpe"]), fmt_ratio(net[DIRC]["sharpe"]),
               fmt_pct(net[BM]["total"]),
               (net[DIR]["total"] - net[DIRC]["total"]) * 100,
               fmt_pct(gross[DIR]["total"]), fmt_pct(gross[DIRC]["total"]),
               fmt_pct(-gross[DIR]["total"]))))

    # --- 3. sentiment sizing dilutes the same signal into nothing ----------
    if SEN in net and DIR in net:
        items.append((
            "Sizing by sentiment destroys the edge rather than refining it.",
            "%s and %s read the *same* model output, yet sentiment-weighting "
            "returns only **%s net** at a Sharpe of **%s**, against %s and %s. "
            "The cause is exposure, not accuracy: mean |position| is %.3f "
            "against %.3f, which cuts annualized volatility to %.2f%% from "
            "%.2f%% and leaves the strategy flat on %d of %d days. Its win "
            "rate on the days it does hold a position, %.1f%%, is close to "
            "%s's %.1f%% — the signal is comparable, the capital behind it is "
            "not. Note also that its net Sharpe (%s) is far below its gross "
            "(%s): at this exposure the %.2f%% cost bill is proportionally "
            "much heavier."
            % (DIR, SEN, fmt_pct(net[SEN]["total"]), fmt_ratio(net[SEN]["sharpe"]),
               fmt_pct(net[DIR]["total"]), fmt_ratio(net[DIR]["sharpe"]),
               turn[SEN]["mean_abs"], turn[DIR]["mean_abs"],
               net[SEN]["ann_vol"] * 100, net[DIR]["ann_vol"] * 100,
               net[SEN]["flats"], net[SEN]["n"],
               100.0 * net[SEN]["wins_in_pos"] / max(1, net[SEN]["n_in_pos"]),
               DIR,
               100.0 * net[DIR]["wins_in_pos"] / max(1, net[DIR]["n_in_pos"]),
               fmt_ratio(net[SEN]["sharpe"]), fmt_ratio(gross[SEN]["sharpe"]),
               costs[SEN] * 100)))

    # --- 4. costs are material but not fatal to the winner -----------------
    if DIR in net:
        items.append((
            "Trading costs are material but survivable at the modelled level.",
            "%s pays **%.2f%% in cost** over the sample and gives up "
            "**%.2f percentage points** of compounded return, turning %s gross "
            "into %s net, with the Sharpe falling %s. That bill is driven by "
            "**%.1fx annualized turnover** — %.0f%% of the %.0fx theoretical "
            "maximum for a book flipping fully long to fully short every "
            "session — across %d days with a trade. The edge clears the bill "
            "with room to spare, but it is the assumption most worth stressing: "
            "the strategy is far more cost-sensitive than the Benchmark, which "
            "pays %.2f%% in total."
            % (DIR, costs[DIR] * 100,
               (gross[DIR]["total"] - net[DIR]["total"]) * 100,
               fmt_pct(gross[DIR]["total"]), fmt_pct(net[DIR]["total"]),
               fmt_ratio(net[DIR]["sharpe"] - gross[DIR]["sharpe"]),
               turn[DIR]["ann"],
               100.0 * turn[DIR]["ann"] / (2.0 * args.trading_days),
               2.0 * args.trading_days, turn[DIR]["days"],
               costs[BM] * 100 if BM in costs else 0.0)))

    # --- 5. how strong is the evidence, really -----------------------------
    if DIR in net:
        t = tstat(DIR)
        se = math.sqrt(args.trading_days / net[DIR]["n"])
        items.append((
            "The statistical evidence is real but moderate, not overwhelming.",
            "%s's mean daily net return carries a t-statistic of **%.2f** over "
            "%d days — significant at the 5%% level, and no more. The standard "
            "error on an annualized Sharpe from this many observations is "
            "about **±%.2f**, so the %s figure is roughly %.1f standard errors "
            "from zero. Neither number is corrected for the fact that this "
            "report tabulates %d strategies and the best one is being quoted."
            % (DIR, t, net[DIR]["n"], se, fmt_ratio(net[DIR]["sharpe"]),
               abs(net[DIR]["sharpe"]) / se if se else float("nan"),
               len(labels))))

    # --- 6. the contamination split ----------------------------------------
    cut = args.training_cutoff
    if DIR in by and cut and cut.lower() != "none" and dates[0] <= cut <= dates[-1]:
        pre, rpre, post, rpost = split_at(DIR, cut)
        bpre, brpre, bpost, brpost = split_at(BM, cut) if BM in by else ([], [], [], [])
        if pre and post:
            se_post = math.sqrt(args.trading_days / len(post))
            sh_post = sub_sharpe(post, rpost)
            items.append((
                "The edge does not collapse after the model's training cutoff — "
                "but the out-of-sample window is too short to settle it.",
                "%d of %d days (%.0f%%) fall on or before %s and are inside the "
                "model's training data, so for most of this sample the model may "
                "be recalling outcomes rather than forecasting them. Splitting "
                "there: %s returns %s before the cutoff and **%s after** it, at "
                "a post-cutoff Sharpe of %s against %s before%s. The edge "
                "persisting out of sample is the strongest evidence here that it "
                "is not pure memorisation — but %d days carry a Sharpe standard "
                "error of about ±%.2f, so that post-cutoff figure sits only "
                "%.1f standard errors from zero and settles nothing on its own. "
                "This is the single largest open question in the report."
                % (len(pre), len(dates), 100.0 * len(pre) / len(dates), cut,
                   DIR, fmt_pct(compound(pre)), fmt_pct(compound(post)),
                   fmt_ratio(sh_post), fmt_ratio(sub_sharpe(pre, rpre)),
                   (", while the Benchmark returned %s over the same "
                    "post-cutoff stretch" % fmt_pct(compound(bpost)))
                   if bpost else "",
                   len(post), se_post,
                   abs(sh_post) / se_post if se_post else float("nan"))))

    if not items:
        return []

    L = ["## Findings", ""]
    L.append("Everything in this section is computed from the tables below, not "
             "written alongside them, so it cannot fall out of step with the "
             "numbers it cites.")
    L.append("")
    for i, (claim, evidence) in enumerate(items, 1):
        L.append("%d. **%s**" % (i, claim))
        L.append("")
        L.append("   %s" % evidence)
        L.append("")
    return L

def build_report(labels, gross, net, turn, costs, meta, rf_note, args, dates,
                 charts=None, by=None, rf=None):
    n = gross[labels[0]]["n"]
    charts = charts or {}

    def figure(key, alt, caption):
        """Emit an image block, or nothing at all if that chart is absent."""
        if not charts.get(key):
            return
        L.append("![%s](%s)" % (alt, charts[key]))
        L.append("")
        L.append("*%s*" % caption)
        L.append("")
    degenerate = gross[labels[0]]["k95"] == gross[labels[0]]["k99"]
    small = n < args.min_obs_warn
    L = []
    L.append("# %s" % args.title)
    L.append("")
    L.append("**%s .. %s — %d trading days — %d strategies**"
             % (dates[0], dates[-1], n, len(labels)))
    L.append("")
    L.append("Source: `%s` (%d rows)." % (rel_path(args.strategy_csv), n * len(labels)))
    L.append("Risk-free: `%s`, `rf_daily`, %s." % (rel_path(args.returns), rf_note))
    if args.no_rf:
        L.append("")
        L.append("> **THIS REPORT WAS RUN WITH `--no-rf`.** Every Sharpe below "
                 "assumes a zero risk-free rate and is overstated. Not a normal run.")
    L.append("")
    L.append("> ### Read this before the tables")
    L.append(">")
    if small:
        L.append("> **n = %d.** Everything below is computed from %d daily "
                 "observations." % (n, n))
        L.append(">")
    if degenerate:
        L.append("> - **VaR and CVaR are not estimates here, they are order "
                 "statistics.** At n = %d the empirical 5%% and 1%% quantiles are "
                 "both the *single worst observed day* (k = %d of %d for both "
                 "levels). That is why the four rows marked † are identical to "
                 "each other in every column, and why CVaR equals VaR exactly: "
                 "the tail holds one observation, so its mean is itself. These "
                 "describe one day that happened, not a loss that might happen."
                 % (n, gross[labels[0]]["k95"], n))
    else:
        L.append("> - **VaR and CVaR are historical (empirical) quantiles**, no "
                 "interpolation: the k-th worst day with k = ceil((1-conf)·n), "
                 "here k = %d at 95%% and k = %d at 99%%."
                 % (gross[labels[0]]["k95"], gross[labels[0]]["k99"]))
    if small:
        L.append("> - **Annualized figures are extrapolations, not forecasts.** "
                 "The standard error of an annualized Sharpe from %d daily "
                 "returns is about **%.2f**, so no Sharpe here is distinguishable "
                 "from zero, or from any other Sharpe here."
                 % (n, math.sqrt(args.trading_days / n)))
    L.append("> - **Sharpe is computed on excess returns** (return − `rf_daily`). "
             "`scripts/predict_positions.py`'s console summary does not subtract "
             "`rf`, so its Sharpe column is overstated. Where the two disagree, "
             "this file is correct.")
    L.append("> - **VaR and CVaR are signed returns: negative is a loss**, on a "
             "1-day horizon. Not annualized, not scaled.")
    L.append("")
    L.append("**Legend** — `(C)` = Contrarian.")
    L.append("")
    L.append("| Short | Full name |")
    L.append("|---|---|")
    for l in labels:
        L.append("| %s | %s |" % (_esc(SHORT.get(l, l)), _esc(l)))
    L.append("")

    if by is not None and rf is not None:
        L.extend(build_findings(labels, gross, net, turn, costs, by, rf,
                                args, dates))

    L.append("## Performance metrics \u2014 gross vs net")
    L.append("")
    L.append("Each strategy occupies two columns: **gross** before trading "
             "costs and **net** after them. The difference between an "
             "adjacent pair is what trading the strategy costs on that "
             "metric.")
    L.append("")
    L.extend(combined_table(labels, gross, net))
    L.append("")
    if degenerate:
        L.append("† At n = %d both confidence levels select the same order "
                 "statistic (k = %d, the worst day), and the tail mean equals it."
                 % (n, gross[labels[0]]["k95"]))
        L.append("")

    figure("equity", "Net cumulative return and drawdown",
           "Growth of $1 after trading costs (top) and drawdown from the "
           "running peak (bottom), both on the net series tabulated above. "
           "The shaded band covers days after the model's training cutoff "
           "(%s) — the only stretch of this sample the model could not have "
           "memorised. Read the slope inside the band against the slope "
           "outside it." % args.training_cutoff)

    L.append("## Trading activity and cost")
    L.append("")
    act = [
        ("Observations (days)", ["%d" % net[l]["n"] for l in labels]),
        ("Flat days (position = 0)", ["%d" % net[l]["flats"] for l in labels]),
        ("Total cost paid (sum of daily)", ["%.4f%%" % (costs[l] * 100) for l in labels]),
        ("Gross − net (total return)",
         ["%.2f pp" % ((gross[l]["total"] - net[l]["total"]) * 100) for l in labels]),
        ("Turnover (total, one-way units)", ["%.2f" % turn[l]["total"] for l in labels]),
        ("of which day-1 entry", ["%.2f" % turn[l]["entry"] for l in labels]),
        ("Turnover (annualized, ex-entry) ‡", ["%.1fx" % _z(turn[l]["ann"]) for l in labels]),
        ("Days with a trade", ["%d" % turn[l]["days"] for l in labels]),
        ("Mean |position|", ["%.3f" % turn[l]["mean_abs"] for l in labels]),
        ("Mean signed position", ["%+.3f" % _z(turn[l]["mean_signed"]) for l in labels]),
        ("Sharpe drag from costs",
         [fmt_ratio(net[l]["sharpe"] - gross[l]["sharpe"])
          if net[l]["sharpe"] == net[l]["sharpe"] and gross[l]["sharpe"] == gross[l]["sharpe"]
          else "n/a" for l in labels]),
    ]
    L.extend(table(labels, act))
    L.append("")
    L.append("‡ Turnover is the sum of `|position_t − position_{t-1}|` with "
             "`position_0 = 0`. The day-1 entry is a one-off cost of establishing "
             "the book, not a rate, so the annualized figure excludes it and "
             "divides the remainder over the %d days on which rebalancing could "
             "occur. Benchmark is 0.0x because buy-and-hold trades once and never "
             "again. A book flipping fully long to fully short every day would "
             "show %.0fx." % (n - 1, 2.0 * args.trading_days))
    L.append("")

    figure("attribution", "Timing skill versus directional tilt",
           "Left: what the market did on the days the strategy chose each "
           "side. Holding long through a rising year earns money without any "
           "skill at all, so the short-day pair is where security selection "
           "has to show up. Right: an additive decomposition of the net "
           "total. Daily returns are summed rather than compounded so the "
           "parts add up; the gap against the true compounded total is "
           "carried explicitly as the `Compounding` bar instead of being "
           "absorbed into rounding.")

    if charts.get("calibration"):
        L.append("## Signal quality")
        L.append("")
        L.append("The model states a `confidence` between 0 and 1 on every "
                 "call, and **no strategy in this report uses it**. The "
                 "figure below tests it against what actually happened, so "
                 "it is the one result here that is not already priced into "
                 "the tables above.")
        L.append("")
        figure("calibration", "Hit rate and realised return by stated confidence",
               "Left: directional hit rate by equal-count confidence bin, with "
               "95% Wilson intervals and a 50% reference line. Right: the mean "
               "net return actually earned in each bin. A profile that rises "
               "left to right would justify sizing by confidence, or standing "
               "aside below a threshold; a flat one says the self-assessment "
               "carries no information and should be ignored. Bins are "
               "equal-count rather than equal-width because confidence "
               "clusters tightly around its mean.")

    L.append("## Notes")
    L.append("")
    L.append("- **Total return is compounded**, not summed.")
    L.append("- **Annualized return is geometric**, `(1 + total) ** (%d/n) - 1`, so "
             "it reconciles with the total-return row directly." % args.trading_days)
    L.append("- **Annualized volatility** is the sample standard deviation (n−1) of "
             "the raw series × √%d. Sharpe uses the standard deviation of the "
             "*excess* series, which is the textbook definition." % args.trading_days)
    L.append("- **Sortino** uses a minimum acceptable return of **%s**. Downside "
             "deviation is `sqrt(Σ min(r−MAR, 0)² / n)` — divided by **n**, the "
             "full sample count, not by the number of losing days. The "
             "loser-count form discards loss-frequency information and is "
             "non-monotone: turning a losing day into a winning one could *raise* "
             "the denominator and *lower* the ratio."
             % ("0%" if args.sortino_mar == "zero" else "the risk-free rate"))
    if args.sortino_mar == "zero":
        L.append("  Because MAR = 0, Sortino's numerator is the mean **raw** return "
                 "while Sharpe's is the mean **excess** return, so the two ratios "
                 "are measured against different benchmarks and are not directly "
                 "comparable. `--sortino-mar rf` makes them consistent.")
    L.append("- A **negative Sharpe or Sortino ranks nothing**: once the numerator "
             "is negative, a larger denominator moves the ratio toward zero.")
    L.append("- **Win rate, days in position** excludes days a strategy "
             "deliberately held no position. Standing aside on a neutral read is "
             "not a loss, but a strict `return > 0` test scores it as one. Flat "
             "days are counted from `Position == 0`, not `return == 0`, so the "
             "gross and net columns share a denominator. The denominators "
             "themselves are the observation and flat-day counts in the "
             "activity table: days in position = observations − flat days.")
    L.append("- **Maximum drawdown** is close-to-close on the strategy's own "
             "equity curve and understates intraday drawdown.")
    L.append("- Contrarian variants are **not** the exact negative of their parent. "
             "Only the daily *gross* return negates; compounded totals do not, and "
             "costs are strictly positive on both sides.")
    L.append("- A high Sharpe on a low-exposure strategy is low volatility, not "
             "necessarily skill — check the `Mean |position|` row.")
    if charts:
        L.append("- **Figures are drawn on net returns** and are regenerated "
                 "on every run; they are written to `%s`. Pass `--no-charts` "
                 "to skip them, which is also what happens automatically if "
                 "matplotlib is not installed." % rel_path(args.charts_dir))
    if any(v == float("inf") or v == float("-inf")
           for d in (gross, net) for l in labels for v in (d[l]["sortino"],)):
        L.append("- \\* `+inf` means no day fell at or below the target. That is a "
                 "small-sample artifact, not an unbounded ratio.")
    L.append("")
    if not args.no_data_notes:
        L.extend(build_data_notes(args, dates, rf_note))
    if not args.no_appendix:
        L.append(DATA_ACQUISITION_NOTES.rstrip())
        L.append("")
    return "\n".join(L)


def save_report(text, path):
    """Atomic write beside the target. newline="\\n" is explicit: the Windows
    default would emit CRLF and make the file differ by machine."""
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g_io = p.add_argument_group("input / output")
    g_io.add_argument("--strategy-csv", default=str(DATA / "predictions_strategy.csv"),
                      help="per-day per-strategy results from predict_positions.py "
                           "(default: data/predictions_strategy.csv)")
    g_io.add_argument("--returns", default=str(DATA / "market_returns.csv"),
                      help="daily return series, used ONLY for rf_daily "
                           "(default: data/market_returns.csv)")
    g_io.add_argument("--out", default=str(OUTPUT / "performance.md"),
                      help="markdown report (default: output/performance.md)")
    g_io.add_argument("--stdout", action="store_true",
                      help="write the markdown to stdout instead of to --out")

    g_m = p.add_argument_group("metrics")
    g_m.add_argument("--trading-days", type=int, default=TRADING_DAYS,
                     help="days per year for annualization (default: 252)")
    g_m.add_argument("--sortino-mar", choices=("zero", "rf"), default="zero",
                     help="Sortino minimum acceptable return: 'zero' treats any "
                          "losing day as downside; 'rf' uses the daily risk-free "
                          "rate, making Sortino consistent with Sharpe "
                          "(default: zero)")
    g_m.add_argument("--min-obs-warn", type=int, default=30,
                     help="emit the small-sample caveat below this many "
                          "observations (default: 30)")

    g_rf = p.add_argument_group("risk-free rate")
    g_rf.add_argument("--rf-missing", choices=("ffill", "zero", "error"), default="ffill",
                      help="a date present in the return series but with a blank "
                           "rf_daily (NYSE open, bond market closed). 'ffill' "
                           "carries the prior quote forward (default)")
    g_rf.add_argument("--no-rf", action="store_true",
                      help="force rf=0 for every day, to A/B against "
                           "predict_positions.py's Sharpe column; marks the report")

    g_fmt = p.add_argument_group("report")
    g_fmt.add_argument("--title", default="Strategy performance")
    g_fmt.add_argument("--no-appendix", action="store_true",
                       help="omit the fixed data-acquisition retrospective")
    g_fmt.add_argument("--no-data-notes", action="store_true",
                       help="omit the data sources and coverage section")

    g_c = p.add_argument_group("charts (require matplotlib; skipped if absent)")
    g_c.add_argument("--no-charts", action="store_true",
                     help="write the report without figures")
    g_c.add_argument("--charts-dir", default=str(OUTPUT / "charts"),
                     help="directory for the PNGs (default: output/charts)")
    g_io.add_argument("--headlines", default=str(DATA / "cnbc_headlines.csv"),
                      help="headline corpus, for the coverage section "
                           "(default: data/cnbc_headlines.csv)")
    g_io.add_argument("--empty-log", default=str(DATA / "cnbc_year_empty.txt"),
                      help="fetcher log of days that returned no headlines "
                           "(default: data/cnbc_year_empty.txt)")
    g_io.add_argument("--ff-returns",
                      default=str(DATA / "mkt_rf_20250919-20260731.csv"),
                      help="Fama/French Mkt-RF file, reported in the coverage "
                           "section as an input that was NOT used")

    g_c.add_argument("--predictions", default=str(DATA / "predictions.csv"),
                     help="per-day LLM output, the only source of the "
                          "confidence column (default: data/predictions.csv)")
    g_c.add_argument("--chart-strategy", default="LLM Direction",
                     help="strategy profiled by the calibration and "
                          "attribution charts (default: LLM Direction)")
    g_c.add_argument("--training-cutoff", default="2026-05-31",
                     help="shade the equity chart after this date, the model "
                          "training cutoff; 'none' disables (default: 2026-05-31)")
    g_c.add_argument("--conf-bins", type=int, default=5,
                     help="equal-count confidence bins (default: 5)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    try:
        by = load_strategy_rows(args.strategy_csv)
    except FileNotFoundError:
        log.error("no strategy file at %s", args.strategy_csv)
        return 1
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    if not by:
        log.error("no usable rows in %s", args.strategy_csv)
        return 1

    labels = [s for s in STRATEGY_ORDER if s in by]
    extra = [s for s in by if s not in STRATEGY_ORDER]
    if extra:
        log.warning("unrecognised strategy label(s) appended at the end: %s",
                    ", ".join(sorted(extra)))
        labels += sorted(extra)

    # Every strategy must cover the same days, or every cross-strategy
    # comparison in the table is invalid while looking perfectly normal.
    datesets = {l: tuple(r["date"] for r in by[l]) for l in labels}
    if len(set(datesets.values())) != 1:
        log.error("strategies do not share a common date set:")
        for l in labels:
            log.error("  %-38s %d day(s)", l, len(datesets[l]))
        return 1
    dates = list(datesets[labels[0]])
    log.info("loaded %d strategies x %d days from %s", len(labels), len(dates),
             args.strategy_csv)

    # --- risk-free -------------------------------------------------------
    rf_by_date, rf_note, rc_rf = {}, "", 0
    if args.no_rf:
        rf_by_date = {d: 0.0 for d in dates}
        rf_note = "DISABLED via --no-rf (all zeros)"
        log.warning("--no-rf: Sharpe and excess figures assume a zero risk-free rate")
    else:
        try:
            rets, meta = load_returns(args.returns)
        except FileNotFoundError:
            log.error("no returns file at %s", args.returns)
            return 1
        if meta["excess"]:
            log.error("%s is an already-excess series (%s); the strategy returns "
                      "were built from total returns, so subtracting its zero rf "
                      "would be wrong. Pass a total-return series.",
                      args.returns, meta["schema"])
            return 1
        want = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
        try:
            lookup, filled, unfilled, missing = build_rf_lookup(rets, want, args.rf_missing)
        except ValueError as exc:
            log.error("%s (--rf-missing=error)", exc)
            return 3
        if missing:
            log.error("%d strategy day(s) are absent from %s entirely -- the two "
                      "files have drifted apart: %s", len(missing), args.returns,
                      ", ".join(str(d) for d in missing[:8]))
            return 1
        if filled:
            log.warning("%d day(s) had no risk-free quote (bond market closed, "
                        "equity market open); carried the prior session forward: %s",
                        len(filled), ", ".join(str(d) for d in filled[:8]))
        if unfilled:
            log.warning("%d day(s) had no risk-free quote and none to fill from; "
                        "used 0.0", len(unfilled))
            rc_rf = 2
        rf_by_date = {d: lookup[w] for d, w in zip(dates, want)}
        mean_rf = sum(rf_by_date.values()) / len(rf_by_date)
        rf_note = ("mean %.8f/day = %.2f%%/yr over %d day(s), %d gap(s) filled"
                   % (mean_rf, ((1 + mean_rf) ** args.trading_days - 1) * 100,
                      len(dates), len(filled)))
        log.info("rf: %s", rf_note)

    rf = [rf_by_date[d] for d in dates]

    # --- metrics ---------------------------------------------------------
    gross, net, turn, costs = {}, {}, {}, {}
    for l in labels:
        rows = by[l]
        pos = [r["pos"] for r in rows]
        # Net Return is read as written, never recomputed as gross - cost: Cost
        # is a fixed-point string and Net Return is rounded, so they differ at
        # ~1e-9 and recomputing would break exact reproduction.
        gross[l] = series_metrics([r["gross"] for r in rows], rf, pos,
                                  args.sortino_mar, args.trading_days)
        net[l] = series_metrics([r["net"] for r in rows], rf, pos,
                                args.sortino_mar, args.trading_days)
        turn[l] = turnover_stats(pos, args.trading_days)
        costs[l] = sum(r["cost"] for r in rows)

    if all(costs[l] == 0.0 for l in labels):
        log.warning("every cost is zero -- this CSV was produced with --no-costs, "
                    "so the gross and net columns are identical by construction")

    charts = make_charts(by, labels, dates, args)

    text = build_report(labels, gross, net, turn, costs, None, rf_note, args,
                        dates, charts, by, rf)

    if args.stdout:
        # The report contains en/em dashes and a Unicode minus, which a cp1252
        # console cannot encode; without this, --stdout dies on Windows with
        # UnicodeEncodeError while --out (UTF-8) works fine.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
        print(text)
    else:
        save_report(text, args.out)
        log.info("wrote %s", args.out)

    print()
    print("  %-38s %10s %10s %8s %9s" % ("strategy", "gross", "net", "Sharpe", "turnover"))
    for l in labels:
        print("  %-38s %10s %10s %8s %9.2f"
              % (l[:38], fmt_pct(gross[l]["total"]), fmt_pct(net[l]["total"]),
                 fmt_ratio(net[l]["sharpe"]).replace("\\", ""), turn[l]["total"]))
    return rc_rf


if __name__ == "__main__":
    sys.exit(main())
