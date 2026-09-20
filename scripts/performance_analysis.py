"""Fund performance metrics for each strategy, written to output/performance.md.

Reads the per-day, per-strategy results produced by predict_positions.py and
reports a full metric set on BOTH gross and net returns, side by side across
strategies, plus a turnover/cost table.

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
from datetime import datetime
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


def metric_rows(labels, stats):
    m = lambda l: stats[l]
    return [
        ("Observations (days)", ["%d" % m(l)["n"] for l in labels]),
        ("Total return", [fmt_pct(m(l)["total"]) for l in labels]),
        ("Annualized return (geometric)", [fmt_pct(m(l)["ann_ret"]) for l in labels]),
        ("Annualized volatility", ["%.2f%%" % (m(l)["ann_vol"] * 100) for l in labels]),
        ("Sharpe (excess, annualized)", [fmt_ratio(m(l)["sharpe"]) for l in labels]),
        ("Sortino (annualized)", [fmt_ratio(m(l)["sortino"]) for l in labels]),
        ("Maximum drawdown", [fmt_pct(m(l)["maxdd"]) for l in labels]),
        ("VaR 95%, historical †", [fmt_pct3(m(l)["var95"]) for l in labels]),
        ("VaR 99%, historical †", [fmt_pct3(m(l)["var99"]) for l in labels]),
        ("CVaR 95% (expected shortfall) †", [fmt_pct3(m(l)["cvar95"]) for l in labels]),
        ("CVaR 99% (expected shortfall) †", [fmt_pct3(m(l)["cvar99"]) for l in labels]),
        ("Win rate, days in position",
         ["%.1f%% (%d/%d)" % (100.0 * m(l)["wins_in_pos"] / m(l)["n_in_pos"]
                              if m(l)["n_in_pos"] else 0.0,
                              m(l)["wins_in_pos"], m(l)["n_in_pos"]) for l in labels]),
        ("Win rate, all days",
         ["%.1f%% (%d/%d)" % (100.0 * m(l)["wins"] / m(l)["n"], m(l)["wins"], m(l)["n"])
          for l in labels]),
        ("Flat days (position = 0)", ["%d" % m(l)["flats"] for l in labels]),
    ]


def build_report(labels, gross, net, turn, costs, meta, rf_note, args, dates,
                 charts=None):
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
    L.append("Source: `%s` (%d rows)." % (args.strategy_csv, n * len(labels)))
    L.append("Risk-free: `%s`, `rf_daily`, %s." % (args.returns, rf_note))
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

    L.append("## Gross returns (before trading costs)")
    L.append("")
    L.extend(table(labels, metric_rows(labels, gross)))
    L.append("")
    if degenerate:
        L.append("† At n = %d both confidence levels select the same order "
                 "statistic (k = %d, the worst day), and the tail mean equals it."
                 % (n, gross[labels[0]]["k95"]))
        L.append("")

    L.append("## Net returns (after trading costs)")
    L.append("")
    L.extend(table(labels, metric_rows(labels, net)))
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
             "gross and net tables share a denominator.")
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
                 "matplotlib is not installed." % args.charts_dir)
    if any(v == float("inf") or v == float("-inf")
           for d in (gross, net) for l in labels for v in (d[l]["sortino"],)):
        L.append("- \\* `+inf` means no day fell at or below the target. That is a "
                 "small-sample artifact, not an unbounded ratio.")
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

    g_c = p.add_argument_group("charts (require matplotlib; skipped if absent)")
    g_c.add_argument("--no-charts", action="store_true",
                     help="write the report without figures")
    g_c.add_argument("--charts-dir", default=str(OUTPUT / "charts"),
                     help="directory for the PNGs (default: output/charts)")
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
                    "so the gross and net tables are identical by construction")

    charts = make_charts(by, labels, dates, args)

    text = build_report(labels, gross, net, turn, costs, None, rf_note, args,
                        dates, charts)

    if args.stdout:
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
