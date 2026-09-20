"""Turn each trading day's pre-close CNBC headlines into a next-day long/short call.

Point-in-time discipline (the whole point of this step):
  * GDELT seendate is UTC; every headline is converted to America/New_York.
  * A headline counts for trading day D only if its ET timestamp falls on D
    BEFORE 16:00 ET. Anything at or after the close is dropped -- it could not
    have informed a trade placed that day.
  * The prediction target is the NEXT trading day, so the label is always
    strictly in the future relative to the information used.
  * The model is only ever shown the surviving headlines. No prices, no dates
    after D, no outcome data appears in the prompt.

Two modes, both on by default; --predict-only / --backtest-only restrict it.

Usage:
    py scripts/predict_positions.py                  # predict, then back-test
    py scripts/predict_positions.py --backtest-only  # free: no API calls
    py scripts/predict_positions.py --dry-run        # inspect filtering, then back-test
    py scripts/performance_analysis.py               # metrics -> output/performance.md

Exit codes: 0 ok | 1 nothing to do | 2 some prediction days failed
            3 prediction aborted | 4 point-in-time audit failed

Input and output default to data/ under the project root, resolved from this
file's own location, so the script can be run from any working directory. A
path given on the command line is used exactly as typed.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from time import sleep
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# --- project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def resolve_prefix(value):
    """--out-prefix names a file basename, not a path, so a bare prefix is
    placed in data/. A prefix that is absolute or carries a directory component
    is honoured as typed."""
    p = Path(value)
    return str(p if p.is_absolute() or len(p.parts) > 1 else DATA / p)


def ensure_dir(path):
    """Create the directory a file is about to be written into."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


ET = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)
MODEL = "claude-opus-5"

# NYSE full-day closures. Weekends are handled separately. This must cover every
# year the headline data spans -- a missing year silently turns a holiday into a
# tradable day and invents a prediction for a day the market never opened.
NYSE_HOLIDAYS = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}

# Days the NYSE opens but closes early, at 13:00 ET. The normal 16:00 cutoff
# would admit post-close news on these days, which is exactly the leak the
# pre-close filter exists to prevent -- so they get their own earlier cutoff.
EARLY_CLOSE = {
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 7, 2), date(2026, 11, 27), date(2026, 12, 24),
}
EARLY_CLOSE_TIME = time(13, 0)

SYSTEM_PROMPT = """You are a portfolio manager at a systematic equity fund.

You will be shown the news headlines published by CNBC on a single trading day,
all of them from BEFORE that day's 4:00 PM ET market close. Based only on those
headlines, you must take a directional position in the broad US equity market
(think S&P 500) to be held for the NEXT trading day.

Rules:
- Judge only from the headlines provided. You have no price data, no knowledge of
  what happened after the close, and no information about later days.
- "long" means you expect the market to rise on the next trading day; "short"
  means you expect it to fall. You must pick one -- there is no flat option.
- sentiment is your read of the news tone on a -5 (maximally bearish) to
  +5 (maximally bullish) integer scale. 0 means genuinely neutral or mixed.
- confidence reflects how strongly the headlines support your call, 0.0 to 1.0.
  Thin or off-topic news days should get low confidence, not a forced strong view.
- Keep the rationale to one or two sentences citing the headlines that drove it.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["long", "short"]},
        # Structured outputs reject minimum/maximum on numeric types, so the
        # -5..+5 range is expressed as an enum and confidence is clamped below.
        "sentiment": {"type": "integer", "enum": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["direction", "sentiment", "confidence", "rationale"],
    "additionalProperties": False,
}

log = logging.getLogger("predict")


def is_trading_day(d):
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def next_trading_day(d):
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def parse_seendate(raw):
    """GDELT seendate '20260819T071500Z' -> aware UTC datetime."""
    return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def load_headlines(path):
    """Group headlines by ET trading day, keeping only those before the close."""
    by_day = {}
    total = dropped_after_close = dropped_non_trading = unparsable = 0

    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += 1
            try:
                et_dt = parse_seendate(row["seendate"]).astimezone(ET)
            except (ValueError, KeyError):
                unparsable += 1
                continue

            day = et_dt.date()
            if not is_trading_day(day):
                dropped_non_trading += 1
                continue
            cutoff = EARLY_CLOSE_TIME if day in EARLY_CLOSE else MARKET_CLOSE
            if et_dt.time() >= cutoff:
                dropped_after_close += 1      # look-ahead guard
                continue

            by_day.setdefault(day, []).append({
                "et_time": et_dt.strftime("%H:%M"),
                "title": (row.get("title") or "").strip(),
                "url": row.get("url", ""),
            })

    for day in by_day:
        by_day[day].sort(key=lambda h: h["et_time"])

    log.info("read %d headline(s) from %s", total, path)
    log.info("  dropped %d at/after 16:00 ET (look-ahead guard)", dropped_after_close)
    log.info("  dropped %d on weekends/holidays (not trading days)", dropped_non_trading)
    if unparsable:
        log.warning("  skipped %d row(s) with an unreadable seendate", unparsable)
    log.info("  kept %d headline(s) across %d trading day(s)",
             sum(len(v) for v in by_day.values()), len(by_day))
    return by_day


FIELDNAMES = ["date", "predicts_for", "direction", "sentiment", "confidence",
              "n_headlines", "last_headline_et", "rationale", "model"]


def save_predictions(rows, path):
    """Write atomically after every call, so an interrupted run keeps what it
    has already paid for. Temp file then replace: a kill mid-write cannot leave
    a half-written CSV behind."""
    ordered = sorted(rows, key=lambda r: r["date"])
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(ordered)
    os.replace(tmp, path)
    return ordered


def load_existing(path):
    """Read predictions already produced, so a resumed run skips them."""
    out = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("date") and r.get("direction"):
                    out.append(r)
    except FileNotFoundError:
        pass
    return out


def save_failed(days, path):
    """Record the days that failed so a later pass can target exactly those."""
    if not days:
        return
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(days)) + "\n")
    os.replace(tmp, path)


def infer_domain(path, override=None):
    """Short source name for the output filename, e.g. cnbc.com -> cnbc.

    Taken from the headline rows rather than a flag so the filename always
    describes the data actually used; --domain overrides it.
    """
    if override:
        return override
    counts = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                d = (r.get("domain") or "").strip().lower()
                if d:
                    counts[d] = counts.get(d, 0) + 1
    except FileNotFoundError:
        return "unknown"
    if not counts:
        return "unknown"
    top = max(counts, key=counts.get)
    if len(counts) > 1:
        log.info("headlines span %d domains; naming the file after the most "
                 "common (%s, %d of %d rows)",
                 len(counts), top, counts[top], sum(counts.values()))
    host = top[4:] if top.startswith("www.") else top
    return host.split(".")[0] or "unknown"


def build_prompt(day, headlines):
    lines = ["%s ET  %s" % (h["et_time"], h["title"]) for h in headlines]
    target = next_trading_day(day)
    return (
        "CNBC headlines published on %s before the 4:00 PM ET close (%d headlines):\n\n"
        % (day.strftime("%A, %B %d, %Y"), len(headlines))
        + "\n".join(lines)
        + "\n\nTake your position for the next trading day, %s."
        % target.strftime("%A, %B %d, %Y")
    )


class FatalAPIError(Exception):
    """An error no amount of retrying will fix (bad key, bad request)."""


class TransientAPIError(Exception):
    """A failure worth retrying, or recording and moving past."""


def ask_model(client, model, prompt, retries=4, base_delay=5.0):
    """One prediction call, with error classification.

    The SDK already retries 429/5xx/connection errors a couple of times; this
    adds a slower outer loop for sustained rate limiting, and -- more
    importantly -- separates errors that retrying can fix from ones it cannot.
    Authentication and request-shape problems abort the whole run rather than
    burning one doomed call per day for 243 days.
    """
    import anthropic

    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError) as exc:
            raise FatalAPIError("%s: %s" % (type(exc).__name__, exc)) from exc
        except (anthropic.RateLimitError, anthropic.APITimeoutError,
                anthropic.APIConnectionError) as exc:
            if attempt == retries:
                raise TransientAPIError("%s after %d attempts: %s"
                                        % (type(exc).__name__, attempt, exc)) from exc
            wait = base_delay * attempt
            log.warning("  %s, attempt %d/%d, retrying in %.0fs",
                        type(exc).__name__, attempt, retries, wait)
            sleep(wait)
            continue
        except anthropic.APIStatusError as exc:
            if exc.status_code < 500:
                raise FatalAPIError("HTTP %s: %s" % (exc.status_code, exc)) from exc
            if attempt == retries:
                raise TransientAPIError("HTTP %s after %d attempts"
                                        % (exc.status_code, attempt)) from exc
            wait = base_delay * attempt
            log.warning("  HTTP %s, attempt %d/%d, retrying in %.0fs",
                        exc.status_code, attempt, retries, wait)
            sleep(wait)
            continue

        # A refusal is a real answer, not a transport failure -- don't retry it.
        if response.stop_reason == "refusal":
            raise TransientAPIError("model refused: %s"
                                    % getattr(response, "stop_details", None))
        try:
            text = next(b.text for b in response.content if b.type == "text")
            return json.loads(text), response.usage
        except (StopIteration, json.JSONDecodeError) as exc:
            raise TransientAPIError("unreadable response: %s: %s"
                                    % (type(exc).__name__, exc)) from exc

    raise TransientAPIError("exhausted %d attempts" % retries)

# ===========================================================================
#  BACKTEST
# ===========================================================================
# --- BEGIN shared by copy with scripts/performance_analysis.py -------------
# Every script in this repo runs standalone -- no package, no cross-script
# imports, no sys.path games -- so the helpers in this fence live in two files
# by design. scripts/performance_analysis.py carries its own copy of
# TRADING_DAYS, max_drawdown, load_returns and fmt_pct. If you change one,
# change the other; divergence is a bug in whichever was edited last.
#
# The guard is numeric, not textual. performance_analysis.py recomputes the
# compounded totals, turnover, drawdown and win rate straight from
# data/predictions_strategy.csv, so the two tools must agree on every strategy:
#     py scripts/predict_positions.py --backtest-only
#     py scripts/performance_analysis.py
# A drift a textual diff would miss shows up there as a mismatched number.
#
# One deliberate divergence, not a drift: performance_analysis.py's copy of
# load_returns records a blank rf_daily as None rather than collapsing it to
# 0.0, so it can forward-fill instead of inheriting an rf=0 bias.
TRADING_DAYS = 252


def max_drawdown(returns):
    """Largest peak-to-trough decline of the compounded equity curve."""
    equity, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def metrics(returns, excess):
    n = len(returns)
    if n == 0:
        return None
    total = 1.0
    for r in returns:
        total *= (1.0 + r)
    total -= 1.0

    mean_x = sum(excess) / n
    if n > 1:
        var = sum((x - mean_x) ** 2 for x in excess) / (n - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    sharpe = (mean_x / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    tstat = (mean_x / (sd / math.sqrt(n))) if sd > 0 else float("nan")

    wins = sum(1 for r in returns if r > 0)
    return {"n": n, "total": total, "mean_daily": sum(returns) / n,
            "sharpe": sharpe, "tstat": tstat, "maxdd": max_drawdown(returns),
            "winrate": wins / n, "wins": wins}


def load_returns(path):
    """Daily return series. Schema A: date,mkt_ret[,rf_daily] (decimals).
    Schema B: date,mkt_rf_pct (Fama/French, percent, ALREADY excess)."""
    out = {}
    meta = {"excess": False, "schema": "mkt_ret (decimal)", "blank_rf": 0}
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
                    if not row.get("rf_daily"):
                        meta["blank_rf"] += 1
                    out[day] = {"mkt_ret": float(row["mkt_ret"]),
                                "rf_daily": float(row["rf_daily"]) if row.get("rf_daily") else 0.0}
            except (ValueError, KeyError):
                continue
    return out, meta


def fmt_pct(x):
    return "%+.2f%%" % (x * 100.0)
# --- END shared by copy with scripts/performance_analysis.py ---------------


def _pos_benchmark(pred):
    return 1.0


def _pos_direction(pred):
    d = pred["direction"]
    if d not in ("long", "short"):
        # Never default to short on an unrecognised value: a typo in the
        # direction column would silently become a real short position.
        # Fail loudly instead.
        raise ValueError("unknown direction %r on %s" % (d, pred["date"]))
    return 1.0 if d == "long" else -1.0


def _pos_sentiment(pred):
    """sentiment/5 -> -1..+1. 0 is genuinely flat, not a small long.
    The clamp enforces the no-leverage-above-1x rule against a hand-edited CSV."""
    s = pred["sentiment"]
    if not -5 <= s <= 5:
        log.warning("sentiment %d on %s outside -5..+5; clamping", s, pred["date"])
    return max(-1.0, min(1.0, s / 5.0))


# (label, base position function, sign). Contrarian rows are the SAME base
# function with sign -1, so there is one position rule per idea, not two.
STRATEGIES = (
    ("Benchmark", _pos_benchmark, 1),
    ("LLM Direction", _pos_direction, 1),
    ("LLM Direction (Contrarian)", _pos_direction, -1),
    ("LLM Sentiment-Weighted", _pos_sentiment, 1),
    ("LLM Sentiment-Weighted (Contrarian)", _pos_sentiment, -1),
)

STRATEGY_FIELDNAMES = ["Date", "Strategy", "Position", "Sentiment Score",
                       "Next Day's Actual Return", "Strategy Return",
                       "Cost", "Net Return"]
VERBOSE_FIELDNAMES = STRATEGY_FIELDNAMES + [
    "Signal Date", "Direction", "Confidence", "N Headlines", "Last Headline ET",
    "Turnover", "Trading Cost", "Borrow Cost", "Expense Cost", "Equity"]


def _z(x):
    """Normalise -0.0 to 0.0. Negating a flat position yields -0.0, which csv
    writes as the literal string '-0.0' -- harmless arithmetically, but it
    lands in the file and breaks byte-identical diffs."""
    return 0.0 if x == 0 else x


def strategy_position(spec, pred):
    _label, fn, sign = spec
    return _z(sign * fn(pred))


def annual_to_daily(annual_rate, day_count=TRADING_DAYS):
    """Annual decimal rate -> per-trading-day rate, geometrically.

    Same conversion fetch_market_returns.py uses to build rf_daily, so borrow
    and expense sit on the same basis as the risk-free series Sharpe subtracts.
    """
    return (1.0 + annual_rate) ** (1.0 / day_count) - 1.0


def build_costs(args):
    if args.no_costs:
        return {"spread_bps": 0.0, "commission_bps": 0.0, "borrow_daily": 0.0,
                "expense_daily": 0.0, "expense_basis": args.expense_basis,
                "enabled": False}
    return {"spread_bps": args.spread_bps, "commission_bps": args.commission_bps,
            "borrow_daily": annual_to_daily(args.borrow_bps_annual / 1e4, args.cost_day_count),
            "expense_daily": annual_to_daily(args.expense_bps_annual / 1e4, args.cost_day_count),
            "expense_basis": args.expense_basis, "enabled": True}


def load_predictions_csv(path):
    """Typed reader. Keeps last_headline_et, which the audit needs."""
    rows, bad = [], 0
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "date": datetime.strptime(r["date"], "%Y-%m-%d").date(),
                    "target": datetime.strptime(r["predicts_for"], "%Y-%m-%d").date(),
                    "direction": (r["direction"] or "").strip().lower(),
                    "sentiment": int(r["sentiment"]),
                    "confidence": float(r["confidence"]),
                    "n_headlines": int(r["n_headlines"]),
                    "last_headline_et": (r.get("last_headline_et") or "").strip(),
                })
            except (ValueError, KeyError, TypeError):
                bad += 1
    if bad:
        log.warning("skipped %d unreadable prediction row(s) in %s", bad, path)
    rows.sort(key=lambda r: r["target"])
    return rows


def audit_point_in_time(preds):
    """Re-verify the look-ahead guarantee from the predictions file alone.

    Uses the same is_trading_day / next_trading_day / EARLY_CLOSE /
    MARKET_CLOSE objects load_headlines enforces the cutoff with, so the audit
    and the filter cannot disagree about a boundary case.
    """
    lines, viol = [], []
    n = len(preds)
    years = {d.year for d in NYSE_HOLIDAYS}

    sig_ok = sum(1 for p in preds if is_trading_day(p["date"]))
    for p in preds:
        if not is_trading_day(p["date"]):
            viol.append("%s: signal day is not a trading day" % p["date"])

    after_ok = sum(1 for p in preds if p["target"] > p["date"])
    for p in preds:
        if p["target"] <= p["date"]:
            viol.append("%s: target %s is not after the signal day" % (p["date"], p["target"]))

    next_ok = next_skip = 0
    for p in preds:
        if p["date"].year not in years or p["target"].year not in years:
            next_skip += 1
            continue
        if p["target"] == next_trading_day(p["date"]):
            next_ok += 1
        else:
            viol.append("%s: target %s is not the next trading day (expected %s)"
                        % (p["date"], p["target"], next_trading_day(p["date"])))

    close_ok = close_skip = 0
    latest = None
    for p in preds:
        raw = p["last_headline_et"]
        try:
            hhmm = datetime.strptime(raw, "%H:%M").time()
        except ValueError:
            close_skip += 1
            continue
        cutoff = EARLY_CLOSE_TIME if p["date"] in EARLY_CLOSE else MARKET_CLOSE
        if hhmm >= cutoff:          # same >= as load_headlines
            viol.append("%s: last headline %s ET is at/after the %s close"
                        % (p["date"], raw, cutoff.strftime("%H:%M")))
        else:
            close_ok += 1
            if latest is None or hhmm > latest[0]:
                latest = (hhmm, p["date"])

    dup_sig = n - len({p["date"] for p in preds})
    dup_tgt = n - len({p["target"] for p in preds})
    if dup_sig:
        viol.append("%d duplicate signal day(s)" % dup_sig)
    if dup_tgt:
        viol.append("%d duplicate target day(s)" % dup_tgt)

    bad_dir = sum(1 for p in preds if p["direction"] not in ("long", "short"))
    if bad_dir:
        viol.append("%d row(s) with an unreadable direction" % bad_dir)
    early = sum(1 for p in preds if p["date"] in EARLY_CLOSE)
    disagree = sum(1 for p in preds if (p["direction"] == "long") != (p["sentiment"] > 0))

    w = 76
    ok_s = lambda got, tot: "OK" if got == tot else "FAIL"
    lines.append("=" * w)
    lines.append("  POINT-IN-TIME AUDIT   (%d rows)" % n)
    lines.append("=" * w)
    lines.append("  Signal day is a trading day ....................... %2d/%d  %s"
                 % (sig_ok, n, ok_s(sig_ok, n)))
    if next_skip:
        lines.append("  Held day is the next trading day .................. %2d/%d  "
                     "(%d SKIPPED: holiday table lacks those years)"
                     % (next_ok, n - next_skip, next_skip))
    else:
        lines.append("  Held day is the next trading day .................. %2d/%d  %s"
                     % (next_ok, n, ok_s(next_ok, n)))
    lines.append("  Held day strictly after signal day ................ %2d/%d  %s"
                 % (after_ok, n, ok_s(after_ok, n)))
    if close_skip == n:
        lines.append("  Last headline before that day's close ............. SKIPPED "
                     "(last_headline_et absent)")
    else:
        lines.append("  Last headline before that day's close ............. %2d/%d  %s"
                     % (close_ok, n - close_skip, ok_s(close_ok, n - close_skip)))
        if latest:
            lines.append("      latest observed %s ET on %s"
                         % (latest[0].strftime("%H:%M"), latest[1]))
    lines.append("  Early-close days in sample (13:00 cutoff) ......... %2d" % early)
    lines.append("  Duplicate signal / held days ...................... %2d / %d"
                 % (dup_sig, dup_tgt))
    lines.append("  Unreadable direction values ....................... %2d" % bad_dir)
    lines.append("  sign(sentiment) disagrees with direction .......... %2d" % disagree)
    lines.append("  " + "-" * (w - 4))
    lines.append("  VERIFIED FROM THIS FILE: every position is taken on information")
    lines.append("  timestamped before the close of the day BEFORE the return it earns.")
    lines.append("  last_headline_et is the MAXIMUM headline time for that day, so a")
    lines.append("  pre-close maximum means the whole set was pre-close.")
    lines.append("  NOT VERIFIABLE HERE -- asserted by the prediction pipeline:")
    lines.append("    * headlines were filtered on GDELT seendate, a crawl time at or")
    lines.append("      after true publication, so the filter errs conservative;")
    lines.append("    * this cannot detect a hand-edited CSV;")
    lines.append("    * the model's training data may postdate these days. No backtest")
    lines.append("      can rule that out.")
    lines.append("=" * w)
    return (not viol), lines, viol


def run_backtest(preds, rets, costs, flat_on_gaps=False, verbose=False):
    """One independent pass per strategy.

    Contrarian variants negate the POSITION and then pay their own costs
    through this same loop. Do NOT shortcut them as -1 x the parent's net
    return: costs are strictly positive for both sides and the borrow leg
    differs (the contrarian is short precisely when the parent is long). Only
    the per-day GROSS return is an exact negation -- compounded totals are not,
    and net returns are not.
    """
    rows = []
    for spec in STRATEGIES:
        label = spec[0]
        prev, equity = 0.0, 1.0
        for i, pr in enumerate(preds):
            pos = strategy_position(spec, pr)
            mkt = rets[pr["target"]]["mkt_ret"]

            if flat_on_gaps and i > 0 and pr["target"] != next_trading_day(preds[i - 1]["target"]):
                prev = 0.0

            traded = abs(pos - prev)
            # --spread-bps is a ROUND TURN, so half is charged per one-way trade.
            c_spread = traded * (costs["spread_bps"] / 2.0) / 1e4
            c_comm = traded * costs["commission_bps"] / 1e4
            c_borrow = max(0.0, -pos) * costs["borrow_daily"]
            basis = pos if costs["expense_basis"] == "signed" else abs(pos)
            c_expense = basis * costs["expense_daily"]

            gross = _z(pos * mkt)               # the requested formula, verbatim
            cost = c_spread + c_comm + c_borrow + c_expense
            net = _z(gross - cost)
            equity *= (1.0 + net)

            row = {
                "Date": pr["target"].isoformat(),
                "Strategy": label,
                "Position": round(_z(pos), 4),
                "Sentiment Score": pr["sentiment"],
                "Next Day's Actual Return": round(_z(mkt), 8),
                "Strategy Return": round(gross, 8),
                # fixed-point: round() renders these as 3.75e-06 in the CSV
                "Cost": "%.10f" % cost,
                "Net Return": round(net, 8),
            }
            if verbose:
                row.update({
                    "Signal Date": pr["date"].isoformat(),
                    "Direction": "long" if pos > 0 else ("short" if pos < 0 else "flat"),
                    "Confidence": pr["confidence"],
                    "N Headlines": pr["n_headlines"],
                    "Last Headline ET": pr["last_headline_et"],
                    "Turnover": round(traded, 4),
                    "Trading Cost": "%.10f" % (c_spread + c_comm),
                    "Borrow Cost": "%.10f" % c_borrow,
                    "Expense Cost": "%.10f" % c_expense,
                    "Equity": round(equity, 8),
                })
            rows.append(row)
            prev = pos

    order = {s[0]: i for i, s in enumerate(STRATEGIES)}
    rows.sort(key=lambda r: (r["Date"], order[r["Strategy"]]))
    return rows


def save_strategy_rows(rows, path, verbose=False):
    """Atomic write beside the target -- os.replace is only atomic same-volume."""
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=VERBOSE_FIELDNAMES if verbose
                           else STRATEGY_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return rows


def summarize(rows, rets, kept, rmeta, costs, unmatched):
    """One row per strategy, one column per metric -- the console summary.

    output/performance.md (scripts/performance_analysis.py) is the transpose:
    one column per strategy, one row per metric, because it carries far more
    metrics than fit across a terminal. NOTE the Sharpe column here is computed
    as metrics(net, net), i.e. WITHOUT subtracting the risk-free rate, so it is
    overstated -- by about 0.40 for the +-1 strategies and about 1.60 for the
    sentiment-weighted ones, whose lower volatility makes the same rf drag
    proportionally larger. performance_analysis.py subtracts rf; where the two
    disagree on Sharpe, that file is right and this one is optimistic."""
    by = {}
    for r in rows:
        by.setdefault(r["Strategy"], []).append(r)
    labels = [s[0] for s in STRATEGIES if s[0] in by]

    # Always-short is the unconditional bearish control: it says whether a good
    # contrarian number is really just a net-short tilt in a falling window.
    # Printed only -- the CSV is the five-strategy deliverable.
    mkt_series = [rets[p["target"]]["mkt_ret"] for p in kept]
    ctrl_net, ctrl_prev, ctrl_cost = [], 0.0, 0.0
    for m in mkt_series:
        pos = -1.0
        c = abs(pos - ctrl_prev) * (costs["spread_bps"] / 2.0) / 1e4 \
            + 1.0 * costs["borrow_daily"] \
            + (pos if costs["expense_basis"] == "signed" else 1.0) * costs["expense_daily"]
        ctrl_net.append(-m - c)
        ctrl_cost += c
        ctrl_prev = pos

    stats = {}
    for lbl in labels:
        rs = by[lbl]
        gross = [float(r["Strategy Return"]) for r in rs]
        net = [float(r["Net Return"]) for r in rs]
        traded = [r for r in rs if abs(float(r["Position"])) > 0]
        g = 1.0
        for x in gross:
            g *= (1.0 + x)
        stats[lbl] = {"m": metrics(net, net), "gross": g - 1.0,
                      "cost": sum(float(r["Cost"]) for r in rs),
                      "turn": sum(abs(float(rs[i]["Position"]) -
                                      (float(rs[i - 1]["Position"]) if i else 0.0))
                                  for i in range(len(rs))),
                      "traded": len(traded),
                      "twins": sum(1 for r in traded if float(r["Net Return"]) > 0),
                      "mabs": sum(abs(float(r["Position"])) for r in rs) / len(rs),
                      "mpos": sum(float(r["Position"]) for r in rs) / len(rs)}
    stats["Always Short (control)"] = {
        "m": metrics(ctrl_net, ctrl_net), "gross": None, "cost": ctrl_cost,
        "turn": 1.0, "traded": len(ctrl_net),
        "twins": sum(1 for x in ctrl_net if x > 0), "mabs": 1.0, "mpos": -1.0}

    n = len(by[labels[0]])
    out = []
    out.append("=" * 118)
    out.append("  STRATEGY BACKTEST   %s .. %s   (%d trading days)"
               % (rows[0]["Date"], rows[-1]["Date"], n))
    out.append("  costs: " + ("DISABLED (--no-costs)" if not costs["enabled"] else
               "%.1f bps round turn | %.0f bps/yr borrow | %.2f bps/yr fee | basis %s"
               % (costs["spread_bps"], costs["borrow_daily"] * TRADING_DAYS * 1e4,
                  costs["expense_daily"] * TRADING_DAYS * 1e4, costs["expense_basis"])))
    if rmeta["excess"]:
        out.append("  returns are EXCESS of the risk-free rate -- net returns are excess")
        out.append("  returns and an equity curve built from them is NOT tradable")
    if unmatched:
        out.append("  WARNING: %d prediction(s) dropped for want of a market return"
                   % len(unmatched))
    out.append("=" * 118)
    hdr = ("  %-37s %9s %9s %9s %7s %8s %13s %6s %8s"
           % ("Strategy", "Gross", "Cost", "NET", "Sharpe", "MaxDD",
              "Win(traded)", "Flat", "Turnover"))
    out.append(hdr)
    out.append("  " + "-" * 114)
    for lbl in labels + ["Always Short (control)"]:
        s = stats[lbl]
        m = s["m"]
        out.append("  %-37s %9s %9s %9s %7.2f %8s %13s %6d %8.2f"
                   % (lbl[:37],
                      "-" if s["gross"] is None else fmt_pct(s["gross"]),
                      "-%.4f%%" % (s["cost"] * 100),
                      fmt_pct(m["total"]), m["sharpe"], fmt_pct(m["maxdd"]),
                      "%.1f%% (%d/%d)" % (100.0 * s["twins"] / s["traded"] if s["traded"]
                                          else 0.0, s["twins"], s["traded"]),
                      m["n"] - s["traded"], s["turn"]))
    out.append("=" * 118)

    # Days where |gross| < cost for BOTH sides of a pair -- the clearest
    # demonstration that the cost model is doing real work.
    for par, con in (("LLM Direction", "LLM Direction (Contrarian)"),
                     ("LLM Sentiment-Weighted", "LLM Sentiment-Weighted (Contrarian)")):
        if par in by and con in by:
            a = {r["Date"]: float(r["Net Return"]) for r in by[par]}
            b = {r["Date"]: float(r["Net Return"]) for r in by[con]}
            both = sum(1 for d in a if a[d] < 0 and b[d] < 0)
            if both:
                out.append("  %s: %d day(s) lost money on BOTH sides -- costs exceeded "
                           "the move" % (par, both))
    if n < 30:
        out.append("")
        out.append("  WARNING: %d observations. An annualized Sharpe from %d daily returns"
                   % (n, n))
        out.append("  is not a meaningful estimate -- its standard error is about %.2f."
                   % math.sqrt(TRADING_DAYS / n))
    return out


def resolve_predictions_path(args, wrote_to):
    if args.predictions:
        return args.predictions
    if wrote_to:
        return wrote_to
    if args.stable_copy:
        return args.stable_copy
    return str(DATA / "predictions.csv")


def do_backtest(args, predictions_path):
    """Score the predictions. Returns an exit code."""
    log.info("back-testing %s", predictions_path)
    try:
        preds = load_predictions_csv(predictions_path)
    except FileNotFoundError:
        log.error("no predictions file at %s", predictions_path)
        return 1
    if not preds:
        log.error("no usable predictions in %s", predictions_path)
        return 1

    ok, report, violations = audit_point_in_time(preds)
    for line in report:
        print(line)
    if not ok:
        for v in violations[:20]:
            log.error("  AUDIT: %s", v)
        if not args.allow_lookahead:
            log.error("point-in-time audit FAILED -- refusing to back-test a file that "
                      "may contain look-ahead. Pass --allow-lookahead to override.")
            return 4
        log.warning("--allow-lookahead: continuing despite %d violation(s)", len(violations))
    if args.audit_only:
        return 0 if ok else 4

    rets, rmeta = load_returns(args.returns)
    log.info("loaded %d daily return(s) from %s (%s)", len(rets), args.returns, rmeta["schema"])
    if rmeta["blank_rf"]:
        log.warning("%d row(s) in the return series have a blank rf_daily, treated as "
                    "0.0 -- this inflates Sharpe on those days", rmeta["blank_rf"])

    kept, unmatched, filtered = [], [], 0
    for pr in preds:
        if pr["n_headlines"] < args.min_headlines or pr["confidence"] < args.min_confidence:
            filtered += 1
            continue
        if pr["target"] not in rets:
            unmatched.append(pr["target"])
            continue
        kept.append(pr)
    if filtered:
        log.info("filtered out %d prediction(s) by --min-headlines/--min-confidence", filtered)
    if unmatched:
        log.warning("%d prediction(s) have no market return for their target day: %s%s",
                    len(unmatched), ", ".join(str(d) for d in sorted(unmatched)[:8]),
                    " ..." if len(unmatched) > 8 else "")
        log.warning("dropping them makes their neighbours adjacent, so turnover across "
                    "the gap is understated (see --flat-on-gaps)")
    if not kept:
        log.error("no predictions could be matched to market returns -- writing no file")
        return 1

    costs = build_costs(args)
    try:
        rows = run_backtest(kept, rets, costs, args.flat_on_gaps, args.verbose_columns)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    save_strategy_rows(rows, args.strategy_out, args.verbose_columns)
    log.info("wrote %d row(s) (%d strategies x %d days) to %s",
             len(rows), len(STRATEGIES), len(kept), args.strategy_out)
    print()
    for line in summarize(rows, rets, kept, rmeta, costs, unmatched):
        print(line)
    return 0


def do_predict(args):
    """The LLM prediction pass. Returns (exit_code, path_written_or_None)."""
    by_day = load_headlines(args.headlines)
    days = sorted(by_day)
    if args.limit:
        days = days[:args.limit]
    if not days:
        log.error("no trading days with pre-close headlines -- nothing to predict")
        return 1, None

    if args.dry_run:
        log.info("DRY RUN -- no API calls, no output file")
        for d in days:
            hs = by_day[d]
            log.info("%s -> predicts %s | %d headline(s), last at %s ET",
                     d, next_trading_day(d), len(hs), hs[-1]["et_time"])
        log.info("would make %d API call(s) to %s", len(days), args.model)
        print("\n--- example prompt (%s) ---" % days[0])
        print(build_prompt(days[0], by_day[days[0]])[:1500])
        return 0, None

    # Credential resolution order matches the SDK's own: env var, then a local
    # key file, then whatever profile `ant auth login` stored. An unset env var
    # is NOT an error -- the zero-arg client picks up the CLI profile itself.
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        log.info("using credentials from the environment")
    elif os.path.exists(args.api_key_file):
        with open(args.api_key_file, encoding="utf-8") as fh:
            os.environ["ANTHROPIC_API_KEY"] = fh.read().strip()
        log.info("loaded API key from %s", args.api_key_file)
    else:
        log.info("no env var or key file; relying on the ant CLI profile")

    import anthropic
    client = anthropic.Anthropic(max_retries=5)   # batch job: be patient

    # Resolve the output filename before the loop, so incremental saves have
    # somewhere to go from the first call onward.
    if args.out is None:
        # predictions_<domain>_<start>_<end>.csv, e.g.
        # predictions_cnbc_20250922_20260911.csv. The dates are the first and
        # last day a prediction is made for, so the name describes the contents.
        domain = infer_domain(args.headlines, args.domain)
        args.out = "%s_%s_%s_%s.csv" % (args.out_prefix, domain,
                                        days[0].strftime("%Y%m%d"),
                                        days[-1].strftime("%Y%m%d"))
    log.info("writing to %s", args.out)

    def persist():
        """Write the range-stamped file, plus the fixed-name copy if enabled."""
        ordered = save_predictions(rows, args.out)
        if args.stable_copy and args.stable_copy != args.out:
            save_predictions(rows, args.stable_copy)
        return ordered

    # Resume seeds from the range-stamped file, falling back to the stable copy.
    # That fallback matters: args.out is computed from the first and last day of
    # the FILTERED headline set, so extending the headline data by even one day
    # changes the filename -- without the fallback a resumed run would find
    # nothing and re-pay for every prediction it had already bought.
    rows = []
    if args.resume:
        rows = load_existing(args.out)
        if not rows and args.stable_copy:
            rows = load_existing(args.stable_copy)
            if rows:
                log.info("seeding resume from the stable copy %s", args.stable_copy)
    if rows:
        done = {r["date"] for r in rows}
        skipped = [d for d in days if d.isoformat() in done]
        days = [d for d in days if d.isoformat() not in done]
        log.info("resuming: %d prediction(s) already present, %d day(s) left",
                 len(skipped), len(days))
        if not days:
            log.info("nothing left to do")
            return 0, args.out

    in_tok, out_tok = 0, 0
    failed, consecutive = [], 0

    for i, d in enumerate(days, 1):
        hs = by_day[d]
        target = next_trading_day(d)
        log.info("[%d/%d] %s (%d headlines) -> call for %s", i, len(days), d, len(hs), target)
        try:
            data, usage = ask_model(client, args.model, build_prompt(d, hs))
        except FatalAPIError as exc:
            # No retry fixes this -- stop rather than burning one doomed call
            # per day for the rest of the run.
            log.error("  FATAL: %s", exc)
            log.error("aborting: this error will not resolve by retrying")
            if rows:
                persist()
                log.info("kept %d prediction(s) already made in %s", len(rows), args.out)
            save_failed(failed + [d.isoformat()], args.failed_log)
            return 3, args.out
        except TransientAPIError as exc:
            log.error("  failed: %s", exc)
            failed.append(d.isoformat())
            save_failed(failed, args.failed_log)
            consecutive += 1
            if consecutive >= args.max_consecutive_failures:
                log.error("aborting: %d consecutive failures -- something is "
                          "systematically wrong, not a run of bad luck", consecutive)
                if rows:
                    persist()
                    log.info("kept %d prediction(s) already made in %s", len(rows), args.out)
                return 3, args.out
            continue

        consecutive = 0
        in_tok += usage.input_tokens
        out_tok += usage.output_tokens
        log.info("  %s | sentiment %+d | confidence %.2f",
                 data["direction"].upper(), data["sentiment"], data["confidence"])
        rows.append({
            "date": d.isoformat(),
            "predicts_for": target.isoformat(),
            "direction": data["direction"],
            "sentiment": data["sentiment"],
            "confidence": round(min(1.0, max(0.0, float(data["confidence"]))), 3),
            "n_headlines": len(hs),
            "last_headline_et": hs[-1]["et_time"],
            "rationale": data["rationale"].replace("\n", " ").strip(),
            "model": args.model,
        })
        persist()                            # persist after every paid-for call

    if not rows:
        # An empty file that looks like valid output is worse than no file.
        log.error("no predictions were produced; not writing an empty file")
        return 3, None

    ordered = persist()
    log.info("wrote %d prediction(s) to %s", len(ordered), args.out)
    if args.stable_copy and args.stable_copy != args.out:
        log.info("stable copy updated: %s", args.stable_copy)
    log.info("tokens: %d in / %d out", in_tok, out_tok)
    if failed:
        log.warning("%d day(s) failed and are absent from the file; "
                    "listed in %s -- re-run with --resume to fill them",
                    len(failed), args.failed_log)
        return 2, args.out
    return 0, args.out

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    mode = p.add_argument_group("mode")
    mx = mode.add_mutually_exclusive_group()
    mx.add_argument("--predict-only", action="store_true",
                    help="run the LLM prediction pass and stop")
    mx.add_argument("--backtest-only", action="store_true",
                    help="skip the LLM and back-test an existing predictions CSV; "
                         "makes no API calls and costs nothing")

    g_pred = p.add_argument_group("predict mode")
    g_pred.add_argument("--headlines", default=str(DATA / "cnbc_headlines.csv"),
                        help="input headlines CSV (default: data/cnbc_headlines.csv)")
    g_pred.add_argument("--out", default=None,
                        help="output CSV path; defaults to "
                             "data/<prefix>_<domain>_<start>_<end>.csv, where the dates are "
                             "the first and last prediction day")
    g_pred.add_argument("--out-prefix", default="predictions",
                        help="basename prefix for the output file; a bare prefix lives "
                             "in data/ (default: predictions)")
    g_pred.add_argument("--domain", default=None,
                        help="source name used in the output filename; inferred from the "
                             "headlines' domain column when omitted (e.g. cnbc.com -> cnbc)")
    g_pred.add_argument("--stable-copy", default=str(DATA / "predictions.csv"),
                        help="also write predictions to this fixed filename so downstream "
                             "scripts have a stable path (default: data/predictions.csv); "
                             "'' disables")
    g_pred.add_argument("--dry-run", action="store_true",
                        help="show what would be sent, make no API calls, write no predictions")
    g_pred.add_argument("--limit", type=int, help="only process the first N trading days")
    g_pred.add_argument("--model", default=MODEL)
    g_pred.add_argument("--resume", action="store_true",
                        help="skip days already present in the output file")
    g_pred.add_argument("--failed-log", default=str(DATA / "predictions_failed.txt"),
                        help="days whose call failed, recorded as they happen "
                             "(default: data/predictions_failed.txt)")
    g_pred.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="abort after this many failures in a row (default: 5)")
    g_pred.add_argument("--api-key-file", default=str(ROOT / ".anthropic_key"),
                        help="file holding the API key, used only if the env var is unset "
                             "(default: .anthropic_key at the project root)")

    g_bt = p.add_argument_group("backtest mode")
    g_bt.add_argument("--predictions", default=None,
                      help="predictions CSV to back-test; defaults to the file the "
                           "predict pass just wrote, else --stable-copy")
    g_bt.add_argument("--returns", default=str(DATA / "market_returns.csv"),
                      help="daily return series (default: data/market_returns.csv)")
    g_bt.add_argument("--strategy-out", default=str(DATA / "predictions_strategy.csv"),
                      help="per-strategy per-day results "
                           "(default: data/predictions_strategy.csv)")
    g_bt.add_argument("--verbose-columns", action="store_true",
                      help="append diagnostic columns (signal date, turnover, the three "
                           "cost legs, equity) to the strategy CSV")
    g_bt.add_argument("--min-headlines", type=int, default=0,
                      help="drop predictions made from fewer than N headlines")
    g_bt.add_argument("--min-confidence", type=float, default=0.0,
                      help="drop predictions below this confidence")
    g_bt.add_argument("--flat-on-gaps", action="store_true",
                      help="treat a missing trading day as a flatten-and-re-enter "
                           "rather than a held position")
    g_bt.add_argument("--audit-only", action="store_true",
                      help="run the point-in-time audit and stop; writes nothing")
    g_bt.add_argument("--allow-lookahead", action="store_true",
                      help="downgrade point-in-time audit failures to warnings")

    g_cost = p.add_argument_group("trading costs")
    g_cost.add_argument("--spread-bps", type=float, default=2.0,
                        help="ROUND-TURN spread + slippage in bps; half is charged per "
                             "one-way trade on |change in position| (default: 2.0 for SPY)")
    g_cost.add_argument("--commission-bps", type=float, default=0.0,
                        help="one-way commission in bps of notional traded (default: 0)")
    g_cost.add_argument("--borrow-bps-annual", type=float, default=30.0,
                        help="annualized stock-borrow fee in bps, charged on short "
                             "exposure only (default: 30.0 = 0.30%%/yr)")
    g_cost.add_argument("--expense-bps-annual", type=float, default=9.45,
                        help="ETF expense ratio in bps/yr (default: 9.45 = SPY's 0.0945%%)")
    g_cost.add_argument("--expense-basis", choices=("signed", "gross"), default="signed",
                        help="'signed' treats the fee as a drag inside the fund NAV, so a "
                             "short is credited it; 'gross' charges |position| always "
                             "(default: signed)")
    g_cost.add_argument("--cost-day-count", type=int, default=252,
                        help="days used to convert annual rates to daily; 252 matches "
                             "rf_daily in market_returns.csv (default: 252)")
    g_cost.add_argument("--no-costs", action="store_true",
                        help="zero every cost, for A/B against the gross numbers")

    args = p.parse_args(argv)
    args.out_prefix = resolve_prefix(args.out_prefix)
    # Bare invocation does both. That means a bare run SPENDS API MONEY.
    args.predict = not args.backtest_only
    args.backtest = not args.predict_only

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    if args.backtest_only:
        for name, val in (("--dry-run", args.dry_run), ("--resume", args.resume),
                          ("--limit", args.limit), ("--out", args.out)):
            if val:
                log.warning("%s has no effect in --backtest-only mode; ignored", name)
    if args.predict and args.backtest and args.limit:
        log.warning("--limit caps the PREDICT pass only; the backtest reads the whole "
                    "predictions file from disk and is not limited")
    if args.predict and not args.dry_run:
        log.info("prediction pass ENABLED -- this run will make API calls "
                 "(use --backtest-only to score existing predictions for free)")

    predict_rc, wrote_to = 0, None
    if args.predict:
        predict_rc, wrote_to = do_predict(args)
        if not args.backtest:
            return predict_rc
        if predict_rc in (1, 3):
            log.error("prediction pass returned %d; skipping the backtest", predict_rc)
            return predict_rc

    bt_rc = do_backtest(args, resolve_predictions_path(args, wrote_to))
    if bt_rc:
        return bt_rc
    return predict_rc


if __name__ == "__main__":
    sys.exit(main())
