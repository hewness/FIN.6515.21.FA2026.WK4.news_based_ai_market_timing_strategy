"""Download daily S&P 500 returns (and a risk-free rate) from FRED.

Ken French's factor library lags by ~7 weeks, so it cannot cover a recent
window. FRED publishes both series with a one-day lag and needs no API key:

    SP500   S&P 500 index close (price index -- excludes dividends)
    DGS1MO  1-month Treasury constant maturity yield, annualized percent

The daily return for date D is close(D) / close(previous trading day) - 1, i.e.
the return realized *during* day D. That is exactly what a position entered at
the prior close earns, which is what scripts/predict_positions.py scores
positions against.

Usage:
    py scripts/fetch_market_returns.py

Output goes to data/ under the project root, resolved from this file's own
location, so the script can be run from any working directory. A path given on
the command line is used exactly as typed.
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# --- project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def ensure_dir(path):
    """Create the directory a file is about to be written into."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s"
# FRED drops connections from unfamiliar agents, so present a browser one.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")
TRADING_DAYS = 252

log = logging.getLogger("returns")


def fetch_series(series_id, local_path=None):
    """Return {date: float} for a FRED series, skipping missing observations.

    FRED serves this file happily to curl run from a shell but drops
    Python-originated connections (both urllib and curl via subprocess), so a
    pre-downloaded local copy is the reliable path; the network code below is
    kept as a convenience for environments where it does work.
    """
    if local_path and os.path.exists(local_path):
        log.info("reading %s from local file %s", series_id, local_path)
        with open(local_path, encoding="utf-8", errors="replace") as fh:
            return parse_series_csv(fh.read(), series_id)

    url = FRED_CSV % series_id
    log.info("downloading %s", url)
    text = None

    # FRED closes connections on Python's urllib handshake but serves curl
    # fine, so curl is the primary transport here and urllib is the fallback.
    try:
        proc = subprocess.run(
            ["curl", "-sL", "--max-time", "120", "-A", USER_AGENT, url],
            capture_output=True, timeout=180,
        )
        if proc.returncode == 0 and proc.stdout:
            text = proc.stdout.decode("utf-8", errors="replace")
        else:
            log.warning("  curl returned %d", proc.returncode)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("  curl unavailable (%s: %s)", type(exc).__name__, exc)

    if text is None:
        headers = {"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"}
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                break
            except Exception as exc:
                log.warning("  urllib attempt %d/3 failed (%s: %s)",
                            attempt, type(exc).__name__, exc)
                time.sleep(2.0 * attempt)

    if text is None:
        raise RuntimeError(
            "could not download FRED series %s -- fetch it in a shell with\n"
            # the path is quoted: every path here contains a space
            '  curl -sL -o "%s" "%s"\n'
            "and pass it with --index-file / --riskfree-file"
            % (series_id, DATA / ("%s.csv" % series_id), url))

    return parse_series_csv(text, series_id)


def parse_series_csv(text, series_id):
    out = {}
    for row in csv.DictReader(text.splitlines()):
        raw_date = row.get("observation_date") or row.get("DATE")
        raw_val = row.get(series_id)
        if not raw_date or raw_val in (None, "", "."):
            continue          # FRED writes "." for holidays and missing days
        try:
            out[datetime.strptime(raw_date, "%Y-%m-%d").date()] = float(raw_val)
        except ValueError:
            continue
    log.info("  %s: %d observation(s), %s .. %s",
             series_id, len(out), min(out) if out else "-", max(out) if out else "-")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DATA / "market_returns.csv"),
                   help="output CSV path (default: data/market_returns.csv)")
    p.add_argument("--index", default="SP500", help="FRED price-index series (default: SP500)")
    p.add_argument("--riskfree", default="DGS1MO", help="FRED risk-free series (default: DGS1MO)")
    p.add_argument("--index-file", default=str(DATA / "SP500.csv"),
                   help="pre-downloaded CSV for the index series "
                        "(default: data/SP500.csv if present, else download)")
    p.add_argument("--riskfree-file", default=str(DATA / "DGS1MO.csv"),
                   help="pre-downloaded CSV for the risk-free series "
                        "(default: data/DGS1MO.csv if present, else download)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    # A download failure is an expected condition (FRED regularly refuses
    # Python-originated requests), so report it as an error with the curl
    # instructions rather than dumping a traceback.
    try:
        closes = fetch_series(args.index, args.index_file)
        rates = fetch_series(args.riskfree, args.riskfree_file)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    if not closes:
        log.error("no index data returned")
        return 1

    days = sorted(closes)
    rows = []
    for prev, day in zip(days, days[1:]):
        ret = closes[day] / closes[prev] - 1.0
        # Annualized percent yield -> per-trading-day rate.
        annual = rates.get(day)
        rf_daily = ((1.0 + annual / 100.0) ** (1.0 / TRADING_DAYS) - 1.0) if annual is not None else ""
        rows.append({
            "date": day.isoformat(),
            "close": round(closes[day], 4),
            "mkt_ret": round(ret, 8),
            "rf_daily": round(rf_daily, 10) if rf_daily != "" else "",
            "mkt_excess": round(ret - rf_daily, 8) if rf_daily != "" else "",
        })

    ensure_dir(args.out)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "close", "mkt_ret", "rf_daily", "mkt_excess"])
        w.writeheader()
        w.writerows(rows)

    log.info("wrote %d daily return(s) to %s", len(rows), args.out)
    log.info("covers %s .. %s", rows[0]["date"], rows[-1]["date"])
    log.warning("note: %s is a PRICE index -- returns exclude dividends", args.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
