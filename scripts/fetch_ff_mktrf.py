"""Download daily Mkt-RF (market excess return) from Ken French's Data Library.

Pulls the Fama/French 3-factor daily CSV, slices it to a date window, and writes
a two-column CSV. Values are kept in the units the library publishes them in:
PERCENT per day (0.68 means +0.68%), hence the mkt_rf_pct column name.

Standard library only.

Usage:
    py scripts/fetch_ff_mktrf.py --latest --days 30

Output goes to data/ under the project root, resolved from this file's own
location, so the script can be run from any working directory. A path given on
the command line is used exactly as typed.
"""

import argparse
import csv
import io
import logging
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
# The library rejects urllib's default agent, so present a browser-ish one.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")
ROW_RE = re.compile(r"^\s*(\d{8})\s*,(.*)$")

log = logging.getLogger("ff")


def download(url):
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    log.info("  got %.1f KB", len(blob) / 1024)
    return blob


def parse_daily(blob):
    """Yield (date, mkt_rf) for every daily row in the zipped factor CSV."""
    # The library serves an HTML maintenance page with HTTP 200 when it is down,
    # so check for the zip signature rather than letting zipfile raise.
    if not blob.startswith(b"PK"):
        snippet = blob[:200].decode("utf-8", errors="replace").replace("\n", " ").strip()
        raise RuntimeError(
            "server did not return a zip file (%d bytes, starts %r). "
            "Ken French's site returns an HTML page when it is unavailable; "
            "first bytes: %s" % (len(blob), blob[:4], snippet[:160]))

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        member = zf.namelist()[0]
        log.info("  reading member %s", member)
        text = zf.read(member).decode("utf-8", errors="replace")

    vintage = ""
    rows = []
    for line in text.splitlines():
        if not vintage and "CRSP database" in line:
            vintage = line.strip()
        m = ROW_RE.match(line)
        if not m:
            continue
        stamp, rest = m.group(1), m.group(2)
        fields = [f.strip() for f in rest.split(",")]
        if not fields or not fields[0]:
            continue
        try:
            day = datetime.strptime(stamp, "%Y%m%d").date()
            mkt_rf = float(fields[0])          # first factor column is Mkt-RF
        except ValueError:
            continue
        rows.append((day, mkt_rf))

    if vintage:
        log.info("  source vintage: %s", vintage)
    log.info("  parsed %d daily observation(s)", len(rows))
    if rows:
        log.info("  library covers %s .. %s", rows[0][0], rows[-1][0])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30, help="window length in days (default: 30)")
    parser.add_argument("--end", help="window end date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--start", help="window start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--latest", action="store_true",
                        help="ignore the window and take the most recent --days trading days present")
    parser.add_argument("--out", default=None,
                        help="output CSV path; defaults to "
                             "data/<prefix>_<start>-<end>.csv naming the date range covered")
    parser.add_argument("--out-prefix", default="mkt_rf",
                        help="basename prefix for the range-stamped output file; "
                             "a bare prefix lives in data/ (default: mkt_rf)")
    parser.add_argument("--url", default=FF_URL, help="source zip URL")
    args = parser.parse_args(argv)
    args.out_prefix = resolve_prefix(args.out_prefix)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    try:
        rows = parse_daily(download(args.url))
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    if not rows:
        log.error("no daily rows parsed -- the file layout may have changed")
        return 1

    if args.latest:
        selected = rows[-args.days:]
        start = selected[0][0] if selected else rows[-1][0]
        end = selected[-1][0] if selected else rows[-1][0]
        log.info("taking the most recent %d trading day(s) available", args.days)
    else:
        end = (datetime.strptime(args.end, "%Y-%m-%d").date() if args.end
               else datetime.now(timezone.utc).date())
        start = (datetime.strptime(args.start, "%Y-%m-%d").date() if args.start
                 else end - timedelta(days=args.days))
        log.info("requested window %s .. %s", start, end)
        selected = [r for r in rows if start <= r[0] <= end]

    # The filename carries the range so a directory listing shows what each file
    # covers. Name it by the data actually written when there is any; fall back
    # to the requested window when the result is empty, so an empty file still
    # records what was asked for rather than being silently unlabelled.
    if args.out is None:
        if selected:
            lo, hi = selected[0][0], selected[-1][0]
        else:
            lo, hi = start, end
        args.out = "%s_%s-%s.csv" % (args.out_prefix,
                                     lo.strftime("%Y%m%d"), hi.strftime("%Y%m%d"))
    log.info("writing to %s", args.out)

    ensure_dir(args.out)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "mkt_rf_pct"])
        for day, val in selected:
            w.writerow([day.isoformat(), f"{val:.2f}"])

    log.info("wrote %d row(s) to %s", len(selected), args.out)
    if selected:
        log.info("output covers %s .. %s", selected[0][0], selected[-1][0])
    else:
        log.warning("NO ROWS in the requested window -- the library's last observation is %s, "
                    "which is %d day(s) before the window start",
                    rows[-1][0], (datetime.now(timezone.utc).date() - rows[-1][0]).days)
        log.warning("Ken French updates daily factors monthly, in arrears; "
                    "re-run with --latest to take the most recent available trading days instead")
    return 0


if __name__ == "__main__":
    sys.exit(main())
