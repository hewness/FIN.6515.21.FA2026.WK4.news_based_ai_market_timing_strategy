"""Download recent cnbc.com headlines from the GDELT 2.0 DOC API into a CSV.

The DOC API caps every response at 250 articles, so the requested window is
walked in chunks (one day by default) and any chunk that comes back saturated
is split in half and retried until it fits under the cap or hits the minimum
chunk size. Standard library only -- no third-party dependencies.

Usage:
    py scripts/fetch_gdelt_headlines.py --days 30
    py scripts/fetch_gdelt_headlines.py --merge --only-missing

Output goes to data/ under the project root, resolved from this file's own
location, so the script can be run from any working directory. A path given on
the command line is used exactly as typed.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def resolve_prefix(value):
    """--out-prefix names a file basename, not a path, so a bare prefix is
    placed in data/. A prefix that is absolute or carries a directory component
    is honoured as typed. This is what keeps `--out-prefix cnbc_year` finding
    the files a previous `--out-prefix cnbc_year` run wrote."""
    p = Path(value)
    return str(p if p.is_absolute() or len(p.parts) > 1 else DATA / p)


def ensure_dir(path):
    """Create the directory a file is about to be written into."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250          # hard ceiling imposed by the DOC API
GDELT_FMT = "%Y%m%d%H%M%S"
USER_AGENT = "news-based-ai/0.1 (GDELT DOC API client)"
MIN_INTERVAL = 5.0         # GDELT asks for no more than one request every 5 seconds
MAX_BACKOFF = 90.0         # ceiling on retry backoff, so unlimited retries stay a steady poll

log = logging.getLogger("gdelt")
_last_response = 0.0   # monotonic time the last response finished arriving
FAILED_WINDOWS = []        # windows abandoned after exhausting retries
CONFIRMED_EMPTY = []       # windows GDELT answered successfully with zero articles
FAILED_LOG_PATH = None     # set in main(); lets fetch_window persist failures as they happen


def throttle(min_interval=MIN_INTERVAL):
    """Block until min_interval seconds have passed since the last response.

    Pacing runs end-to-start: the clock is stamped by mark_response() when a
    response finishes arriving, not when a request is sent. That guarantees a
    full quiet gap after every response, however long the request itself took.
    Send-to-start pacing would let a slow response be followed instantly by the
    next request, which looks like back-to-back traffic to a rate limiter.
    """
    wait = min_interval - (time.monotonic() - _last_response)
    if wait > 0:
        time.sleep(wait)


def mark_response():
    """Start the quiet period. Called as soon as a response is in hand."""
    global _last_response
    _last_response = time.monotonic()


def gdelt_time(dt):
    return dt.astimezone(timezone.utc).strftime(GDELT_FMT)


def save_rows(rows, path):
    """Write the CSV atomically: temp file then replace.

    Called after every chunk, so an interrupted run keeps everything fetched up
    to that point. Writing via a temp file means a kill mid-write cannot leave a
    half-written CSV behind -- the old file stays intact until the new one is
    complete.
    """
    ordered = sorted(rows.values(), key=lambda r: r.get("seendate", ""))
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seendate", "title", "domain", "url"])
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(tmp, path)
    return ordered


def save_failed_windows(path):
    """Persist abandoned windows so a killed run doesn't lose the record.

    Without this the list lives only in memory, and a process kill loses the
    knowledge of exactly which windows failed -- leaving days that are present
    but half-fetched, which --only-missing would then skip forever.
    """
    if path is None:
        return
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for a, b in FAILED_WINDOWS:
            fh.write("%s,%s\n" % (gdelt_time(a), gdelt_time(b)))
    os.replace(tmp, path)


def load_failed_windows(path):
    """Read back previously abandoned windows as (start, end) datetimes."""
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or "," not in line:
                    continue
                a, b = line.split(",", 1)
                try:
                    out.append((
                        datetime.strptime(a, GDELT_FMT).replace(tzinfo=timezone.utc),
                        datetime.strptime(b, GDELT_FMT).replace(tzinfo=timezone.utc),
                    ))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return out


def save_empty_log(known_empty, new_empty, path):
    """Persist confirmed-empty days incrementally, same atomic pattern."""
    merged = sorted(set(known_empty) | set(new_empty))
    if not merged:
        return merged
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(merged) + "\n")
    os.replace(tmp, path)
    return merged


def fetch_window(query, start, end, sleep, retries=0):
    """Fetch one time window. Returns the list of article dicts.

    retries=0 means retry forever: GDELT's 429s and connection resets are
    transient, so giving up just converts a recoverable stall into a permanent
    hole in the data. Backoff is capped (MAX_BACKOFF) so an endless retry
    settles into a steady slow poll rather than growing without bound.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(MAX_RECORDS),
        "sort": "datedesc",
        "startdatetime": gdelt_time(start),
        "enddatetime": gdelt_time(end),
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"

    attempt = 0
    while True:
        attempt += 1
        throttle(sleep)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace").strip()
            mark_response()
        except urllib.error.HTTPError as exc:
            mark_response()      # a 429 or 5xx is still a response from GDELT
            # 429 means we out-ran the rate limit; back off hard before trying again.
            wait = min(15.0 * attempt if exc.code == 429 else sleep * attempt * 2, MAX_BACKOFF)
            if retries and attempt >= retries:
                break
            log.warning("  HTTP %s, attempt %d%s, retrying in %.0fs",
                        exc.code, attempt, ("/%d" % retries) if retries else "", wait)
            time.sleep(wait)
            continue
        except OSError as exc:
            # Catches URLError, TimeoutError, ConnectionResetError and every
            # other socket/SSL failure. A reset mid-read arrives as a bare
            # OSError subclass rather than a URLError, and used to kill the run.
            wait = min(sleep * attempt * 2, MAX_BACKOFF)
            if retries and attempt >= retries:
                break
            log.warning("  request failed (%s: %s), attempt %d%s, retrying in %.1fs",
                        type(exc).__name__, exc, attempt,
                        ("/%d" % retries) if retries else "", wait)
            time.sleep(wait)
            continue

        if not body:
            # GDELT returns an empty body when a window genuinely has no articles.
            return []
        try:
            return json.loads(body).get("articles", []) or []
        except json.JSONDecodeError:
            # Rate limiting and query errors come back as plain text, not JSON.
            wait = min(15.0 * attempt if "5 seconds" in body else sleep * attempt * 2,
                       MAX_BACKOFF)
            if retries and attempt >= retries:
                break
            log.warning("  non-JSON response (%s), attempt %d%s, retrying in %.1fs",
                        body[:120].replace("\n", " "), attempt,
                        ("/%d" % retries) if retries else "", wait)
            time.sleep(wait)

    log.error("  giving up on window %s .. %s", gdelt_time(start), gdelt_time(end))
    if (start, end) not in FAILED_WINDOWS:
        FAILED_WINDOWS.append((start, end))
    save_failed_windows(FAILED_LOG_PATH)      # persist immediately, not at run end
    return []


def collect(query, start, end, sleep, min_minutes, depth=0, sink=None, retries=0):
    """Fetch a window, splitting it when the response hits the 250-article cap.

    `sink` is called with each leaf window's articles as soon as they arrive, so
    a long chunk that gets killed partway keeps the sub-windows already fetched
    instead of discarding the lot.
    """
    indent = "  " * depth
    log.info("%sfetching %s -> %s", indent, start.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M"))
    articles = fetch_window(query, start, end, sleep, retries)

    span_minutes = (end - start).total_seconds() / 60
    if len(articles) >= MAX_RECORDS and span_minutes > min_minutes:
        mid = start + (end - start) / 2
        log.info("%s  hit the %d-article cap, splitting at %s",
                 indent, MAX_RECORDS, mid.strftime("%Y-%m-%d %H:%M"))
        return (collect(query, start, mid, sleep, min_minutes, depth + 1, sink, retries)
                + collect(query, mid, end, sleep, min_minutes, depth + 1, sink, retries))

    if len(articles) >= MAX_RECORDS:
        log.warning("%s  window is at the cap but already down to %.0f min; "
                    "some articles in it are unreachable", indent, span_minutes)
    log.info("%s  got %d article(s)", indent, len(articles))
    if sink is not None and articles:
        sink(articles)
    return articles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="cnbc.com", help="source domain (default: cnbc.com)")
    parser.add_argument("--days", type=int, default=30, help="days of history to pull (default: 30)")
    parser.add_argument("--out", default=None,
                        help="output CSV path; defaults to "
                             "data/<prefix>_<start>-<end>.csv naming the requested range")
    parser.add_argument("--out-prefix", default="cnbc_headlines",
                        help="basename prefix for the range-stamped output file; "
                             "a bare prefix lives in data/ (default: cnbc_headlines)")
    parser.add_argument("--stable-copy", default=str(DATA / "cnbc_headlines.csv"),
                        help="also write the merged rows to this fixed filename so "
                             "downstream scripts have a stable path "
                             "(default: data/cnbc_headlines.csv); '' disables")
    parser.add_argument("--merge", action="store_true",
                        help="load the existing --out file and add to it instead of overwriting")
    parser.add_argument("--only-missing", action="store_true",
                        help="with --merge, fetch only days that currently have no rows at all")
    parser.add_argument("--failed-log", default=str(DATA / "cnbc_failed_windows.txt"),
                        help="windows abandoned after retries; persisted as they happen "
                             "(default: data/cnbc_failed_windows.txt)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry only the windows listed in --failed-log, then exit")
    parser.add_argument("--empty-log", default=str(DATA / "cnbc_empty_days.txt"),
                        help="days GDELT confirmed as having no articles; skipped by "
                             "--only-missing (default: data/cnbc_empty_days.txt)")
    parser.add_argument("--chunk-hours", type=float, default=24.0,
                        help="initial window size in hours (default: 24)")
    parser.add_argument("--min-minutes", type=float, default=15.0,
                        help="smallest window the splitter will go down to (default: 15)")
    parser.add_argument("--retries", type=int, default=0,
                        help="attempts per window before giving up; 0 = retry forever (default)")
    parser.add_argument("--sleep", type=float, default=MIN_INTERVAL,
                        help="minimum seconds between API calls; GDELT asks for 5 (default: %.0f)"
                             % MIN_INTERVAL)
    args = parser.parse_args(argv)
    args.out_prefix = resolve_prefix(args.out_prefix)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    global FAILED_LOG_PATH
    FAILED_LOG_PATH = args.failed_log
    FAILED_WINDOWS.extend(load_failed_windows(args.failed_log))
    if FAILED_WINDOWS:
        log.info("carrying %d previously failed window(s) from %s",
                 len(FAILED_WINDOWS), args.failed_log)

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    query = f"domainis:{args.domain}"

    # The filename carries the requested range, so a directory listing shows
    # what each file covers without opening it.
    if args.out is None:
        args.out = "%s_%s-%s.csv" % (args.out_prefix,
                                     start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))

    log.info("GDELT DOC API query %r", query)
    log.info("range %s .. %s UTC (%d days)", start.strftime("%Y-%m-%d %H:%M"),
             end.strftime("%Y-%m-%d %H:%M"), args.days)
    log.info("writing to %s", args.out)

    rows = {}                       # url -> row, de-duplicates overlapping windows
    if args.merge or args.only_missing:
        try:
            # Range-stamped filenames mean prior runs live in sibling files, so
            # resume has to read all of them, not just this run's output path.
            # out_prefix is absolute by now, so the search is rooted at the data
            # directory rather than at whatever the working directory happens to
            # be -- which is what made --merge silently find nothing when this
            # script was invoked from elsewhere. Path.glob treats the directory
            # as literal and only the argument as a pattern, so a project path
            # containing glob metacharacters cannot mis-match.
            prefix_path = Path(args.out_prefix)
            sources = sorted(set(
                [str(q) for q in prefix_path.parent.glob(prefix_path.name + "_*.csv")]
                + ([args.stable_copy] if args.stable_copy else [])
                + [args.out]
            ))
            loaded_from = []
            for src in sources:
                try:
                    with open(src, newline="", encoding="utf-8") as fh:
                        n_before = len(rows)
                        for r in csv.DictReader(fh):
                            if r.get("url"):
                                rows[r["url"]] = r
                        if len(rows) > n_before:
                            loaded_from.append("%s (+%d)" % (src, len(rows) - n_before))
                except FileNotFoundError:
                    continue
            if loaded_from:
                log.info("merged prior rows from: %s", ", ".join(loaded_from))
            log.info("starting from %d existing row(s)", len(rows))
        except OSError as exc:
            log.info("could not read prior rows (%s); starting fresh", exc)

    covered = {r["seendate"][:8] for r in rows.values() if r.get("seendate")}

    # Days GDELT has already answered for, with nothing to give. Re-asking for
    # these forever is what turns a retry loop into pointless load.
    known_empty = set()
    try:
        with open(args.empty_log, encoding="utf-8") as fh:
            known_empty = {ln.strip() for ln in fh if ln.strip()}
    except FileNotFoundError:
        pass
    if args.only_missing:
        log.info("only-missing mode: skipping %d covered day(s) and %d confirmed-empty day(s)",
                 len(covered), len(known_empty))

    def add_articles(articles):
        """Fold a batch of articles into rows. Keyed by url, so re-adding is safe."""
        added = 0
        for art in articles:
            url = art.get("url")
            if not url:
                continue
            if url not in rows:
                added += 1
            rows[url] = {
                "seendate": art.get("seendate", ""),
                "title": (art.get("title") or "").strip(),
                "domain": art.get("domain", ""),
                "url": url,
            }
        return added

    def checkpoint(articles):
        """Sink for collect(): persist each sub-window as soon as it lands."""
        add_articles(articles)
        persist()

    def persist():
        """Write the range-stamped file, plus the fixed-name copy if enabled."""
        ordered = save_rows(rows, args.out)
        if args.stable_copy and args.stable_copy != args.out:
            save_rows(rows, args.stable_copy)
        return ordered

    # --retry-failed: work only the windows a previous run abandoned, and drop
    # each from the log once it succeeds.
    if args.retry_failed:
        pending = load_failed_windows(args.failed_log)
        if not pending:
            log.info("no failed windows recorded in %s -- nothing to retry", args.failed_log)
            return 0
        log.info("retrying %d previously abandoned window(s)", len(pending))
        FAILED_WINDOWS.clear()
        FAILED_WINDOWS.extend(pending)
        for i, (a, b) in enumerate(list(pending), 1):
            log.info("[retry %d/%d] %s -> %s", i, len(pending),
                     a.strftime("%Y-%m-%d %H:%M"), b.strftime("%Y-%m-%d %H:%M"))
            before = len(FAILED_WINDOWS)
            got = collect(query, a, b, args.sleep, args.min_minutes, sink=checkpoint, retries=args.retries)
            # Succeeded (no new abandonment) -> this window is no longer pending.
            if len(FAILED_WINDOWS) == before and (a, b) in FAILED_WINDOWS:
                FAILED_WINDOWS.remove((a, b))
                save_failed_windows(args.failed_log)
                log.info("  cleared from the failed log (%d article(s))", len(got))
        ordered = persist()
        log.info("wrote %d row(s) to %s; %d window(s) still failing",
                 len(ordered), args.out, len(FAILED_WINDOWS))
        return 0 if not FAILED_WINDOWS else 2

    chunk = timedelta(hours=args.chunk_hours)
    cursor = start
    chunk_no = 0
    total_chunks = max(1, int((end - start) / chunk + 0.999))

    while cursor < end:
        stop = min(cursor + chunk, end)
        chunk_no += 1
        # A chunk may span several days. Skip it only when EVERY day in it is
        # already covered or confirmed empty -- checking just the first day
        # would silently drop the rest of the week.
        if args.only_missing:
            span_days = []
            probe = cursor
            while probe < stop:
                span_days.append(probe.strftime("%Y%m%d"))
                probe += timedelta(days=1)
            outstanding = [d for d in span_days if d not in covered and d not in known_empty]
            if not outstanding:
                log.info("[chunk %d/%d] %s .. %s fully covered, skipping",
                         chunk_no, total_chunks, cursor.strftime("%Y-%m-%d"),
                         stop.strftime("%Y-%m-%d"))
                cursor = stop
                continue
            log.info("[chunk %d/%d] %s .. %s -- %d day(s) outstanding",
                     chunk_no, total_chunks, cursor.strftime("%Y-%m-%d"),
                     stop.strftime("%Y-%m-%d"), len(outstanding))
        log.info("[chunk %d/%d] %s", chunk_no, total_chunks, cursor.strftime("%Y-%m-%d"))
        before_failures = len(FAILED_WINDOWS)
        # One unexpected failure should cost a chunk, not the whole run.
        try:
            got = collect(query, cursor, stop, args.sleep, args.min_minutes, sink=checkpoint, retries=args.retries)
        except Exception as exc:                          # noqa: BLE001 - deliberate
            log.error("[chunk %d/%d] crashed (%s: %s); recording it and moving on",
                      chunk_no, total_chunks, type(exc).__name__, exc)
            if (cursor, stop) not in FAILED_WINDOWS:
                FAILED_WINDOWS.append((cursor, stop))
            save_failed_windows(FAILED_LOG_PATH)
            persist()
            cursor = stop
            continue

        # If the whole chunk was served without a single abandoned window, then
        # any day in it that produced no articles genuinely has none in GDELT.
        # Record those so later passes stop asking for them.
        if len(FAILED_WINDOWS) == before_failures:
            days_with_articles = {a.get("seendate", "")[:8] for a in got}
            probe = cursor
            while probe < stop:
                stamp = probe.strftime("%Y%m%d")
                if stamp not in days_with_articles and stamp not in known_empty:
                    CONFIRMED_EMPTY.append(stamp)
                probe += timedelta(days=1)

        add_articles(got)
        # Persist after every chunk so an interrupted run keeps its progress.
        persist()
        save_empty_log(known_empty, CONFIRMED_EMPTY, args.empty_log)
        log.info("[chunk %d/%d] running total: %d unique article(s) (saved)",
                 chunk_no, total_chunks, len(rows))
        cursor = stop

    if CONFIRMED_EMPTY:
        # Use the helper rather than a bare open(): it writes atomically and
        # creates the directory, matching the incremental save in the loop.
        merged = save_empty_log(known_empty, CONFIRMED_EMPTY, args.empty_log)
        log.info("recorded %d newly confirmed-empty day(s) in %s (%d total)",
                 len(CONFIRMED_EMPTY), args.empty_log, len(merged))

    ordered = persist()
    log.info("wrote %d row(s) to %s", len(ordered), args.out)
    if args.stable_copy and args.stable_copy != args.out:
        log.info("stable copy updated: %s", args.stable_copy)
    if ordered:
        log.info("earliest: %s  |  latest: %s", ordered[0]["seendate"], ordered[-1]["seendate"])
    else:
        log.warning("no articles returned -- check the domain or widen the date range")

    # An abandoned window is a hole in the data, not a day with no news. Say so loudly.
    if FAILED_WINDOWS:
        log.error("INCOMPLETE: %d window(s) were abandoned after exhausting retries:",
                  len(FAILED_WINDOWS))
        for a, b in FAILED_WINDOWS:
            log.error("    %s .. %s", gdelt_time(a), gdelt_time(b))
        log.error("re-run with --merge --only-missing once the API is healthy to fill them")
        return 2

    empty_days = []
    day = start
    have = {r["seendate"][:8] for r in ordered if r.get("seendate")}
    confirmed = known_empty | set(CONFIRMED_EMPTY)
    while day <= end:
        stamp = day.strftime("%Y%m%d")
        if stamp not in have and stamp not in confirmed:
            empty_days.append(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)
    if empty_days:
        log.warning("%d day(s) in the window have no rows: %s", len(empty_days), ", ".join(empty_days))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
