"""Scrape arXiv econ.GN metadata into a CSV.

Sanctioned use of arXiv's public API (1 req / 3 s, single connection).
Run from anywhere: `python sankalp/example/code/scrape_econ_gn.py`.
"""

import logging
import sys
import time
from pathlib import Path

import feedparser  # arXiv returns Atom XML; feedparser handles namespaced fields like arxiv_doi
import pandas as pd
import requests

# arXiv's public Atom query endpoint. Documented at https://info.arxiv.org/help/api/user-manual.html.
BASE = "http://export.arxiv.org/api/query"

# arXiv asks API users to identify themselves so abuse can be traced before the IP is blocked.
# Format follows their guidance: app name + contact email.
HEADERS = {"User-Agent": "USC-Econ-Workshop/1.0 (mailto:ssharma9@usc.edu)"}

# arXiv caps a single response at ~2000 entries; 200 is a conservative page size that
# keeps individual responses small (faster parse, less memory) without blowing the request budget.
PAGE = 200

# Rate limit is "no more than 1 request every 3 seconds" per arXiv's terms of use.
# The 0.1s cushion absorbs clock drift so we never accidentally fall under the floor.
DELAY = 3.1

# Hard ceiling on records pulled in one run. econ.GN currently has well under this many submissions,
# so this also acts as a safety brake against an infinite loop if pagination logic regresses.
MAX_RECORDS = 6000

# Network is flaky and arXiv occasionally returns 503/empty payloads under load.
# Three attempts with backoff is enough to ride out a brief blip without masking a real outage.
RETRIES = 3
RETRY_SLEEP = 10

# Output lives next to the script's sibling `output/` folder so the scrape is self-contained
# regardless of where the user invokes Python from. resolve() defends against symlinks/relative paths.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "econ_gn.csv"
LOG_PATH = OUTPUT_DIR / "scrape.log"

# Log to both the file (audit trail / post-mortem) and stderr (live progress while the user watches).
# mode="w" truncates the log each run so the file always reflects only the latest scrape.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("arxiv")


def fetch(start: int) -> str:
    """Pull one page of econ.GN results starting at offset `start`. Returns raw Atom XML."""
    params = {
        # cat:econ.GN restricts to the General Economics arXiv category.
        "search_query": "cat:econ.GN",
        "start": start,
        "max_results": PAGE,
        # Sort ascending by submission date so paging is stable: new arXiv submissions
        # land at the END of the result set, not the start, which means re-running the script
        # later won't shift earlier offsets and corrupt comparisons across runs.
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    for attempt in range(1, RETRIES + 1):
        try:
            # 30s timeout balances arXiv's occasional slow responses against detecting a hung socket.
            resp = requests.get(BASE, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.text
            # Non-200 (typically 503 "service unavailable" during maintenance windows) is logged
            # but not raised here so the retry loop gets a chance.
            log.warning("start=%d status=%d attempt=%d", start, resp.status_code, attempt)
        except requests.RequestException as exc:
            # Catches DNS failures, connection resets, read timeouts, SSL errors, etc.
            # Anything narrower would let transient network errors crash a multi-hour scrape.
            log.warning("start=%d error=%s attempt=%d", start, exc, attempt)
        time.sleep(RETRY_SLEEP)
    # Out of retries. Raise so the caller (main) terminates and the partial CSV isn't silently truncated.
    raise RuntimeError(f"failed to fetch start={start} after {RETRIES} attempts")


def parse_entry(e) -> dict:
    """Flatten one feedparser entry into a CSV-friendly dict."""
    # arXiv exposes multiple <link> elements per entry; the one with title="pdf" is the paper PDF.
    # next(..., None) handles the rare case where the PDF link is missing (withdrawn papers).
    pdf = next((l.href for l in e.links if l.get("title") == "pdf"), None)
    return {
        # e.id is a URL like "http://arxiv.org/abs/2501.12345v1"; we keep just the trailing arXiv ID.
        "id": e.id.rsplit("/", 1)[-1],
        "submitted": e.published,  # original submission timestamp
        "updated": e.updated,       # last revision timestamp (differs when authors post v2, v3, ...)
        # arXiv titles and abstracts arrive with hard-wrapped newlines and indented continuation lines.
        # split()+join() collapses any run of whitespace into a single space, giving clean CSV cells.
        "title": " ".join(e.title.split()),
        "authors": "; ".join(a.name for a in e.authors),  # semicolon-delimited so commas inside names survive CSV
        "primary": e.arxiv_primary_category["term"],  # e.g., "econ.GN"
        "categories": "; ".join(t.term for t in e.tags),  # all cross-listed categories
        "abstract": " ".join(e.summary.split()),
        "pdf": pdf,
        # DOI and journal_ref are populated only after publication; getattr handles unpublished preprints.
        "doi": getattr(e, "arxiv_doi", None),
        "journal_ref": getattr(e, "arxiv_journal_ref", None),
    }


def main() -> None:
    rows: list[dict] = []
    # Walk the result set in PAGE-sized windows. range stops at MAX_RECORDS as a hard ceiling.
    for start in range(0, MAX_RECORDS, PAGE):
        # monotonic() (not time()) is immune to wall-clock adjustments mid-run; safer for measuring elapsed.
        t0 = time.monotonic()
        feed = feedparser.parse(fetch(start))
        elapsed = time.monotonic() - t0
        n = len(feed.entries)
        log.info("start=%d entries=%d elapsed=%.2fs total=%d", start, n, elapsed, len(rows) + n)
        # Empty page = arXiv has nothing more to give us; we've reached the end of econ.GN.
        # We still sleep before exiting so a subsequent quick re-run doesn't violate the rate limit.
        if n == 0:
            time.sleep(DELAY)
            break
        rows.extend(parse_entry(e) for e in feed.entries)
        # Sleep AFTER the request so the rate limit is enforced between consecutive fetches,
        # not before the first one (which would waste 3s on every run).
        time.sleep(DELAY)

    df = pd.DataFrame(rows)
    # index=False: pandas' default integer index is meaningless here; the arXiv `id` column is the real key.
    df.to_csv(CSV_PATH, index=False)
    log.info("wrote %d rows to %s", len(df), CSV_PATH)
    print(f"\n{len(df)} papers -> {CSV_PATH}")


if __name__ == "__main__":
    main()
