#!/usr/bin/env python3
"""
SEC EDGAR scraper.

Endpoints and record counts documented in sec_edgar_endpoints.md.

SAFETY DEFAULT: running this with no arguments downloads exactly ONE small
filing document (under 500 KB) and stops. Every bulk mode is opt-in and the
two multi-gigabyte modes additionally require --yes.

    python3 sec_edgar_scraper.py                      # one random small filing
    python3 sec_edgar_scraper.py demo --seed 42       # same, reproducible
    python3 sec_edgar_scraper.py demo --dry-run       # print URLs, fetch nothing
    python3 sec_edgar_scraper.py selftest             # offline assertions

    python3 sec_edgar_scraper.py tickers              # 10,403 ticker/CIK rows
    python3 sec_edgar_scraper.py submissions --ticker AAPL --ticker MSFT
    python3 sec_edgar_scraper.py frames --tag Assets --period CY2023Q1I
    python3 sec_edgar_scraper.py fts --query '"climate risk"' --forms 10-K \\
                                     --start 2023-01-01 --end 2023-12-31
    python3 sec_edgar_scraper.py index --year 2024 --quarter 1
    python3 sec_edgar_scraper.py filings --manifest data/index/2024Q1.jsonl \\
                                         --form 10-K --max-items 25
    python3 sec_edgar_scraper.py bulk --which submissions --yes   # 1.56 GB

The SEC requires a User-Agent naming a real person and email. Set it here or
via the SEC_UA environment variable. Without it every request returns 403.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import certifi

#############################################
# CONFIGURATION                             #
#############################################

# identify the operator to the SEC; a generic agent string is refused with 403
DEFAULT_UA = "Sankalp Sharma (USC) sankalp.sharma437@gmail.com"
USER_AGENT = os.environ.get("SEC_UA", DEFAULT_UA)

# the SEC publishes a 10 requests/second ceiling; stay under it
RATE_LIMIT_PER_SEC = 8.0

WWW = "https://www.sec.gov"
DATA = "https://data.sec.gov"
EFTS = "https://efts.sec.gov/LATEST/search-index"

# the full-text index refuses from + size above this value
FTS_WINDOW_MAX = 10_000
FTS_PAGE_SIZE = 100

# EDGAR quarterly indexes start here; the scraper clamps requests to this floor
FIRST_INDEX_YEAR = 1993

OUT_ROOT = Path(__file__).resolve().parent / "data"

# python.org Python cannot verify sec.gov without an explicit CA bundle
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


#############################################
# HTTP LAYER                                #
#############################################


@dataclass
class Fetcher:
    """Rate-limited HTTP client with retry, byte caps, and a request counter."""

    user_agent: str = USER_AGENT
    rate_limit: float = RATE_LIMIT_PER_SEC
    max_retries: int = 5
    timeout: int = 60
    verbose: bool = True
    n_requests: int = field(default=0, init=False)
    _last_call: float = field(default=0.0, init=False)

    def _throttle(self) -> None:
        """Sleep just long enough to hold the configured requests-per-second."""

        min_gap = 1.0 / self.rate_limit
        elapsed = time.monotonic() - self._last_call
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        self._last_call = time.monotonic()

    def get(self, url: str, max_bytes: int | None = None) -> bytes:
        """Fetch one URL, retrying on transient failures with growing backoff."""

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                    "Accept": "*/*",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=SSL_CONTEXT
                ) as response:
                    self.n_requests += 1
                    raw = (
                        response.read()
                        if max_bytes is None
                        else response.read(max_bytes)
                    )
                    return _maybe_gunzip(raw, response.headers.get("Content-Encoding"))
            except urllib.error.HTTPError as err:
                # a 403 here almost always means the User-Agent was rejected
                if err.code == 403:
                    raise RuntimeError(
                        f"403 from {url}. Set SEC_UA to 'Your Name your@email' "
                        "and retry; the SEC blocks undeclared automated tools."
                    ) from err
                if err.code == 404 or err.code < 500 and err.code != 429:
                    raise
                last_error = err
            except (urllib.error.URLError, TimeoutError, ssl.SSLError) as err:
                last_error = err

            backoff = 1.5 * (2**attempt) + random.uniform(0, 0.5)
            if self.verbose:
                print(f"  retry {attempt + 1}/{self.max_retries} in {backoff:.1f}s")
            time.sleep(backoff)

        raise RuntimeError(f"gave up on {url}: {last_error!r}")

    def head_size(self, url: str) -> int | None:
        """Return Content-Length for a URL without downloading the body."""

        self._throttle()
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": self.user_agent}
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=SSL_CONTEXT
            ) as response:
                self.n_requests += 1
                length = response.headers.get("Content-Length")
                return int(length) if length else None
        except urllib.error.HTTPError:
            return None

    def get_json(self, url: str) -> dict:
        """Fetch a URL and parse it as JSON."""

        return json.loads(self.get(url).decode("utf-8"))


def _maybe_gunzip(raw: bytes, encoding: str | None) -> bytes:
    """Decompress a response body when the server gzipped it."""

    if encoding and "gzip" in encoding.lower():
        import gzip

        return gzip.decompress(raw)
    return raw


#############################################
# OUTPUT HELPERS                            #
#############################################


def write_atomic(target_path: Path, payload: bytes) -> Path:
    """Write to a sibling .tmp file then rename, so a crash leaves no half file."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(target_path)
    return target_path


def already_done(target_path: Path) -> bool:
    """Treat a non-empty existing file as complete, which makes runs resumable."""

    return target_path.exists() and target_path.stat().st_size > 0


#############################################
# URL BUILDERS (pure, covered by selftest)  #
#############################################


def pad_cik(cik: int | str) -> str:
    """Zero-pad a CIK to the 10 digits that data.sec.gov requires."""

    return str(int(str(cik).lstrip("CIK").lstrip("0") or 0)).zfill(10)


def accession_nodash(accession: str) -> str:
    """Strip the dashes from an accession number for Archives path building."""

    return accession.replace("-", "")


def url_submissions(cik: int | str) -> str:
    return f"{DATA}/submissions/CIK{pad_cik(cik)}.json"


def url_companyfacts(cik: int | str) -> str:
    return f"{DATA}/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json"


def url_frames(tag: str, period: str, taxonomy: str = "us-gaap", unit: str = "USD") -> str:
    return f"{DATA}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"


def url_filing_doc(cik: int | str, accession: str, document: str) -> str:
    """Build the static Archives URL for one document inside one filing."""

    return (
        f"{WWW}/Archives/edgar/data/{int(cik)}/"
        f"{accession_nodash(accession)}/{document}"
    )


def url_filing_index(cik: int | str, accession: str) -> str:
    return (
        f"{WWW}/Archives/edgar/data/{int(cik)}/"
        f"{accession_nodash(accession)}/index.json"
    )


def url_quarter_index(year: int, quarter: int, name: str = "master.zip") -> str:
    return f"{WWW}/Archives/edgar/full-index/{year}/QTR{quarter}/{name}"


#############################################
# MODE: DEMO (the default)                  #
#############################################


def parse_recent_filings(submissions_dict: dict) -> list[dict]:
    """Flip the parallel arrays in filings.recent into one dict per filing."""

    recent_dict = submissions_dict["filings"]["recent"]
    field_list = list(recent_dict.keys())
    n_rows = len(recent_dict["accessionNumber"])

    filing_list = []
    for row in range(n_rows):
        filing_list.append({name: recent_dict[name][row] for name in field_list})
    return filing_list


def pick_small_filing(filing_list: list[dict], max_bytes: int, rng: random.Random) -> dict | None:
    """Choose one filing whose primary document is small and actually named."""

    candidate_list = [
        f
        for f in filing_list
        if f.get("primaryDocument")
        and str(f.get("size") or 0).isdigit()
        and 0 < int(f["size"]) <= max_bytes
    ]
    if not candidate_list:
        return None
    return rng.choice(candidate_list)


def mode_demo(args, fetcher: Fetcher) -> int:
    """Download exactly one small filing document and print what it is.

    Three requests total: the ticker table, one company's submission history,
    and one filing document. Nothing here touches the bulk archives.
    """

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir or (OUT_ROOT / "demo"))

    print("SEC EDGAR demo: fetching ONE small filing.\n")

    # step 1: pull the ticker table and choose a company at random
    tickers_url = f"{WWW}/files/company_tickers.json"
    print(f"[1/3] company universe   {tickers_url}")
    if args.dry_run:
        print("      (dry run, stopping before any download)")
        print(f"      next would be   {url_submissions(320193)}")
        return 0

    tickers_dict = fetcher.get_json(tickers_url)
    company_list = list(tickers_dict.values())
    print(f"      {len(company_list):,} companies with a ticker")

    # step 2: walk random companies until one has a filing under the size cap
    chosen_company, chosen_filing = None, None
    for attempt in range(args.max_tries):
        company = rng.choice(company_list)
        subs_url = url_submissions(company["cik_str"])
        print(f"[2/3] filing history     {company['ticker']}  {subs_url}")

        try:
            submissions_dict = fetcher.get_json(subs_url)
        except (RuntimeError, urllib.error.HTTPError) as err:
            print(f"      skipping, {err}")
            continue

        filing_list = parse_recent_filings(submissions_dict)
        filing = pick_small_filing(filing_list, args.max_bytes, rng)
        if filing is None:
            print(f"      no document under {args.max_bytes:,} bytes, trying another")
            continue

        chosen_company, chosen_filing = company, filing
        print(f"      {len(filing_list):,} recent filings, picked one")
        break

    if chosen_filing is None:
        print(f"\nNo small filing found in {args.max_tries} tries. Raise --max-bytes.")
        return 1

    # step 3: download that one document and write it atomically
    cik = int(chosen_company["cik_str"])
    doc_url = url_filing_doc(cik, chosen_filing["accessionNumber"], chosen_filing["primaryDocument"])
    # primaryDocument sometimes carries a subdirectory, so flatten it
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", chosen_filing["primaryDocument"])
    target_path = (
        out_dir
        / f"{chosen_company['ticker']}_{chosen_filing['form']}"
        f"_{chosen_filing['accessionNumber']}_{safe_name}"
    )

    print(f"[3/3] document           {doc_url}")
    if already_done(target_path):
        print(f"      already on disk, skipping: {target_path}")
    else:
        body = fetcher.get(doc_url, max_bytes=args.max_bytes * 4)
        write_atomic(target_path, body)

    # record what was fetched so the run is auditable
    meta_path = target_path.with_suffix(target_path.suffix + ".meta.json")
    write_atomic(
        meta_path,
        json.dumps(
            {
                "ticker": chosen_company["ticker"],
                "company": chosen_company["title"],
                "cik": cik,
                "url": doc_url,
                "bytes_on_disk": target_path.stat().st_size,
                "seed": args.seed,
                **{
                    k: chosen_filing.get(k)
                    for k in ("accessionNumber", "form", "filingDate", "reportDate", "size", "primaryDocDescription")
                },
            },
            indent=2,
        ).encode(),
    )

    print("\nDone.")
    print(f"  company    {chosen_company['title']} ({chosen_company['ticker']}, CIK {cik})")
    print(f"  form       {chosen_filing['form']}  filed {chosen_filing['filingDate']}")
    print(f"  saved      {target_path}  ({target_path.stat().st_size:,} bytes)")
    print(f"  metadata   {meta_path}")
    print(f"  requests   {fetcher.n_requests}")
    return 0


#############################################
# MODE: TICKERS                             #
#############################################


def mode_tickers(args, fetcher: Fetcher) -> int:
    """Save the ticker-to-CIK table and the exchange-annotated version."""

    out_dir = Path(args.output_dir or (OUT_ROOT / "reference"))

    for name, url in [
        ("company_tickers.json", f"{WWW}/files/company_tickers.json"),
        ("company_tickers_exchange.json", f"{WWW}/files/company_tickers_exchange.json"),
    ]:
        target_path = out_dir / name
        if already_done(target_path) and not args.force:
            print(f"skip (exists) {target_path}")
            continue
        write_atomic(target_path, fetcher.get(url))
        print(f"wrote {target_path}  ({target_path.stat().st_size:,} bytes)")

    return 0


#############################################
# MODE: SUBMISSIONS                         #
#############################################


def resolve_ciks(ticker_list: list[str], cik_list: list[str], fetcher: Fetcher) -> list[tuple[int, str]]:
    """Turn a mix of tickers and CIKs into a deduplicated (cik, label) list."""

    resolved_list: list[tuple[int, str]] = []

    if ticker_list:
        tickers_dict = fetcher.get_json(f"{WWW}/files/company_tickers.json")
        by_ticker = {v["ticker"].upper(): v for v in tickers_dict.values()}
        for ticker in ticker_list:
            row = by_ticker.get(ticker.upper())
            if row is None:
                print(f"warning: ticker {ticker} not in company_tickers.json")
                continue
            resolved_list.append((int(row["cik_str"]), row["ticker"]))

    for cik in cik_list:
        resolved_list.append((int(cik), f"CIK{pad_cik(cik)}"))

    seen_set: set[int] = set()
    unique_list = []
    for cik, label in resolved_list:
        if cik not in seen_set:
            seen_set.add(cik)
            unique_list.append((cik, label))
    return unique_list


def mode_submissions(args, fetcher: Fetcher) -> int:
    """Save the submissions JSON for each requested company, shards included."""

    out_dir = Path(args.output_dir or (OUT_ROOT / "submissions"))
    target_list = resolve_ciks(args.ticker, args.cik, fetcher)

    if not target_list:
        print("nothing to do: pass --ticker or --cik")
        return 1

    for cik, label in target_list:
        target_path = out_dir / f"CIK{pad_cik(cik)}.json"
        if already_done(target_path) and not args.force:
            print(f"skip (exists) {label}")
            continue

        body = fetcher.get(url_submissions(cik))
        write_atomic(target_path, body)
        submissions_dict = json.loads(body)
        n_recent = len(submissions_dict["filings"]["recent"]["accessionNumber"])
        print(f"{label}: {n_recent:,} recent filings -> {target_path.name}")

        # older filings live in shard files listed under filings.files
        if args.shards:
            for shard in submissions_dict["filings"].get("files", []):
                shard_path = out_dir / shard["name"]
                if already_done(shard_path) and not args.force:
                    continue
                write_atomic(shard_path, fetcher.get(f"{DATA}/submissions/{shard['name']}"))
                print(f"  shard {shard['name']}: {shard['filingCount']:,} filings")

    return 0


#############################################
# MODE: FRAMES (cross-section of one tag)   #
#############################################


def mode_frames(args, fetcher: Fetcher) -> int:
    """Pull one XBRL tag for every filer in one period, which is the cheap panel."""

    out_dir = Path(args.output_dir or (OUT_ROOT / "frames"))
    target_path = out_dir / f"{args.taxonomy}_{args.tag}_{args.unit}_{args.period}.json"

    if already_done(target_path) and not args.force:
        print(f"skip (exists) {target_path}")
        return 0

    url = url_frames(args.tag, args.period, args.taxonomy, args.unit)
    print(f"GET {url}")
    body = fetcher.get(url)
    write_atomic(target_path, body)

    frame_dict = json.loads(body)
    print(f"wrote {target_path}  ({len(frame_dict['data']):,} companies)")
    return 0


#############################################
# MODE: FULL-TEXT SEARCH                    #
#############################################


def fts_page(fetcher: Fetcher, query: str, forms: str | None, start: str | None,
             end: str | None, offset: int) -> dict:
    """Request one 100-hit page from the Elasticsearch front end."""

    param_dict = {"q": query, "from": offset}
    if forms:
        param_dict["forms"] = forms
    if start:
        param_dict["startdt"] = start
    if end:
        param_dict["enddt"] = end
    return fetcher.get_json(f"{EFTS}?{urllib.parse.urlencode(param_dict)}")


def mode_fts(args, fetcher: Fetcher) -> int:
    """Page a full-text query into JSONL, refusing to cross the 10,000 ceiling.

    The index rejects from + size above 10,000. When a query is larger than
    that, this stops and tells you to narrow the date window rather than
    silently returning a truncated set.
    """

    out_dir = Path(args.output_dir or (OUT_ROOT / "fts"))
    slug = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_")[:40]
    target_path = out_dir / f"{slug}_{args.start or 'all'}_{args.end or 'all'}.jsonl"

    if already_done(target_path) and not args.force:
        print(f"skip (exists) {target_path}")
        return 0

    first_page = fts_page(fetcher, args.query, args.forms, args.start, args.end, 0)
    total_dict = first_page["hits"]["total"]
    print(f"total hits: {total_dict['value']:,} ({total_dict['relation']})")

    if total_dict["relation"] != "eq" or total_dict["value"] > FTS_WINDOW_MAX:
        print(
            f"\nThis query exceeds the {FTS_WINDOW_MAX:,} result ceiling.\n"
            "Narrow it with --start and --end (try one month at a time) and\n"
            "run again per slice. Paging past the ceiling returns an error,\n"
            "not more results."
        )
        return 1

    hit_list = list(first_page["hits"]["hits"])
    offset = FTS_PAGE_SIZE
    while offset < total_dict["value"] and offset < FTS_WINDOW_MAX:
        page = fts_page(fetcher, args.query, args.forms, args.start, args.end, offset)
        page_hits = page["hits"]["hits"]
        if not page_hits:
            break
        hit_list.extend(page_hits)
        offset += FTS_PAGE_SIZE
        print(f"  {len(hit_list):,}/{total_dict['value']:,}")

    payload = "\n".join(json.dumps(h["_source"] | {"_id": h["_id"]}) for h in hit_list)
    write_atomic(target_path, payload.encode())
    print(f"wrote {target_path}  ({len(hit_list):,} hits)")
    return 0


#############################################
# MODE: QUARTERLY INDEX                     #
#############################################


def parse_master_idx(raw: bytes) -> list[dict]:
    """Turn a pipe-delimited master index into one dict per filing.

    The file carries a text preamble; data starts after the dashed rule under
    the CIK|Company Name|Form Type|Date Filed|Filename header.
    """

    text = raw.decode("latin-1")
    line_list = text.splitlines()

    start = 0
    for pos, line in enumerate(line_list):
        if line.startswith("---"):
            start = pos + 1
            break

    row_list = []
    for line in line_list[start:]:
        part_list = line.split("|")
        if len(part_list) != 5 or not part_list[0].strip().isdigit():
            continue
        row_list.append(
            {
                "cik": int(part_list[0]),
                "company": part_list[1].strip(),
                "form": part_list[2].strip(),
                "date_filed": part_list[3].strip(),
                "path": part_list[4].strip(),
                "accession": Path(part_list[4]).stem,
            }
        )
    return row_list


def mode_index(args, fetcher: Fetcher) -> int:
    """Download quarterly master indexes as zips and flatten them to JSONL.

    The zip is roughly one eighth the size of the .idx, so this always takes
    the zip. One quarter of recent data is about 4 MB down, 370k rows out.
    """

    out_dir = Path(args.output_dir or (OUT_ROOT / "index"))
    this_year = time.gmtime().tm_year

    year_list = [args.year] if args.year else list(range(FIRST_INDEX_YEAR, this_year + 1))
    quarter_list = [args.quarter] if args.quarter else [1, 2, 3, 4]

    n_written = 0
    for year in year_list:
        for quarter in quarter_list:
            target_path = out_dir / f"{year}Q{quarter}.jsonl"
            if already_done(target_path) and not args.force:
                print(f"skip (exists) {year}Q{quarter}")
                continue

            url = url_quarter_index(year, quarter, "master.zip")
            try:
                body = fetcher.get(url)
            except urllib.error.HTTPError as err:
                # future or pre-EDGAR quarters simply do not exist
                print(f"{year}Q{quarter}: not available ({err.code})")
                continue

            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                inner_name = archive.namelist()[0]
                row_list = parse_master_idx(archive.read(inner_name))

            payload = "\n".join(json.dumps(r) for r in row_list)
            write_atomic(target_path, payload.encode())
            n_written += 1
            print(f"{year}Q{quarter}: {len(row_list):,} filings -> {target_path.name}")

    print(f"\n{n_written} quarter files written to {out_dir}")
    return 0


#############################################
# MODE: FILING DOCUMENTS                    #
#############################################


def mode_filings(args, fetcher: Fetcher) -> int:
    """Download primary documents for filings selected from a quarter manifest.

    Reads the JSONL produced by the index mode, filters it, then fetches each
    filing's complete submission text file. --max-items is mandatory in spirit:
    it defaults to 10 so a mistyped filter cannot start a million-file crawl.
    """

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}")
        print("run:  python3 sec_edgar_scraper.py index --year 2024 --quarter 1")
        return 1

    out_dir = Path(args.output_dir or (OUT_ROOT / "filings"))

    row_list = []
    with manifest_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if args.form and row["form"] != args.form:
                continue
            if args.cik and row["cik"] not in {int(c) for c in args.cik}:
                continue
            row_list.append(row)

    print(f"{len(row_list):,} filings match; taking the first {args.max_items}")
    row_list = row_list[: args.max_items]

    n_ok, n_skip, n_fail = 0, 0, 0
    for pos, row in enumerate(row_list, 1):
        target_path = out_dir / str(row["cik"]) / f"{row['accession']}.txt"
        if already_done(target_path) and not args.force:
            n_skip += 1
            continue

        url = f"{WWW}/Archives/{row['path']}"
        try:
            body = fetcher.get(url, max_bytes=args.max_bytes)
        except (RuntimeError, urllib.error.HTTPError) as err:
            # one bad filing never kills the run
            n_fail += 1
            print(f"  [{pos}/{len(row_list)}] FAIL {row['accession']}: {err}")
            continue

        write_atomic(target_path, body)
        n_ok += 1
        if pos % 10 == 0 or pos == len(row_list):
            print(f"  [{pos}/{len(row_list)}] {row['company'][:40]} {row['form']}")

    print(f"\ndownloaded {n_ok}, skipped {n_skip}, failed {n_fail} -> {out_dir}")
    return 0


#############################################
# MODE: BULK ARCHIVES                       #
#############################################

BULK_DICT = {
    "submissions": (f"{WWW}/Archives/edgar/daily-index/bulkdata/submissions.zip", 1_559_612_838),
    "companyfacts": (f"{WWW}/Archives/edgar/daily-index/xbrl/companyfacts.zip", 1_407_131_132),
}


def mode_bulk(args, fetcher: Fetcher) -> int:
    """Stream one of the two multi-gigabyte archives to disk after confirmation."""

    url, known_bytes = BULK_DICT[args.which]
    out_dir = Path(args.output_dir or (OUT_ROOT / "bulk"))
    target_path = out_dir / f"{args.which}.zip"

    live_bytes = fetcher.head_size(url) or known_bytes
    print(f"{args.which}.zip is {live_bytes / 1e9:.2f} GB")
    print(f"  from  {url}")
    print(f"  to    {target_path}")

    if not args.yes:
        print("\nRefusing to download without --yes. This is intentional.")
        return 1

    if already_done(target_path) and not args.force:
        print("already on disk, skipping")
        return 0

    # stream rather than buffering 1.5 GB in memory
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".zip.tmp")
    request = urllib.request.Request(url, headers={"User-Agent": fetcher.user_agent})

    with urllib.request.urlopen(request, timeout=300, context=SSL_CONTEXT) as response, tmp_path.open("wb") as handle:
        n_done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            n_done += len(chunk)
            if n_done % (100 << 20) < (1 << 20):
                print(f"  {n_done / 1e9:.2f} / {live_bytes / 1e9:.2f} GB")

    tmp_path.replace(target_path)
    print(f"done: {target_path} ({target_path.stat().st_size:,} bytes)")
    return 0


#############################################
# SELFTEST                                  #
#############################################


def mode_selftest(args, fetcher: Fetcher) -> int:
    """Assert the pure helpers offline; no network, no files touched."""

    assert pad_cik(320193) == "0000320193"
    assert pad_cik("0000320193") == "0000320193"
    assert pad_cik("CIK0000320193") == "0000320193"

    assert accession_nodash("0000320193-24-000123") == "000032019324000123"

    assert url_submissions(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert url_frames("Assets", "CY2023Q1I") == (
        "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY2023Q1I.json"
    )
    assert url_filing_doc(320193, "0000320193-24-000123", "aapl-20240928.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000123/aapl-20240928.htm"
    )

    sample_idx = (
        b"Description: Master Index\n"
        b"CIK|Company Name|Form Type|Date Filed|Filename\n"
        b"------------------------------------------\n"
        b"1000045|NICHOLAS FINANCIAL INC|10-Q|2024-02-13|edgar/data/1000045/0000950170-24-014566.txt\n"
        b"garbage line that should be ignored\n"
    )
    row_list = parse_master_idx(sample_idx)
    assert len(row_list) == 1, row_list
    assert row_list[0]["cik"] == 1000045
    assert row_list[0]["form"] == "10-Q"
    assert row_list[0]["accession"] == "0000950170-24-014566"

    fake_submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["a-1", "a-2"],
                "form": ["8-K", "10-K"],
                "size": [5000, 9_000_000],
                "primaryDocument": ["small.htm", "huge.htm"],
            }
        }
    }
    filing_list = parse_recent_filings(fake_submissions)
    assert len(filing_list) == 2
    picked = pick_small_filing(filing_list, max_bytes=500_000, rng=random.Random(0))
    assert picked["primaryDocument"] == "small.htm", picked
    assert pick_small_filing(filing_list, max_bytes=100, rng=random.Random(0)) is None

    print("selftest: all assertions passed")
    return 0


#############################################
# CLI                                       #
#############################################


def build_parser() -> argparse.ArgumentParser:
    # flags every mode accepts, attached to each subparser as a parent
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--user-agent", default=USER_AGENT, help="name and email for the SEC")
    common.add_argument("--rate", type=float, default=RATE_LIMIT_PER_SEC, help="requests per second")
    common.add_argument("--output-dir", default=None, help="override the output directory")
    common.add_argument("--force", action="store_true", help="refetch files already on disk")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )

    sub = parser.add_subparsers(dest="mode")

    p_demo = sub.add_parser("demo", parents=[common], help="download ONE small filing (default)")
    p_demo.add_argument("--seed", type=int, default=None, help="make the pick reproducible")
    p_demo.add_argument("--max-bytes", type=int, default=500_000, help="size cap for the document")
    p_demo.add_argument("--max-tries", type=int, default=8, help="companies to try before giving up")
    p_demo.add_argument("--dry-run", action="store_true", help="print URLs, fetch nothing")
    p_demo.set_defaults(func=mode_demo)

    p_tick = sub.add_parser("tickers", parents=[common], help="save the ticker/CIK reference tables")
    p_tick.set_defaults(func=mode_tickers)

    p_subs = sub.add_parser("submissions", parents=[common], help="per-company filing history JSON")
    p_subs.add_argument("--ticker", action="append", default=[])
    p_subs.add_argument("--cik", action="append", default=[])
    p_subs.add_argument("--shards", action="store_true", help="also fetch pre-2015 shard files")
    p_subs.set_defaults(func=mode_submissions)

    p_fr = sub.add_parser("frames", parents=[common], help="one XBRL tag across every filer for one period")
    p_fr.add_argument("--tag", required=True, help="for example Assets or Revenues")
    p_fr.add_argument("--period", required=True, help="CY2023Q1I, CY2023Q1, or CY2023")
    p_fr.add_argument("--taxonomy", default="us-gaap")
    p_fr.add_argument("--unit", default="USD")
    p_fr.set_defaults(func=mode_frames)

    p_fts = sub.add_parser("fts", parents=[common], help="full-text search, 2001 to present")
    p_fts.add_argument("--query", required=True, help='quote phrases: \'"climate risk"\'')
    p_fts.add_argument("--forms", default=None, help="for example 10-K")
    p_fts.add_argument("--start", default=None, help="YYYY-MM-DD")
    p_fts.add_argument("--end", default=None, help="YYYY-MM-DD")
    p_fts.set_defaults(func=mode_fts)

    p_idx = sub.add_parser("index", parents=[common], help="quarterly master index to JSONL")
    p_idx.add_argument("--year", type=int, default=None, help="omit for every year since 1993")
    p_idx.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], default=None)
    p_idx.set_defaults(func=mode_index)

    p_fil = sub.add_parser("filings", parents=[common], help="download documents listed in a quarter manifest")
    p_fil.add_argument("--manifest", required=True, help="a JSONL file written by the index mode")
    p_fil.add_argument("--form", default=None, help="filter to one form type")
    p_fil.add_argument("--cik", action="append", default=[], help="filter to specific CIKs")
    p_fil.add_argument("--max-items", type=int, default=10, help="hard cap on documents fetched")
    p_fil.add_argument("--max-bytes", type=int, default=25_000_000, help="per-document size cap")
    p_fil.set_defaults(func=mode_filings)

    p_bulk = sub.add_parser("bulk", parents=[common], help="the two multi-gigabyte archives")
    p_bulk.add_argument("--which", choices=sorted(BULK_DICT), required=True)
    p_bulk.add_argument("--yes", action="store_true", help="required; confirms the size")
    p_bulk.set_defaults(func=mode_bulk)

    p_test = sub.add_parser("selftest", parents=[common], help="offline assertions on the pure helpers")
    p_test.set_defaults(func=mode_selftest)

    return parser


SUBCOMMAND_SET = {
    "demo", "tickers", "submissions", "frames", "fts", "index", "filings", "bulk", "selftest",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    token_list = list(sys.argv[1:] if argv is None else argv)

    # no subcommand anywhere means the safe single-file demo
    if not any(token in SUBCOMMAND_SET for token in token_list):
        token_list.insert(0, "demo")

    args = parser.parse_args(token_list)

    fetcher = Fetcher(user_agent=args.user_agent, rate_limit=args.rate)

    if args.user_agent == DEFAULT_UA and args.mode != "selftest":
        print(f"User-Agent: {args.user_agent}")
        print("(override with SEC_UA or --user-agent)\n")

    return args.func(args, fetcher)


if __name__ == "__main__":
    sys.exit(main())
