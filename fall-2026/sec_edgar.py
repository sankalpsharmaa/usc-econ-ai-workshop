#!/usr/bin/env python3
"""
SEC EDGAR: from recon to scraper.

A worked example for the USC Economics AI Workshop. It has two halves that
mirror how you should approach any unfamiliar data source.

  RECON   `probe` calls candidate endpoints and reports what is really there.
          You run this before writing a single line of scraper.

  SCRAPE  every other mode pulls actual data, cheapest path first.

The lesson EDGAR teaches: most of the time the "scraper" you need is not a
scraper. The SEC publishes its own backend. Two zip files replace roughly
20,000 API calls. Finding that out took nine minutes of probing and saved
days of crawling. Do the recon.

SAFE BY DEFAULT
    Running this with no arguments downloads exactly ONE filing under 500 KB
    and stops. Bulk modes are opt-in; the two multi-gigabyte ones need --yes.

    python3 sec_edgar.py                     one small filing, 3 requests
    python3 sec_edgar.py --dry-run           print the URLs, fetch nothing
    python3 sec_edgar.py selftest            offline assertions, no network

RECON
    python3 sec_edgar.py probe               call 20 endpoints, print a table

PULLING DATA (cheapest to most expensive)
    python3 sec_edgar.py bulk --which submissions --yes    1.6 GB, everything
    python3 sec_edgar.py frames --tag Assets --period CY2023Q1I
    python3 sec_edgar.py submissions --ticker AAPL --ticker MSFT --shards
    python3 sec_edgar.py index --year 2024 --quarter 1
    python3 sec_edgar.py filings --manifest data/index/2024Q1.jsonl \
                                 --form 10-K --max-items 25
    python3 sec_edgar.py fts --query '"climate risk"' --forms 10-K \
                             --start 2023-01-01 --end 2023-03-31

THE ONE HARD REQUIREMENT
    The SEC blocks any request whose User-Agent does not name a real person
    and email. Set SEC_UA or pass --user-agent. Everything 403s without it.

Endpoint details and measured record counts: sec_edgar_endpoints.md
"""

from __future__ import annotations

import argparse
import gzip
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

# the SEC refuses generic agent strings with a 403 and an HTML scolding
DEFAULT_UA = "Sankalp Sharma (USC) sankalp.sharma437@gmail.com"
USER_AGENT = os.environ.get("SEC_UA", DEFAULT_UA)

# the published ceiling is 10 requests/second, so leave headroom
RATE_LIMIT_PER_SEC = 8.0

WWW = "https://www.sec.gov"
DATA = "https://data.sec.gov"
EFTS = "https://efts.sec.gov/LATEST/search-index"

# the full-text index rejects from + size above this; measured, not guessed
FTS_WINDOW_MAX = 10_000
FTS_PAGE_SIZE = 100

# EDGAR quarterly indexes begin here
FIRST_INDEX_YEAR = 1993

# outputs land next to this script, so moving the file moves its data
OUT_ROOT = Path(__file__).resolve().parent / "data"

# python.org Python ships no CA bundle that satisfies sec.gov; certifi does
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


#############################################
# HTTP LAYER                                #
#############################################


@dataclass
class Fetcher:
    """One rate-limited HTTP client, shared by every mode.

    Four things every scraper needs and most first drafts skip: a real
    throttle, retry with growing backoff, a byte cap so one huge response
    cannot eat memory, and a request counter so you can see what a run cost.
    """

    user_agent: str = USER_AGENT
    rate_limit: float = RATE_LIMIT_PER_SEC
    max_retries: int = 5
    timeout: int = 60
    verbose: bool = True
    n_requests: int = field(default=0, init=False)
    _last_call: float = field(default=0.0, init=False)

    def _throttle(self) -> None:
        """Sleep only as long as needed to hold the requests-per-second rate.

        Sleeping a fixed amount after every call wastes time when the server
        was already slow. Measuring the gap since the last call does not.
        """
        min_gap = 1.0 / self.rate_limit
        elapsed = time.monotonic() - self._last_call
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        self._last_call = time.monotonic()

    def get(self, url: str, max_bytes: int | None = None) -> bytes:
        """Fetch one URL, retrying only on failures that retrying can fix.

        A 404 means the file is not there and never will be, so retrying it
        five times just wastes five requests. A 500 or a dropped connection
        is worth another try. Telling those apart is most of the work.
        """
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
                    raw = response.read() if max_bytes is None else response.read(max_bytes)
                    if "gzip" in (response.headers.get("Content-Encoding") or "").lower():
                        raw = gzip.decompress(raw)
                    return raw

            except urllib.error.HTTPError as err:
                # a 403 here is almost always the User-Agent, so say so plainly
                if err.code == 403:
                    raise RuntimeError(
                        f"403 from {url}\n"
                        "The SEC blocks undeclared automated tools. Set a real "
                        "contact string:\n"
                        "  export SEC_UA='Your Name your@email.edu'"
                    ) from err

                # anything else in the 4xx range will not improve on a retry
                if err.code < 500 and err.code != 429:
                    raise
                last_error = err

            except (urllib.error.URLError, TimeoutError, ssl.SSLError) as err:
                last_error = err

            # back off further each time, with jitter so parallel runs desync
            backoff = 1.5 * (2**attempt) + random.uniform(0, 0.5)
            if self.verbose:
                print(f"    retry {attempt + 1}/{self.max_retries} in {backoff:.1f}s")
            time.sleep(backoff)

        raise RuntimeError(f"gave up on {url} after {self.max_retries} tries: {last_error!r}")

    def get_json(self, url: str) -> dict:
        """Fetch a URL and parse the body as JSON."""
        return json.loads(self.get(url).decode("utf-8"))

    def head(self, url: str) -> tuple[int | str, int | None, str]:
        """Ask for size and content type without downloading the body.

        This is how you find out a file is 1.6 GB before you start saving it.
        """
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
                return response.status, int(length) if length else None, (
                    response.headers.get("Content-Type") or ""
                )
        except urllib.error.HTTPError as err:
            return err.code, None, ""
        except Exception as err:
            return f"ERR {type(err).__name__}", None, ""


#############################################
# OUTPUT HELPERS                            #
#############################################


def write_atomic(target_path: Path, payload: bytes) -> Path:
    """Write to a .tmp sibling, then rename.

    Rename is atomic on the same filesystem. Without this, a crash halfway
    through a write leaves a short file that the resume check will happily
    treat as finished, and you will not notice until the analysis is wrong.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(target_path)
    return target_path


def already_done(target_path: Path) -> bool:
    """Treat any non-empty existing file as finished, which makes runs resumable.

    Checking the filesystem beats keeping a separate progress database: there
    is no second source of truth to fall out of sync.
    """
    return target_path.exists() and target_path.stat().st_size > 0


def safe_filename(name: str) -> str:
    """Flatten a name that might contain path separators into one filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


#############################################
# URL BUILDERS                              #
#############################################
# These are pure functions with no I/O, which is why selftest can check every
# one of them offline. Keeping URL construction separate from fetching is the
# single cheapest way to make a scraper testable.


def pad_cik(cik: int | str) -> str:
    """Zero-pad a CIK to the 10 digits data.sec.gov insists on.

    Apple is CIK 320193 in the archive paths and CIK0000320193 in the JSON
    API. Mixing the two up is the most common EDGAR mistake.
    """
    digits = str(cik).upper().removeprefix("CIK").lstrip("0")
    return (digits or "0").zfill(10)


def accession_nodash(accession: str) -> str:
    """Strip dashes from an accession number for archive path building.

    The same filing is 0000320193-24-000123 in metadata and
    000032019324000123 in its URL.
    """
    return accession.replace("-", "")


def url_submissions(cik: int | str) -> str:
    return f"{DATA}/submissions/CIK{pad_cik(cik)}.json"


def url_companyfacts(cik: int | str) -> str:
    return f"{DATA}/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json"


def url_frames(tag: str, period: str, taxonomy: str = "us-gaap", unit: str = "USD") -> str:
    return f"{DATA}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"


def url_filing_doc(cik: int | str, accession: str, document: str) -> str:
    return f"{WWW}/Archives/edgar/data/{int(cik)}/{accession_nodash(accession)}/{document}"


def url_quarter_index(year: int, quarter: int, name: str = "master.zip") -> str:
    return f"{WWW}/Archives/edgar/full-index/{year}/QTR{quarter}/{name}"


#############################################
# PARSERS                                   #
#############################################


def parse_recent_filings(submissions_dict: dict) -> list[dict]:
    """Flip filings.recent from parallel arrays into one dict per filing.

    The API stores 16 same-length arrays rather than a list of records, which
    is compact over the wire and unusable in a loop. Transpose it once at the
    boundary and every downstream line gets simpler.
    """
    recent_dict = submissions_dict["filings"]["recent"]
    field_list = list(recent_dict.keys())
    n_rows = len(recent_dict["accessionNumber"])

    return [{name: recent_dict[name][row] for name in field_list} for row in range(n_rows)]


def parse_master_idx(raw: bytes) -> list[dict]:
    """Turn a quarterly master index into one dict per filing.

    The file opens with a plain-text preamble, so the parser finds the dashed
    rule and starts after it rather than skipping a hardcoded line count. The
    preamble length has changed before.
    """
    line_list = raw.decode("latin-1").splitlines()

    start = 0
    for pos, line in enumerate(line_list):
        if line.startswith("---"):
            start = pos + 1
            break

    row_list = []
    for line in line_list[start:]:
        part_list = line.split("|")

        # skip anything that is not five fields starting with a numeric CIK
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


#############################################
# MODE: PROBE (do this before scraping)     #
#############################################

# One row per endpoint worth knowing about. HEAD where the body is large and
# only the size matters; GET where the body answers the question.
PROBE_LIST = [
    ("GET", "ticker universe", f"{WWW}/files/company_tickers.json"),
    ("HEAD", "all filers (name to CIK)", f"{WWW}/Archives/edgar/cik-lookup-data.txt"),
    ("GET", "one company's filings", url_submissions(320193)),
    # GET not HEAD: data.sec.gov answers HEAD on this path with a 403,
    # which looks like a User-Agent problem and is not one
    ("GET", "one company's XBRL facts", url_companyfacts(320193)),
    ("GET", "one tag, every filer", url_frames("Assets", "CY2023Q1I")),
    ("GET", "full-text search", f"{EFTS}?q=%22climate+risk%22&forms=10-K"),
    ("HEAD", "quarterly index (raw)", url_quarter_index(2024, 1, "master.idx")),
    ("HEAD", "quarterly index (zip)", url_quarter_index(2024, 1, "master.zip")),
    ("GET", "index directory listing", f"{WWW}/Archives/edgar/full-index/index.json"),
    ("HEAD", "BULK all submissions", f"{WWW}/Archives/edgar/daily-index/bulkdata/submissions.zip"),
    ("HEAD", "BULK all XBRL facts", f"{WWW}/Archives/edgar/daily-index/xbrl/companyfacts.zip"),
    ("HEAD", "financial statements 2024q1", f"{WWW}/files/dera/data/financial-statement-data-sets/2024q1.zip"),
    ("HEAD", "insider trades 2024q1", f"{WWW}/files/structureddata/data/insider-transactions-data-sets/2024q1_form345.zip"),
    ("HEAD", "13F holdings (wrong guess)", f"{WWW}/files/structureddata/data/form-13f-data-sets/2024q1_form13f.zip"),
    ("GET", "live filing feed", f"{WWW}/cgi-bin/browse-edgar?action=getcurrent&output=atom"),
]


def mode_probe(args, fetcher: Fetcher) -> int:
    """Call every candidate endpoint and report what is actually there.

    This is the step people skip. Fifteen requests here told us the SEC
    publishes a 1.6 GB zip of every company's filing history, which made the
    per-company crawler we were about to write unnecessary.

    Read the output for three things: what returns 200 without auth, how big
    the bulk files are, and which guessed URLs 404. That last one matters.
    The 13F row is left in on purpose as a wrong guess that fails loudly.
    """
    print(f"Probing {len(PROBE_LIST)} endpoints at {fetcher.rate_limit:.0f} req/s.\n")
    print(f"{'status':>7}  {'size':>14}  what")
    print("-" * 78)

    result_list = []
    for method, label, url in PROBE_LIST:
        if method == "HEAD":
            status, n_bytes, _ = fetcher.head(url)
            note = ""
        else:
            try:
                body = fetcher.get(url, max_bytes=8_000_000)
                status, n_bytes = 200, len(body)
                note = _describe_json(url, body)
            except Exception as err:
                status, n_bytes, note = getattr(err, "code", "ERR"), None, str(err)[:40]

        size_text = f"{n_bytes:,}" if n_bytes else "-"
        print(f"{str(status):>7}  {size_text:>14}  {label}{('  ' + note) if note else ''}")
        result_list.append(
            {"label": label, "method": method, "url": url, "status": status, "bytes": n_bytes}
        )

    out_path = Path(args.output_dir or (OUT_ROOT / "probe")) / "probe_results.json"
    write_atomic(out_path, json.dumps(result_list, indent=1, default=str).encode())

    n_ok = sum(1 for r in result_list if r["status"] == 200)
    print(f"\n{n_ok}/{len(result_list)} returned 200. Raw results: {out_path}")
    print("\nRead the two BULK rows. Those two files replace about 20,000 API calls.")
    return 0


def _describe_json(url: str, body: bytes) -> str:
    """Say something useful about a JSON body without printing all of it."""
    try:
        payload = json.loads(body)
    except Exception:
        return ""

    if "company_tickers" in url:
        return f"{len(payload):,} companies"
    if "/frames/" in url:
        return f"{len(payload['data']):,} filers in one call"
    if "search-index" in url:
        total_dict = payload["hits"]["total"]
        return f"{total_dict['value']:,} hits ({total_dict['relation']})"
    if "/submissions/" in url:
        return f"{len(payload['filings']['recent']['accessionNumber']):,} recent filings"
    if "/companyfacts/" in url:
        n_tags = sum(len(v) for v in payload["facts"].values())
        return f"{n_tags:,} distinct tags for one company"
    if "directory" in payload:
        return f"{len(payload['directory']['item'])} entries"
    return ""


#############################################
# MODE: DEMO (the default)                  #
#############################################


def pick_small_filing(filing_list: list[dict], max_bytes: int, rng: random.Random) -> dict | None:
    """Choose one filing whose primary document is named and under the cap."""
    candidate_list = [
        f
        for f in filing_list
        if f.get("primaryDocument")
        and str(f.get("size") or 0).isdigit()
        and 0 < int(f["size"]) <= max_bytes
    ]
    return rng.choice(candidate_list) if candidate_list else None


def mode_demo(args, fetcher: Fetcher) -> int:
    """Download exactly one small filing, in three requests.

    Deliberately the default. A scraper whose no-argument behaviour is
    "start downloading 26 million filings" is a scraper someone will run by
    accident. Make the safe path the lazy path.

    The three requests trace the whole hierarchy: which companies exist, what
    one company filed, and the bytes of one document.
    """
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir or (OUT_ROOT / "demo"))

    print("Fetching ONE small filing. Three requests, nothing else.\n")

    tickers_url = f"{WWW}/files/company_tickers.json"
    print(f"  [1/3] which companies exist   {tickers_url}")

    if args.dry_run:
        print("        dry run, stopping here")
        print(f"  [2/3] would then call        {url_submissions(320193)}")
        print("  [3/3] then one document from  /Archives/edgar/data/<cik>/<accession>/")
        return 0

    tickers_dict = fetcher.get_json(tickers_url)
    company_list = list(tickers_dict.values())
    print(f"        {len(company_list):,} companies with a ticker")

    # try companies until one has a document under the size cap
    chosen_company, chosen_filing = None, None
    for _ in range(args.max_tries):
        company = rng.choice(company_list)
        print(f"  [2/3] what {company['ticker']} filed", " " * max(0, 12 - len(company["ticker"])), url_submissions(company["cik_str"]))

        try:
            submissions_dict = fetcher.get_json(url_submissions(company["cik_str"]))
        except RuntimeError as err:
            print(f"        skipping: {err}")
            continue

        filing_list = parse_recent_filings(submissions_dict)
        filing = pick_small_filing(filing_list, args.max_bytes, rng)
        if filing is None:
            print(f"        nothing under {args.max_bytes:,} bytes, trying another company")
            continue

        chosen_company, chosen_filing = company, filing
        print(f"        {len(filing_list):,} recent filings, one fits")
        break

    if chosen_filing is None:
        print(f"\nNo small filing in {args.max_tries} tries. Raise --max-bytes.")
        return 1

    cik = int(chosen_company["cik_str"])
    doc_url = url_filing_doc(cik, chosen_filing["accessionNumber"], chosen_filing["primaryDocument"])
    target_path = out_dir / (
        f"{chosen_company['ticker']}_{chosen_filing['form']}"
        f"_{chosen_filing['accessionNumber']}_{safe_filename(chosen_filing['primaryDocument'])}"
    )

    print(f"  [3/3] the document            {doc_url}")
    if already_done(target_path):
        print("        already on disk, skipping the download")
    else:
        write_atomic(target_path, fetcher.get(doc_url, max_bytes=args.max_bytes * 4))

    # save what was fetched so the run can be audited later
    write_atomic(
        target_path.with_suffix(target_path.suffix + ".meta.json"),
        json.dumps(
            {
                "ticker": chosen_company["ticker"],
                "company": chosen_company["title"],
                "cik": cik,
                "url": doc_url,
                "seed": args.seed,
                **{
                    name: chosen_filing.get(name)
                    for name in ("accessionNumber", "form", "filingDate", "size")
                },
            },
            indent=2,
        ).encode(),
    )

    print(f"\n  {chosen_company['title']} ({chosen_company['ticker']}, CIK {cik})")
    print(f"  {chosen_filing['form']} filed {chosen_filing['filingDate']}")
    print(f"  {target_path}  ({target_path.stat().st_size:,} bytes)")
    print(f"  {fetcher.n_requests} requests total")
    return 0


#############################################
# MODE: BULK (the cheapest path)            #
#############################################

BULK_DICT = {
    "submissions": (
        f"{WWW}/Archives/edgar/daily-index/bulkdata/submissions.zip",
        "every company's complete filing history",
    ),
    "companyfacts": (
        f"{WWW}/Archives/edgar/daily-index/xbrl/companyfacts.zip",
        "every company's parsed XBRL financial facts",
    ),
}


def mode_bulk(args, fetcher: Fetcher) -> int:
    """Stream one of the two prebuilt archives to disk.

    Try this before writing any per-company loop. Roughly 20,000 API calls
    at 8 per second is about 40 minutes of requests and 20,000 chances to
    hit a transient error. One 1.6 GB download is neither.

    Streaming in 1 MB chunks rather than reading the whole body keeps memory
    flat, which matters once a file passes what your laptop can hold.
    """
    url, description = BULK_DICT[args.which]
    out_dir = Path(args.output_dir or (OUT_ROOT / "bulk"))
    target_path = out_dir / f"{args.which}.zip"

    _, n_bytes, _ = fetcher.head(url)
    print(f"{args.which}.zip: {description}")
    print(f"  size  {(n_bytes or 0) / 1e9:.2f} GB")
    print(f"  from  {url}")
    print(f"  to    {target_path}")

    if not args.yes:
        print("\nRefusing without --yes. Confirm the size above first.")
        return 1

    if already_done(target_path) and not args.force:
        print("\nalready on disk, skipping")
        return 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".zip.tmp")
    request = urllib.request.Request(url, headers={"User-Agent": fetcher.user_agent})

    print()
    with urllib.request.urlopen(request, timeout=300, context=SSL_CONTEXT) as response, tmp_path.open("wb") as handle:
        n_done = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            n_done += len(chunk)
            if n_done % (100 << 20) < (1 << 20):
                print(f"  {n_done / 1e9:.2f} / {(n_bytes or 0) / 1e9:.2f} GB")

    tmp_path.replace(target_path)
    print(f"\ndone: {target_path} ({target_path.stat().st_size:,} bytes)")
    return 0


#############################################
# MODE: FRAMES (one call, a whole panel)    #
#############################################


def mode_frames(args, fetcher: Fetcher) -> int:
    """Pull one XBRL tag for every filer in one period.

    The endpoint economists should know about. Asking each company for its
    assets is one request per company. Asking the frames endpoint for the
    Assets tag in one quarter returns roughly 6,300 companies in a single
    call. A firm-quarter panel is a few hundred requests, not a few hundred
    thousand.

    Period grammar: CY2023Q1I for stocks (balance sheet, measured at an
    instant), CY2023Q1 for flows (income statement, measured over a period),
    CY2023 for the annual figure.
    """
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
    print(f"{len(frame_dict['data']):,} companies in one request -> {target_path}")
    print(f"first row: {json.dumps(frame_dict['data'][0])}")
    return 0


#############################################
# MODE: SUBMISSIONS (one company at a time) #
#############################################


def resolve_ciks(ticker_list: list[str], cik_list: list[str], fetcher: Fetcher) -> list[tuple[int, str]]:
    """Turn a mix of tickers and CIKs into a deduplicated (cik, label) list."""
    resolved_list: list[tuple[int, str]] = []

    if ticker_list:
        tickers_dict = fetcher.get_json(f"{WWW}/files/company_tickers.json")
        by_ticker = {row["ticker"].upper(): row for row in tickers_dict.values()}
        for ticker in ticker_list:
            row = by_ticker.get(ticker.upper())
            if row is None:
                print(f"warning: {ticker} is not in company_tickers.json")
                continue
            resolved_list.append((int(row["cik_str"]), row["ticker"]))

    resolved_list.extend((int(cik), f"CIK{pad_cik(cik)}") for cik in cik_list)

    seen_set: set[int] = set()
    unique_list = []
    for cik, label in resolved_list:
        if cik not in seen_set:
            seen_set.add(cik)
            unique_list.append((cik, label))
    return unique_list


def mode_submissions(args, fetcher: Fetcher) -> int:
    """Save the filing history for named companies.

    Worth knowing: filings.recent holds only the most recent 1,000 filings.
    Everything older sits in shard files listed under filings.files. A
    scraper that reads only the recent block silently truncates history at
    2015 for an active filer, and nothing in the response says so.
    """
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
        shard_list = submissions_dict["filings"].get("files", [])
        n_recent = len(submissions_dict["filings"]["recent"]["accessionNumber"])
        print(f"{label}: {n_recent:,} recent filings, {len(shard_list)} older shard(s)")

        if not args.shards:
            if shard_list:
                print("       pass --shards to also pull the older history")
            continue

        for shard in shard_list:
            shard_path = out_dir / shard["name"]
            if already_done(shard_path) and not args.force:
                continue
            write_atomic(shard_path, fetcher.get(f"{DATA}/submissions/{shard['name']}"))
            print(f"       {shard['name']}: {shard['filingCount']:,} filings "
                  f"({shard['filingFrom']} to {shard['filingTo']})")

    return 0


#############################################
# MODE: INDEX (the manifest)                #
#############################################


def mode_index(args, fetcher: Fetcher) -> int:
    """Download quarterly master indexes and flatten them to JSONL.

    This is the manifest step, and it is what separates a crawl from a
    targeted pull. Every filing since 1993 appears here with its CIK, form
    type, date, and path. Fetch the manifest, filter it locally, then fetch
    only the documents you decided you wanted.

    Always take master.zip over master.idx. Same content, roughly one eighth
    the bytes: 4 MB against 32 MB for a recent quarter.
    """
    out_dir = Path(args.output_dir or (OUT_ROOT / "index"))
    this_year = time.gmtime().tm_year

    year_list = [args.year] if args.year else list(range(FIRST_INDEX_YEAR, this_year + 1))
    quarter_list = [args.quarter] if args.quarter else [1, 2, 3, 4]

    n_written, n_filings = 0, 0
    for year in year_list:
        for quarter in quarter_list:
            target_path = out_dir / f"{year}Q{quarter}.jsonl"
            if already_done(target_path) and not args.force:
                print(f"skip (exists)  {year}Q{quarter}")
                continue

            try:
                body = fetcher.get(url_quarter_index(year, quarter, "master.zip"))
            except urllib.error.HTTPError as err:
                # quarters before EDGAR and quarters not yet filed simply do not exist
                print(f"{year}Q{quarter}: not available ({err.code})")
                continue

            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                row_list = parse_master_idx(archive.read(archive.namelist()[0]))

            write_atomic(target_path, "\n".join(json.dumps(r) for r in row_list).encode())
            n_written += 1
            n_filings += len(row_list)
            print(f"{year}Q{quarter}: {len(row_list):,} filings -> {target_path.name}")

    print(f"\n{n_written} quarters, {n_filings:,} filings indexed in {out_dir}")
    return 0


#############################################
# MODE: FILINGS (fetch from the manifest)   #
#############################################


def mode_filings(args, fetcher: Fetcher) -> int:
    """Download documents for filings selected from a quarter manifest.

    --max-items defaults to 10 on purpose. A typo in --form should cost you
    ten requests, not a million. Raise it once you have looked at what the
    filter actually matched.

    One failed filing never stops the run. At this scale some fraction of
    requests will fail for reasons that have nothing to do with your code,
    and a crash on filing 4,000 of 20,000 is a bad way to find that out.
    """
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}")
        print("build one first:  python3 sec_edgar.py index --year 2024 --quarter 1")
        return 1

    out_dir = Path(args.output_dir or (OUT_ROOT / "filings"))
    cik_set = {int(c) for c in args.cik}

    row_list = []
    with manifest_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if args.form and row["form"] != args.form:
                continue
            if cik_set and row["cik"] not in cik_set:
                continue
            row_list.append(row)

    print(f"{len(row_list):,} filings match the filter; taking {args.max_items}\n")
    row_list = row_list[: args.max_items]

    n_ok, n_skip, n_fail = 0, 0, 0
    for pos, row in enumerate(row_list, 1):
        target_path = out_dir / str(row["cik"]) / f"{row['accession']}.txt"

        if already_done(target_path) and not args.force:
            n_skip += 1
            continue

        try:
            body = fetcher.get(f"{WWW}/Archives/{row['path']}", max_bytes=args.max_bytes)
        except (RuntimeError, urllib.error.HTTPError) as err:
            n_fail += 1
            print(f"  [{pos}/{len(row_list)}] FAILED {row['accession']}: {err}")
            continue

        write_atomic(target_path, body)
        n_ok += 1
        print(f"  [{pos}/{len(row_list)}] {row['company'][:44]:<44} {row['form']}")

    print(f"\ndownloaded {n_ok}, already had {n_skip}, failed {n_fail} -> {out_dir}")
    return 0


#############################################
# MODE: FULL-TEXT SEARCH                    #
#############################################


def mode_fts(args, fetcher: Fetcher) -> int:
    """Page a full-text query into JSONL, stopping at the real ceiling.

    The index refuses from + size above 10,000 and returns an error, not a
    partial result. Rather than paging until something breaks, this checks
    the reported total first and refuses to start a query it cannot finish.

    Watch the "relation" field. "eq" means the total is exact. "gte" means
    the query is larger than the window and the number you see is a floor.
    Treating a gte total as the true count is how people quietly lose data.

    The fix is date slicing: run the same query one month at a time until
    every slice reports eq.
    """
    out_dir = Path(args.output_dir or (OUT_ROOT / "fts"))
    slug = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_")[:40]
    target_path = out_dir / f"{slug}_{args.start or 'all'}_{args.end or 'all'}.jsonl"

    if already_done(target_path) and not args.force:
        print(f"skip (exists) {target_path}")
        return 0

    def fetch_page(offset: int) -> dict:
        param_dict = {"q": args.query, "from": offset}
        if args.forms:
            param_dict["forms"] = args.forms
        if args.start:
            param_dict["startdt"] = args.start
        if args.end:
            param_dict["enddt"] = args.end
        return fetcher.get_json(f"{EFTS}?{urllib.parse.urlencode(param_dict)}")

    first_page = fetch_page(0)
    total_dict = first_page["hits"]["total"]
    print(f"total hits: {total_dict['value']:,} (relation: {total_dict['relation']})")

    if total_dict["relation"] != "eq" or total_dict["value"] > FTS_WINDOW_MAX:
        print(
            f"\nThis query is past the {FTS_WINDOW_MAX:,} result ceiling, so the\n"
            "number above is a floor, not a count. Narrow it with --start and\n"
            "--end (one month at a time works) and run each slice separately."
        )
        return 1

    hit_list = list(first_page["hits"]["hits"])
    offset = FTS_PAGE_SIZE
    while offset < total_dict["value"]:
        page_hits = fetch_page(offset)["hits"]["hits"]
        if not page_hits:
            break
        hit_list.extend(page_hits)
        offset += FTS_PAGE_SIZE
        print(f"  {len(hit_list):,}/{total_dict['value']:,}")

    payload = "\n".join(json.dumps(hit["_source"] | {"_id": hit["_id"]}) for hit in hit_list)
    write_atomic(target_path, payload.encode())
    print(f"\n{len(hit_list):,} hits -> {target_path}")
    return 0


#############################################
# SELFTEST                                  #
#############################################


def mode_selftest(args, fetcher: Fetcher) -> int:
    """Check every pure function offline. No network, no files written.

    The point of separating URL building and parsing from fetching: all of
    this runs in well under a second and catches the errors that would
    otherwise show up as a confusing 404 forty minutes into a crawl.
    """
    # CIK padding, the most common EDGAR mistake
    assert pad_cik(320193) == "0000320193"
    assert pad_cik("0000320193") == "0000320193"
    assert pad_cik("CIK0000320193") == "0000320193"
    assert pad_cik(0) == "0000000000"

    assert accession_nodash("0000320193-24-000123") == "000032019324000123"
    assert safe_filename("xsl144X01/primary_doc.xml") == "xsl144X01_primary_doc.xml"

    # URL shapes, checked against real working URLs
    assert url_submissions(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert url_frames("Assets", "CY2023Q1I") == (
        "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY2023Q1I.json"
    )
    assert url_filing_doc(320193, "0000320193-24-000123", "aapl-20240928.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )

    # the index parser must survive the preamble and ignore junk lines
    sample_idx = (
        b"Description: Master Index of EDGAR Dissemination Feed\n"
        b"CIK|Company Name|Form Type|Date Filed|Filename\n"
        b"------------------------------------------------\n"
        b"1000045|NICHOLAS FINANCIAL INC|10-Q|2024-02-13|edgar/data/1000045/0000950170-24-014566.txt\n"
        b"a line that is not a filing and must be dropped\n"
    )
    row_list = parse_master_idx(sample_idx)
    assert len(row_list) == 1, row_list
    assert row_list[0]["cik"] == 1000045
    assert row_list[0]["form"] == "10-Q"
    assert row_list[0]["accession"] == "0000950170-24-014566"

    # the parallel-array transpose, and the size filter that keeps the demo small
    fake_submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["a-1", "a-2"],
                "form": ["8-K", "10-K"],
                "size": [5_000, 9_000_000],
                "primaryDocument": ["small.htm", "huge.htm"],
            }
        }
    }
    filing_list = parse_recent_filings(fake_submissions)
    assert len(filing_list) == 2
    assert filing_list[0] == {
        "accessionNumber": "a-1", "form": "8-K", "size": 5_000, "primaryDocument": "small.htm",
    }
    assert pick_small_filing(filing_list, 500_000, random.Random(0))["primaryDocument"] == "small.htm"
    assert pick_small_filing(filing_list, 100, random.Random(0)) is None

    print("selftest: all assertions passed")
    return 0


#############################################
# CLI                                       #
#############################################

SUBCOMMAND_SET = {
    "probe", "demo", "bulk", "frames", "submissions", "index", "filings", "fts", "selftest",
}


def build_parser() -> argparse.ArgumentParser:
    # flags every mode accepts, attached to each subparser as a parent so that
    # `sec_edgar.py --force` works the same as `sec_edgar.py demo --force`
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--user-agent", default=USER_AGENT, help="your name and email, required by the SEC")
    common.add_argument("--rate", type=float, default=RATE_LIMIT_PER_SEC, help="requests per second")
    common.add_argument("--output-dir", default=None, help="override where output lands")
    common.add_argument("--force", action="store_true", help="refetch files already on disk")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = parser.add_subparsers(dest="mode")

    p_probe = sub.add_parser("probe", parents=[common], help="RECON: what is actually there")
    p_probe.set_defaults(func=mode_probe)

    p_demo = sub.add_parser("demo", parents=[common], help="download ONE small filing (default)")
    p_demo.add_argument("--seed", type=int, default=None, help="make the random pick reproducible")
    p_demo.add_argument("--max-bytes", type=int, default=500_000, help="size cap for the document")
    p_demo.add_argument("--max-tries", type=int, default=8, help="companies to try before giving up")
    p_demo.add_argument("--dry-run", action="store_true", help="print the URLs, fetch nothing")
    p_demo.set_defaults(func=mode_demo)

    p_bulk = sub.add_parser("bulk", parents=[common], help="the two prebuilt multi-GB archives")
    p_bulk.add_argument("--which", choices=sorted(BULK_DICT), required=True)
    p_bulk.add_argument("--yes", action="store_true", help="required; confirms you saw the size")
    p_bulk.set_defaults(func=mode_bulk)

    p_frames = sub.add_parser("frames", parents=[common], help="one XBRL tag, every filer, one period")
    p_frames.add_argument("--tag", required=True, help="Assets, Revenues, EarningsPerShareBasic")
    p_frames.add_argument("--period", required=True, help="CY2023Q1I (stock), CY2023Q1 (flow), CY2023")
    p_frames.add_argument("--taxonomy", default="us-gaap")
    p_frames.add_argument("--unit", default="USD")
    p_frames.set_defaults(func=mode_frames)

    p_subs = sub.add_parser("submissions", parents=[common], help="filing history for named companies")
    p_subs.add_argument("--ticker", action="append", default=[])
    p_subs.add_argument("--cik", action="append", default=[])
    p_subs.add_argument("--shards", action="store_true", help="also pull history older than the recent 1,000")
    p_subs.set_defaults(func=mode_submissions)

    p_index = sub.add_parser("index", parents=[common], help="quarterly filing manifest to JSONL")
    p_index.add_argument("--year", type=int, default=None, help="omit for every year since 1993")
    p_index.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], default=None)
    p_index.set_defaults(func=mode_index)

    p_filings = sub.add_parser("filings", parents=[common], help="fetch documents listed in a manifest")
    p_filings.add_argument("--manifest", required=True, help="a JSONL file written by the index mode")
    p_filings.add_argument("--form", default=None, help="filter to one form type, e.g. 10-K")
    p_filings.add_argument("--cik", action="append", default=[], help="filter to specific CIKs")
    p_filings.add_argument("--max-items", type=int, default=10, help="hard cap on documents fetched")
    p_filings.add_argument("--max-bytes", type=int, default=25_000_000, help="per-document size cap")
    p_filings.set_defaults(func=mode_filings)

    p_fts = sub.add_parser("fts", parents=[common], help="full-text search, 2001 to present")
    p_fts.add_argument("--query", required=True, help="quote phrases: '\"climate risk\"'")
    p_fts.add_argument("--forms", default=None, help="e.g. 10-K")
    p_fts.add_argument("--start", default=None, help="YYYY-MM-DD")
    p_fts.add_argument("--end", default=None, help="YYYY-MM-DD")
    p_fts.set_defaults(func=mode_fts)

    p_test = sub.add_parser("selftest", parents=[common], help="offline assertions, no network")
    p_test.set_defaults(func=mode_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    token_list = list(sys.argv[1:] if argv is None else argv)

    # no subcommand anywhere means the safe single-file demo
    if not any(token in SUBCOMMAND_SET for token in token_list):
        token_list.insert(0, "demo")

    args = parser.parse_args(token_list)
    fetcher = Fetcher(user_agent=args.user_agent, rate_limit=args.rate)

    if args.mode != "selftest":
        print(f"User-Agent: {args.user_agent}  (override with SEC_UA)\n")

    return args.func(args, fetcher)


if __name__ == "__main__":
    sys.exit(main())
