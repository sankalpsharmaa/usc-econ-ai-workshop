#!/usr/bin/env python3
"""
SEC EDGAR: from recon to scraper.

A worked example for the USC Economics AI Workshop. Read this file top to
bottom. The code is the lesson: first find out what the site will give you,
then pull data on the cheapest path that actually exists.

Two halves, on purpose:

  RECON   `probe` calls candidate URLs and prints what is really there.
          Run this before you write a scraper. Fifteen requests told us
          the crawler we were about to write was unnecessary.

  SCRAPE  every other mode pulls data, cheapest path first. `bulk` takes
          two zip files the SEC already built. `filings` is last: it
          fetches only the documents a filtered manifest named.

The lesson EDGAR teaches: most of the time the "scraper" you need is not a
scraper. The search page is a front end. Behind it the SEC publishes JSON,
indexes, and two multi-gigabyte zips. Those two zips replace roughly
20,000 API calls. Finding that out took nine minutes.

HOW TO READ THIS FILE
    Configuration, then the HTTP client every mode shares, then helpers,
    then URL builders and parsers (no network — that is why they are
    testable), then the modes from cheapest to dearest.

    Short path: CONFIGURATION, Fetcher.get, write_atomic, PROBE_LIST,
    mode_demo. That is the whole argument.

SAFE BY DEFAULT
    No arguments downloads exactly ONE filing under 500 KB and stops.
    Bulk modes are opt-in; the two multi-gigabyte ones need --yes.

    python3 sec_edgar.py                     one small filing, 3 requests
    python3 sec_edgar.py --dry-run           print the URLs, fetch nothing
    python3 sec_edgar.py selftest            offline assertions, no network

RECON
    python3 sec_edgar.py probe               call every candidate URL, print a table

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
# Put the rules of the site in one place, not sprinkled through the fetch
# code. If the SEC changes a host or a rate limit, this is the only block
# that should move.
#
# Three public hosts. The pages at sec.gov/edgar are a front end over these.
#   www.sec.gov/Archives  — the files themselves, plus bulk zips and indexes
#   data.sec.gov          — JSON: filing histories and parsed financials
#   efts.sec.gov          — full-text search (Elasticsearch, exposed as-is)

# A generic "python-urllib/3" User-Agent gets a 403 and an HTML scolding.
# Name a real person and email. Override with SEC_UA so you do not commit
# a classmate's contact string by accident.
DEFAULT_UA = "Sankalp Sharma (USC) sankalp.sharma437@gmail.com"
USER_AGENT = os.environ.get("SEC_UA", DEFAULT_UA)

# The SEC's published cap is 10 requests per second. Stay under it. The
# throttle is client-side: the API does not send a Retry-After header.
RATE_LIMIT_PER_SEC = 8.0

WWW = "https://www.sec.gov"
DATA = "https://data.sec.gov"
EFTS = "https://efts.sec.gov/LATEST/search-index"

# Full-text search will not page past this. Measured by hitting the boundary,
# not taken from docs. Treat it as a hard window, not a suggestion.
FTS_WINDOW_MAX = 10_000
FTS_PAGE_SIZE = 100

# Quarterly indexes start in 1993Q1. Earlier years 404; that is expected.
FIRST_INDEX_YEAR = 1993

# Outputs sit next to this file, so moving the script moves its data.
# The folder is gitignored. Nothing downloaded belongs in the repo.
OUT_ROOT = Path(__file__).resolve().parent / "data"

# Python from python.org ships no CA bundle that sec.gov accepts.
# certifi does. Without this, every request dies as an SSL error that
# looks like a network problem and is not one.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


#############################################
# HTTP LAYER                                #
#############################################
# Every mode goes through this client. That is the point. Rate limits,
# retries, the User-Agent, and the SSL context live here, not in each
# download function. If probe, demo, and bulk each rolled their own
# urllib call, one of them would forget the User-Agent and fail in a
# way that looks random.


@dataclass
class Fetcher:
    """One HTTP client, shared by every mode.

    Four things every scraper needs and most first drafts skip:

      1. A real throttle. Not "sleep 1 second every time." Sleep only
         the leftover gap so a slow server does not add extra delay.
      2. Retry with growing backoff, but only on errors that retrying
         can fix. A 404 will not become a 200 if you ask again.
      3. A byte cap, so one huge response cannot eat memory.
      4. A request counter, so you can see what a run actually cost.

    Read get() next. That is the whole client.
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
        """Fetch one URL. Retry only failures that retrying can fix.

        A 404 means the file is not there and never will be. Retrying it
        five times wastes five requests. A 500 or a dropped connection is
        worth another try. Telling those apart is most of the work.

        403 is special on this site. It almost always means the User-Agent
        is missing or generic. The error message says that, because the
        HTTP status alone will not.
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
                    # max_bytes is a safety cap, not a range request.
                    # We still pay for the whole response on the wire;
                    # we just refuse to hold more than this in memory.
                    raw = response.read() if max_bytes is None else response.read(max_bytes)
                    if "gzip" in (response.headers.get("Content-Encoding") or "").lower():
                        raw = gzip.decompress(raw)
                    return raw

            except urllib.error.HTTPError as err:
                if err.code == 403:
                    raise RuntimeError(
                        f"403 from {url}\n"
                        "The SEC blocks undeclared automated tools. Set a real "
                        "contact string:\n"
                        "  export SEC_UA='Your Name your@email.edu'"
                    ) from err

                # 4xx (except 429, "slow down") will not improve on a retry.
                if err.code < 500 and err.code != 429:
                    raise
                last_error = err

            except (urllib.error.URLError, TimeoutError, ssl.SSLError) as err:
                last_error = err

            # Grow the wait each time. A little jitter keeps two parallel
            # runs from retrying in lockstep and hitting the cap together.
            backoff = 1.5 * (2**attempt) + random.uniform(0, 0.5)
            if self.verbose:
                print(f"    retry {attempt + 1}/{self.max_retries} in {backoff:.1f}s")
            time.sleep(backoff)

        raise RuntimeError(f"gave up on {url} after {self.max_retries} tries: {last_error!r}")

    def get_json(self, url: str) -> dict:
        """Fetch a URL and parse the body as JSON.

        A one-liner so call sites stay readable. Failures still go through
        get(), so the throttle, retries, and 403 message apply here too.
        """
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
# Two habits that save a night when a run dies at 80%.
#
#   write_atomic   write a sibling .tmp, then rename. Rename is one step.
#                  A crash mid-write cannot leave a short file that looks done.
#
#   already_done   if the file exists and is non-empty, skip it. Re-running
#                  the same command continues where it left off. The files
#                  on disk are the progress log. There is no second database
#                  to fall out of sync.


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
    """Turn a document name that may contain slashes into one filename.

    EDGAR primary documents can look like xsl144X01/primary_doc.xml.
    Written as-is, that would create a subdirectory you did not mean to.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


#############################################
# URL BUILDERS                              #
#############################################
# Pure functions: string in, URL out, no network. That is why selftest can
# check every one of them on a plane. Keeping URL construction separate
# from fetching is the cheapest way to make a scraper testable.
#
# The bugs these catch are the ones that otherwise show up as a confusing
# 404 forty minutes into a crawl. Apple is CIK 320193 in archive paths and
# CIK0000320193 in the JSON API. Mixing the two up is the most common
# EDGAR mistake, which is why pad_cik exists.


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
    """Filing history for one company, as JSON.

    filings.recent is capped at about 1,000 rows. Older filings live in
    shard files named under filings.files. See mode_submissions.
    """
    return f"{DATA}/submissions/CIK{pad_cik(cik)}.json"


def url_companyfacts(cik: int | str) -> str:
    """Every XBRL fact one company has ever reported. Apple's file is ~4 MB."""
    return f"{DATA}/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json"


def url_companyconcept(cik: int | str, tag: str, taxonomy: str = "us-gaap") -> str:
    """One XBRL tag for one company, every period.

    The sibling of companyfacts. Use this when you want Revenues for Apple
    and not the 4 MB blob of every tag Apple has ever reported.
    """
    return f"{DATA}/api/xbrl/companyconcept/CIK{pad_cik(cik)}/{taxonomy}/{tag}.json"


def url_frames(tag: str, period: str, taxonomy: str = "us-gaap", unit: str = "USD") -> str:
    """One tag, one period, every filer.

    This is the panel builder. One call returned 6,289 firms for Assets
    in 2023Q1. Period codes: CY2023Q1I (balance sheet, a point in time),
    CY2023Q1 (income statement, a span of time), CY2023 (annual).
    """
    return f"{DATA}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"


def url_filing_doc(cik: int | str, accession: str, document: str) -> str:
    """The raw document in the archive.

    The archive path uses the short CIK (no leading zeros) and the
    accession with dashes stripped. The JSON API uses the opposite.
    """
    return f"{WWW}/Archives/edgar/data/{int(cik)}/{accession_nodash(accession)}/{document}"


def url_quarter_index(year: int, quarter: int, name: str = "master.zip") -> str:
    """The quarterly filing manifest. Prefer master.zip over master.idx.

    Same rows, about one eighth the bytes. 4 MB against 32 MB for a
    recent quarter. Always take the zip.
    """
    return f"{WWW}/Archives/edgar/full-index/{year}/QTR{quarter}/{name}"


#############################################
# PARSERS                                   #
#############################################
# Also pure: bytes or a dict in, a list of records out. No network.
# selftest feeds them a tiny fake file and checks the rows.
#
# Do the ugly shape-shifting once, at the boundary. Downstream code then
# loops over ordinary dicts and never has to know that the API stored
# sixteen parallel arrays, or that the index file starts with a preamble.


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
# You are here in the story: you do not have a scraper yet. You have a
# list of URLs that might be the real backend. Hit each one. Write down
# status, size, and (for JSON) a count. Then decide what to build.
#
# HEAD when you only need the size. GET when the body answers the
# question (how many companies, how many tags). One exception is noted
# on the companyfacts row: that host answers HEAD with a 403, which
# looks like a User-Agent bug and is not one.
#
# The 13F row is a wrong guess, left in on purpose. A URL copied from a
# sibling dataset looked right. It 404s. That is how scrapers ship and
# quietly return nothing.

PROBE_LIST = [
    ("GET", "ticker universe", f"{WWW}/files/company_tickers.json"),
    ("GET", "ticker universe + exchange", f"{WWW}/files/company_tickers_exchange.json"),
    # HEAD here is a known dud: gzip makes Content-Length 20 or missing.
    # The file is ~40 MB uncompressed. GET it when you need the size.
    ("HEAD", "all filers (name to CIK)", f"{WWW}/Archives/edgar/cik-lookup-data.txt"),
    ("GET", "one company's filings", url_submissions(320193)),
    # GET not HEAD: data.sec.gov answers HEAD on this path with a 403,
    # which looks like a User-Agent problem and is not one
    ("GET", "one company's XBRL facts", url_companyfacts(320193)),
    ("GET", "one tag, one company", url_companyconcept(320193, "Revenues")),
    ("GET", "one tag, every filer", url_frames("Assets", "CY2023Q1I")),
    # URL taken from js/edgar_full_text_search.js on /edgar/search/, not guessed
    ("GET", "full-text search", f"{EFTS}?q=%22climate+risk%22&forms=10-K"),
    ("HEAD", "quarterly index (raw)", url_quarter_index(2024, 1, "master.idx")),
    ("HEAD", "quarterly index (zip)", url_quarter_index(2024, 1, "master.zip")),
    ("GET", "index directory listing", f"{WWW}/Archives/edgar/full-index/index.json"),
    ("GET", "one filing's document list", f"{WWW}/Archives/edgar/data/320193/000032019324000123/index.json"),
    ("HEAD", "BULK all submissions", f"{WWW}/Archives/edgar/daily-index/bulkdata/submissions.zip"),
    ("HEAD", "BULK all XBRL facts", f"{WWW}/Archives/edgar/daily-index/xbrl/companyfacts.zip"),
    ("HEAD", "financial statements 2024q1", f"{WWW}/files/dera/data/financial-statement-data-sets/2024q1.zip"),
    ("HEAD", "insider trades 2024q1", f"{WWW}/files/structureddata/data/insider-transactions-data-sets/2024q1_form345.zip"),
    ("HEAD", "13F holdings (wrong guess)", f"{WWW}/files/structureddata/data/form-13f-data-sets/2024q1_form13f.zip"),
    ("GET", "live filing feed", f"{WWW}/cgi-bin/browse-edgar?action=getcurrent&output=atom"),
]


def mode_probe(args, fetcher: Fetcher) -> int:
    """Call every candidate URL and print what is actually there.

    This is the step people skip. A short probe told us the SEC
    publishes a 1.6 GB zip of every company's filing history. That made
    the per-company crawler we were about to write unnecessary.

    Read the table for three things:
      - what returns 200 with no login
      - how big the bulk files are
      - which guessed URLs 404

    That last one matters. The 13F row fails loudly on purpose.
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
    """One short note for the probe table: a count, not a dump of the body.

    The probe is supposed to be readable in a terminal. Printing 4 MB of
    Apple's companyfacts JSON would hide the number that matters (505 tags).
    Each branch below is "what question was this URL meant to answer."
    """
    try:
        payload = json.loads(body)
    except Exception:
        return ""

    if url.rstrip("/").endswith("company_tickers.json"):
        return f"{len(payload):,} companies"
    if "company_tickers_exchange" in url:
        rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        return f"{len(rows):,} companies"
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
    if "/companyconcept/" in url:
        n_obs = sum(len(v) for v in payload["units"].values())
        return f"{n_obs:,} observations of {payload.get('tag', 'tag')}"
    if "directory" in payload:
        return f"{len(payload['directory']['item'])} entries"
    return ""


#############################################
# MODE: DEMO (the default)                  #
#############################################
# You are here in the story: probe already showed the hierarchy.
# This mode walks it for one file, in three requests:
#
#   1. company_tickers.json     who exists
#   2. submissions/CIK....json  what one company filed
#   3. Archives/.../document    the bytes of one document
#
# It is the default on purpose. A scraper whose no-argument behaviour is
# "start downloading 26 million filings" will eventually be run by accident.
# Make the safe path the lazy path.


def pick_small_filing(filing_list: list[dict], max_bytes: int, rng: random.Random) -> dict | None:
    """Pick one filing whose primary document is named and under the size cap.

    The demo is a teaching default, not a random sample of EDGAR. We only
    want a file small enough to open. 8-K HTML often fits; a 10-K does not.
    """
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

    Request 1: who exists. Request 2: what one company filed.
    Request 3: the bytes of one document. Stop.
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

    # Many companies' recent filings are all large (10-Ks, 10-Qs). Skip
    # those and try another ticker rather than downloading a 9 MB file
    # "because it was first." The size cap is the point of the demo.
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

    # Sidecar metadata so you can see later what was fetched, from which
    # seed, without reverse-engineering the filename. The document alone
    # does not name the company.
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
# You are here in the story: probe found two zip files. submissions.zip
# is every company's filing history. companyfacts.zip is every parsed
# XBRL fact. Together they are about 3 GB. That is two requests.
#
# The alternative is one API call per company, about 20,000 of them, at
# 8 per second: 40 minutes and 20,000 chances to hit a blip. Take the zip.
#
# --yes is required. Printing the size and then refusing is the feature.
# A default of "start a 1.6 GB download" is how you fill a laptop by
# tab-completing the wrong command.

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

    # Stream 1 MB at a time. Reading the whole body first would hold 1.6 GB
    # in RAM, then write it. Chunking keeps memory flat. Same atomic habit
    # as write_atomic: land in .tmp, rename when the last byte is on disk.
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
# You are here in the story: you want a firm-quarter panel of one line
# item, the way Compustat is used. Asking each company for its assets is
# one request per company. Asking this endpoint for the Assets tag in one
# quarter returns roughly 6,300 companies in a single call.
#
# A full panel is a few hundred requests (one per tag-quarter), not a few
# hundred thousand. If you are building a dataset of "assets for every
# filer," start here, not with submissions.


def mode_frames(args, fetcher: Fetcher) -> int:
    """Pull one XBRL tag for every filer in one period.

    Period codes: CY2023Q1I for stocks (balance sheet, a point in time),
    CY2023Q1 for flows (income statement, a span of time), CY2023 for the
    annual figure. The I suffix is easy to drop and then you get zeros,
    because a stock has no duration value.
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
# You are here in the story: you want Apple's filing history, not every
# company's. This is the per-company JSON from data.sec.gov.
#
# The trap: filings.recent holds only the most recent 1,000 filings.
# Everything older sits in shard files listed under filings.files. A
# scraper that reads only the recent block silently truncates history
# at 2015 for an active filer, and nothing in the response says so.
# Pass --shards or you will not notice.


def resolve_ciks(ticker_list: list[str], cik_list: list[str], fetcher: Fetcher) -> list[tuple[int, str]]:
    """Turn a mix of tickers and CIKs into a deduplicated (cik, label) list.

    Tickers are what people type. CIKs are what the API wants. The lookup
    file is the same company_tickers.json the demo uses. A ticker that is
    not in it is usually a private filer; use --cik for those.
    """
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

    Without --shards this writes the recent 1,000 and prints how many
    older shards exist. That print is the warning. Use it.
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
# You are here in the story: you want some of the documents, not all of
# them. Do not crawl first. Build a list of every filing, filter it on
# disk, then fetch only what you kept.
#
# Each quarter is one zip: CIK, company, form, date, path. 1993Q1 through
# last quarter is about 135 files, ~290 MB as zips, ~27 million rows.
# Filter "10-K in 2024" locally. Then mode_filings does the downloads.
#
# Quarters that do not exist (before EDGAR, or not yet filed) 404. That
# is expected. The loop prints "not available" and continues.


def mode_index(args, fetcher: Fetcher) -> int:
    """Download quarterly master indexes and flatten them to JSONL.

    Always fetches master.zip, not master.idx. Same rows, about one eighth
    the bytes.
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
# You are here in the story: the manifest exists. You have already decided
# which rows you want. This mode downloads those documents and nothing else.
#
# --max-items defaults to 10. A typo in --form should cost you ten
# requests, not a million. Raise it once you have looked at what the
# filter actually matched.
#
# One failed filing never stops the run. At this scale some fraction of
# requests fail for reasons that have nothing to do with your code.
# Crashing on filing 4,000 of 20,000 is a bad way to find that out.


def mode_filings(args, fetcher: Fetcher) -> int:
    """Download documents for filings selected from a quarter manifest."""
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
            # Log it, count it, keep going. The summary at the end is how
            # you notice a bad URL pattern, not a traceback on item 4,000.
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
# You are here in the story: you want filings that mention a phrase, not
# every 10-K. efts.sec.gov is Elasticsearch, exposed directly.
#
# Two traps, both measured:
#
#   1. You cannot page past 10,000 hits. from + size above that returns
#      an error, not a partial page.
#   2. The total field has a "relation". "eq" means the number is exact.
#      "gte" means it is a floor. Treating 10000/gte as "there are 10,000
#      hits" silently drops the rest of the sample.
#
# This mode refuses to start a query it cannot finish. Slice by date
# (one month at a time) until each slice reports eq, then page through.


def mode_fts(args, fetcher: Fetcher) -> int:
    """Page a full-text query into JSONL. Refuse to start if it cannot finish."""
    out_dir = Path(args.output_dir or (OUT_ROOT / "fts"))
    slug = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_")[:40]
    target_path = out_dir / f"{slug}_{args.start or 'all'}_{args.end or 'all'}.jsonl"

    if already_done(target_path) and not args.force:
        print(f"skip (exists) {target_path}")
        return 0

    def fetch_page(offset: int) -> dict:
        # `from` is Elasticsearch's offset, not a date. 100 hits per page.
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
# You are here in the story: before any network call, check the parts that
# do not need one. URL builders and parsers are pure functions. If pad_cik
# is wrong, every submissions URL 404s. Better to find that in half a
# second on a fake Apple CIK than forty minutes into a crawl.
#
# Run this first: python3 sec_edgar.py selftest


def mode_selftest(args, fetcher: Fetcher) -> int:
    """Check every pure function offline. No network, no files written.

    Three families of bugs, all of which look like "the site is down" when
    they are actually a string:
      - CIK padding (archive vs JSON API)
      - URL shapes (wrong host, missing CIK zeros, leftover dashes)
      - parsers (index preamble, parallel-array transpose, size filter)
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
    assert url_companyconcept(320193, "Revenues") == (
        "https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/Revenues.json"
    )
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
# argparse only. No hidden flags. The module docstring is --help.
#
# Defaulting to demo is a code choice, not a docs choice: if the user
# types the filename and hits enter, they get one small file, not 26
# million. Inserting "demo" when no subcommand is present is how.


SUBCOMMAND_SET = {
    "probe", "demo", "bulk", "frames", "submissions", "index", "filings", "fts", "selftest",
}


def build_parser() -> argparse.ArgumentParser:
    # Shared flags are a parent parser so `sec_edgar.py --force` and
    # `sec_edgar.py demo --force` mean the same thing. Easy to get wrong
    # if each subparser declares --force itself.
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

    # No subcommand anywhere means the safe single-file demo.
    # `python3 sec_edgar.py` and `python3 sec_edgar.py --dry-run` both
    # land here. Bulk cannot.
    if not any(token in SUBCOMMAND_SET for token in token_list):
        token_list.insert(0, "demo")

    args = parser.parse_args(token_list)
    fetcher = Fetcher(user_agent=args.user_agent, rate_limit=args.rate)

    if args.mode != "selftest":
        # Print this first so a 403 is diagnosed before anything else runs.
        print(f"User-Agent: {args.user_agent}  (override with SEC_UA)\n")

    return args.func(args, fetcher)


if __name__ == "__main__":
    sys.exit(main())
