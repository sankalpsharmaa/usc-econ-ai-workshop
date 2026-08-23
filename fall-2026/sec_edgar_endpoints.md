# SEC EDGAR: API Endpoint Discovery

Recon date: 2026-08-22. All endpoints below were called live from this machine. Status codes, byte counts, and record counts are measured, not quoted from documentation.

## Architecture

There is nothing to reverse-engineer. EDGAR is not a portal wrapped around a hidden backend; the SEC publishes the backend itself as three public services with no keys, no cookies, no captcha, and no session state.

| Host | What it is | Serves |
|-|-|-|
| `www.sec.gov/Archives/` | Static file archive (S3-backed) | Every filing document ever submitted, plus quarterly and daily index files |
| `data.sec.gov` | AWS API Gateway JSON API | Per-company filing histories and parsed XBRL financial facts |
| `efts.sec.gov/LATEST/search-index` | Elasticsearch, exposed directly | Full-text search across 2001-present filings |

The `www.sec.gov/edgar/` pages you see in a browser are thin front-ends over these three. `cgi-bin/browse-edgar` still exists and still works, but every dataset it returns is available in bulk elsewhere.

### The one hard requirement

Every request needs a `User-Agent` header that identifies you by name and email. A generic UA is refused:

```
GET https://data.sec.gov/submissions/CIK0000320193.json
User-Agent: python-urllib/3
→ 403  "Your Request Originates from an Undeclared Automated Tool"
```

With `User-Agent: Sankalp Sharma (USC) sankalp.sharma437@gmail.com` the same request returns 200. The SEC's stated cap is 10 requests per second. No rate-limit headers are returned, so throttle client-side.

Python note: `urllib` on this machine fails TLS verification against sec.gov. Pass `ssl.create_default_context(cafile=certifi.where())`.

## High-value endpoints

### 1. Company universe

**`GET https://www.sec.gov/files/company_tickers.json`**
- Auth: none. Verified 200.
- Records: 10,403 companies with a ticker.
- Fields: `cik_str`, `ticker`, `title`.

**`GET https://www.sec.gov/files/company_tickers_exchange.json`**
- Same 10,403 rows, plus `exchange`. Fields: `cik`, `name`, `ticker`, `exchange`.

**`GET https://www.sec.gov/Archives/edgar/cik-lookup-data.txt`**
- 40,002,136 bytes, 1,056,221 lines. Format `COMPANY NAME:CIK:`.
- This is the full EDGAR filer universe, not just listed companies. Use it when you need private issuers, funds, and individual insider filers.

### 2. Per-company filing history

**`GET https://data.sec.gov/submissions/CIK##########.json`** (CIK zero-padded to 10 digits)
- Auth: none. Verified on Apple (CIK 0000320193).
- Company metadata: `sic`, `sicDescription`, `ein`, `lei`, `stateOfIncorporation`, `addresses`, `tickers`, `exchanges`, `formerNames`, `fiscalYearEnd`, `category`, `entityType`.
- `filings.recent` holds the most recent 1,000 filings (Apple returned 1,001 rows) with 16 parallel arrays: `accessionNumber`, `filingDate`, `reportDate`, `acceptanceDateTime`, `act`, `form`, `fileNumber`, `filmNumber`, `items`, `core_type`, `size`, `isXBRL`, `isInlineXBRL`, `isXBRLNumeric`, `primaryDocument`, `primaryDocDescription`.
- Pagination: older filings sit in shard files named in `filings.files`. Apple's shard is `CIK0000320193-submissions-001.json` covering 1,240 filings from 1994-01-26 to 2015-06-02, fetched from the same `/submissions/` path.

### 3. Parsed XBRL financials

**`GET https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`**
- Every reported XBRL fact for one company, every period, every filing. Apple's file is 3,789,099 bytes and holds 503 `us-gaap` tags plus 2 `dei` tags.

**`GET https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/{tag}.json`**
- One tag for one company. Apple + `Revenues` returned 2,252 bytes. Use when you want one line item and not the 3.8 MB blob.

**`GET https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json`**
- The cross-sectional cut: one tag, one period, every filer at once. `us-gaap/Assets/USD/CY2023Q1I` returned 6,289 companies in one call.
- Period format: `CY2023Q1I` for instantaneous (balance sheet), `CY2023Q1` for duration (income statement), `CY2023` for annual.
- This is the endpoint to build a panel from. 6,289 firms per call means a full Compustat-style panel is a few hundred calls, not a few hundred thousand.

### 4. Full-text search (2001-present)

**`GET https://efts.sec.gov/LATEST/search-index?q={query}&forms={forms}&startdt=&enddt=&from={offset}`**
- Raw Elasticsearch response, `hits.hits` array, 100 per page.
- Query params confirmed working: `q` (quoted phrases supported), `forms`, `startdt`, `enddt`, `from`, `dateRange`, `ciks`.
- Aggregations returned free: `entity_filter`, `sic_filter`, `biz_states_filter`, `form_filter`.
- Hit fields: `adsh` (accession), `ciks`, `display_names`, `form`, `root_forms`, `file_date`, `period_ending`, `file_type`, `file_description`, `sics`, `biz_states`, `biz_locations`, `inc_states`, `film_num`, `items`, `sequence`.

**Hard pagination ceiling, verified:**
```
?q="climate risk"&from=9990  → search_phase_execution_exception, window too large
?q="climate risk"&from=10000 → same
```
`from + size` must be ≤ 10,000. `"climate risk"` alone reports `total: {value: 10000, relation: "gte"}`, meaning truncated. Narrowed to `forms=10-K&startdt=2023-01-01&enddt=2023-12-31` it reports `total: {value: 278, relation: "eq"}`, an exact count.

**Consequence for bulk work:** you cannot page past 10,000 hits. Slice any broad query by date (month or week) until each slice returns under 10,000, then page through it.

### 5. The complete archive (this is where "all the data" lives)

**`https://www.sec.gov/Archives/edgar/full-index/{YYYY}/QTR{1-4}/master.idx`**
- Pipe-delimited, 11 header lines then data: `CIK|Company Name|Form Type|Date Filed|Filename`.
- 2024Q1: 33,207,145 bytes, 370,322 lines, first data row `1000045|NICHOLAS FINANCIAL INC|10-Q|2024-02-13|edgar/data/1000045/0000950170-24-014566.txt`.
- Coverage: 135 quarters, 1993Q1 through 2026Q3, verified by walking `full-index/index.json` and each quarter's `index.json`.
- Total `master.idx` size across all 135 quarters: 2,346,325 KB (2.24 GB). At the 2024Q1 rate of ~89.7 bytes per row that is roughly **26.8 million filings**. Treat that as an estimate; pre-2000 rows are shorter, so the true count is somewhat higher.
- Same content in four flavours per quarter: `master.idx` (by CIK), `company.idx` (by name, 54 MB), `form.idx` (by form type, 54 MB), `crawler.idx` (with URLs, 66 MB), `xbrl.idx` (XBRL filings only, 2.3 MB). Each has `.zip`, `.gz`, and `.Z` compressed twins at roughly one eighth the size (`master.zip` is 4,114 KB vs 32,097 KB uncompressed). **Always pull the .zip.**

**`https://www.sec.gov/Archives/edgar/daily-index/{YYYY}/QTR{n}/`**
- Same four index types per business day. 2024Q1 held 310 files. Use this for incremental updates once the historical pull is done.

**Directory listing as JSON, at every level:**
- `https://www.sec.gov/Archives/edgar/full-index/index.json` (years)
- `https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/index.json` (files with sizes)
- `https://www.sec.gov/Archives/edgar/data/{cik}/index.json` (all accessions for a filer)
- `https://www.sec.gov/Archives/edgar/data/{cik}/{accession-no-dashes}/index.json` (every document in one filing, with per-file byte sizes, including a prebuilt `{accession}-xbrl.zip`)

That last one is the workhorse. Given an accession number from `master.idx`, one JSON call lists every exhibit in that filing, and each is a plain static file.

### 6. Prebuilt bulk archives (start here, not with the crawler)

| File | Size (verified) | Contents |
|-|-|-|
| `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip` | 1,559,612,838 bytes (1.56 GB) | Every company's `submissions` JSON, in one download |
| `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` | 1,407,131,132 bytes (1.41 GB) | Every company's full XBRL fact set |

3 GB of downloads replaces roughly 20,000 individual API calls. Refreshed nightly.

### 7. DERA / structured-data quarterly datasets

Flat CSV/TSV inside quarterly zips. Enumerated by scraping each landing page for `.zip` hrefs.

| Dataset | Link count | URL pattern | Latest seen |
|-|-|-|-|
| Financial Statement Data Sets | 69 | `/files/dera/data/financial-statement-data-sets/{YYYY}q{N}.zip` | 2026q1 |
| Financial Statement Notes | 79 | `/files/dera/data/financial-statement-notes-data-sets/{YYYY}q{N}_notes.zip` | 2009q1 onward |
| Insider Transactions (Forms 3/4/5) | 82 | `/files/structureddata/data/insider-transactions-data-sets/{YYYY}q{N}_form345.zip` | 2026q2 (now under `/files/datastandardsinnovation/`) |
| Form D (private placements) | 74 | `/files/structureddata/data/form-d-data-sets/{YYYY}q{N}_d.zip` | 2026q2 |
| Form 13F (institutional holdings) | 53 | `/files/structureddata/data/form-13f-data-sets/{01mmmYYYY-ddmmmYYYY}_form13f.zip` | 2026 |
| Form N-PORT (fund holdings) | 27 | `/files/dera/data/form-n-port-data-sets/{YYYY}q{N}_nport.zip` | 2019q4 onward |
| Crowdfunding Offerings | 41 | `/files/dera/data/crowdfunding-offerings-data-sets/{YYYY}q{N}_cf.zip` | 2016q2 onward |

Verified sizes: `2024q1.zip` financial statements = 124,336,804 bytes; `2024q1_form345.zip` insider = 13,874,620 bytes.

**Do not hardcode these paths.** Three different prefixes are in use (`/files/dera/data/`, `/files/structureddata/data/`, `/files/datastandardsinnovation/data/`) and the SEC has migrated files between them. Scrape the landing page for hrefs each run.

Other dataset families on the hub page at `https://www.sec.gov/data-research/sec-markets-data`, not probed in detail: BDC data sets, Form N-CEN, Form N-MFP (money market funds), Mutual Fund Prospectus Risk/Return (note the URL is `mutual-fund-prospectus-riskreturn-summary-data-sets`, no hyphen in "riskreturn"), Regulation A, Transfer Agent, Variable Insurance Product, and six market-structure series (MIDAS).

### 8. Legacy and feed endpoints (still live)

- `GET https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&output=atom&count=10` → 200, `application/atom+xml`, 17,485 bytes. Superseded by `/submissions/`.
- `GET https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom` → 200, live feed of filings as they arrive. Useful for a real-time watcher.
- `https://www.sec.gov/Archives/edgar/monthly/xbrlrss-{YYYY}-{MM}.xml` → 200. Monthly RSS of every XBRL filing.

## Dead ends

These 404'd during recon and should not be coded against:

- `/files/dera/data/financial-statement-notes-data-sets/2024_01_notes.zip` (notes use `{YYYY}q{N}_notes.zip`, not month-based)
- `/files/structureddata/data/form-13f-data-sets/2024q1_form13f.zip` (13F is date-range named, not quarter named)
- `/files/structureddata/data/form-n-port-data-sets/2024q1_nport.zip` (N-PORT lives under `/files/dera/data/`)
- `/files/EDGAR_LogFileData_thru_Jun2017.zip` (the EDGAR log file data sets page loads but exposes no download links; the series appears retired)
- `/data-research/sec-markets-data/mutual-fund-prospectus-risk-return-summary-data-sets` (correct slug drops the hyphen: `riskreturn`)
- `/data-research/sec-markets-data/money-market-fund-data` and `.../asset-backed-securities-data-sets`

## Key findings

1. **Everything is public and unauthenticated.** No key, no login, no captcha, no session. The only gate is a self-declared `User-Agent`, enforced with a 403.
2. **The archive is roughly 26.8 million filings across 135 quarters.** The index alone is 2.24 GB uncompressed, 290 MB as zips.
3. **Two zips get you 90% of the structured data.** `submissions.zip` (1.56 GB) and `companyfacts.zip` (1.41 GB) hold every company's filing history and every parsed financial fact.
4. **Full-text search caps at 10,000 results per query.** Confirmed by hitting the exact boundary. Any bulk text search must be sliced by date.
5. **The `frames` endpoint is the cheapest panel builder.** One call returned 6,289 firms for a single tag and quarter.
6. **DERA URL prefixes are unstable.** Files have moved between `/files/dera/`, `/files/structureddata/`, and `/files/datastandardsinnovation/`. Enumerate from the landing page, never from a hardcoded pattern.

## Recommended extraction path

1. Pull `submissions.zip` and `companyfacts.zip`. Two requests, 3 GB, gives the full company and financials layer.
2. Pull 135 `master.zip` files from `full-index/{YYYY}/QTR{n}/`. About 290 MB, gives the complete filing manifest with paths to every document.
3. Pull DERA quarterly zips for whichever forms matter (13F, Form D, insider, N-PORT).
4. Only then crawl `Archives/edgar/data/{cik}/{accession}/` for the raw document text of the specific filings you actually need. Filtered from the manifest first, this is thousands of requests, not millions.
5. Keep a daily job on `daily-index/` for incremental updates.

Throttle to 8 requests per second and set the `User-Agent` on every call.

## Verification artifacts

Every number above came from a live call. Reproduce the core of it with:

```bash
python3 sec_edgar.py probe
```

That calls 15 endpoints, prints status and size for each, and writes the raw results to `data/probe/probe_results.json`. The scraper built from this recon is `sec_edgar.py` in the same folder; it defaults to downloading one small filing.
