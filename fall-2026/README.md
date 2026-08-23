# Agentic Web Scraping: SEC EDGAR

Worked example for the USC Economics AI Workshop. One script, one writeup.

```
sec_edgar.py               the code, recon and scraping in one file
sec_edgar_endpoints.md     what the recon found, with measured numbers
```

## Run it

```bash
cd fall-2026
export SEC_UA="Your Name your@email.edu"     # the SEC blocks requests without this

python3 sec_edgar.py selftest                # offline checks, no network
python3 sec_edgar.py --dry-run               # see the URLs, fetch nothing
python3 sec_edgar.py                         # download ONE small filing
```

The default is deliberately harmless: three requests, one document under 500 KB. Every bulk mode is opt-in.

## The point of the example

The instinct when you need data from a website is to write a scraper against the pages you can see. On EDGAR that instinct costs you days. The SEC publishes its own backend, and one of the files it publishes is a 1.6 GB zip holding every company's complete filing history.

Finding that out took fifteen requests. `probe` mode is those fifteen requests:

```bash
python3 sec_edgar.py probe
```

```
 status            size  what
------------------------------------------------------------------------------
    200         796,148  ticker universe  10,403 companies
    200               -  all filers (name to CIK)
    200         164,439  one company's filings  1,001 recent filings
    200       3,789,099  one company's XBRL facts  505 distinct tags
    200         839,867  one tag, every filer  6,289 filers in one call
    200          60,081  full-text search  1,453 hits (eq)
    200      33,207,145  quarterly index (raw)
    200       4,262,545  quarterly index (zip)
    200   1,559,612,838  BULK all submissions
    200   1,407,131,132  BULK all XBRL facts
    404               -  13F holdings (wrong guess)
```

Four things that table tells you, none of which you learn by reading the HTML:

1. Nothing needs a login. The only gate is a `User-Agent` naming a real person.
2. The two BULK rows replace roughly 20,000 individual API calls.
3. `quarterly index (zip)` is 4 MB against 33 MB for the same content. Always take the zip.
4. One row 404s. That URL was a reasonable guess and it was wrong. Guessing URL patterns from a sibling dataset is how people ship scrapers that quietly return nothing.

## What the script does

| Mode | What it teaches |
|-|-|
| `probe` | Find out what is really there before writing code. |
| `demo` | The safe default. Three requests, one small file. |
| `bulk` | Download the whole thing instead of crawling it. Needs `--yes`. |
| `frames` | One request returns ~6,300 firms. This is how you build a panel. |
| `submissions` | Per-company history, and the 1,000-filing truncation trap. |
| `index` | Build a manifest of every filing, then filter locally. |
| `filings` | Fetch only the documents the manifest says you want. |
| `fts` | Full-text search, and the 10,000-result ceiling nobody documents. |
| `selftest` | Assertions that run offline in under a second. |

## Five habits worth copying

**Recon before code.** Fifteen requests told us the crawler we were about to write was unnecessary.

**Make the safe path the lazy path.** No arguments means one small file. A scraper whose default is "start downloading 26 million filings" will eventually be run by accident.

**Write atomically.** Write to a `.tmp` file and rename. A crash mid-write otherwise leaves a short file that the resume check treats as finished, and you find out when the analysis is wrong.

**Never let one bad record kill a run.** At any real scale some requests fail for reasons unrelated to your code. Crashing on item 4,000 of 20,000 is a bad way to discover that.

**Keep URL building separate from fetching.** Every URL builder and parser in `sec_edgar.py` is a pure function, which is why `selftest` checks all of them offline in under a second. Those are exactly the bugs that otherwise appear as a confusing 404 forty minutes into a crawl.

## Reading the API's own hedges

The full-text endpoint reports totals like this:

```json
{"value": 1453,  "relation": "eq"}      exact count
{"value": 10000, "relation": "gte"}     a floor, not a count
```

`gte` means the query is bigger than the 10,000-result window and the number shown is a lower bound. Treating it as the true count silently loses data. `sec_edgar.py fts` refuses to run a `gte` query and tells you to slice by date instead.

## Scope

Roughly 26.8 million filings across 135 quarters, 1993Q1 to 2026Q3. The index alone is 2.24 GB uncompressed, 290 MB as zips.

Recommended order if you want everything: the two bulk zips first (3 GB, 2 requests), then the 135 quarterly manifests (290 MB), then crawl only the documents your filtered manifest names.

Full endpoint list, field names, and measured record counts: [`sec_edgar_endpoints.md`](sec_edgar_endpoints.md).

## Requirements

Python 3.10 or newer and `certifi`. Everything else is standard library. On macOS, Python from python.org cannot verify `sec.gov` without `certifi`, which is why the script builds its SSL context explicitly.

Downloads land in `data/` next to the script and are gitignored.
