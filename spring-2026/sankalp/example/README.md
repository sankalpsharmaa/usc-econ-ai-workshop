# arXiv econ.GN Metadata Scrape

A self-contained example for the USC Economics AI Workshop ("Agentic Web Scraping with Claude Code"). Pulls every paper in the arXiv `econ.GN` (General Economics) category — roughly 3-5k records since the category opened in 2017 — into a single CSV in about a minute.

This example exists because arXiv explicitly sanctions metadata scraping through its public API. The only rules are a 3-second delay between requests, a single connection, and a User-Agent header identifying who you are. Compare against NBER or SSRN, where bulk download is prohibited by terms of service.

## Run it

```bash
cd sankalp/example
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/scrape_econ_gn.py
```

Outputs land in `output/`:

- `econ_gn.csv` — one row per paper, openable in Excel or VS Code
- `scrape.log` — per-page progress (timestamps, counts, retries)

## Output schema

| Column | Description |
|-|-|
| `id` | arXiv ID (e.g. `1706.04101v1`) |
| `submitted` | First submission timestamp (ISO 8601) |
| `updated` | Latest revision timestamp |
| `title` | Paper title, whitespace-collapsed |
| `authors` | Authors joined with `; ` |
| `primary` | Primary arXiv category (usually `econ.GN`) |
| `categories` | All cross-listed categories joined with `; ` |
| `abstract` | Abstract, whitespace-collapsed (LaTeX commands left raw) |
| `pdf` | Direct PDF link on arxiv.org |
| `doi` | Publisher DOI when supplied by the author |
| `journal_ref` | Journal citation when supplied by the author |

## Three concepts the script demonstrates

1. **Identification.** A `User-Agent` header with a contact email — what a polite client looks like.
2. **Pagination.** `start` and `max_results` parameters, looping until the response is empty.
3. **Rate limiting.** `time.sleep(3.1)` between requests. The 3-second floor is collective across all your machines, so never parallelize.

## Extending the demo

- Date filter inside `search_query`: `submittedDate:[202501010000 TO 202604300000]`
- Cross-category sweep: `cat:econ.GN OR cat:econ.EM OR cat:econ.TH`
- Author trail: `au:"Banerjee_A"`
- Topic clustering: run `sentence-transformers` over the `abstract` column

## Caveats

- Abstracts contain raw LaTeX (`$\alpha$`, `\textit{...}`). Cleaning is a separate exercise.
- arXiv occasionally returns transient 503s. The script retries 3 times with a 10-second backoff.
- A few records have no PDF link or no DOI; those columns are empty.
