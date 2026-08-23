#!/usr/bin/env python3
"""Second EDGAR pass: record counts, pagination limits, and correct URLs for the 404s."""
import json, time, ssl, certifi, urllib.request, urllib.error, re

CTX = ssl.create_default_context(cafile=certifi.where())
UA = "Sankalp Sharma (USC) sankalp.sharma437@gmail.com"


def get(url, method="GET", cap=3_000_000):
    req = urllib.request.Request(url, method=method,
                                 headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            return r.status, r.headers, (r.read(cap) if method == "GET" else b"")
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read(500)
    except Exception as e:
        return "ERR", {}, repr(e).encode()


def j(url):
    s, h, b = get(url)
    if s != 200:
        return s, None
    try:
        return s, json.loads(b)
    except Exception:
        return s, None


rep = {}
time.sleep(0.2)

# 1. ticker universe
s, d = j("https://www.sec.gov/files/company_tickers.json")
rep["company_tickers"] = {"status": s, "n": len(d) if d else 0,
                          "row0": d["0"] if d else None}

s, d = j("https://www.sec.gov/files/company_tickers_exchange.json")
rep["company_tickers_exchange"] = {"status": s,
                                   "fields": d.get("fields") if d else None,
                                   "n": len(d.get("data", [])) if d else 0}

# 2. submissions doc: recent block + older shards
s, d = j("https://data.sec.gov/submissions/CIK0000320193.json")
if d:
    rec = d["filings"]["recent"]
    rep["submissions"] = {
        "status": s,
        "top_keys": list(d.keys()),
        "recent_fields": list(rec.keys()),
        "recent_n": len(rec["accessionNumber"]),
        "older_shards": d["filings"].get("files"),
        "tickers": d.get("tickers"), "sic": d.get("sic"),
        "sicDescription": d.get("sicDescription"),
    }

# 3. companyfacts size + taxonomy/concept counts
s, h, b = get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json", cap=40_000_000)
if s == 200:
    d = json.loads(b)
    rep["companyfacts"] = {
        "status": s, "bytes": len(b),
        "taxonomies": {k: len(v) for k, v in d["facts"].items()},
    }

# 4. frames: how many companies in one frame
s, d = j("https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY2023Q1I.json")
if d:
    rep["frames"] = {"status": s, "n_companies": len(d["data"]),
                     "keys": list(d.keys()), "row0": d["data"][0]}

# 5. full-text search: totals + pagination behaviour
for label, url in [
    ("fts_basic", "https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22"),
    ("fts_forms_dates", "https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22&forms=10-K&startdt=2023-01-01&enddt=2023-12-31"),
    ("fts_from_9990", "https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22&from=9990"),
    ("fts_from_10000", "https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22&from=10000"),
]:
    s, d = j(url)
    if d and "hits" in d:
        rep[label] = {"status": s,
                      "total": d["hits"]["total"],
                      "returned": len(d["hits"]["hits"]),
                      "agg_keys": list(d.get("aggregations", {}).keys())}
    else:
        s2, h2, b2 = get(url)
        rep[label] = {"status": s2, "err": b2[:200].decode("utf-8", "replace")}
    time.sleep(0.2)

# sample hit shape
s, d = j("https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22&forms=10-K")
if d:
    rep["fts_hit_shape"] = d["hits"]["hits"][0]

# 6. index listings
s, d = j("https://www.sec.gov/Archives/edgar/full-index/index.json")
if d:
    rep["full_index_years"] = [x["name"] for x in d["directory"]["item"]]

s, d = j("https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/index.json")
if d:
    rep["full_index_2024Q1_files"] = [(x["name"], x.get("size")) for x in d["directory"]["item"]]

s, d = j("https://www.sec.gov/Archives/edgar/daily-index/2024/QTR1/index.json")
if d:
    items = d["directory"]["item"]
    rep["daily_index_2024Q1"] = {"n": len(items), "first": items[:3], "last": items[-3:]}

# master.idx line count for one quarter (streamed cap)
s, h, b = get("https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.idx", cap=40_000_000)
if s == 200:
    lines = b.decode("latin-1").splitlines()
    rep["master_idx_2024Q1"] = {"bytes": len(b), "lines": len(lines),
                                "header": lines[:12], "row": lines[12] if len(lines) > 12 else None}

# 7. hunt correct URLs for the datasets that 404'd
for label, page in [
    ("dera_finstmt", "https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets"),
    ("dera_notes", "https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets"),
    ("dera_13f", "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"),
    ("dera_nport", "https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets"),
    ("dera_insider", "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"),
    ("dera_mfrr", "https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-risk-return-summary-data-sets"),
    ("dera_logs", "https://www.sec.gov/data-research/sec-markets-data/edgar-log-file-data-sets"),
    ("dera_index", "https://www.sec.gov/data-research/sec-markets-data"),
]:
    s, h, b = get(page, cap=2_000_000)
    if s == 200:
        html = b.decode("utf-8", "replace")
        zips = sorted(set(re.findall(r'href="([^"]+\.(?:zip|tsv|csv|json))"', html)))
        rep[label] = {"status": s, "n_links": len(zips), "sample": zips[:6]}
    else:
        rep[label] = {"status": s}
    time.sleep(0.2)

with open("/private/tmp/claude-501/-Users-sankalpsharma-research-notes/c10be4a8-fadb-4239-b34e-bb8dd6658d2b/scratchpad/edgar_probe2.json", "w") as f:
    json.dump(rep, f, indent=1, default=str)
print(json.dumps(rep, indent=1, default=str)[:12000])
