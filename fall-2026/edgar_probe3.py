#!/usr/bin/env python3
"""Third EDGAR pass: total archive scale, filing-level listings, remaining dataset pages."""
import json, time, ssl, certifi, urllib.request, urllib.error, re

CTX = ssl.create_default_context(cafile=certifi.where())
UA = "Sankalp Sharma (USC) sankalp.sharma437@gmail.com"

def get(url, method="GET", cap=5_000_000):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            return r.status, dict(r.headers), (r.read(cap) if method == "GET" else b"")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(400)
    except Exception as e:
        return "ERR", {}, repr(e).encode()

rep = {}

# total scale: sum master.idx bytes over every quarter directory
total_kb, quarters = 0, []
for year in range(1993, 2027):
    for qtr in (1, 2, 3, 4):
        s, h, b = get(f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/index.json")
        if s != 200:
            continue
        try:
            items = json.loads(b)["directory"]["item"]
        except Exception:
            continue
        for it in items:
            if it["name"] == "master.idx":
                kb = int(it["size"].split()[0])
                total_kb += kb
                quarters.append((f"{year}Q{qtr}", kb))
        time.sleep(0.12)
rep["master_idx_total"] = {"n_quarters": len(quarters), "total_KB": total_kb,
                           "total_GB": round(total_kb/1048576, 2),
                           "first": quarters[:2], "last": quarters[-2:]}
# bytes-per-row calibrated on 2024Q1 (33207145 bytes / 370310 data rows)
rep["est_total_filings"] = int(total_kb * 1024 / (33207145/370310))

# filing-level directory listing
s, h, b = get("https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/index.json")
rep["filing_index_json"] = {"status": s, "sample": b[:700].decode("utf-8","replace")}

# response headers from data.sec.gov (rate-limit / cache clues)
s, h, b = get("https://data.sec.gov/submissions/CIK0000320193.json", cap=2000)
rep["data_sec_headers"] = {"status": s, "headers": h}

# no-User-Agent behaviour
try:
    req = urllib.request.Request("https://data.sec.gov/submissions/CIK0000320193.json",
                                 headers={"User-Agent": "python-urllib/3"})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        rep["no_ua"] = {"status": r.status, "note": "generic UA accepted"}
except urllib.error.HTTPError as e:
    rep["no_ua"] = {"status": e.code, "body": e.read(300).decode("utf-8","replace")}
except Exception as e:
    rep["no_ua"] = {"status": "ERR", "err": repr(e)[:200]}

# cik-lookup-data.txt row count
s, h, b = get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", cap=60_000_000)
if s == 200:
    rep["cik_lookup"] = {"bytes": len(b), "lines": b.count(b"\n"),
                         "head": b[:200].decode("latin-1")}

# remaining dataset landing pages
for label, page in [
    ("mfrr", "https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-risk-return-summary-data-sets"),
    ("logs", "https://www.sec.gov/data-research/sec-markets-data/edgar-log-file-data-sets"),
    ("crowdfunding", "https://www.sec.gov/data-research/sec-markets-data/crowdfunding-offerings-data-sets"),
    ("regd", "https://www.sec.gov/data-research/sec-markets-data/form-d-data-sets"),
    ("mmf", "https://www.sec.gov/data-research/sec-markets-data/money-market-fund-data"),
    ("abs", "https://www.sec.gov/data-research/sec-markets-data/asset-backed-securities-data-sets"),
]:
    s, h, b = get(page, cap=2_000_000)
    html = b.decode("utf-8","replace") if s == 200 else ""
    zips = sorted(set(re.findall(r'href="([^"]+\.(?:zip|tsv|csv|json|xlsx))"', html)))
    rep[label] = {"status": s, "n_links": len(zips), "sample": zips[:4]}
    time.sleep(0.15)

with open("edgar_probe3.json","w") as f:
    json.dump(rep, f, indent=1, default=str)
print(json.dumps(rep, indent=1, default=str)[:7000])
