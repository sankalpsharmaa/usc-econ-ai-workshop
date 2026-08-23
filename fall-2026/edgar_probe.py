#!/usr/bin/env python3
"""Probe SEC EDGAR backend endpoints. Recon only: HEAD/GET, 8 req/sec cap."""
import json, time, sys
import urllib.request, urllib.error
import ssl, certifi
CTX = ssl.create_default_context(cafile=certifi.where())

UA = "Sankalp Sharma (USC) sankalp.sharma437@gmail.com"
HDRS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": None}

TARGETS = [
    # (label, method, url)
    ("company_tickers", "GET", "https://www.sec.gov/files/company_tickers.json"),
    ("company_tickers_exchange", "GET", "https://www.sec.gov/files/company_tickers_exchange.json"),
    ("cik_lookup_txt", "HEAD", "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"),
    ("submissions_AAPL", "GET", "https://data.sec.gov/submissions/CIK0000320193.json"),
    ("companyfacts_AAPL", "GET", "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"),
    ("companyconcept_AAPL", "GET", "https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/Revenues.json"),
    ("frames_CY2023Q1I", "GET", "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY2023Q1I.json"),
    ("fts_efts", "GET", "https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22&forms=10-K"),
    ("fts_count", "GET", "https://efts.sec.gov/LATEST/search-index?q=%22apple%22"),
    ("browse_edgar_atom", "GET", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&dateb=&owner=include&count=10&output=atom"),
    ("full_index_2024Q1", "HEAD", "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.idx"),
    ("full_index_1993Q1", "HEAD", "https://www.sec.gov/Archives/edgar/full-index/1993/QTR1/master.idx"),
    ("full_index_json", "GET", "https://www.sec.gov/Archives/edgar/full-index/index.json"),
    ("daily_index_json", "GET", "https://www.sec.gov/Archives/edgar/daily-index/index.json"),
    ("bulk_submissions_zip", "HEAD", "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"),
    ("bulk_companyfacts_zip", "HEAD", "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"),
    ("filing_dir_json", "GET", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&output=atom&count=1"),
    ("archives_cik_dir_json", "GET", "https://www.sec.gov/Archives/edgar/data/320193/index.json"),
    ("finstmt_2024q1", "HEAD", "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2024q1.zip"),
    ("finstmt_notes_2024q1", "HEAD", "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets/2024_01_notes.zip"),
    ("form13f_2024q1", "HEAD", "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2024q1_form13f.zip"),
    ("insider_2024q1", "HEAD", "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2024q1_form345.zip"),
    ("nport_2024q1", "HEAD", "https://www.sec.gov/files/structureddata/data/form-n-port-data-sets/2024q1_nport.zip"),
    ("mfrt_2024q1", "HEAD", "https://www.sec.gov/files/structureddata/data/mutual-fund-prospectus-risk-return-summary-data-sets/2024q1.zip"),
    ("edgar_log_2017", "HEAD", "https://www.sec.gov/files/EDGAR_LogFileData_thru_Jun2017.zip"),
    ("rss_recent_all", "GET", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&start=0&count=40&output=atom"),
    ("xbrl_rss_2024_01", "HEAD", "https://www.sec.gov/Archives/edgar/monthly/xbrlrss-2024-01.xml"),
    ("frames_index", "GET", "https://data.sec.gov/api/xbrl/frames/us-gaap/EarningsPerShareBasic/USD-per-shares/CY2023Q1.json"),
]


def probe(label, method, url):
    req = urllib.request.Request(url, method=method, headers={
        "User-Agent": UA, "Accept-Encoding": "identity",
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=45, context=CTX) as r:
            body = r.read(400000) if method == "GET" else b""
            return {
                "label": label, "url": url, "method": method,
                "status": r.status,
                "ctype": r.headers.get("Content-Type", ""),
                "clen": r.headers.get("Content-Length", ""),
                "ms": int((time.time() - t0) * 1000),
                "sample": body[:600].decode("utf-8", "replace"),
                "bodylen": len(body),
            }
    except urllib.error.HTTPError as e:
        return {"label": label, "url": url, "method": method, "status": e.code,
                "ctype": e.headers.get("Content-Type", ""), "clen": "", "ms": 0,
                "sample": e.read(300).decode("utf-8", "replace"), "bodylen": 0}
    except Exception as e:
        return {"label": label, "url": url, "method": method, "status": "ERR",
                "ctype": "", "clen": "", "ms": 0, "sample": repr(e)[:300], "bodylen": 0}


out = []
for label, method, url in TARGETS:
    r = probe(label, method, url)
    out.append(r)
    print(f"{r['status']:>5}  {label:<28} clen={r['clen']:<12} {r['ctype'][:30]}")
    sys.stdout.flush()
    time.sleep(0.15)

with open("/private/tmp/claude-501/-Users-sankalpsharma-research-notes/c10be4a8-fadb-4239-b34e-bb8dd6658d2b/scratchpad/edgar_probe.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote edgar_probe.json")
