"""Testet, welche ETF-Anbieter ihre Positionslisten von GitHub aus herausgeben (nur Protokoll, keine Daten)."""
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Accept": "text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,*/*"}
CANDIDATES = [
    ("Xtrackers MSCI World (GBR)", "https://etf.dws.com/etfdata/export/GBR/ENG/excel/product/constituent/IE00BJ0KDQ92/"),
    ("Xtrackers MSCI World (DEU)", "https://etf.dws.com/etfdata/export/DEU/DEU/excel/product/constituent/IE00BJ0KDQ92/"),
    ("Xtrackers S&P 500 EW (DEU)", "https://etf.dws.com/etfdata/export/DEU/DEU/excel/product/constituent/IE00BLNMYC90/"),
    ("SPDR MSCI ACWI IMI", "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-spyi-gy.xlsx"),
    ("SPDR Global Div. Aristocrats", "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-zprg-gy.xlsx"),
    ("iShares Core MSCI World (DE)", "https://www.ishares.com/de/privatanleger/de/produkte/251882/ishares-msci-world-ucits-etf-acc-fund/1478358465952.ajax?fileType=csv&fileName=EUNL_holdings&dataType=fund"),
    ("iShares Core MSCI World (UK)", "https://www.ishares.com/uk/individual/en/products/251882/ishares-msci-world-ucits-etf-acc-fund/1506575576011.ajax?fileType=csv&fileName=SWDA_holdings&dataType=fund"),
    ("iShares Core S&P 500 (US-Seite IVV)", "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"),
    ("Vanguard FTSE All-World (UK)", "https://www.vanguard.co.uk/professional/product/etf/equity/9679/ftse-all-world-ucits-etf-usd-accumulating"),
]
for name, url in CANDIDATES:
    try:
        r = requests.get(url, headers=UA, timeout=40, allow_redirects=True)
        ct = r.headers.get("content-type", "")
        head = r.content[:120]
        kind = "XLSX" if head[:2] == b"PK" else ("HTML" if b"<html" in head.lower() or b"<!doctype" in head.lower() else "TEXT/CSV")
        print(f"[TEST] {name}: HTTP {r.status_code} · {ct[:60]} · {len(r.content)} Bytes · {kind} · {head[:90]!r}")
    except Exception as e:
        print(f"[TEST] {name}: FEHLER {e}")
