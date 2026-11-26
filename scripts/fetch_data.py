"""
Datensammler für das Research-Terminal.

Läuft in GitHub Actions (oder lokal) und schreibt JSON-Dateien nach ./data:
  market.json         Indizes, Watchlist, Devisen/Rohstoffe
  macro.json          FRED (USA) + EZB (Euroraum/Österreich)
  news.json           Schlagzeilen (Yahoo + Google News) mit FinBERT-Stimmung
  tickers/<SYM>.json  Detailseite je Watchlist-Titel (Kurse, Kennzahlen, Bilanzen, News)

Jede Quelle ist einzeln abgesichert: fällt eine aus, laufen die anderen weiter.
"""
from __future__ import annotations

import io
import json
import math
import re
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TICK = DATA / "tickers"
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
NOW = datetime.now(timezone.utc)

HTTP = requests.Session()
HTTP.headers["User-Agent"] = "Mozilla/5.0 (research-terminal)"


# ----------------------------------------------------------------- helpers
def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def clean(x):
    """NaN/inf -> None, numpy -> python, rekursiv."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            x = x.item()
        except Exception:
            pass
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


# ----------------------------------------------------------------- Verschlüsselung
# Ist das GitHub-Secret TERMINAL_PASSWORD gesetzt, werden alle Daten-Dateien mit
# AES-256-GCM verschlüsselt (Schlüssel per PBKDF2-SHA256 aus dem Passwort).
# Die Webseite entschlüsselt sie im Browser erst nach Eingabe des Passworts.
import base64
import os

PASSWORD = os.environ.get("TERMINAL_PASSWORD", "").strip()
KDF_ITER = 250_000
_KEY = _SALT = None
if PASSWORD:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    _SALT = os.urandom(16)
    _KEY = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=_SALT, iterations=KDF_ITER).derive(PASSWORD.encode())


def encode(obj) -> str:
    raw = json.dumps(clean(obj), ensure_ascii=False, separators=(",", ":"))
    if not _KEY:
        return raw
    iv = os.urandom(12)
    ct = AESGCM(_KEY).encrypt(iv, raw.encode("utf-8"), None)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return json.dumps({"enc": 1, "iter": KDF_ITER, "salt": b64(_SALT), "iv": b64(iv), "data": b64(ct)})


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode(obj), encoding="utf-8")
    log("geschrieben:", path.relative_to(ROOT))


def safe(fn, default=None, label=""):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        log(f"WARNUNG {label}: {e}")
        return default


def rnd(v, n=4):
    try:
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), n)
    except Exception:
        return None


def file_safe(sym: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sym)


# ----------------------------------------------------------------- sentiment
class Sentiment:
    """FinBERT (ProsusAI/finbert). Fällt bei Problemen auf 'neutral' zurück."""

    def __init__(self):
        self.pipe = None
        try:
            from transformers import pipeline

            self.pipe = pipeline("text-classification", model=CFG.get("sentiment_model", "ProsusAI/finbert"),
                                 truncation=True, max_length=128)
            log("FinBERT geladen")
        except Exception as e:  # noqa: BLE001
            log("FinBERT nicht verfügbar – Stimmung wird übersprungen:", e)

    def score(self, texts: list[str]) -> list[dict]:
        if not self.pipe or not texts:
            return [{"label": "n/a", "score": None, "value": 0.0} for _ in texts]
        out = []
        for i in range(0, len(texts), 32):
            for r in self.pipe(texts[i:i + 32]):
                lab = r["label"].lower()
                val = r["score"] if lab == "positive" else -r["score"] if lab == "negative" else 0.0
                out.append({"label": lab, "score": round(r["score"], 3), "value": round(val, 3)})
        return out


# ----------------------------------------------------------------- yfinance
import yfinance as yf  # noqa: E402


def history_block(syms: list[str], period="1y", interval="1d") -> dict[str, pd.Series]:
    """Lädt Schlusskurse für viele Ticker in einem Rutsch."""
    out = {}
    if not syms:
        return out
    df = yf.download(syms, period=period, interval=interval, auto_adjust=False, progress=False,
                     group_by="ticker", threads=True, timeout=30)
    for s in syms:
        try:
            col = df[s]["Close"] if len(syms) > 1 else df["Close"]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            col = col.dropna()
            if len(col):
                out[s] = col
        except Exception:
            pass
    return out


def quote_row(sym, name, closes: pd.Series | None, extra=None):
    row = {"sym": sym, "name": name, "price": None, "chg": None, "chgPct": None,
           "ytd": None, "y1": None, "spark": []}
    if closes is not None and len(closes) >= 2:
        last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
        row.update(price=rnd(last), chg=rnd(last - prev), chgPct=rnd((last / prev - 1) * 100, 3))
        ytd_base = closes[closes.index.year < NOW.year]
        if len(ytd_base):
            row["ytd"] = rnd((last / float(ytd_base.iloc[-1]) - 1) * 100, 2)
        row["y1"] = rnd((last / float(closes.iloc[0]) - 1) * 100, 2)
        row["spark"] = [rnd(v, 3) for v in closes.iloc[-60:].tolist()]
        row["date"] = closes.index[-1].strftime("%Y-%m-%d")
    if extra:
        row.update(extra)
    return row


# ----------------------------------------------------------------- SEC EDGAR
SEC_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "grossProfit": ["GrossProfit"],
    "operatingIncome": ["OperatingIncomeLoss"],
    "netIncome": ["NetIncomeLoss", "ProfitLoss"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "totalAssets": ["Assets"],
    "totalLiabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "longTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "operatingCashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "dividendsPaid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}
INSTANT = {"totalAssets", "totalLiabilities", "equity", "cash", "longTermDebt"}


class Edgar:
    def __init__(self):
        self.h = {"User-Agent": CFG.get("sec_user_agent", "research terminal contact@example.com")}
        self.map = safe(self._load_map, {}, "SEC Tickerliste")

    def _load_map(self):
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=self.h, timeout=30)
        r.raise_for_status()
        return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}

    def financials(self, sym: str):
        cik = self.map.get(sym.upper().replace("-", "."), self.map.get(sym.upper()))
        if not cik:
            return None
        time.sleep(0.15)  # SEC erlaubt max. 10 Anfragen/Sekunde
        r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=self.h, timeout=60)
        r.raise_for_status()
        facts = r.json().get("facts", {})
        gaap = facts.get("us-gaap", {})
        res: dict[str, dict[int, float]] = {}
        for key, tags in SEC_TAGS.items():
            for tag in tags:
                node = gaap.get(tag)
                if not node:
                    continue
                units = node.get("units", {})
                vals = units.get("USD") or units.get("USD/shares") or next(iter(units.values()), [])
                series: dict[int, tuple[str, float]] = {}
                for f in vals:
                    if f.get("form") not in ("10-K", "10-K/A", "20-F", "40-F"):
                        continue
                    end = f.get("end")
                    if not end:
                        continue
                    if key not in INSTANT:
                        st = f.get("start")
                        if not st:
                            continue
                        days = (datetime.fromisoformat(end) - datetime.fromisoformat(st)).days
                        if not 340 <= days <= 380:
                            continue
                    yr = int(end[:4])
                    filed = f.get("filed", "")
                    if yr not in series or filed > series[yr][0]:
                        series[yr] = (filed, f["val"])
                if series:
                    merged = res.setdefault(key, {})
                    for yr, (_, v) in series.items():
                        merged.setdefault(yr, v)  # erster passender Tag hat Vorrang
            # nächstes Feld
        if not res:
            return None
        years = sorted({y for s in res.values() for y in s})[-10:]
        out = {"source": "SEC EDGAR (10-K)", "currency": "USD", "years": years}
        for key, s in res.items():
            out[key] = [s.get(y) for y in years]
        if "operatingCashflow" in out and "capex" in out:
            out["freeCashflow"] = [None if a is None or b is None else a - b
                                   for a, b in zip(out["operatingCashflow"], out["capex"])]
        return out


def yahoo_financials(t: yf.Ticker):
    """Fallback (und für Nicht-US-Titel): Jahresabschlüsse aus Yahoo."""
    inc, bal, cf = t.financials, t.balance_sheet, t.cashflow
    if inc is None or inc.empty:
        return None
    cols = sorted(set(inc.columns) | set(bal.columns if bal is not None else []))[-6:]
    years = [c.year for c in cols]

    def pick(df, names):
        if df is None or df.empty:
            return [None] * len(cols)
        for n in names:
            if n in df.index:
                return [rnd(df.at[n, c], 2) if c in df.columns else None for c in cols]
        return [None] * len(cols)

    out = {
        "source": "Yahoo Finance", "currency": safe(lambda: t.fast_info["currency"], None, "Währung"),
        "years": years,
        "revenue": pick(inc, ["Total Revenue", "Operating Revenue"]),
        "grossProfit": pick(inc, ["Gross Profit"]),
        "operatingIncome": pick(inc, ["Operating Income", "EBIT"]),
        "netIncome": pick(inc, ["Net Income", "Net Income Common Stockholders"]),
        "eps": pick(inc, ["Diluted EPS", "Basic EPS"]),
        "totalAssets": pick(bal, ["Total Assets"]),
        "totalLiabilities": pick(bal, ["Total Liabilities Net Minority Interest"]),
        "equity": pick(bal, ["Stockholders Equity", "Common Stock Equity"]),
        "cash": pick(bal, ["Cash And Cash Equivalents"]),
        "longTermDebt": pick(bal, ["Long Term Debt"]),
        "operatingCashflow": pick(cf, ["Operating Cash Flow"]),
        "capex": pick(cf, ["Capital Expenditure"]),
        "freeCashflow": pick(cf, ["Free Cash Flow"]),
        "dividendsPaid": pick(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
    }
    # Yahoo liefert Capex/Dividenden negativ -> als positive Auszahlung zeigen
    for k in ("capex", "dividendsPaid"):
        out[k] = [None if v is None else abs(v) for v in out[k]]
    return out


# ----------------------------------------------------------------- News
def yahoo_news(t: yf.Ticker, sym: str, n: int):
    items = []
    for it in (t.news or [])[:n]:
        c = it.get("content", it)
        title = c.get("title")
        if not title:
            continue
        link = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or (c.get("clickThroughUrl") or {}).get("url")
        prov = (c.get("provider") or {}).get("displayName") or c.get("publisher") or "Yahoo Finance"
        pub = c.get("pubDate") or c.get("displayTime")
        if not pub and it.get("providerPublishTime"):
            pub = datetime.fromtimestamp(it["providerPublishTime"], tz=timezone.utc).isoformat()
        items.append({"title": title, "link": link, "source": prov, "published": pub, "ticker": sym, "via": "Yahoo"})
    return items


def google_news(query: str, n: int, tag: str | None = None):
    import feedparser

    url = f"https://news.google.com/rss/search?q={quote_plus(query)}+when:7d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(HTTP.get(url, timeout=30).content)
    items = []
    for e in feed.entries[:n]:
        title = re.sub(r"\s+-\s+[^-]+$", "", e.title)  # " - Quelle" abschneiden
        src = getattr(getattr(e, "source", None), "title", None) or e.title.rsplit(" - ", 1)[-1]
        pub = None
        if getattr(e, "published_parsed", None):
            pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        items.append({"title": title, "link": e.link, "source": src, "published": pub,
                      "ticker": tag, "topic": None if tag else query, "via": "Google News"})
    return items


def dedupe(items):
    seen, out = set(), []
    for it in items:
        k = re.sub(r"\W+", "", it["title"].lower())[:80]
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# ----------------------------------------------------------------- Makro
FRED = [
    # id, Name, Einheit, Transformation, Gruppe
    ("FEDFUNDS", "Fed Funds Rate", "%", None, "Zinsen USA"),
    ("DGS2", "US-Staatsanleihe 2J", "%", None, "Zinsen USA"),
    ("DGS10", "US-Staatsanleihe 10J", "%", None, "Zinsen USA"),
    ("T10Y2Y", "Zinskurve 10J–2J", "%-Pkt.", None, "Zinsen USA"),
    ("BAMLH0A0HYM2", "High-Yield-Spread", "%-Pkt.", None, "Kredit"),
    ("CPIAUCSL", "US-Inflation (VPI)", "% J/J", "yoy", "Inflation"),
    ("PCEPILFE", "US-Kern-PCE", "% J/J", "yoy", "Inflation"),
    ("UNRATE", "US-Arbeitslosenquote", "%", None, "Arbeitsmarkt"),
    ("PAYEMS", "US-Beschäftigung (Δ Monat)", "Tsd.", "diff", "Arbeitsmarkt"),
    ("A191RL1Q225SBEA", "US-BIP real (annualisiert)", "% Q/Q", None, "Wachstum"),
    ("INDPRO", "US-Industrieproduktion", "% J/J", "yoy", "Wachstum"),
    ("UMCSENT", "Uni Michigan Konsumklima", "Index", None, "Stimmung"),
    ("VIXCLS", "VIX Volatilität", "Index", None, "Stimmung"),
    ("DTWEXBGS", "US-Dollar-Index (breit)", "Index", None, "Währung"),
    ("M2SL", "Geldmenge M2", "% J/J", "yoy", "Liquidität"),
]
ECB = [
    ("FM/D.U2.EUR.4F.KR.DFR.LEV", "EZB-Einlagensatz", "%", "Zinsen Euroraum"),
    ("FM/D.U2.EUR.4F.KR.MRR_FR.LEV", "EZB-Hauptrefinanzierungssatz", "%", "Zinsen Euroraum"),
    ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", "Euro-Rendite 10J (AAA-Kurve)", "%", "Zinsen Euroraum"),
    ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y", "Euro-Rendite 2J (AAA-Kurve)", "%", "Zinsen Euroraum"),
    ("FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", "3M-Euribor", "%", "Zinsen Euroraum"),
    ("ICP/M.U2.N.000000.4.ANR", "Euroraum-Inflation (HVPI)", "% J/J", "Inflation"),
    ("ICP/M.U2.N.XEF000.4.ANR", "Euroraum-Kerninflation", "% J/J", "Inflation"),
    ("ICP/M.AT.N.000000.4.ANR", "Österreich-Inflation (HVPI)", "% J/J", "Inflation"),
    ("ICP/M.DE.N.000000.4.ANR", "Deutschland-Inflation (HVPI)", "% J/J", "Inflation"),
    ("LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T", "Euroraum-Arbeitslosenquote", "%", "Arbeitsmarkt"),
    ("EXR/D.USD.EUR.SP00.A", "EUR/USD (EZB-Referenz)", "USD", "Währung"),
]


def fred_series(sid: str, start="2000-01-01") -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
    df = pd.read_csv(io.StringIO(HTTP.get(url, timeout=30).text))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.dropna().set_index(pd.to_datetime(df.dropna()["date"]))["v"]


def ecb_series(key: str, start="2000-01") -> pd.Series:
    flow, k = key.split("/", 1)
    url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{k}?format=csvdata&startPeriod={start}"
    df = pd.read_csv(io.StringIO(HTTP.get(url, timeout=30).text))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(df["TIME_PERIOD"].astype(str), errors="coerce")).dropna()
    return s[s.index.notna()].sort_index()


def series_block(s: pd.Series, meta: dict) -> dict:
    s = s.dropna()
    # Tägliche Reihen auf Wochenwerte ausdünnen (kleinere Dateien)
    if len(s) > 1500:
        s = s.resample("W").last().dropna()
    s = s[s.index >= pd.Timestamp(NOW.year - 15, 1, 1)]
    last, prev = float(s.iloc[-1]), float(s.iloc[-2]) if len(s) > 1 else None
    y_ago = s[s.index <= s.index[-1] - pd.DateOffset(years=1)]
    return {**meta, "last": rnd(last, 3), "prev": rnd(prev, 3),
            "yearAgo": rnd(float(y_ago.iloc[-1]), 3) if len(y_ago) else None,
            "date": s.index[-1].strftime("%Y-%m-%d"),
            "history": [[d.strftime("%Y-%m-%d"), rnd(v, 3)] for d, v in s.items()]}


def build_macro():
    out = []
    for sid, name, unit, tr, grp in FRED:
        def f():
            s = fred_series(sid)
            if tr == "yoy":
                s = (s / s.shift(12) - 1) * 100
            elif tr == "diff":
                s = s.diff()
            return series_block(s, {"id": sid, "name": name, "unit": unit, "group": grp,
                                    "region": "USA", "source": "FRED"})
        r = safe(f, None, f"FRED {sid}")
        if r:
            out.append(r)
    for key, name, unit, grp in ECB:
        region = "Österreich" if ".AT." in key else "Deutschland" if ".DE." in key else "Euroraum"
        r = safe(lambda: series_block(ecb_series(key), {"id": key, "name": name, "unit": unit, "group": grp,
                                                        "region": region, "source": "EZB"}), None, f"EZB {key}")
        if r:
            out.append(r)
    save(DATA / "macro.json", {"updated": NOW.isoformat(), "series": out})


# ----------------------------------------------------------------- Ticker-Details
def build_ticker(sym: str, edgar: Edgar, closes_1y: pd.Series | None):
    t = yf.Ticker(sym)
    info = safe(lambda: t.info, {}, f"info {sym}") or {}
    hist5 = safe(lambda: t.history(period="5y", interval="1wk", auto_adjust=False)["Close"].dropna(), None, f"5y {sym}")
    hist1 = closes_1y if closes_1y is not None else safe(lambda: t.history(period="1y")["Close"].dropna(), None, sym)

    is_us = "." not in sym and info.get("country") in (None, "United States")
    fin = safe(lambda: edgar.financials(sym), None, f"SEC {sym}") if is_us else None
    if not fin:
        fin = safe(lambda: yahoo_financials(t), None, f"Yahoo-Bilanz {sym}")

    def g(*keys):
        for k in keys:
            if info.get(k) is not None:
                return info[k]
        return None

    price = float(hist1.iloc[-1]) if hist1 is not None and len(hist1) else g("currentPrice", "regularMarketPrice")
    prev = float(hist1.iloc[-2]) if hist1 is not None and len(hist1) > 1 else g("previousClose")
    stats = {
        "Marktkap.": g("marketCap"), "Unternehmenswert": g("enterpriseValue"),
        "KGV (ttm)": g("trailingPE"), "KGV (erw.)": g("forwardPE"), "KBV": g("priceToBook"),
        "KUV": g("priceToSalesTrailing12Months"), "EV/EBITDA": g("enterpriseToEbitda"),
        "PEG": g("trailingPegRatio", "pegRatio"),
        "Div.-Rendite %": g("dividendYield"), "Ausschüttungsquote": g("payoutRatio"),
        "Beta": g("beta"), "EPS (ttm)": g("trailingEps"), "EPS (erw.)": g("forwardEps"),
        "Bruttomarge": g("grossMargins"), "Op. Marge": g("operatingMargins"), "Nettomarge": g("profitMargins"),
        "EK-Rendite": g("returnOnEquity"), "Verschuldung/EK": g("debtToEquity"),
        "52W Hoch": g("fiftyTwoWeekHigh"), "52W Tief": g("fiftyTwoWeekLow"),
        "Ø Volumen": g("averageVolume"), "Aktien ausst.": g("sharesOutstanding"),
    }
    analyst = {"rating": g("recommendationKey"), "n": g("numberOfAnalystOpinions"),
               "target": g("targetMeanPrice"), "high": g("targetHighPrice"), "low": g("targetLowPrice")}

    news = safe(lambda: yahoo_news(t, sym, CFG.get("news_per_ticker", 8)), [], f"Yahoo-News {sym}")
    name = g("longName", "shortName") or sym
    news += safe(lambda: google_news(f'"{name.split(",")[0]}" stock', 6, sym), [], f"Google-News {sym}")

    return {
        "symbol": sym, "name": name, "currency": g("currency"), "exchange": g("fullExchangeName", "exchange"),
        "sector": g("sector"), "industry": g("industry"), "country": g("country"),
        "website": g("website"), "employees": g("fullTimeEmployees"),
        "description": g("longBusinessSummary"),
        "price": rnd(price), "chg": rnd(price - prev) if price and prev else None,
        "chgPct": rnd((price / prev - 1) * 100, 3) if price and prev else None,
        "stats": stats, "analyst": analyst,
        "history1y": [[d.strftime("%Y-%m-%d"), rnd(v)] for d, v in hist1.items()] if hist1 is not None else [],
        "history5y": [[d.strftime("%Y-%m-%d"), rnd(v)] for d, v in hist5.items()] if hist5 is not None else [],
        "financials": fin, "news": dedupe(news), "updated": NOW.isoformat(),
    }


# ----------------------------------------------------------------- main
def main():
    DATA.mkdir(exist_ok=True)
    TICK.mkdir(exist_ok=True)
    log("Verschlüsselung:", "AN" if _KEY else "AUS (Secret TERMINAL_PASSWORD fehlt)")
    # Alte Ticker-Dateien entfernen, damit keine unverschlüsselten Reste liegen bleiben
    for old in TICK.glob("*.json"):
        old.unlink()
    wl, idx, mkt = CFG["watchlist"], CFG["indices"], CFG["markets"]

    log("Kurse laden …")
    closes = history_block(list(idx) + list(mkt) + wl, period="1y")

    edgar = Edgar()
    details = {}
    for s in wl:
        log("Detail:", s)
        d = safe(lambda: build_ticker(s, edgar, closes.get(s)), None, f"Ticker {s}")
        if d:
            details[s] = d

    market = {
        "updated": NOW.isoformat(),
        "indices": [quote_row(s, n, closes.get(s)) for s, n in idx.items()],
        "markets": [quote_row(s, n, closes.get(s)) for s, n in mkt.items()],
        "watchlist": [quote_row(s, (details.get(s) or {}).get("name", s), closes.get(s), {
            "currency": (details.get(s) or {}).get("currency"),
            "sector": (details.get(s) or {}).get("sector"),
            "pe": ((details.get(s) or {}).get("stats") or {}).get("KGV (ttm)"),
            "mcap": ((details.get(s) or {}).get("stats") or {}).get("Marktkap."),
        }) for s in wl],
    }

    log("Makro laden …")
    safe(build_macro, None, "Makro")

    log("News + Stimmung …")
    sent = Sentiment()
    general = []
    for q in CFG.get("news_queries", []):
        general += safe(lambda: google_news(q, 15), [], f"Google-News {q}")
    general = dedupe(general)
    all_items = general + [n for d in details.values() for n in d["news"]]
    scores = sent.score([n["title"] for n in all_items])
    for n, sc in zip(all_items, scores):
        n["sentiment"] = sc

    summary = {}
    for s, d in details.items():
        vals = [n["sentiment"]["value"] for n in d["news"] if n.get("sentiment", {}).get("score") is not None]
        labs = [n["sentiment"]["label"] for n in d["news"]]
        summary[s] = {"score": rnd(sum(vals) / len(vals), 3) if vals else None, "n": len(d["news"]),
                      "pos": labs.count("positive"), "neg": labs.count("negative"), "neu": labs.count("neutral")}
        d["sentiment"] = summary[s]
        save(TICK / f"{file_safe(s)}.json", d)
    for row in market["watchlist"]:
        row["sentiment"] = (summary.get(row["sym"]) or {}).get("score")
    save(DATA / "market.json", market)

    def sort_key(n):
        return n.get("published") or ""

    save(DATA / "news.json", {
        "updated": NOW.isoformat(),
        "model": CFG.get("sentiment_model") if sent.pipe else None,
        "general": sorted(general, key=sort_key, reverse=True),
        "tickers": sorted([n for d in details.values() for n in d["news"]], key=sort_key, reverse=True)[:200],
        "summary": summary,
    })
    save(DATA / "meta.json", {"updated": NOW.isoformat(), "tickers": {s: file_safe(s) for s in details}})
    log("Fertig.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
