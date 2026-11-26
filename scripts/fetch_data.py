"""
Datensammler für das Research-Terminal.

Zwei Modi (Umgebungsvariable MODE):
  fast     alle 30 Minuten: Indizes, Märkte, Watchlist-Details, Kurse des ganzen Universums, News
  nightly  einmal pro Nacht: Indexlisten aktualisieren, Kennzahlen + Detailseiten für einen Teil
           des Universums (rotierend), Makrodaten, Kalender – danach zusätzlich alles aus "fast"

Ausgabe nach $DATA_DIR (Standard ./data). Ist TERMINAL_PASSWORD gesetzt, wird jede Datei verschlüsselt.
Jede Quelle ist einzeln abgesichert: fällt eine aus, laufen die anderen weiter.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
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
DATA = Path(os.environ.get("DATA_DIR") or ROOT / "data").resolve()
TICK = DATA / "tickers"
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
MODE = (os.environ.get("MODE") or "fast").strip().lower()
NOW = datetime.now(timezone.utc)
T0 = time.time()

HTTP = requests.Session()
HTTP.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


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


def safe(fn, default=None, label=""):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        log(f"WARNUNG {label}: {str(e)[:200]}")
        return default


def rnd(v, n=4):
    try:
        if v is None:
            return None
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, n)
    except Exception:
        return None


def file_safe(sym: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sym)


def iso_date(ts):
    """Unix-Zeitstempel / Datum -> 'YYYY-MM-DD'."""
    try:
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return str(ts)[:10]
    except Exception:
        return None


# ----------------------------------------------------------------- Verschlüsselung
# Ist das GitHub-Secret TERMINAL_PASSWORD gesetzt, werden alle Daten-Dateien mit
# AES-256-GCM verschlüsselt (Schlüssel per PBKDF2-SHA256 aus dem Passwort).
PASSWORD = os.environ.get("TERMINAL_PASSWORD", "").strip()
KDF_ITER = 250_000
_KEYS: dict[str, bytes] = {}
if PASSWORD:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_SALT = os.urandom(16)


def _key(salt: bytes, it: int = KDF_ITER) -> bytes:
    k = base64.b64encode(salt).decode() + str(it)
    if k not in _KEYS:
        _KEYS[k] = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=it).derive(PASSWORD.encode())
    return _KEYS[k]


def encode(obj) -> str:
    raw = json.dumps(clean(obj), ensure_ascii=False, separators=(",", ":"))
    if not PASSWORD:
        return raw
    iv = os.urandom(12)
    ct = AESGCM(_key(_SALT)).encrypt(iv, raw.encode("utf-8"), None)
    b = lambda x: base64.b64encode(x).decode()  # noqa: E731
    return json.dumps({"enc": 1, "iter": KDF_ITER, "salt": b(_SALT), "iv": b(iv), "data": b(ct)})


def save(path: Path, obj, quiet=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode(obj), encoding="utf-8")
    if not quiet:
        log("geschrieben:", path.relative_to(DATA))


def load(path: Path, default=None):
    """Liest eine (ggf. verschlüsselte) Datei aus einem früheren Lauf."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if isinstance(obj, dict) and obj.get("enc"):
        if not PASSWORD:
            return default
        try:
            key = _key(base64.b64decode(obj["salt"]), int(obj.get("iter", KDF_ITER)))
            raw = AESGCM(key).decrypt(base64.b64decode(obj["iv"]), base64.b64decode(obj["data"]), None)
            return json.loads(raw)
        except Exception:
            log("Hinweis: alte Datei mit anderem Passwort, wird neu erstellt:", path.name)
            return default
    return obj


# ----------------------------------------------------------------- Stimmung
HALF_LIFE_DAYS = 3.0
WINDOW_DAYS = 14


class Sentiment:
    """FinBERT (ProsusAI/finbert): Wahrscheinlichkeit positiv/neutral/negativ je Schlagzeile."""

    def __init__(self):
        self.pipe = None
        try:
            from transformers import pipeline

            self.pipe = pipeline("text-classification", model=CFG.get("sentiment_model", "ProsusAI/finbert"),
                                 truncation=True, max_length=128, top_k=None)
            log("FinBERT geladen")
        except Exception as e:  # noqa: BLE001
            log("FinBERT nicht verfügbar – Stimmung wird übersprungen:", e)

    def score(self, texts: list[str]) -> list[dict]:
        if not self.pipe or not texts:
            return [{"label": "n/a", "score": None, "value": 0.0} for _ in texts]
        out = []
        for i in range(0, len(texts), 32):
            for r in self.pipe(texts[i:i + 32]):
                p = {d["label"].lower(): d["score"] for d in (r if isinstance(r, list) else [r])}
                pos, neg, neu = p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)
                lab = max(("positive", pos), ("negative", neg), ("neutral", neu), key=lambda t: t[1])[0]
                out.append({"label": lab, "score": round(max(pos, neg, neu), 3), "value": round(pos - neg, 3),
                            "p": [round(pos, 3), round(neu, 3), round(neg, 3)]})
        return out


def aggregate_sentiment(items: list[dict]) -> dict:
    """Zeitgewichteter Durchschnitt: Halbwertszeit 3 Tage, nur die letzten 14 Tage."""
    num = den = 0.0
    pos = neg = neu = n = 0
    for it in items:
        s = it.get("sentiment") or {}
        if s.get("score") is None:
            continue
        age = 0.0
        try:
            age = max(0.0, (NOW - datetime.fromisoformat(str(it.get("published")).replace("Z", "+00:00"))).total_seconds() / 86400)
        except Exception:
            age = WINDOW_DAYS / 2
        if age > WINDOW_DAYS:
            continue
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        num += w * s["value"]
        den += w
        n += 1
        pos += s["label"] == "positive"
        neg += s["label"] == "negative"
        neu += s["label"] == "neutral"
    return {"score": rnd(num / den, 3) if den else None, "n": n, "pos": pos, "neg": neg, "neu": neu}


# ----------------------------------------------------------------- yfinance
import yfinance as yf  # noqa: E402


def history_block(syms: list[str], period="1y", interval="1d", chunk=150) -> dict[str, pd.Series]:
    """Lädt Schlusskurse für viele Ticker, in Paketen."""
    out = {}
    syms = list(dict.fromkeys(syms))
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        try:
            df = yf.download(part, period=period, interval=interval, auto_adjust=False, progress=False,
                             group_by="ticker", threads=True, timeout=30)
        except Exception as e:  # noqa: BLE001
            log("WARNUNG Download:", str(e)[:150])
            continue
        for s in part:
            try:
                col = df[s]["Close"] if len(part) > 1 else df["Close"]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                col = col.dropna()
                if len(col):
                    out[s] = col
            except Exception:
                pass
        if len(syms) > chunk:
            log(f"  Kurse {min(i + chunk, len(syms))}/{len(syms)}")
            time.sleep(1.5)
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
        row["hi52"], row["lo52"] = rnd(closes.max()), rnd(closes.min())
    if extra:
        row.update(extra)
    return row


def series_pairs(s: pd.Series | None, n=4):
    if s is None:
        return []
    return [[d.strftime("%Y-%m-%d"), rnd(v, n)] for d, v in s.dropna().items()]


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
        self.last_gaap = (sym, gaap)
        res: dict[str, dict[int, float]] = {}
        eps_filed: dict[int, str] = {}
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
                    for yr, (fd, v) in series.items():
                        if yr not in merged and key == "eps":
                            eps_filed[yr] = fd
                        merged.setdefault(yr, v)  # erster passender Tag hat Vorrang
            # nächstes Feld
        if not res:
            return None
        years = sorted({y for s in res.values() for y in s})[-10:]
        out = {"source": "SEC EDGAR (10-K)", "currency": "USD", "years": years}
        for key, s in res.items():
            out[key] = [s.get(y) for y in years]
        if eps_filed:
            out["epsFiled"] = [eps_filed.get(y) for y in years]
        if "operatingCashflow" in out and "capex" in out:
            out["freeCashflow"] = [None if a is None or b is None else a - b
                                   for a, b in zip(out["operatingCashflow"], out["capex"])]
        return out


Q_TAGS = {"revenue": SEC_TAGS["revenue"], "netIncome": SEC_TAGS["netIncome"],
          "operatingIncome": SEC_TAGS["operatingIncome"], "eps": SEC_TAGS["eps"]}


def edgar_quarterly(gaap: dict, n=16):
    """Quartalswerte aus 10-Q/10-K; viertes Quartal = Geschäftsjahr minus Q1–Q3."""
    out: dict[str, dict[str, float]] = {}
    for key, tags in Q_TAGS.items():
        for tag in tags:
            node = gaap.get(tag)
            if not node:
                continue
            units = node.get("units", {})
            vals = units.get("USD") or units.get("USD/shares") or next(iter(units.values()), [])
            q, fy = {}, []
            for f in vals:
                if f.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A") or not f.get("start") or not f.get("end"):
                    continue
                st, en = datetime.fromisoformat(f["start"]), datetime.fromisoformat(f["end"])
                days = (en - st).days
                if 80 <= days <= 100:
                    k = en.strftime("%Y-%m-%d")
                    if k not in q or f.get("filed", "") > q[k][0]:
                        q[k] = (f.get("filed", ""), float(f["val"]), st)
                elif 340 <= days <= 380:
                    fy.append((st, en, float(f["val"])))
            for st, en, v in fy:  # Q4 ableiten
                k = en.strftime("%Y-%m-%d")
                if k in q:
                    continue
                inside = [x for kk, x in q.items() if x[2] >= st - timedelta(days=5) and datetime.fromisoformat(kk) <= en]
                if len(inside) == 3:
                    q[k] = ("derived", v - sum(x[1] for x in inside), en - timedelta(days=91))
            if q:
                for k, (_, v, _) in q.items():
                    out.setdefault(k, {}).setdefault(key, v)
                break
    if not out:
        return None
    ends = sorted(out)[-n:]
    lab = lambda e: f"{e[:4]}-Q{(int(e[5:7]) - 1) // 3 + 1}"  # noqa: E731
    return {"source": "SEC EDGAR (10-Q/10-K)", "currency": "USD", "periods": [lab(e) for e in ends], "ends": ends,
            **{k: [out[e].get(k) for e in ends] for k in Q_TAGS}}


def yahoo_quarterly(t: yf.Ticker):
    q = t.quarterly_income_stmt
    if q is None or q.empty:
        return None
    cols = sorted(q.columns)[-8:]

    def pick(names):
        for n in names:
            if n in q.index:
                return [rnd(q.at[n, c], 4) if c in q.columns else None for c in cols]
        return [None] * len(cols)
    return {"source": "Yahoo Finance", "currency": safe(lambda: t.fast_info["currency"], None, "Währung"),
            "periods": [f"{c.year}-Q{(c.month - 1) // 3 + 1}" for c in cols], "ends": [c.strftime("%Y-%m-%d") for c in cols],
            "revenue": pick(["Total Revenue", "Operating Revenue"]), "netIncome": pick(["Net Income", "Net Income Common Stockholders"]),
            "operatingIncome": pick(["Operating Income", "EBIT"]), "eps": pick(["Diluted EPS", "Basic EPS"])}


def earnings_surprises(t: yf.Ticker, n=12):
    df = t.get_earnings_dates(limit=n + 4)
    if df is None or df.empty:
        return []
    out = []
    for d, r in df.iterrows():
        est, rep = r.get("EPS Estimate"), r.get("Reported EPS")
        sur = r.get("Surprise(%)")
        out.append({"d": pd.Timestamp(d).strftime("%Y-%m-%d"), "est": rnd(est, 3), "rep": rnd(rep, 3),
                    "sur": rnd(sur, 2) if sur is not None and not (isinstance(sur, float) and math.isnan(sur)) else None})
    return sorted(out, key=lambda x: x["d"])[-n:]


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




# ----------------------------------------------------------------- Universum (Indexlisten)
EXCHANGE_SUFFIX = [
    ("amsterdam", ".AS"), ("paris", ".PA"), ("brussels", ".BR"), ("lisbon", ".LS"), ("dublin", ".IR"), ("irish", ".IR"),
    ("xetra", ".DE"), ("deutsche b", ".DE"), ("frankfurt", ".F"), ("borsa italiana", ".MI"), ("milan", ".MI"),
    ("madrid", ".MC"), ("mercado continuo", ".MC"), ("london", ".L"), ("swiss", ".SW"), ("six ", ".SW"),
    ("stockholm", ".ST"), ("copenhagen", ".CO"), ("helsinki", ".HE"), ("oslo", ".OL"), ("vienna", ".VI"), ("wiener", ".VI"),
    ("tokyo", ".T"), ("hong kong", ".HK"), ("toronto", ".TO"), ("asx", ".AX"), ("australian", ".AX"),
    ("korea", ".KS"), ("taiwan", ".TW"), ("singapore", ".SI"), ("shanghai", ".SS"), ("shenzhen", ".SZ"),
    ("national stock exchange of india", ".NS"), ("india", ".NS"), ("johannesburg", ".JO"), ("b3", ".SA"), ("bovespa", ".SA"),
    ("mexic", ".MX"), ("saudi", ".SR"), ("tel aviv", ".TA"), ("warsaw", ".WA"), ("new zealand", ".NZ"), ("indonesia", ".JK"),
    ("thailand", ".BK"), ("malaysia", ".KL"), ("bursa", ".KL"), ("istanbul", ".IS"), ("santiago", ".SN"), ("qatar", ".QA"),
    ("athens", ".AT"), ("prague", ".PR"), ("budapest", ".BD"), ("philippine", ".PS"),
    ("kosdaq", ".KQ"), ("taipei exchange", ".TWO"), ("sao paulo", ".SA"),
    ("nasdaq", ""), ("new york", ""), ("nyse", ""), ("cboe", ""), ("bats", ""),
]
US_FIX = {"BRKB": "BRK-B", "BFB": "BF-B", "LENB": "LEN-B", "HEIA": "HEI-A", "MOGA": "MOG-A", "UHALB": "UHAL-B"}
SECTOR_DE = {"information technology": "Technologie", "technology": "Technologie", "financials": "Finanzen",
             "financial services": "Finanzen", "health care": "Gesundheit", "healthcare": "Gesundheit",
             "consumer discretionary": "Zyklischer Konsum", "consumer cyclical": "Zyklischer Konsum",
             "consumer staples": "Basiskonsum", "consumer defensive": "Basiskonsum", "industrials": "Industrie",
             "communication": "Kommunikation", "communication services": "Kommunikation", "energy": "Energie",
             "materials": "Rohstoffe", "basic materials": "Rohstoffe", "utilities": "Versorger", "real estate": "Immobilien",
             "informationstechnologie": "Technologie", "finanzen": "Finanzen", "gesundheitsversorgung": "Gesundheit",
             "zyklische konsumgüter": "Zyklischer Konsum", "nichtzyklische konsumgüter": "Basiskonsum",
             "industrie": "Industrie", "kommunikation": "Kommunikation", "energie": "Energie", "materialien": "Rohstoffe",
             "versorger": "Versorger", "immobilien": "Immobilien"}


def sector_de(s):
    return SECTOR_DE.get(str(s or "").strip().lower(), s or None)


def yahoo_symbol(ticker: str, exchange: str) -> str | None:
    t = str(ticker or "").strip().upper()
    ex = str(exchange or "").strip().lower()
    if not t or t in ("-", "NAN") or not ex:
        return None
    suf = next((s for k, s in EXCHANGE_SUFFIX if k in ex), None)
    if suf is None:
        return None
    if suf == "":
        return US_FIX.get(t, t.replace(".", "-").replace(" ", "-").replace("/", "-"))
    t = t.rstrip(".").replace(" ", "-").replace("/", "-")
    if suf == ".L":
        t = t.replace(".", "-")
    elif suf != "":
        t = t.replace(".", "-")
    if suf == ".HK" and t.isdigit():
        t = t.zfill(4)
    if suf in (".SS", ".SZ", ".KS") and t.isdigit():
        t = t.zfill(6)
    return t + suf


def ishares_holdings(url: str, label: str) -> list[dict]:
    r = HTTP.get(url, timeout=60, headers={"Accept": "text/csv,application/octet-stream,*/*",
                                           "Referer": url.split(".ajax")[0].rsplit("/", 1)[0] + "/"})
    txt = r.text
    lines = txt.splitlines()
    idx = next((i for i, ln in enumerate(lines[:40]) if re.match(r'^\ufeff?"?(Ticker|Emittententicker)"?[,;]', ln)), None)
    if idx is None:
        snippet = re.sub(r"\s+", " ", txt[:160])
        raise ValueError(f"keine Positionsliste (HTTP {r.status_code}, Antwort beginnt mit: {snippet!r})")
    body = "\n".join(lines[idx:])
    sep = ";" if lines[idx].count(";") > lines[idx].count(",") else ","
    df = pd.read_csv(io.StringIO(body), sep=sep, dtype=str, on_bad_lines="skip")
    df.columns = [c.strip().strip('"').lstrip("﻿") for c in df.columns]
    col = lambda *names: next((c for c in df.columns if c in names), None)  # noqa: E731
    ct, cn = col("Ticker", "Emittententicker"), col("Name")
    cs, ca = col("Sector", "Sektor"), col("Asset Class", "Anlageklasse")
    cl, ce = col("Location", "Standort"), col("Exchange", "Börse")
    out = []
    for _, r in df.iterrows():
        if ca and str(r.get(ca, "")).strip().lower() not in ("equity", "aktien"):
            continue
        sym = yahoo_symbol(r.get(ct), r.get(ce))
        if sym:
            out.append({"s": sym, "n": str(r.get(cn) or sym).strip().title() if str(r.get(cn) or "").isupper() else str(r.get(cn) or sym).strip(),
                        "sec": sector_de(r.get(cs)), "c": (str(r.get(cl)).strip() if cl and r.get(cl) else None), "src": [label]})
    log(f"  {label}: {len(out)} Titel")
    return out


YAHOO_SUFFIXES = tuple(sorted({s for _, s in EXCHANGE_SUFFIX if s}))


def wiki_list(url: str, label: str, suffix: str = "", country: str | None = None, min_rows: int = 15) -> list[dict]:
    """Indexmitglieder aus einer Wikipedia-Tabelle (Spalte Ticker/Symbol)."""
    clean_h = lambda c: re.sub(r"\[.*?\]", "", str(c)).strip()  # noqa: E731
    tables = pd.read_html(io.StringIO(HTTP.get(url, timeout=60).text))
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(str(x) for x in c if "Unnamed" not in str(x)) for c in t.columns]
        cols = [str(c) for c in t.columns]
        sc = next((c for c in cols if clean_h(c) in ("Symbol", "Ticker", "Ticker symbol", "EPIC", "Code", "Stock code")), None)
        if not sc or len(t) < min_rows:
            continue
        nc = next((c for c in cols if clean_h(c) in ("Security", "Company", "Name", "Company name")), None)
        sec = next((c for c in cols if "ector" in c or "Industry" in c), None)
        out = []
        for _, r in t.iterrows():
            raw = re.sub(r"\[.*?\]", "", str(r[sc])).strip().upper()
            if ":" in raw:  # z. B. "SEHK: 5"
                raw = raw.split(":")[-1].strip()
            if not raw or raw == "NAN":
                continue
            if suffix == ".HK" and raw.isdigit():
                raw = raw.zfill(4)
            if raw.endswith(YAHOO_SUFFIXES):
                sym = raw
            elif suffix:
                sym = raw.rstrip(".").replace(".", "-").replace(" ", "-") + suffix
            else:
                sym = raw.replace(".", "-")
            out.append({"s": sym, "n": str(r[nc]).strip() if nc else sym, "sec": sector_de(r[sec]) if sec else None,
                        "c": country, "src": [label]})
        log(f"  {label}: {len(out)} Titel")
        return out
    raise ValueError("Tabelle nicht gefunden")


def build_universe_list(old: list[dict]) -> list[dict]:
    uc = CFG.get("universe", {})
    found: dict[str, dict] = {}

    def add(items):
        for it in items:
            cur = found.get(it["s"])
            if cur:
                cur["src"] = sorted(set(cur["src"]) | set(it["src"]))
                for k in ("n", "sec", "c"):
                    cur[k] = cur.get(k) or it.get(k)
            else:
                found[it["s"]] = dict(it)

    for src in uc.get("ishares", []):
        add(safe(lambda: ishares_holdings(src["url"], src["name"]), [], f"Indexliste {src['name']}"))
    def with_fallback(fn, label):
        """Zweiter Versuch nach kurzer Pause; schlägt beides fehl, gilt die Liste vom letzten Lauf."""
        res = safe(fn, None, f"Indexliste {label}")
        if not res:
            time.sleep(4)
            res = safe(fn, None, f"Indexliste {label} (2. Versuch)")
        if not res:
            res = [dict(r, src=[label]) for r in old if label in (r.get("src") or [])]
            if res:
                log(f"  {label}: {len(res)} Titel aus dem letzten Lauf übernommen")
        return res or []

    for src in uc.get("wikipedia", []):
        add(with_fallback(lambda: wiki_list(src["url"], src["name"], src.get("suffix", ""), src.get("country")),
                          src["name"]))
    if uc.get("em_korea_taiwan", True):
        add(with_fallback(em_korea_taiwan, "MSCI EM (Korea/Taiwan)"))
    add([{"s": s, "n": None, "sec": None, "c": None, "src": ["Zusatz"]} for s in uc.get("extra", [])])
    add([{"s": s, "n": None, "sec": None, "c": None, "src": ["Watchlist"]} for s in CFG.get("watchlist", [])])
    if len(found) < 100 and old:  # noqa: SIM102
        log("Indexlisten lieferten zu wenig Titel – vorherige Liste wird weiterverwendet")
        add(old)
    return sorted(found.values(), key=lambda x: x["s"])


# ----------------------------------------------------------------- Makro
FRED = [
    # id, Name, Einheit, Transformation, Gruppe, Referenzlinie
    ("FEDFUNDS", "Fed Funds Rate", "%", None, "Zinsen USA", None),
    ("DGS2", "US-Staatsanleihe 2J", "%", None, "Zinsen USA", None),
    ("DGS10", "US-Staatsanleihe 10J", "%", None, "Zinsen USA", None),
    ("T10Y2Y", "Zinskurve 10J–2J", "%-Pkt.", None, "Zinsen USA", 0),
    ("BAMLH0A0HYM2", "High-Yield-Spread USA", "%-Pkt.", None, "Kredit", None),
    ("CPIAUCSL", "US-Inflation (VPI)", "% J/J", "yoy", "Inflation", 2),
    ("PCEPILFE", "US-Kern-PCE", "% J/J", "yoy", "Inflation", 2),
    ("UNRATE", "US-Arbeitslosenquote", "%", None, "Arbeitsmarkt", None),
    ("PAYEMS", "US-Beschäftigung (Δ Monat)", "Tsd.", "diff", "Arbeitsmarkt", 0),
    ("A191RL1Q225SBEA", "US-BIP real (annualisiert)", "% Q/Q", None, "Wachstum", 0),
    ("INDPRO", "US-Industrieproduktion", "% J/J", "yoy", "Wachstum", 0),
    ("GACDFSA066MSFRBPHI", "Philly-Fed Industrie (USA)", "Saldo", None, "Frühindikatoren", 0),
    ("GACDISA066MSFRBNY", "Empire State Industrie (USA)", "Saldo", None, "Frühindikatoren", 0),
    ("CFNAI", "Chicago Fed National Activity", "Index", None, "Frühindikatoren", 0),
    ("UMCSENT", "Uni Michigan Konsumklima", "Index", None, "Stimmung", None),
    ("VIXCLS", "VIX Volatilität", "Index", None, "Stimmung", 20),
    ("DTWEXBGS", "US-Dollar-Index (breit)", "Index", None, "Währung", None),
    ("M2SL", "Geldmenge M2 USA", "% J/J", "yoy", "Liquidität", 0),
]
ECB = [
    ("FM/D.U2.EUR.4F.KR.DFR.LEV", "EZB-Einlagensatz", "%", "Zinsen Euroraum"),
    ("FM/D.U2.EUR.4F.KR.MRR_FR.LEV", "EZB-Hauptrefinanzierungssatz", "%", "Zinsen Euroraum"),
    ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", "Euro-Rendite 10J (AAA-Kurve)", "%", "Zinsen Euroraum"),
    ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y", "Euro-Rendite 2J (AAA-Kurve)", "%", "Zinsen Euroraum"),
    ("FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", "3M-Euribor", "%", "Zinsen Euroraum"),
    ("IRS/M.AT.L.L40.CI.0000.EUR.N.Z", "Österreich Staatsanleihe 10J", "%", "Staatsanleihen Österreich"),
    ("IRS/M.DE.L.L40.CI.0000.EUR.N.Z", "Deutschland Bund 10J", "%", "Staatsanleihen Österreich"),
    ("IRS/M.IT.L.L40.CI.0000.EUR.N.Z", "Italien Staatsanleihe 10J", "%", "Staatsanleihen Österreich"),
    ("IRS/M.FR.L.L40.CI.0000.EUR.N.Z", "Frankreich Staatsanleihe 10J", "%", "Staatsanleihen Österreich"),
    ("ICP/M.U2.N.000000.4.ANR", "Euroraum-Inflation (HVPI)", "% J/J", "Inflation"),
    ("ICP/M.U2.N.XEF000.4.ANR", "Euroraum-Kerninflation", "% J/J", "Inflation"),
    ("ICP/M.AT.N.000000.4.ANR", "Österreich-Inflation (HVPI)", "% J/J", "Inflation"),
    ("ICP/M.DE.N.000000.4.ANR", "Deutschland-Inflation (HVPI)", "% J/J", "Inflation"),
    ("LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T", "Euroraum-Arbeitslosenquote", "%", "Arbeitsmarkt"),
    ("EXR/D.USD.EUR.SP00.A", "EUR/USD (EZB-Referenz)", "USD", "Währung"),
]
# EU-Kommission (Eurostat, Datensatz ei_bssi_m_r2): Konjunkturumfragen – frei verfügbare PMI-Alternative
EUROSTAT = [
    ("BS-ESI-I", "Wirtschaftsstimmung ESI", "Index", 100),
    ("BS-ICI-BAL", "Industrievertrauen", "Saldo", 0),
    ("BS-SCI-BAL", "Dienstleistungsvertrauen", "Saldo", 0),
    ("BS-CSMCI-BAL", "Verbrauchervertrauen", "Saldo", 0),
]
EUROSTAT_GEO = [("EA20", "Euroraum"), ("AT", "Österreich"), ("DE", "Deutschland")]


FRED_STATE = {"fails": 0}


def fred_series(sid: str, start="2000-01-01") -> pd.Series:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}"
               f"&file_type=json&observation_start={start}")
        obs = HTTP.get(url, timeout=40).json()["observations"]
        s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs}).dropna()
        return s.sort_index()
    if FRED_STATE["fails"] >= 2:
        raise RuntimeError("übersprungen (FRED antwortet nicht – Secret FRED_API_KEY anlegen)")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
    try:
        txt = HTTP.get(url, timeout=20).text
    except Exception:
        FRED_STATE["fails"] += 1
        raise
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    return pd.Series(df["v"].values, index=pd.to_datetime(df["date"]))


def ecb_series(key: str, start="2000-01") -> pd.Series:
    flow, k = key.split("/", 1)
    url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{k}?format=csvdata&startPeriod={start}"
    df = pd.read_csv(io.StringIO(HTTP.get(url, timeout=30).text))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(df["TIME_PERIOD"].astype(str), errors="coerce")).dropna()
    return s[s.index.notna()].sort_index()


def eurostat_series(indic: str, geo: str) -> pd.Series:
    url = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bssi_m_r2"
           f"?format=JSON&lang=EN&geo={geo}&indic={indic}&s_adj=SA&sinceTimePeriod=2005-01")
    j = HTTP.get(url, timeout=40).json()
    tidx = j["dimension"]["time"]["category"]["index"]
    vals = j.get("value", {})
    pts = {}
    for lab, pos in tidx.items():
        v = vals.get(str(pos)) if isinstance(vals, dict) else (vals[pos] if pos < len(vals) else None)
        if v is not None:
            pts[pd.Timestamp(lab.replace("M", "-") + "-01" if len(lab) <= 7 else lab)] = float(v)
    if not pts:
        raise ValueError("leer")
    return pd.Series(pts).sort_index()


def manual_series(editions: list[dict] | None = None) -> list[dict]:
    """PMI-Werte aus den Marktupdates und aus manual/pmi.csv."""
    data: dict[str, dict] = {}
    for ed in reversed(editions or []):  # ältere zuerst, neuere überschreiben
        for pt in ed.get("pmi", []) or []:
            try:
                data.setdefault(str(pt["reihe"]).strip(), {})[pd.Timestamp(str(pt["monat"])[:7] + "-01")] = float(pt["wert"])
            except Exception:
                continue
    p = ROOT / "manual" / "pmi.csv"
    rows = [ln for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip() and not ln.lstrip().startswith("#")] \
        if p.exists() else []
    for r in csv.reader(rows[1:] if rows and rows[0].lower().startswith("monat") else rows, delimiter=";"):
        if len(r) < 3:
            continue
        try:
            d = pd.Timestamp(r[0].strip()[:7] + "-01")
            v = float(r[2].strip().replace(",", "."))
        except Exception:
            continue
        data.setdefault(r[1].strip(), {})[d] = v
    out = []
    for name, pts in data.items():
        s = pd.Series(pts).sort_index()
        region = "Euroraum" if "euro" in name.lower() else "USA" if any(k in name.lower() for k in ("usa", "us ", "ism")) \
            else "Österreich" if "österr" in name.lower() else "Deutschland" if "deutsch" in name.lower() else "Global"
        out.append(series_block(s, {"id": "manual:" + name, "name": name, "unit": "Index", "group": "Einkaufsmanager (Marktupdate)",
                                    "region": region, "source": "Marktupdate / eigene Eingabe", "ref": 50}, min_len=1))
    return out


def series_block(s: pd.Series, meta: dict, min_len=2) -> dict:
    s = s.dropna()
    if len(s) < min_len:
        raise ValueError("zu wenig Werte")
    if len(s) > 1500:
        s = s.resample("W").last().dropna()
    s = s[s.index >= pd.Timestamp(NOW.year - 15, 1, 1)]
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else None
    y_ago = s[s.index <= s.index[-1] - pd.DateOffset(years=1)]
    return {**meta, "last": rnd(last, 3), "prev": rnd(prev, 3),
            "yearAgo": rnd(float(y_ago.iloc[-1]), 3) if len(y_ago) else None,
            "date": s.index[-1].strftime("%Y-%m-%d"), "history": series_pairs(s, 3)}


def build_macro():
    out = []
    raw: dict[str, pd.Series] = {}
    for sid, name, unit, tr, grp, ref in FRED:
        def f():
            s = fred_series(sid)
            if tr == "yoy":
                s = (s / s.shift(12) - 1) * 100
            elif tr == "diff":
                s = s.diff()
            return series_block(s, {"id": sid, "name": name, "unit": unit, "group": grp,
                                    "region": "USA", "source": "FRED", "ref": ref})
        r = safe(f, None, f"FRED {sid}")
        if r:
            out.append(r)
    for key, name, unit, grp in ECB:
        region = next((lab for code, lab in ((".AT.", "Österreich"), (".DE.", "Deutschland"), (".IT.", "Italien"),
                                              (".FR.", "Frankreich")) if code in key), "Euroraum")

        def f():
            s = ecb_series(key)
            raw[key] = s
            return series_block(s, {"id": key, "name": name, "unit": unit, "group": grp, "region": region,
                                    "source": "EZB", "ref": 2 if "ICP" in key else None})
        r = safe(f, None, f"EZB {key}")
        if r:
            out.append(r)
    at, de = raw.get("IRS/M.AT.L.L40.CI.0000.EUR.N.Z"), raw.get("IRS/M.DE.L.L40.CI.0000.EUR.N.Z")
    if at is not None and de is not None:
        sp = (at - de).dropna() * 100
        r = safe(lambda: series_block(sp, {"id": "SPREAD_AT_DE", "name": "Renditeaufschlag Österreich ggü. Deutschland",
                                           "unit": "Bp.", "group": "Staatsanleihen Österreich", "region": "Österreich",
                                           "source": "EZB (berechnet)", "ref": 0}), None, "Spread AT-DE")
        if r:
            out.append(r)
    for geo, region in EUROSTAT_GEO:
        for indic, name, unit, ref in EUROSTAT:
            def f():
                try:
                    s = eurostat_series(indic, geo)
                except Exception:
                    if geo != "EA20":
                        raise
                    s = eurostat_series(indic, "EA")
                return series_block(s, {"id": f"ESTAT:{indic}:{geo}", "name": f"{name} ({region})", "unit": unit,
                                        "group": "Frühindikatoren", "region": region, "source": "EU-Kommission", "ref": ref})
            r = safe(f, None, f"Eurostat {indic} {geo}")
            if r:
                out.append(r)
    old = (load(DATA / "macro.json", {}) or {}).get("series", [])
    have = {s["id"] for s in out}
    kept = [s for s in old if s.get("id") not in have and not str(s.get("id", "")).startswith("manual:")]
    if kept:
        log(f"Makro: {len(kept)} Reihen aus dem letzten erfolgreichen Abruf übernommen")
    save(DATA / "macro.json", {"updated": NOW.isoformat(), "series": out + kept})


def merge_manual_macro(editions: list[dict]):
    """PMI-Reihen aus Marktupdate + pmi.csv in macro.json einsetzen (bei jedem Lauf)."""
    m = load(DATA / "macro.json", None)
    if not m:
        return
    m["series"] = [s for s in m.get("series", []) if not str(s.get("id", "")).startswith("manual:")]
    m["series"] += safe(lambda: manual_series(editions), [], "PMI aus Marktupdate")
    save(DATA / "macro.json", m)


# ----------------------------------------------------------------- Österreichische Bundesanleihen
OEKB_STRIP = "https://www.oekb.at/en/capital-market-services/government-bond-and-atb-auctions/stripping-of-government-bonds-of-the-republic-of-austria.html"
WB_GOV = "https://www.wienerborse.at/en/trading/market-segmentation/bond-market/government-bonds/"


def bond_master() -> list[dict]:
    """Alle umlaufenden Bundesanleihen (OeKB-Liste): ISIN, Kupon, Laufzeitjahre, Volumen."""
    tables = pd.read_html(io.StringIO(HTTP.get(OEKB_STRIP, timeout=40).text))
    t = next(t for t in tables if any("ISIN" in str(c) for c in t.columns))
    t.columns = [str(c) for c in t.columns]
    ci = next(c for c in t.columns if "ISIN" in c)
    cn = next(c for c in t.columns if "Issue" in c or "Emission" in c or "Bezeichnung" in c)
    cv = next((c for c in t.columns if "Outstanding" in c or "Aushaftend" in c), None)
    out = []
    for _, r in t.iterrows():
        isin, name = str(r[ci]).strip(), str(r[cn]).strip()
        if not isin.startswith("AT"):
            continue
        m = re.search(r"(\d+[.,]\d+)\s*%", name)
        y = re.search(r"(\d{4})-(\d{4})", name)
        vol = pd.to_numeric(str(r[cv]).replace(",", ""), errors="coerce") if cv else None
        out.append({"isin": isin, "name": name, "kupon": float(m.group(1).replace(",", ".")) if m else None,
                    "von": int(y.group(1)) if y else None, "bis": int(y.group(2)) if y else None,
                    "volumen": rnd(vol, 2), "gruen": "(G)" in name})
    return out


def wb_prices() -> dict[str, dict]:
    """Kurse der Bundesanleihen von der Wiener Börse (15 Min. verzögert)."""
    page = HTTP.get(WB_GOV, timeout=40).text
    m = re.search(r'href="([^"]*?/en/bonds/\?[^"]*?XVIE_8164[^"]*)"', page)
    if not m:
        raise ValueError("Link zur Kursliste nicht gefunden")
    url = m.group(1).replace("&amp;", "&")
    if url.startswith("/"):
        url = "https://www.wienerborse.at" + url
    prices = {}
    for pg in range(1, 4):
        html = HTTP.get(url + (f"&c7928-page={pg}" if pg > 1 else "") + "&per-page=50", timeout=40).text
        found = 0
        for t in pd.read_html(io.StringIO(html)):
            cols = [str(c) for c in t.columns]
            if not any("ISIN" in c for c in cols):
                continue
            ci = next(c for c in cols if "ISIN" in c)
            cl = next((c for c in cols if c.startswith("Last") or c.startswith("Letzt")), None)
            cm = next((c for c in cols if "Maturity" in c or "Fälligkeit" in c), None)
            cd = next((c for c in cols if "Date" in c or "Datum" in c), None)
            for _, r in t.iterrows():
                isin = str(r[ci]).strip()
                if not isin.startswith("AT"):
                    continue
                px = pd.to_numeric(str(r[cl]).replace(",", ""), errors="coerce") if cl else None
                mat = pd.to_datetime(str(r[cm]), format="%m/%d/%Y", errors="coerce") if cm else None
                prices[isin] = {"kurs": rnd(px, 3), "faellig": mat.strftime("%Y-%m-%d") if mat is not None and not pd.isna(mat) else None,
                                "zeit": str(r[cd]) if cd else None}
                found += 1
        if found < 50:
            break
    if not prices:
        raise ValueError("keine Kurse in der Liste")
    return prices


def bond_yield(price: float, coupon: float, maturity: datetime, today: datetime):
    """Rendite auf Verfall (jährlicher Kupon, ACT/ACT), modifizierte Duration. Kurs = Clean-Preis."""
    if not price or not maturity or maturity <= today:
        return None, None
    cfs, d = [], maturity
    while d > today:
        cfs.append(d)
        d = d.replace(year=d.year - 1)
    prev = d
    cfs.reverse()
    accrued = coupon * (today - prev).days / max(1, (cfs[0] - prev).days)
    dirty = price + accrued
    t = [(c - today).days / 365.25 for c in cfs]
    amt = [coupon] * len(cfs)
    amt[-1] += 100

    def pv(y):
        return sum(a / (1 + y) ** ti for a, ti in zip(amt, t))
    lo, hi = -0.05, 0.5
    for _ in range(200):
        mid = (lo + hi) / 2
        if pv(mid) > dirty:
            lo = mid
        else:
            hi = mid
    y = (lo + hi) / 2
    mac = sum(ti * a / (1 + y) ** ti for a, ti in zip(amt, t)) / pv(y)
    return round(y * 100, 3), round(mac / (1 + y), 2)


def build_bonds():
    master = bond_master()
    prices = safe(wb_prices, {}, "Kurse Wiener Börse")
    today = datetime(NOW.year, NOW.month, NOW.day)
    rows = []
    for b in master:
        p = prices.get(b["isin"], {})
        mat = None
        if p.get("faellig"):
            mat = datetime.fromisoformat(p["faellig"])
        y, dur = bond_yield(p.get("kurs"), b["kupon"] or 0.0, mat, today) if mat else (None, None)
        rows.append({**b, **p, "rendite": y, "duration": dur,
                     "rlz": round((mat - today).days / 365.25, 2) if mat else (b["bis"] - NOW.year if b["bis"] else None)})
    rows.sort(key=lambda r: (r["rlz"] is None, r["rlz"] or 0))
    save(DATA / "bonds.json", {"updated": NOW.isoformat(), "rows": rows,
                               "quellen": {"liste": "OeKB (Stripping-Übersicht)", "kurse": "Wiener Börse, 15 Min. verzögert"},
                               "mitKurs": sum(1 for r in rows if r.get("kurs"))})
    log(f"Bundesanleihen: {len(rows)} Titel, {sum(1 for r in rows if r.get('kurs'))} mit Kurs")


# ----------------------------------------------------------------- ETFs
ETF_INDEX = {"SXR8.DE": "S&P 500", "VUSA.DE": "S&P 500", "XDEW.DE": "S&P 500", "EXS1.DE": "DAX", "DBXD.DE": "DAX",
             "EUN2.DE": "Euro Stoxx 50", "SXRV.DE": "Nasdaq 100", "EQQQ.DE": "Nasdaq 100", "EXXX.DE": "ATX"}


def build_etfs():
    cfg = json.loads((ROOT / "manual" / "etfs.json").read_text(encoding="utf-8"))
    etfs = cfg.get("etfs", [])
    tick = [e["ticker"] for e in etfs if e.get("ticker")]
    d1 = history_block(tick, period="1y")
    w5 = history_block(tick, period="5y", interval="1wk")
    old = {e["isin"]: e for e in (load(DATA / "etfs.json", {}) or {}).get("etfs", [])}
    out = []
    for e in etfs:
        t = e.get("ticker")
        c, w = d1.get(t), w5.get(t)
        row = {**e, "index": ETF_INDEX.get(t)}
        if MODE == "nightly" or e["isin"] not in old or old[e["isin"]].get("ter") is None:
            info = safe(lambda: yf.Ticker(t).info or {}, {}, f"ETF {t}") if t else {}
            ter = info.get("netExpenseRatio") or info.get("annualReportExpenseRatio")
            if ter is not None and ter < 0.05:  # Bruchteil (0,002) -> Prozent (0,2)
                ter = ter * 100
            row.update(ter=rnd(ter, 3) if ter else None,
                       aum=info.get("totalAssets"), cur=info.get("currency"), ydiv=rnd((info.get("yield") or 0) * 100, 2) if info.get("yield") else None)
        else:
            row.update({k: old[e["isin"]].get(k) for k in ("ter", "aum", "cur", "ydiv")})
        if c is not None and len(c) > 2:
            px = float(c.iloc[-1])
            yb = c[c.index.year < NOW.year]
            ret = c.pct_change().dropna()
            peak = c.cummax()
            row.update(px=rnd(px), chg=rnd((px / float(c.iloc[-2]) - 1) * 100, 2),
                       m1=rnd((px / float(c.iloc[-22]) - 1) * 100, 1) if len(c) > 22 else None,
                       ytd=rnd((px / float(yb.iloc[-1]) - 1) * 100, 1) if len(yb) else None,
                       y1=rnd((px / float(c.iloc[0]) - 1) * 100, 1),
                       vola=rnd(float(ret.std()) * (252 ** 0.5) * 100, 1),
                       mdd=rnd(float((c / peak - 1).min()) * 100, 1), hist1y=series_pairs(c))
        if w is not None and len(w) > 60:
            pxw = float(w.iloc[-1])
            for yrs, key in ((3, "y3"), (5, "y5")):
                base = w[w.index <= w.index[-1] - pd.DateOffset(years=yrs)]
                if len(base):
                    row[key] = rnd(((pxw / float(base.iloc[-1])) ** (1 / yrs) - 1) * 100, 1)
            row["hist5y"] = series_pairs(w)
        out.append(row)
    save(DATA / "etfs.json", {"updated": NOW.isoformat(), "geprueft": cfg.get("geprueft"), "hinweis": cfg.get("_hinweis"), "etfs": out})
    log(f"ETFs: {len(out)}, mit Kurs {sum(1 for r in out if r.get('px'))}")



# ----------------------------------------------------------------- ETF-Positionen
NAME_STOP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "AG", "SE", "SA", "NV",
             "AB", "ASA", "OYJ", "SPA", "HOLDINGS", "HOLDING", "GROUP", "GRUPPE", "CLASS", "A", "B", "C", "THE", "REG", "REGISTERED",
             "SHS", "ORD", "ADR", "COM", "KGAA", "AKTIENGESELLSCHAFT", "PREF", "VZ", "ST", "N", "V", "S", "P", "DE", "LA", "PLC.",
             "CL", "SHARES", "NPV", "BHD", "TBK", "PCL", "AS", "A/S", "SAB", "CV", "OY", "BV", "LP", "ETF"}


def name_key(n: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", str(n or "")).encode("ascii", "ignore").decode()
    w = re.sub(r"[^A-Z0-9 ]", " ", n.upper().replace("&", " AND "))
    return " ".join(t for t in w.split() if t not in NAME_STOP)


def read_holdings(e: dict) -> list[tuple[str, float]]:
    """(Name, Gewicht in %) aus der Anbieterdatei."""
    return [(r["name"], r["w"]) for r in read_holdings_full(e)]


def read_holdings_full(e: dict) -> list[dict]:
    """Alle Positionen mit Name, ISIN, Börse und Gewicht in %."""
    import openpyxl
    src = e.get("holdings") or {}
    if src.get("anbieter") == "xtrackers":
        url = f"https://etf.dws.com/etfdata/export/DEU/DEU/excel/product/constituent/{e['isin']}/"
    elif src.get("anbieter") == "spdr":
        url = f"https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-{e['ticker'].split('.')[0].lower()}-gy.xlsx"
    else:
        return []
    wb = openpyxl.load_workbook(io.BytesIO(HTTP.get(url, timeout=60).content), read_only=True, data_only=True)
    ws = wb.worksheets[0]
    head, out = None, []
    for row in ws.iter_rows(values_only=True):
        cells = [str(c).strip() if c is not None else "" for c in row]
        if head is None:
            if "ISIN" in cells and any(c in cells for c in ("Name", "Security Name")):
                head = {c: i for i, c in enumerate(cells)}
            continue
        ni = head.get("Name", head.get("Security Name"))
        wi = head.get("Weighting", head.get("Percent of Fund"))
        if ni is None or wi is None or ni >= len(row) or not row[ni]:
            continue
        try:
            w = float(row[wi])
        except Exception:
            continue
        gi = head.get("ISIN")
        xi = head.get("Exchange", head.get("Trade Country Name"))
        out.append({"name": str(row[ni]), "w": w * 100 if src.get("anbieter") == "xtrackers" else w,
                    "isin": str(row[gi]) if gi is not None and gi < len(row) and row[gi] else "",
                    "ex": str(row[xi]) if xi is not None and xi < len(row) and row[xi] else ""})
    return out


def em_korea_taiwan(min_weight=0.02) -> list[dict]:
    """Korea- und Taiwan-Titel aus der MSCI-EM-Positionsliste (Börsencode steckt in der ISIN)."""
    e = {"isin": "IE00BTJRMP35", "ticker": "XMME.DE", "holdings": {"anbieter": "xtrackers"}}
    out = []
    for r in read_holdings_full(e):
        isin, ex = r["isin"].upper(), r["ex"].upper()
        if r["w"] < min_weight:
            continue
        if isin.startswith("KR7") and len(isin) == 12:
            sym = isin[3:9] + (".KQ" if "KOSDAQ" in ex else ".KS")
            c = "South Korea"
        elif isin.startswith("TW000") and len(isin) == 12 and isin[5:9].isdigit():
            sym = isin[5:9] + (".TWO" if ("GRETAI" in ex or "TAIPEI EXCHANGE" in ex or "TPEX" in ex) else ".TW")
            c = "Taiwan"
        else:
            continue
        out.append({"s": sym, "n": r["name"].title(), "sec": None, "c": c, "src": ["MSCI EM (Korea/Taiwan)"]})
    # Börsenplatz prüfen: Taipei Exchange (.TWO) bzw. KOSDAQ (.KQ) sind in der Positionsliste nicht immer erkennbar
    alt = {".TW": ".TWO", ".TWO": ".TW", ".KS": ".KQ", ".KQ": ".KS"}
    have = history_block([r["s"] for r in out], period="5d", interval="1d")
    miss = [r for r in out if have.get(r["s"]) is None or not len(have[r["s"]])]
    if miss:
        swap = {r["s"]: r["s"].rsplit(".", 1)[0] + alt["." + r["s"].rsplit(".", 1)[1]] for r in miss}
        have2 = history_block(list(swap.values()), period="5d", interval="1d")
        for r in miss:
            n = swap[r["s"]]
            r["s"] = n if (have2.get(n) is not None and len(have2[n])) else None
    dropped = sum(1 for r in out if not r["s"])
    out = [r for r in out if r["s"]]
    log(f"  MSCI EM (Korea/Taiwan): {len(out)} Titel ({len(miss)} Börsenplatz geprüft, {dropped} ohne Kurse verworfen)")
    return out


def build_holdings(uni_rows: list[dict], fund: dict):
    cfg = json.loads((ROOT / "manual" / "etfs.json").read_text(encoding="utf-8"))
    keys: dict[str, str] = {}
    first: dict[str, list[str]] = {}
    for r in uni_rows:
        n = (fund.get(r["s"]) or {}).get("n") or r.get("n")
        k = name_key(n)
        if k:
            keys.setdefault(k, r["s"])
            first.setdefault(k.split()[0], []).append(r["s"])
    by_sym: dict[str, list] = {}
    tops = {}
    for e in cfg.get("etfs", []):
        src = e.get("holdings") or {}
        if src.get("index"):
            for r in uni_rows:
                if src["index"] in (r.get("src") or []):
                    by_sym.setdefault(r["s"], []).append([e["isin"], None])
            continue
        if not src.get("anbieter"):
            continue
        rows = safe(lambda: read_holdings(e), [], f"Positionen {e['ticker']}")
        if not rows:
            continue
        tops[e["isin"]] = [[n, rnd(w, 2)] for n, w in sorted(rows, key=lambda x: -x[1])[:10]]
        agg: dict[str, float] = {}
        hit = 0
        for n, w in rows:
            k = name_key(n)
            sym = keys.get(k)
            if not sym and k:
                cand = first.get(k.split()[0], [])
                sym = cand[0] if len(cand) == 1 else None
            if sym:
                agg[sym] = agg.get(sym, 0) + w
                hit += 1
        for sym, w in agg.items():
            by_sym.setdefault(sym, []).append([e["isin"], rnd(w, 3)])
        log(f"  Positionen {e['ticker']}: {len(rows)} Titel, {hit} im Universum zugeordnet")
    save(DATA / "holdings.json", {"updated": NOW.isoformat(), "bySymbol": by_sym, "top": tops})


# ----------------------------------------------------------------- Langfrist-Historie (Beratung)
def build_long_history():
    idx = CFG.get("long_indices", {})
    # Eintrag: "SYMBOL": "Name"  oder  "SCHLÜSSEL": {"name": …, "try": [Kandidaten]} – genommen wird die längste Reihe
    cand = {k: (v.get("try") if isinstance(v, dict) else [k]) for k, v in idx.items()}
    data = history_block([s for c in cand.values() for s in c], period="max", interval="1mo", chunk=20)
    out = {}
    for key, v in idx.items():
        name = v["name"] if isinstance(v, dict) else v
        best = max((data.get(s) for s in cand[key] if data.get(s) is not None), key=len, default=None)
        if best is None or len(best) < 24:
            log(f"  Langfrist: {name} ohne Daten")
            continue
        src = next(s for s in cand[key] if data.get(s) is best)
        log(f"  Langfrist: {name} aus {src} ab {best.index[0].date()}")
        out[key] = {"name": name, "src": src, "m": series_pairs(best, 2)}
    save(DATA / "long.json", {"updated": NOW.isoformat(), "series": out})
    log(f"Langfrist-Historie: {len(out)} Reihen")

# ----------------------------------------------------------------- Marktupdate
def build_marktupdate() -> list[dict]:
    """Entschlüsselt manual/marktupdate/*.enc (Secret MKT_PRIVATE_KEY) und veröffentlicht sie verschlüsselt."""
    files = sorted((ROOT / "manual" / "marktupdate").glob("*.enc"))
    if not files:
        return []
    pem = os.environ.get("MKT_PRIVATE_KEY", "").strip()
    if not pem:
        log("Marktupdate-Dateien vorhanden, aber das Secret MKT_PRIVATE_KEY fehlt – werden übersprungen")
        return []
    from cryptography.hazmat.primitives import hashes as _h, serialization as _ser
    from cryptography.hazmat.primitives.asymmetric import padding as _pad
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AES

    priv = _ser.load_pem_private_key(pem.encode(), None)
    eds = []
    for f in files:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            key = priv.decrypt(base64.b64decode(j["ek"]),
                               _pad.OAEP(mgf=_pad.MGF1(_h.SHA256()), algorithm=_h.SHA256(), label=None))
            eds.append(json.loads(_AES(key).decrypt(base64.b64decode(j["iv"]), base64.b64decode(j["data"]), None)))
        except Exception as e:  # noqa: BLE001
            log(f"WARNUNG Marktupdate {f.name}: {e}")
    eds.sort(key=lambda e: str(e.get("datum", "")), reverse=True)
    save(DATA / "marktupdate.json", {"updated": NOW.isoformat(), "editions": eds})
    log(f"Marktupdate: {len(eds)} Ausgaben")
    return eds


# ----------------------------------------------------------------- Ticker-Details
CUR_FIX = {"GBp": ("GBP", 100), "ZAc": ("ZAR", 100), "ILA": ("ILS", 100)}


def build_ticker(sym: str, edgar: "Edgar", closes_1y: pd.Series | None, sent: "Sentiment", google=True):
    """Detailseite + Kurzprofil (für Screener/Kalender) eines Titels."""
    t = yf.Ticker(sym)
    info = t.info or {}
    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None and not info.get("longName")):
        raise ValueError("keine Daten bei Yahoo")
    hist10 = safe(lambda: t.history(period="10y", interval="1wk", auto_adjust=False)["Close"].dropna(), None, f"10y {sym}")
    hist5 = hist10[hist10.index >= hist10.index[-1] - pd.Timedelta(days=5 * 365 + 2)] if hist10 is not None and len(hist10) else None
    hist1 = closes_1y if closes_1y is not None and len(closes_1y) > 5 else \
        safe(lambda: t.history(period="1y", auto_adjust=False)["Close"].dropna(), None, f"1y {sym}")

    is_us = "." not in sym and info.get("country") in (None, "United States")
    edgar.last_gaap = None
    fin = safe(lambda: edgar.financials(sym), None, f"SEC {sym}") if is_us else None
    quarterly = None
    if fin and getattr(edgar, "last_gaap", None) and edgar.last_gaap[0] == sym:
        quarterly = safe(lambda: edgar_quarterly(edgar.last_gaap[1]), None, f"SEC-Quartale {sym}")
    if not fin:
        fin = safe(lambda: yahoo_financials(t), None, f"Yahoo-Bilanz {sym}")
    if not quarterly:
        quarterly = safe(lambda: yahoo_quarterly(t), None, f"Yahoo-Quartale {sym}")
    surprises = safe(lambda: earnings_surprises(t), [], f"Gewinnüberraschungen {sym}")

    def g(*keys):
        for k in keys:
            if info.get(k) is not None:
                return info[k]
        return None

    price = float(hist1.iloc[-1]) if hist1 is not None and len(hist1) else g("currentPrice", "regularMarketPrice")
    prev = float(hist1.iloc[-2]) if hist1 is not None and len(hist1) > 1 else g("previousClose")
    dy = None
    if g("dividendRate") and price:
        dy = g("dividendRate") / price * 100
    elif g("trailingAnnualDividendYield") is not None:
        dy = g("trailingAnnualDividendYield") * 100
    stats = {
        "Marktkap.": g("marketCap"), "Unternehmenswert": g("enterpriseValue"),
        "KGV (ttm)": g("trailingPE"), "KGV (erw.)": g("forwardPE"), "KBV": g("priceToBook"),
        "KUV": g("priceToSalesTrailing12Months"), "EV/EBITDA": g("enterpriseToEbitda"),
        "PEG": g("trailingPegRatio", "pegRatio"),
        "Div.-Rendite %": rnd(dy, 2), "Ausschüttungsquote": g("payoutRatio"),
        "Beta": g("beta"), "EPS (ttm)": g("trailingEps"), "EPS (erw.)": g("forwardEps"),
        "Bruttomarge": g("grossMargins"), "Op. Marge": g("operatingMargins"), "Nettomarge": g("profitMargins"),
        "EK-Rendite": g("returnOnEquity"), "Verschuldung/EK": g("debtToEquity"),
        "52W Hoch": g("fiftyTwoWeekHigh"), "52W Tief": g("fiftyTwoWeekLow"),
        "Ø Volumen": g("averageVolume"), "Aktien ausst.": g("sharesOutstanding"),
    }
    analyst = {"rating": g("recommendationKey"), "n": g("numberOfAnalystOpinions"),
               "target": g("targetMeanPrice"), "high": g("targetHighPrice"), "low": g("targetLowPrice")}
    name = g("longName", "shortName") or sym
    news = safe(lambda: yahoo_news(t, sym, CFG.get("news_per_ticker", 8)), [], f"Yahoo-News {sym}")
    if google:
        news += safe(lambda: google_news(f'"{name.split(",")[0]}" stock', 6, sym), [], f"Google-News {sym}")
    news = dedupe(news)
    for n, sc in zip(news, sent.score([n["title"] for n in news])):
        n["sentiment"] = sc
    agg = aggregate_sentiment(news)
    earn = g("earningsTimestampStart", "earningsTimestamp")

    detail = {
        "symbol": sym, "name": name, "currency": g("currency"), "exchange": g("fullExchangeName", "exchange"),
        "sector": sector_de(g("sector")), "industry": g("industry"), "country": g("country"),
        "website": g("website"), "employees": g("fullTimeEmployees"), "description": g("longBusinessSummary"),
        "price": rnd(price), "chg": rnd(price - prev) if price and prev else None,
        "chgPct": rnd((price / prev - 1) * 100, 3) if price and prev else None,
        "stats": stats, "analyst": analyst,
        "dates": {"earnings": iso_date(earn), "exDividend": iso_date(g("exDividendDate")),
                  "dividendPay": iso_date(g("dividendDate"))},
        "history1y": series_pairs(hist1), "history5y": series_pairs(hist5),
        "financials": fin, "quarterly": quarterly, "surprises": surprises,
        "news": news, "sentiment": agg, "updated": NOW.isoformat(),
    }
    prof = {"n": name, "sec": detail["sector"], "ind": g("industry"), "c": g("country"), "cur": g("currency"),
            "mcap": g("marketCap"), "pe": g("trailingPE"), "fpe": g("forwardPE"), "pb": g("priceToBook"),
            "dy": rnd(dy, 2), "roe": g("returnOnEquity"), "pm": g("profitMargins"), "beta": g("beta"),
            "er": iso_date(earn), "exd": iso_date(g("exDividendDate")), "divd": iso_date(g("dividendDate")),
            "rating": g("recommendationKey"), "tgt": g("targetMeanPrice"),
            "ni": g("netIncomeToCommon"), "eg": g("earningsGrowth", "earningsQuarterlyGrowth"), "rg": g("revenueGrowth"),
            "feg": (g("forwardEps") / g("trailingEps") - 1) if (g("forwardEps") and g("trailingEps") and g("trailingEps") > 0) else None,
            "rmean": g("recommendationMean"), "nan": g("numberOfAnalystOpinions"),
            "up": (g("targetMeanPrice") / price - 1) if (g("targetMeanPrice") and price) else None,
            "sent": agg["score"], "f": file_safe(sym), "upd": NOW.isoformat()}
    peh = safe(lambda: pe_history(fin, hist10, t.splits, g("trailingPE")), None, f"KGV-Historie {sym}")
    if peh:
        prof["peh"] = peh
        detail["peHistory"] = peh
    return detail, prof


def pe_history(fin: dict | None, wk: pd.Series | None, splits, pe_now) -> dict | None:
    """KGV je Geschäftsjahr: Jahresschlusskurs ÷ verwässerter Gewinn je Aktie (split-bereinigt)."""
    if not fin or wk is None or not len(wk) or not fin.get("eps"):
        return None
    idx = wk.index.tz_localize(None) if getattr(wk.index, "tz", None) is not None else wk.index
    wk = pd.Series(wk.values, index=idx)
    sp = []
    if splits is not None and len(splits):
        for d, r in splits.items():
            if r and r > 0:
                sp.append(((d.tz_localize(None) if getattr(d, "tzinfo", None) else d).to_pydatetime(), float(r)))
    filed = fin.get("epsFiled") or [None] * len(fin["years"])
    out = {}
    for y, e, fd in zip(fin["years"], fin["eps"], filed):
        if not e or e <= 0:
            continue
        px = wk[wk.index.year == int(y)]
        if not len(px):
            continue
        # SEC-Werte stehen so, wie sie zuletzt eingereicht wurden: nur spätere Splits herausrechnen.
        # Yahoo-Jahreszahlen sind bereits auf den heutigen Aktienstand umgestellt.
        cut = datetime.fromisoformat(fd) if fd else None
        adj = 1.0
        if cut:
            for d, r in sp:
                if d > cut:
                    adj *= r
        pe = float(px.iloc[-1]) / (e / adj)
        if 0 < pe <= 150:
            out[str(y)] = round(pe, 1)
    if not out:
        return None
    if pe_now and pe_now > 0:
        last = out[max(out)]
        if not 0.33 <= last / pe_now <= 3:  # Währungs- oder Split-Unstimmigkeit → lieber weglassen
            return None
    return out


def fx_to_eur(currencies: set[str]) -> dict[str, float]:
    """Wie viele Einheiten der Währung entsprechen 1 EUR."""
    need = {CUR_FIX.get(c, (c, 1))[0] for c in currencies if c} - {"EUR"}
    pairs = {f"EUR{c}=X": c for c in need}
    fx = {"EUR": 1.0}
    for sym, s in history_block(list(pairs), period="5d").items():
        fx[pairs[sym]] = float(s.iloc[-1])
    return fx


def to_eur(v, cur, fx):
    if v is None or not cur:
        return None
    base, div = CUR_FIX.get(cur, (cur, 1))
    rate = fx.get(base)
    return v / div / rate if rate else None


def build_calendar(fund: dict, watch: set[str], editions: list[dict] | None = None) -> dict:
    today = NOW.date()
    lo, hi = today - timedelta(days=3), today + timedelta(days=90)
    ev = []
    for c in CFG.get("central_banks", []):
        try:
            d = datetime.fromisoformat(c["date"]).date()
        except Exception:
            continue
        if lo <= d <= hi:
            ev.append({"d": c["date"], "t": "cb", "title": f'{c["bank"]}: {c.get("note", "Zinsentscheid")}', "bank": c["bank"]})
    for t in ((editions or [{}])[0].get("termine") or []) if editions else []:
        try:
            d = datetime.fromisoformat(str(t.get("d"))[:10]).date()
        except Exception:
            continue
        if lo <= d <= hi:
            ev.append({"d": d.isoformat(), "t": "mkt", "title": str(t.get("title") or "")})
    for s, f in fund.items():
        for key, typ in (("er", "earn"), ("exd", "exdiv"), ("divd", "div")):
            dv = f.get(key)
            if not dv:
                continue
            try:
                d = datetime.fromisoformat(dv).date()
            except Exception:
                continue
            if lo <= d <= hi:
                ev.append({"d": dv, "t": typ, "sym": s, "name": f.get("n") or s, "mcap": f.get("mcapEUR"),
                           "dy": f.get("dy"), "wl": s in watch})
    ev.sort(key=lambda e: (e["d"], e["t"] != "cb", -(e.get("mcap") or 0)))
    return {"updated": NOW.isoformat(), "events": ev}


# ----------------------------------------------------------------- main
def elapsed_min():
    return (time.time() - T0) / 60


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    TICK.mkdir(exist_ok=True)
    log(f"Modus: {MODE} · Verschlüsselung: {'AN' if PASSWORD else 'AUS (Secret TERMINAL_PASSWORD fehlt)'} · Ziel: {DATA}")
    wl, idx, mkt = CFG["watchlist"], CFG["indices"], CFG["markets"]
    watch = set(wl)
    sent = Sentiment()
    edgar = Edgar()

    # --- gespeicherter Zustand aus dem letzten Lauf
    uni_list = (load(DATA / "universe_list.json", {}) or {}).get("rows", [])
    fund: dict = load(DATA / "fundamentals.json", {}) or {}
    refs: dict = load(DATA / "refs.json", {}) or {}

    # --- Nachtlauf: Indexlisten, Jahreskurse, rotierende Detailseiten
    closes_uni: dict[str, pd.Series] = {}
    if MODE == "nightly" or not uni_list:
        log("Indexlisten laden …")
        uni_list = build_universe_list(uni_list)
        save(DATA / "universe_list.json", {"updated": NOW.isoformat(), "rows": uni_list})
    uni_syms = [r["s"] for r in uni_list]
    log(f"Universum: {len(uni_syms)} Titel")

    if MODE == "nightly" or not refs:
        log("Jahreskurse des Universums …")
        closes_uni = history_block(uni_syms, period="1y")
        refs = {}
        for s, c in closes_uni.items():
            yb = c[c.index.year < NOW.year]
            refs[s] = {"yb": rnd(float(yb.iloc[-1])) if len(yb) else None, "y1b": rnd(float(c.iloc[0])),
                       "hi": rnd(float(c.max())), "lo": rnd(float(c.min())),
                       "m1b": rnd(float(c.iloc[-22])) if len(c) > 22 else None}
        save(DATA / "refs.json", refs)

    if MODE == "nightly":
        budget = float(CFG.get("nightly_minutes", 90))
        batch = int(CFG.get("nightly_batch", 600))
        known = [s for s in uni_syms if s in refs and s not in watch]
        todo = [s for s in known if s not in fund] + sorted([s for s in known if s in fund],
                                                           key=lambda s: fund[s].get("upd") or "")
        log(f"Detailseiten: bis zu {batch} von {len(todo)} offenen Titeln (Zeitbudget {budget:.0f} Min.)")
        done = fails = 0
        for s in todo[:batch]:
            if elapsed_min() > budget:
                log("Zeitbudget erreicht – Rest folgt in der nächsten Nacht")
                break
            try:
                d, p = build_ticker(s, edgar, closes_uni.get(s), sent, google=False)
                save(TICK / f"{file_safe(s)}.json", d, quiet=True)
                fund[s] = p
                done += 1
                fails = 0
            except Exception as e:  # noqa: BLE001
                fails += 1
                fund.setdefault(s, {})["upd"] = NOW.isoformat()  # nicht sofort erneut versuchen
                if fails >= 25:
                    log("Viele Fehler hintereinander (Yahoo-Sperre?) – Abbruch für heute:", str(e)[:120])
                    break
            if done and done % 50 == 0:
                log(f"  {done} Detailseiten fertig ({elapsed_min():.0f} Min.)")
                save(DATA / "fundamentals.json", fund, quiet=True)
            time.sleep(0.4)
        log(f"Detailseiten erstellt: {done}")

    # --- Indizes, Märkte, Watchlist
    log("Kurse Indizes/Märkte/Watchlist …")
    closes = history_block(list(idx) + list(mkt) + wl, period="1y")
    w5 = history_block(list(idx) + list(mkt), period="5y", interval="1wk")
    details = {}
    for s in wl:
        log("Watchlist:", s)
        r = safe(lambda: build_ticker(s, edgar, closes.get(s), sent, google=True), None, f"Ticker {s}")
        if r:
            details[s], fund[s] = r
            save(TICK / f"{file_safe(s)}.json", r[0], quiet=True)

    market = {
        "updated": NOW.isoformat(),
        "indices": [quote_row(s, n, closes.get(s)) for s, n in idx.items()],
        "markets": [quote_row(s, n, closes.get(s)) for s, n in mkt.items()],
        "watchlist": [quote_row(s, (details.get(s) or {}).get("name") or (fund.get(s) or {}).get("n") or s, closes.get(s), {
            "currency": (fund.get(s) or {}).get("cur"), "sector": (fund.get(s) or {}).get("sec"),
            "pe": (fund.get(s) or {}).get("pe"), "mcap": (fund.get(s) or {}).get("mcap"),
            "sentiment": (fund.get(s) or {}).get("sent")}) for s in wl],
    }
    save(DATA / "market.json", market)
    save(DATA / "history.json", {"updated": NOW.isoformat(), "series": {
        s: {"name": n, "d": series_pairs(closes.get(s)), "w": series_pairs(w5.get(s))}
        for s, n in {**idx, **mkt}.items()}})

    # --- Kurse des gesamten Universums
    log("Tageskurse Universum …")
    last = closes_uni if closes_uni else history_block(uni_syms, period="5d")
    fx = safe(lambda: fx_to_eur({(fund.get(s) or {}).get("cur") for s in uni_syms} | {"USD", "GBP", "CHF", "JPY"}),
              {"EUR": 1.0}, "Devisenkurse")
    rows = []
    meta_map = {}
    for u in uni_list:
        s = u["s"]
        c = last.get(s)
        f = fund.get(s) or {}
        rf = refs.get(s) or {}
        if c is None or len(c) < 1:
            continue
        px = float(c.iloc[-1])
        prev = float(c.iloc[-2]) if len(c) > 1 else None
        if f.get("mcap"):
            f["mcapEUR"] = rnd(to_eur(f["mcap"], f.get("cur"), fx), 0)
        det = (TICK / f"{f.get('f') or file_safe(s)}.json").exists()
        if det:
            meta_map[s] = f.get("f") or file_safe(s)
        rows.append({
            "s": s, "n": f.get("n") or u.get("n") or s, "sec": f.get("sec") or u.get("sec"), "c": f.get("c") or u.get("c"),
            "cur": f.get("cur"), "px": rnd(px), "chg": rnd((px / prev - 1) * 100, 2) if prev else None,
            "m1": rnd((px / rf["m1b"] - 1) * 100, 1) if rf.get("m1b") else None,
            "ytd": rnd((px / rf["yb"] - 1) * 100, 1) if rf.get("yb") else None,
            "y1": rnd((px / rf["y1b"] - 1) * 100, 1) if rf.get("y1b") else None,
            "hi": rf.get("hi"), "mcap": f.get("mcapEUR"), "pe": rnd(f.get("pe"), 1), "fpe": rnd(f.get("fpe"), 1),
            "pb": rnd(f.get("pb"), 2), "dy": f.get("dy"), "roe": rnd((f.get("roe") or 0) * 100, 1) if f.get("roe") is not None else None,
            "beta": rnd(f.get("beta"), 2), "sent": f.get("sent"), "rating": f.get("rating"),
            "ind": f.get("ind"), "ni": rnd(to_eur(f.get("ni"), f.get("cur"), fx), 0) if f.get("ni") is not None else None,
            "eg": rnd((f.get("eg") or 0) * 100, 1) if f.get("eg") is not None else None,
            "rg": rnd((f.get("rg") or 0) * 100, 1) if f.get("rg") is not None else None,
            "feg": rnd((f.get("feg") or 0) * 100, 1) if f.get("feg") is not None else None,
            "up": rnd((f["tgt"] / px - 1) * 100, 1) if f.get("tgt") and px else None,
            "rmean": rnd(f.get("rmean"), 2), "nan": f.get("nan"), "peh": f.get("peh"),
            "pm": rnd((f.get("pm") or 0) * 100, 1) if f.get("pm") is not None else None,
            "er": f.get("er"), "src": u.get("src", []), "det": det, "wl": s in watch,
        })
    for s in wl:
        if (TICK / f"{file_safe(s)}.json").exists():
            meta_map[s] = file_safe(s)
    save(DATA / "universe.json", {"updated": NOW.isoformat(), "n": len(rows), "rows": rows,
                                  "sources": sorted({x for r in rows for x in r["src"]})})
    save(DATA / "fundamentals.json", fund, quiet=True)
    editions = safe(build_marktupdate, [], "Marktupdate") or []
    save(DATA / "calendar.json", build_calendar(fund, watch, editions))

    # --- Makro (nachts oder wenn noch nicht vorhanden)
    macro_old = load(DATA / "macro.json", None)
    fred_missing = bool(os.environ.get("FRED_API_KEY")) and macro_old is not None and \
        not any(x.get("source") == "FRED" for x in macro_old.get("series", []))
    if MODE == "nightly" or macro_old is None or fred_missing:
        log("Makro laden …")
        safe(build_macro, None, "Makro")
    safe(lambda: merge_manual_macro(editions), None, "PMI-Zusammenführung")

    safe(build_bonds, None, "Bundesanleihen")
    safe(build_etfs, None, "ETFs")
    if MODE == "nightly" or not set(CFG.get("long_indices", {})) <= set((load(DATA / "long.json") or {}).get("series", {})):
        safe(build_long_history, None, "Langfrist-Historie")
    if MODE == "nightly" or not (DATA / "holdings.json").exists():
        safe(lambda: build_holdings(uni_list, fund), None, "ETF-Positionen")

    # --- News + Stimmung
    log("News + Stimmung …")
    general = []
    for q in CFG.get("news_queries", []):
        general += safe(lambda: google_news(q, 15), [], f"Google-News {q}")
    general = dedupe(general)
    for n, sc in zip(general, sent.score([n["title"] for n in general])):
        n["sentiment"] = sc
    by_date = lambda n: n.get("published") or ""  # noqa: E731
    summary = {s: d["sentiment"] for s, d in details.items()}
    save(DATA / "news.json", {
        "updated": NOW.isoformat(), "model": CFG.get("sentiment_model") if sent.pipe else None,
        "method": {"halfLifeDays": HALF_LIFE_DAYS, "windowDays": WINDOW_DAYS},
        "general": sorted(general, key=by_date, reverse=True),
        "tickers": sorted([n for d in details.values() for n in d["news"]], key=by_date, reverse=True)[:250],
        "summary": summary,
    })
    save(DATA / "meta.json", {"updated": NOW.isoformat(), "mode": MODE, "tickers": meta_map,
                              "universe": len(rows), "details": len(meta_map), "marktupdate": len(editions)})
    log(f"Fertig nach {elapsed_min():.1f} Min. · Universum {len(rows)} · Detailseiten {len(meta_map)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

