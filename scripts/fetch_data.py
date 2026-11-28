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


WARNINGS: list[dict] = []


def safe(fn, default=None, label=""):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        log(f"WARNUNG {label}: {str(e)[:200]}")
        WARNINGS.append({"bereich": label, "fehler": str(e)[:200]})
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


def write_briefing(market: dict, movers: list, calendar: dict, rows: list[dict]):
    """Kompakte Datei fürs Morgen-Briefing: ausschließlich öffentliche Marktdaten (keine Kunden-, keine RR-Inhalte).
    Liegt unverschlüsselt unter data/public/briefing.json, damit die geplante Claude-Aufgabe sie lesen kann."""
    today = NOW.astimezone(timezone(timedelta(hours=1))).date()
    days = {today.isoformat(), (today + timedelta(days=1)).isoformat()}
    by = {r["s"]: r for r in rows}
    ev = [e for e in (calendar or {}).get("events", []) if e.get("d") in days and
          (e.get("t") in ("cb", "mkt") or (e.get("t") == "earn" and (e.get("mcap") or 0) >= 2e10))]
    ra = (load(DATA / "analyst_recent.json", {}) or {}).get("items", [])
    cut = (today - timedelta(days=3)).isoformat()
    big = []
    for a in ra:
        r = by.get(a["s"]) or {}
        ch = (a["pt"] / a["ppt"] - 1) * 100 if a.get("pt") and a.get("ppt") else None
        if a.get("d", "") >= cut and (r.get("mcap") or 0) >= 2e10 and (a.get("act") in ("up", "down") or (ch is not None and abs(ch) >= 10)):
            big.append({**a, "n": a.get("n") or r.get("n"), "chgPt": rnd(ch, 1), "mcapEUR": r.get("mcap")})
    big.sort(key=lambda a: (a.get("d", ""), abs(a.get("chgPt") or 0)), reverse=True)
    out = {
        "updated": NOW.isoformat(), "hinweis": "Öffentliche Marktdaten aus dem Research-Terminal (Yahoo Finance u. a.), verzögert.",
        "indices": [{k: q.get(k) for k in ("sym", "name", "price", "chgPct", "ytd", "date")} for q in market.get("indices", [])],
        "markets": [{k: q.get(k) for k in ("sym", "name", "price", "chgPct", "ytd", "date")} for q in market.get("markets", [])],
        "movers": [{"s": m["s"], "n": m.get("n"), "chg": m.get("chg"), "dir": m.get("dir"),
                    "news": [{"title": x["title"], "source": x.get("source"), "link": x.get("link")} for x in (m.get("news") or [])[:3]]}
                   for m in movers or []],
        "termine": ev[:25],
        "analysten": big[:15],
    }
    path = DATA / "public" / "briefing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  Briefing: {len(out['termine'])} Termine, {len(out['analysten'])} Analystenmeldungen")


def mover_news(rows: list[dict], sent: "Sentiment", k=3, min_mcap=5e9) -> list[dict]:
    """Die je 3 stärksten Gewinner und Verlierer des Tages (ab 5 Mrd. € Marktwert) mit ihren Schlagzeilen."""
    cand = [r for r in rows if r.get("chg") is not None and (r.get("mcap") or 0) >= min_mcap]
    cand.sort(key=lambda r: r["chg"])
    pick = [("up", r) for r in reversed(cand[-k:]) if r["chg"] > 0] + [("down", r) for r in cand[:k] if r["chg"] < 0]
    out = []
    for d, r in pick:
        items = safe(lambda: yahoo_news(yf.Ticker(r["s"]), r["s"], 6), [], f"Mover-News {r['s']}")
        if not items:
            items = safe(lambda: google_news(f'"{r.get("n") or r["s"]}" stock', 5, r["s"]), [], f"Mover-Google {r['s']}")
            for it in items:
                it["ticker"] = r["s"]
        items = dedupe(items)[:6]
        for it, sc in zip(items, sent.score([it["title"] for it in items]) if items else []):
            it["sentiment"] = sc
        out.append({"s": r["s"], "n": r.get("n"), "chg": r["chg"], "px": r.get("px"), "cur": r.get("cur"),
                    "dir": d, "news": sorted(items, key=lambda n: n.get("published") or "", reverse=True)})
    log(f"  Top-Mover-News: {len(out)} Titel")
    return out


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
    tables = None
    for attempt, u in enumerate([url, url + ("&" if "?" in url else "?") + "action=render"]):
        try:
            tables = pd.read_html(io.StringIO(HTTP.get(u, timeout=60).text))
            break
        except ValueError:  # Wikipedia liefert bei Last gelegentlich eine Seite ohne Tabellen
            if attempt == 0:
                time.sleep(6)
    if tables is None:
        raise ValueError("No tables found")
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


_REGIONAL = (".F", ".DU", ".MU", ".BE", ".SG", ".HM", ".HA", ".VI", ".MX", ".BA", ".SN", ".NE", ".CN", ".TI", ".IL")


def isin_to_yahoo(isin: str, ex: str = "") -> str | None:
    """Yahoo-Symbol zu einer ISIN über die Yahoo-Suche; bevorzugt den Heimatbörsenplatz."""
    try:
        q = [x for x in (yf.Search(isin, max_results=8).quotes or []) if x.get("quoteType") in ("EQUITY", "ETF", "MUTUALFUND")]
    except Exception:
        return None
    if not q:
        return None
    syms = [x["symbol"] for x in q]
    want = next((suf for k, suf in EXCHANGE_SUFFIX if k in (ex or "").lower()), None)
    if want is not None:
        hit = next((s_ for s_ in syms if (s_.endswith(want) if want else "." not in s_)), None)
        if hit:
            return hit
    if isin.startswith("US"):
        hit = next((s_ for s_ in syms if "." not in s_), None)
        if hit:
            return hit
    return next((s_ for s_ in syms if not s_.endswith(_REGIONAL)), syms[0])


_UNI_SEARCH = [0]


def xtrackers_universe(src: dict, have: set[str], have_isin: set[str]) -> list[dict]:
    """Zusätzliche Titel aus einer Xtrackers-Positionsliste (z. B. MSCI World), Symbol per ISIN-Suche (Cache)."""
    rows = read_holdings_full({"isin": src["isin"], "ticker": "", "holdings": {"anbieter": "xtrackers"}})
    cache = load(DATA / "isin_yahoo.json") or {}
    out, new = [], 0
    for r in sorted(rows, key=lambda x: -x["w"]):
        isin = (r.get("isin") or "").upper()
        if r["w"] < src.get("min_w", 0) or not isin_ok(isin) or isin in have_isin:
            continue
        c = cache.get(isin)
        if c is None or (not c[0] and (NOW - datetime.fromisoformat(c[1])).days > 60):
            if _UNI_SEARCH[0] <= 0:
                continue
            _UNI_SEARCH[0] -= 1
            c = [isin_to_yahoo(isin, r.get("ex")) or "", NOW.isoformat()]
            cache[isin] = c
            new += 1
            time.sleep(0.2)
        if c[0] and c[0] not in have:
            out.append({"s": c[0], "n": nice_name(r["name"]), "sec": None, "c": None, "src": [src["name"]], "i": isin})
    save(DATA / "isin_yahoo.json", cache, quiet=True)
    log(f"  {src['name']} (Xtrackers): {len(out)} zusätzliche Titel ({new} ISINs neu gesucht)")
    return out


_INV_ISIN: dict[str, str] = {}


def build_universe_list(old: list[dict]) -> list[dict]:
    uc = CFG.get("universe", {})
    found: dict[str, dict] = {}

    def add(items):
        for it in items:
            cur = found.get(it["s"])
            if cur:
                cur["src"] = sorted(set(cur["src"]) | set(it["src"]))
                for k in ("n", "sec", "c", "i"):
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
    _UNI_SEARCH[0] = int(uc.get("isin_search_per_night", 700))
    have_isin = set((load(DATA / "holdings.json") or {}).get("isin", {}).keys())
    for src in uc.get("xtrackers", []):
        add(with_fallback(lambda: xtrackers_universe(src, set(found), have_isin), src["name"]))
    add([{"s": s, "n": None, "sec": None, "c": None, "src": ["ATX"]} for s in uc.get("extra", [])])
    add([{"s": s, "n": None, "sec": None, "c": None, "src": ["Watchlist"]} for s in CFG.get("watchlist", [])])
    if len(found) < 100 and old:  # noqa: SIM102
        log("Indexlisten lieferten zu wenig Titel – vorherige Liste wird weiterverwendet")
        add(old)
    return sorted(found.values(), key=lambda x: x["s"])


# ----------------------------------------------------------------- Makro
FRED = [
    # id, Name, Einheit, Transformation, Gruppe, Referenzlinie  – bewusst schlank gehalten
    ("FEDFUNDS", "Fed Funds Rate", "%", None, "Zinsen USA", None),
    ("DGS10", "US-Staatsanleihe 10J", "%", None, "Zinsen USA", None),
    ("CPIAUCSL", "US-Inflation (VPI)", "% J/J", "yoy", "Inflation", 2),
    ("PCEPILFE", "US-Kerninflation (PCE, Fed-Zielgröße)", "% J/J", "yoy", "Inflation", 2),
    ("UNRATE", "US-Arbeitslosenquote", "%", None, "Arbeitsmarkt", None),
    ("A191RL1Q225SBEA", "US-Wirtschaftswachstum (BIP, annualisiert)", "% Q/Q", None, "Wachstum", 0),
    # Risikoaufschläge ggü. Staatsanleihen (ICE BofA, optionsbereinigt)
    ("BAMLC0A0CM", "Aufschlag Unternehmensanleihen USA (Investment Grade)", "%-Pkt.", None, "Kreditaufschläge", None),
    ("BAMLH0A0HYM2", "Aufschlag Hochzinsanleihen USA (High Yield)", "%-Pkt.", None, "Kreditaufschläge", None),
    ("BAMLHE00EHYIOAS", "Aufschlag Hochzinsanleihen Euro (High Yield)", "%-Pkt.", None, "Kreditaufschläge", None),
]
ECB = [
    ("FM/D.U2.EUR.4F.KR.DFR.LEV", "EZB-Einlagensatz", "%", "Zinsen Euroraum"),
    ("FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", "3M-Euribor", "%", "Zinsen Euroraum"),
    ("IRS/M.AT.L.L40.CI.0000.EUR.N.Z", "Österreich Staatsanleihe 10J", "%", "Staatsanleihen"),
    ("IRS/M.DE.L.L40.CI.0000.EUR.N.Z", "Deutschland Bund 10J", "%", "Staatsanleihen"),
    ("ICP/M.U2.N.000000.4.ANR", "Euroraum-Inflation (HVPI)", "% J/J", "Inflation"),
    ("ICP/M.AT.N.000000.4.ANR", "Österreich-Inflation (HVPI)", "% J/J", "Inflation"),
    ("LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T", "Euroraum-Arbeitslosenquote", "%", "Arbeitsmarkt"),
]
# EU-Kommission (Eurostat, Datensatz ei_bssi_m_r2): Wirtschaftsstimmung als Ergänzung zu den PMIs
EUROSTAT = [
    ("BS-ESI-I", "Wirtschaftsstimmung ESI", "Index", 100),
]
EUROSTAT_GEO = [("EA20", "Euroraum"), ("AT", "Österreich")]


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
                                    "region": ("Euroraum" if "Euro " in name else "USA"), "source": "FRED", "ref": ref})
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
                                           "unit": "Bp.", "group": "Staatsanleihen", "region": "Österreich",
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
                                        "group": "Stimmung", "region": region, "source": "EU-Kommission", "ref": ref})
            r = safe(f, None, f"Eurostat {indic} {geo}")
            if r:
                out.append(r)
    old = (load(DATA / "macro.json", {}) or {}).get("series", [])
    have = {s["id"] for s in out}
    allowed = ({f[0] for f in FRED} | {e[0] for e in ECB} | {"SPREAD_AT_DE"}
               | {f"ESTAT:{i[0]}:{g}" for i in EUROSTAT for g, _ in EUROSTAT_GEO})
    kept = [s for s in old if s.get("id") in allowed and s.get("id") not in have]
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
    for pg in range(1, 2):  # nur die erste Seite: robots.txt der Wiener Börse sperrt "-page="-Parameter (alle ~35 Bundesanleihen passen darauf)
        html = HTTP.get(url, timeout=40).text
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


# ----------------------------------------------------------------- Raiffeisen Zertifikate (Anlageprodukte)
RZ_BASE = "https://www.raiffeisenzertifikate.at"
RZ_LABELS = ["Basiswert", "Kapitalschutzbetrag", "Partizipationsfaktor", "Cap", "Fixer Zinssatz jährlich", "Zinssatz Gesamtlaufzeit",
             "Startwert", "Basispreis", "Barriere", "Abstand zur Barriere", "Barriere verletzt", "Beobachtung Barriere", "Bonuslevel",
             "Bonusbetrag", "Erster Bewertungstag", "Emissionstag", "Letzter Bewertungstag", "Rückzahlungstermin", "Produkt-Klassifikation",
             "Wertpapier-Typ (Eusipa-Nr.)", "Land / Region Basiswert", "Geeignete Markterwartung", "Währung Produkt", "Rückzahlungsart",
             "Max. Rendite p.a.", "Max. Rendite p.a. bis zum nächsten Bewertungstag", "Emissionspreis", "Max. Rendite Restlaufzeit (%)",
             "Max. Rendite bis zum nächsten Bewertungstag", "Discount (%)", "Nachhaltigkeitspräferenz", "Abstand zum Basispreis",
             "Abstand zum nächsten Auszahlungslevel"]


def _html_lines(page: str) -> list[str]:
    import html as _h
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    return [x.strip() for x in _h.unescape(re.sub(r"<[^>]+>", "\n", t)).split("\n") if x.strip()]


def rz_detail(isin: str) -> dict:
    """Eckdaten einer Produktseite (/zertifikat/{ISIN} leitet auf die Detailseite weiter)."""
    L = _html_lines(HTTP.get(f"{RZ_BASE}/zertifikat/{isin}", timeout=60).text)
    i = L.index("Eckdaten") if "Eckdaten" in L else 0
    L = L[i:i + 600]
    out = {}
    for k, ln in enumerate(L):
        if ln in RZ_LABELS and ln not in out:
            for v in L[k + 1:k + 6]:
                if v == ln or v == "Schliessen" or (len(v) > 45 and v.endswith(".")):
                    continue
                out[ln] = v
                break
    return out


def build_certificates():
    """Alle Anlageprodukte von Raiffeisen Zertifikate (EUR): Liste nachts vollständig, Eckdaten je ISIN rotierend (Cache 30 Tage)."""
    import html as _h
    old = load(DATA / "zertifikate.json") or {}
    det = load(DATA / "rz_details.json") or {}
    rows: dict[str, dict] = {}
    if MODE == "nightly" or not old.get("rows"):
        start = HTTP.get(f"{RZ_BASE}/unsere-zertifikate/anlageprodukte", timeout=60).text
        links: dict[int, str] = {}
        harvest = lambda pg: [links.setdefault(int(n), _h.unescape(u)) for u, n in re.findall(r'href="([^"]*c5081-page=(\d+)[^"]*)"', pg)]  # noqa: E731
        harvest(start)
        seen: set[int] = set()
        while len(seen) < 120:
            todo = [n for n in sorted(links) if n not in seen]
            if not todo:
                break
            n = todo[0]
            seen.add(n)
            try:
                pg = HTTP.get(RZ_BASE + links[n], timeout=60).text
            except Exception as ex:
                log(f"  Zertifikate Seite {n}: {ex}")
                continue
            harvest(pg)
            for tr in pg.split('<tr data-key')[1:]:
                mi = re.search(r'class="sxp-no-ajax anchor isin-link" href="([^"]+)">([A-Z0-9]{12})<', tr)
                if not mi:
                    continue
                tds = [re.sub(r"\s+", " ", _h.unescape(re.sub(r"<[^>]+>", " ", x))).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
                pct = re.findall(r"(\d[\d\.]*,\d+)\s*%", tds[4] if len(tds) > 4 else "")
                mpdf = re.search(r'href="(/pdf/[^"]+)"', tr)
                rows[mi.group(2)] = {"isin": mi.group(2), "url": RZ_BASE + mi.group(1), "n": tds[1] if len(tds) > 1 else "",
                                     "pdf": RZ_BASE + _h.unescape(mpdf.group(1)) if mpdf else None,
                                     "typ": tds[2] if len(tds) > 2 else "", "faellig": tds[3] if len(tds) > 3 else "",
                                     "geld": float(pct[0].replace(".", "").replace(",", ".")) if pct else None,
                                     "brief": float(pct[1].replace(".", "").replace(",", ".")) if len(pct) > 1 else None,
                                     "cur": tds[5] if len(tds) > 5 else ""}
            time.sleep(0.4)
        log(f"  Zertifikate: {len(rows)} Anlageprodukte aus {len(seen)} Listenseiten")
    if not rows:
        rows = {r["isin"]: r for r in old.get("rows", [])}
    # Fällige Produkte entfernen (falls die Liste sie noch kurz führt); Eckdaten nicht mehr gelisteter Produkte verwerfen
    def _matured(r):
        m = re.match(r"(\d\d)\.(\d\d)\.(\d{4})$", r.get("faellig") or "")
        return bool(m) and datetime(int(m[3]), int(m[2]), int(m[1]), tzinfo=NOW.tzinfo) < NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    n0 = len(rows)
    rows = {k: r for k, r in rows.items() if not _matured(r)}
    det = {k: v for k, v in det.items() if k in rows}
    first = {r["isin"]: r.get("seit") for r in old.get("rows", [])}
    for k, r in rows.items():
        r["seit"] = first[k] if k in first else (NOW.date().isoformat() if old.get("rows") else None)  # None = schon vor Einführung gelistet
    if n0 != len(rows):
        log(f"  Zertifikate: {n0 - len(rows)} fällige entfernt")
    left = 300 if MODE == "nightly" else 25
    for isin in sorted(rows, key=lambda x: det.get(x, {}).get("_d", "")):
        d = det.get(isin) or {}
        if d.get("_d") and (NOW - datetime.fromisoformat(d["_d"])).days < 7:  # Renditen ändern sich mit dem Kurs: wöchentlich auffrischen
            continue
        if left <= 0:
            break
        left -= 1
        try:
            det[isin] = {**rz_detail(isin), "_d": NOW.isoformat()}
        except Exception as ex:
            log(f"  Zertifikat {isin}: {ex}")
        time.sleep(0.6)
    num = lambda v: float(re.sub(r"[^\d,\-]", "", v).replace(",", ".")) if v and re.search(r"\d", v) else None  # noqa: E731
    out = []
    for isin, r in rows.items():
        d = det.get(isin) or {}
        kl = d.get("Produkt-Klassifikation", "")
        out.append({**r, "basis": d.get("Basiswert"), "land": d.get("Land / Region Basiswert"), "klasse": kl,
                    "schutz": num(d.get("Kapitalschutzbetrag")), "kupon": num(d.get("Fixer Zinssatz jährlich")),
                    "barriere": d.get("Barriere"), "abst": num(d.get("Abstand zur Barriere")), "verletzt": d.get("Barriere verletzt"),
                    "cap": d.get("Cap"), "beob": d.get("Beobachtung Barriere"), "rueck": d.get("Rückzahlungsart"),
                    "eusipa": d.get("Wertpapier-Typ (Eusipa-Nr.)"), "markt": d.get("Geeignete Markterwartung"),
                    "maxp": num(d.get("Max. Rendite p.a.") or d.get("Max. Rendite p.a. bis zum nächsten Bewertungstag")),
                    "maxr": num(d.get("Max. Rendite Restlaufzeit (%)") or d.get("Max. Rendite bis zum nächsten Bewertungstag")),
                    "maxnext": bool(d.get("Max. Rendite bis zum nächsten Bewertungstag") and not d.get("Max. Rendite Restlaufzeit (%)")),
                    "disc": num(d.get("Discount (%)")), "pai": bool(d.get("Nachhaltigkeitspräferenz")),
                    "dstand": (d.get("_d") or "")[:10] or None,
                    "emission": d.get("Emissionstag"), "detail": bool(d)})
    save(DATA / "rz_details.json", det, quiet=True)
    save(DATA / "zertifikate.json", {"updated": NOW.isoformat(), "quelle": "raiffeisenzertifikate.at", "rows": out})
    log(f"Zertifikate: {len(out)}, mit Eckdaten {sum(1 for r in out if r['detail'])}")


# ----------------------------------------------------------------- Bank- & Wohnbauanleihen (Stammdaten über OpenFIGI)
def openfigi(isins: list[str], cache: dict) -> dict:
    """Stammdaten je ISIN über die öffentliche OpenFIGI-API (Emittent, Kupon, Fälligkeit); Cache 60 Tage."""
    todo = [i for i in isins if not (cache.get(i) and (NOW - datetime.fromisoformat(cache[i]["_d"])).days < 60)]
    for k in range(0, len(todo), 10):
        chunk = todo[k:k + 10]
        try:
            r = HTTP.post("https://api.openfigi.com/v3/mapping", json=[{"idType": "ID_ISIN", "idValue": i} for i in chunk], timeout=40)
            if r.status_code == 429:
                time.sleep(30)
                r = HTTP.post("https://api.openfigi.com/v3/mapping", json=[{"idType": "ID_ISIN", "idValue": i} for i in chunk], timeout=40)
            res = r.json()
        except Exception as ex:
            log(f"  OpenFIGI: {ex}")
            break
        for i, x in zip(chunk, res):
            d = (x.get("data") or [{}])[0] if isinstance(x, dict) else {}
            cache[i] = {"name": d.get("name"), "ticker": d.get("ticker"), "sektor": d.get("marketSector"), "typ": d.get("securityType2"),
                        "boerse": d.get("exchCode"), "_d": NOW.isoformat()}
        time.sleep(2.6)  # ohne API-Schlüssel max. 25 Anfragen pro Minute
    return cache


def parse_bond_ticker(t: str | None) -> tuple[float | None, str | None]:
    """'RFLBTI 2.9 09/07/32' -> (2.9, '2032-09-07')"""
    if not t:
        return None, None
    m = re.search(r"\s(\d+(?:\.\d+)?|\d+ \d/\d+|0)\s+(\d\d)/(\d\d)/(\d\d)\b", t)
    if not m:
        return None, None
    c = m.group(1)
    if " " in c:
        w, fr = c.split(" ")
        a, b = fr.split("/")
        c = float(w) + float(a) / float(b)
    yy = int(m.group(4))
    return float(c), f"{2000 + yy if yy < 70 else 1900 + yy}-{m.group(2)}-{m.group(3)}"


def build_bank_bonds():
    """Raiffeisen-, Bank- und Wohnbauanleihen: ISINs von den öffentlichen Emissionsseiten der Emittenten, Stammdaten über OpenFIGI."""
    srcs = CFG.get("anleihen_quellen", [])
    man = ROOT / "manual" / "anleihen.json"
    extra = json.loads(man.read_text(encoding="utf-8")).get("anleihen", []) if man.exists() else []
    old = {r["isin"]: r for r in (load(DATA / "anleihen.json") or {}).get("rows", [])}
    found: dict[str, dict] = {}
    for src in srcs:
        try:
            page = HTTP.get(src["url"], timeout=60).text
        except Exception as ex:
            log(f"  Anleihen-Quelle {src.get('emittent')}: {ex}")
            for i, r in old.items():
                if r.get("quelle") == src["url"]:
                    found.setdefault(i, r)
            continue
        lines = _html_lines(page)
        for k, ln in enumerate(lines):
            for isin in re.findall(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b", ln):
                if isin_ok(isin) and isin not in found:
                    txt = " ".join(lines[k:k + 2]) if len(ln) < 20 else ln
                    found[isin] = {"isin": isin, "emittent": src.get("emittent"), "art": src.get("art", "Bankanleihe"),
                                   "bez": re.sub(r"\s+", " ", txt)[:160], "quelle": src["url"]}
    for e in extra:
        found.setdefault(e["isin"], {"isin": e["isin"], "emittent": e.get("emittent"), "art": e.get("art", "Anleihe"),
                                     "bez": e.get("bez", ""), "quelle": "manual/anleihen.json"})
    cache = openfigi(sorted(found), load(DATA / "openfigi.json") or {})
    save(DATA / "openfigi.json", cache, quiet=True)
    rows = []
    for isin, r in found.items():
        f = cache.get(isin) or {}
        if f.get("sektor") and f["sektor"] not in ("Corp", "Govt", "Muni", "Mtge"):
            continue  # z. B. Fonds oder Aktien auf derselben Seite
        kup, fae = parse_bond_ticker(f.get("ticker"))
        if kup is None:
            m = re.search(r"(\d{1,2}(?:,\d{1,3})?)\s?%", r.get("bez", ""))
            kup = float(m.group(1).replace(",", ".")) if m else None
        stufen = bool(re.search(r"Stufenzins|\d\s?%\s?-\s?\d", r.get("bez", "")))
        var = bool(re.search(r"[Vv]ario|[Vv]ariabel", r.get("bez", "")))
        if fae and fae < NOW.strftime("%Y-%m-%d"):
            continue  # bereits getilgt
        rows.append({**r, "name": f.get("name") or r.get("emittent"), "ticker": f.get("ticker"), "kupon": kup, "faellig": fae,
                     "stufen": stufen, "variabel": var, "boerse": f.get("boerse"),
                     "wohnbau": "Wohnbau" in (r.get("art") or "") or "WOHNBAU" in (f.get("name") or "").upper()})
    save(DATA / "anleihen.json", {"updated": NOW.isoformat(), "rows": sorted(rows, key=lambda x: (x.get("emittent") or "", x.get("faellig") or ""))})
    log(f"Anleihen (Bank/Wohnbau): {len(rows)} aus {len(srcs)} Quellen, {sum(1 for r in rows if r['faellig'])} mit Fälligkeit")


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
ETF_INDEX = {"IE00B5BMR087": "S&P 500", "IE00BLNMYC90": "S&P 500", "DE0005933931": "DAX", "IE00B53L3W79": "Euro Stoxx 50",
             "IE00BYVQ9F29": "Nasdaq 100", "LU0659579063": "ATX"}


def resolve_etf_tickers(etfs: list[dict]):
    """Fehlende Yahoo-Ticker über die ISIN suchen (Xetra bevorzugt) und merken."""
    cache = load(DATA / "etf_tickers.json", {}) or {}
    for e in etfs:
        if e.get("ticker"):
            continue
        if cache.get(e["isin"]) and (not e.get("try") or cache[e["isin"]] in e["try"]):
            e["ticker"] = cache[e["isin"]]
            continue
        def find():
            syms = list(e.get("try") or [])
            try:
                syms += [q["symbol"] for q in (yf.Search(e["isin"], max_results=10).quotes or []) if q.get("symbol")]
            except Exception:
                pass
            for sym in dict.fromkeys(syms):  # Reihenfolge der Kandidaten aus etfs.json zuerst
                try:
                    h = yf.Ticker(sym).history(period="1mo")["Close"].dropna()
                except Exception:
                    continue
                if len(h) > 5:
                    return sym
            return None
        sym = safe(find, None, f"Tickersuche {e['isin']}")
        log(f"  ETF-Ticker {e['isin']}: {sym or 'nicht gefunden'}")
        if sym:
            e["ticker"] = cache[e["isin"]] = sym
    save(DATA / "etf_tickers.json", cache, quiet=True)


def perf_fields(c: pd.Series | None, w: pd.Series | None) -> dict:
    """Kurs, Tages-/Monats-/YTD-/1J-Performance, Vola, Max. Drawdown, 3J/5J p. a. und Verläufe (für ETFs und Fonds)."""
    row = {}
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
                   mdd=rnd(float((c / peak - 1).min()) * 100, 1), hist1y=series_pairs(c),
                   pxd=iso_date(c.index[-1]))
    if w is not None and len(w) > 60:
        pxw = float(w.iloc[-1])
        for yrs, key in ((3, "y3"), (5, "y5")):
            target = w.index[-1] - pd.DateOffset(years=yrs)
            base = w[w.index <= target]
            if not len(base) and (w.index[0] - target).days <= 14:  # Reihe beginnt knapp nach dem Stichtag
                base = w.iloc[:1]
            if len(base):
                row[key] = rnd(((pxw / float(base.iloc[-1])) ** (1 / yrs) - 1) * 100, 1)
        row["hist5y"] = series_pairs(w)
    return row


def fund_symbols(fonds: list[dict]) -> dict[str, dict]:
    """Yahoo-Symbol (NAV-Reihe „0P…") je Fonds über die Tranchen-ISINs; Cache, Fehltreffer nach 30 Tagen neu suchen."""
    cache = load(DATA / "fund_yahoo.json", {}) or {}
    left = 200 if MODE == "nightly" else 20
    for f in fonds:
        isins = [i for i in (f.get("isins") or []) if isin_ok(i)]
        if not isins:
            continue
        c = cache.get(f["name"]) or {}
        if c.get("sym") and c.get("isin") in isins:
            continue
        if c.get("d") and not c.get("sym") and (NOW - datetime.fromisoformat(c["d"])).days < 30:
            continue
        if left <= 0:
            break
        found = None
        for isin in isins[:4]:
            left -= 1
            try:
                qs = yf.Search(isin, max_results=10).quotes or []
            except Exception:
                qs = []
            cands = [q["symbol"] for q in qs if q.get("symbol") and (q["symbol"].startswith("0P") or q.get("quoteType") == "MUTUALFUND")]
            for sym in cands:
                try:
                    h = yf.Ticker(sym).history(period="1y")["Close"].dropna()
                except Exception:
                    continue
                if len(h) > 150:
                    found = {"sym": sym, "isin": isin}
                    break
            if found:
                break
            time.sleep(0.5)
        cache[f["name"]] = {**(found or {}), "d": NOW.isoformat()}
        log(f"  Fonds-Kurs {f['name']}: {found['sym'] + ' (' + found['isin'] + ')' if found else 'kein NAV bei Yahoo'}")
    save(DATA / "fund_yahoo.json", cache, quiet=True)
    return cache


def build_funds():
    """Fondskatalog mit NAV-Performance (Yahoo „0P…") und Verwaltungsgebühr aus dem letzten Bericht anreichern."""
    fk = ROOT / "manual" / "fonds.json"
    if not fk.exists():
        return
    cat = json.loads(fk.read_text(encoding="utf-8"))
    fonds = cat.get("fonds", [])
    syms = safe(lambda: fund_symbols(fonds), {}, "Fonds-Symbole") or {}
    tick = sorted({v["sym"] for v in syms.values() if v.get("sym")})
    d1 = history_block(tick, period="1y") if tick else {}
    w5 = history_block(tick, period="5y", interval="1wk") if tick else {}
    reps = load(DATA / "fund_reports.json", {}) or {}
    n_px = n_fee = 0
    for f in fonds:
        y = syms.get(f["name"]) or {}
        if y.get("sym"):
            perf = perf_fields(d1.get(y["sym"]), w5.get(y["sym"]))
            if perf.get("px"):
                f.update(perf, nav_sym=y["sym"], nav_isin=y["isin"])
                n_px += 1
        g = (reps.get(f["name"]) or {}).get("gebuehr")
        if g:
            f["gebuehr"] = g
            f["gebuehr_quelle"] = (reps.get(f["name"]) or {}).get("typ")
            n_fee += 1
    cat["updated"] = NOW.isoformat()
    save(DATA / "fonds.json", cat)
    log(f"Fonds: {len(fonds)}, mit NAV {n_px}, mit Verwaltungsgebühr {n_fee}")


def build_etfs():
    cfg = json.loads((ROOT / "manual" / "etfs.json").read_text(encoding="utf-8"))
    etfs = cfg.get("etfs", [])
    resolve_etf_tickers(etfs)
    tick = [e["ticker"] for e in etfs if e.get("ticker")]
    d1 = history_block(tick, period="1y")
    w5 = history_block(tick, period="5y", interval="1wk")
    old = {e["isin"]: e for e in (load(DATA / "etfs.json", {}) or {}).get("etfs", [])}
    out = []
    for e in etfs:
        t = e.get("ticker")
        c, w = d1.get(t), w5.get(t)
        row = {**e, "index": ETF_INDEX.get(e["isin"])}
        if MODE == "nightly" or e["isin"] not in old or old[e["isin"]].get("ter") is None:
            info = safe(lambda: yf.Ticker(t).info or {}, {}, f"ETF {t}") if t else {}
            ter = info.get("netExpenseRatio") or info.get("annualReportExpenseRatio")
            if ter is not None and ter < 0.05:  # Bruchteil (0,002) -> Prozent (0,2)
                ter = ter * 100
            row.update(ter=rnd(ter, 3) if ter else None,
                       aum=info.get("totalAssets"), cur=info.get("currency"), ydiv=rnd((info.get("yield") or 0) * 100, 2) if info.get("yield") else None)
        else:
            row.update({k: old[e["isin"]].get(k) for k in ("ter", "aum", "cur", "ydiv")})
        row.update(perf_fields(c, w))
        out.append(row)
    save(DATA / "etfs.json", {"updated": NOW.isoformat(), "geprueft": cfg.get("geprueft"), "hinweis": cfg.get("_hinweis"), "etfs": out})
    safe(build_funds, None, "Fonds")
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


_HOLD_CACHE: dict[str, bytes] = {}


def read_holdings_full(e: dict) -> list[dict]:
    """Alle Positionen mit Name, ISIN, Börse und Gewicht in %."""
    import openpyxl
    src = e.get("holdings") or {}
    # Bei Näherungen ("isin"/"ticker" in holdings) die Positionsliste eines Fonds auf denselben/ähnlichen Index lesen
    if src.get("anbieter") == "xtrackers":
        url = f"https://etf.dws.com/etfdata/export/DEU/DEU/excel/product/constituent/{src.get('isin') or e['isin']}/"
    elif src.get("anbieter") == "spdr":
        url = f"https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-{(src.get('ticker') or e['ticker']).split('.')[0].lower()}-gy.xlsx"
    else:
        return []
    if url not in _HOLD_CACHE:
        _HOLD_CACHE[url] = HTTP.get(url, timeout=60).content
    wb = openpyxl.load_workbook(io.BytesIO(_HOLD_CACHE[url]), read_only=True, data_only=True)
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
        elif isin.startswith("BR") and len(isin) == 12 and isin[2:6].isalpha():
            # B3-Kürzel aus der ISIN: BRPETRACNPR6 → PETR4 (ON=3, PN=4, PNA=5, PNB=6, Units=11)
            cls_ = {"ACNOR": "3", "ACNPR": "4", "ACNPA": "5", "ACNPB": "6", "CDAM1": "11", "UNT00": "11"}.get(isin[6:11])
            if not cls_ and isin[6:10] == "CDAM":
                cls_ = "11"
            if not cls_:
                continue
            sym = isin[2:6] + cls_ + ".SA"
            c = "Brazil"
        else:
            continue
        out.append({"s": sym, "n": r["name"].title(), "sec": None, "c": c,
                    "src": ["MSCI EM (Brasilien)" if c == "Brazil" else "MSCI EM (Korea/Taiwan)"]})
    # Börsenplatz prüfen: Taipei Exchange (.TWO) bzw. KOSDAQ (.KQ) sind in der Positionsliste nicht immer erkennbar
    alt = {".TW": ".TWO", ".TWO": ".TW", ".KS": ".KQ", ".KQ": ".KS", ".SA": ".SA"}
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
    log(f"  MSCI EM (Korea/Taiwan/Brasilien): {len(out)} Titel, davon Brasilien {sum(1 for r in out if r['c'] == 'Brazil')} ({len(miss)} Börsenplatz geprüft, {dropped} ohne Kurse verworfen)")
    return out


EUROPE = {"Germany", "France", "United Kingdom", "Switzerland", "Netherlands", "Spain", "Italy", "Sweden", "Denmark", "Finland",
          "Norway", "Belgium", "Ireland", "Austria", "Portugal", "Luxembourg", "Poland"}
RENEW_RE = re.compile(r"\b(vestas|[oø]rsted|nordex|siemens energy|edp renov|verbund|enphase|first solar|solaredge|sma solar|nextera|"
                      r"acciona energ|neoen|scatec|brookfield renew|plug power|array tech|nextracker|iberdrola|encavis|grenergy|"
                      r"solaria|voltalia|china longyuan|xinyi solar|sungrow|longi|adani green|suzlon|ormat|clearway|bloom energy)", re.I)


def filter_holdings(f: dict, uni_rows: list[dict], fund: dict) -> list[tuple[str, float]]:
    """Index-Näherung aus dem Universum: Auswahl und Gewichtung nach den Grundregeln des Index."""
    rows = []
    for r in uni_rows:
        d = {**r, **(fund.get(r["s"]) or {})}
        cap = d.get("mcapEUR")
        if not cap or cap < f.get("min_mcap", 0):
            continue
        if f.get("region") == "Europa" and d.get("c") not in EUROPE:
            continue
        if f.get("sector") and d.get("sec") != f["sector"]:
            continue
        if f.get("exclude_sec") and d.get("sec") in f["exclude_sec"]:
            continue
        if f.get("industry_re") and not re.search(f["industry_re"], str(d.get("ind") or ""), re.I):
            continue
        if f.get("theme") == "renew" and not (re.search(r"renewable|solar|wind", str(d.get("ind") or ""), re.I)
                                              or RENEW_RE.search(str(d.get("n") or ""))):
            continue
        if f.get("sort") == "dy" or f.get("weight") == "dy" or f.get("dy_above_median"):
            if not d.get("dy"):
                continue
        rows.append(d)
    if f.get("dy_above_median") and rows:
        med = sorted(r["dy"] for r in rows)[len(rows) // 2]
        rows = [r for r in rows if r["dy"] > med]
    key = "dy" if f.get("sort") == "dy" else "mcapEUR"
    rows.sort(key=lambda r: -(r.get(key) or 0))
    if f.get("top"):
        rows = rows[: f["top"]]
    wkey = "dy" if f.get("weight") == "dy" else "mcapEUR"
    tot = sum(r.get(wkey) or 0 for r in rows)
    if not tot:
        return []
    w = {r["s"]: (r.get(wkey) or 0) / tot * 100 for r in rows}
    cap = f.get("max_w")
    for _ in range(20 if cap else 0):  # Obergrenze je Titel, Überschuss anteilig verteilen
        over = {k: v for k, v in w.items() if v > cap}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        free = {k: v for k, v in w.items() if v < cap}
        fs = sum(free.values()) or 1
        for k in over:
            w[k] = cap
        for k, v in free.items():
            w[k] = v + excess * v / fs
    return sorted(w.items(), key=lambda x: -x[1])


_SYMC: dict = {}
_SEARCH_LEFT = [0]
_EXTRA: set = set()


def build_holdings(uni_rows: list[dict], fund: dict):
    _SYMC.clear()
    _SYMC.update(load(DATA / "fund_symbols.json") or {})
    _SEARCH_LEFT[0] = 250 if MODE == "nightly" else 120
    _RES_LEFT[0] = 700 if MODE == "nightly" else 200
    _EXTRA.clear()
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
    meta: dict[str, dict] = {}
    isin_map: dict[str, str] = {}
    reps = safe(load_fund_reports, {}, "Fondsberichte") or {}
    auto = safe(lambda: auto_etfs(reps, cfg), [], "ETFs aus Dachfonds") or []
    cfg["etfs"] = cfg.get("etfs", []) + auto
    for e in cfg.get("etfs", []):
        src = e.get("holdings") or {}
        if src.get("index"):
            n = 0
            for r in uni_rows:
                sec = (fund.get(r["s"]) or {}).get("sec") or r.get("sec")
                if (src["index"] in (r.get("src") or []) or (src["index"] == "ATX" and "Zusatz" in (r.get("src") or []))) and (not src.get("sektor") or sec == src["sektor"]):
                    by_sym.setdefault(r["s"], []).append([e["isin"], None])
                    n += 1
            meta[e["isin"]] = {"art": "index", "quelle": src["index"] + (f" · {src['sektor']}" if src.get("sektor") else ""),
                               "naeherung": bool(src.get("naeherung")), "n": n}
            continue
        if src.get("filter"):
            fh = safe(lambda: filter_holdings(src["filter"], uni_rows, fund), [], f"Index-Näherung {e.get('ticker')}")
            if not fh:
                continue
            for sym, w in fh:
                by_sym.setdefault(sym, []).append([e["isin"], rnd(w, 3)])
            names = {r["s"]: (fund.get(r["s"]) or {}).get("n") or r.get("n") or r["s"] for r in uni_rows}
            tops[e["isin"]] = [[names.get(sym, sym), rnd(w, 2)] for sym, w in fh[:10]]
            meta[e["isin"]] = {"art": "liste", "quelle": src.get("quelle") or "Index-Näherung aus dem Universum", "naeherung": True,
                               "n": len(fh), "hit": len(fh), "wsum": 100.0, "wall": 100.0}
            log(f"  Index-Näherung {e.get('ticker')}: {len(fh)} Titel")
            continue
        if not src.get("anbieter"):
            continue
        full = safe(lambda: read_holdings_full(e), [], f"Positionen {e.get('ticker') or e['isin']}")
        rows = [(r["name"], r["w"]) for r in full]
        if not rows:
            continue
        tops[e["isin"]] = [[n, rnd(w, 2)] for n, w in sorted(rows, key=lambda x: -x[1])[:10]]
        agg: dict[str, float] = {}
        hit = 0
        for (n, w), fr in zip(rows, full):
            k = name_key(n)
            fis = (fr.get("isin") or "").strip().upper()
            sym = isin_map.get(fis) or keys.get(k)  # gleiche ISIN -> immer dasselbe Symbol (keine Doppelzählung über mehrere ETFs)
            if not sym and k:
                cand = first.get(k.split()[0], [])
                sym = cand[0] if len(cand) == 1 else None
            fisin = (fr.get("isin") or "").strip().upper()
            if not sym and fisin and w >= 0.15 and not src.get("subset"):
                # Titel außerhalb des Universums: Yahoo-Symbol über ISIN-Cache bzw. Namenssuche (begrenzt je Lauf)
                c = _SYMC.get(fisin)
                if c and (c[0] or (NOW - datetime.fromisoformat(c[1])).days < 30):
                    sym = c[0] or None
                elif _SEARCH_LEFT[0] > 0:
                    _SEARCH_LEFT[0] -= 1
                    sym = search_symbol(n, {})
                    _SYMC[fisin] = [sym or "", NOW.isoformat(), 1]
                if sym:
                    _EXTRA.add(sym)
            if sym:
                agg[sym] = agg.get(sym, 0) + w
                hit += 1
                isin = (fr.get("isin") or "").strip().upper()
                if len(isin) == 12:
                    isin_map.setdefault(isin, sym)
        sub = src.get("subset")
        if sub:
            # Sektor-/Branchenindex = Teilmenge des Mutterindex: Liste des Mutter-ETFs filtern und neu gewichten
            rowmap = {r["s"]: {**r, **(fund.get(r["s"]) or {})} for r in uni_rows}
            ok = lambda d: d and (not sub.get("sector") or d.get("sec") == sub["sector"]) and (  # noqa: E731
                not sub.get("industry_re") or re.search(sub["industry_re"], str(d.get("ind") or ""), re.I))
            agg = {s_: w for s_, w in agg.items() if ok(rowmap.get(s_))}
            tot = sum(agg.values()) or 1
            agg = {s_: w / tot * 100 for s_, w in agg.items()}
            hit = len(agg)
            tops[e["isin"]] = [[rowmap[s_].get("n") or s_, rnd(w, 2)] for s_, w in sorted(agg.items(), key=lambda x: -x[1])[:10]]
            rows = [(s_, w) for s_, w in agg.items()]
        for sym, w in agg.items():
            by_sym.setdefault(sym, []).append([e["isin"], rnd(w, 3)])
        meta[e["isin"]] = {"art": "liste", "quelle": src.get("quelle") or "Positionsliste des Anbieters",
                           "naeherung": bool(src.get("isin") or src.get("ticker")), "n": len(rows), "hit": hit,
                           "wsum": rnd(sum(agg.values()), 1), "wall": rnd(sum(w for _, w in rows), 1)}
        log(f"  Positionen {e.get('ticker') or e['isin']}: {len(rows)} Titel, {hit} im Universum zugeordnet")
    save(DATA / "fund_symbols.json", _SYMC, quiet=True)
    extra: set[str] = set(_EXTRA)
    safe(lambda: add_factsheets(cfg, keys, by_sym, tops, meta, extra, uni_rows, fund), None, "Factsheets")
    safe(lambda: add_ish_reports(cfg, keys, first, by_sym, tops, meta, isin_map, extra), None, "iShares-Berichte")
    safe(lambda: add_report_holdings(cfg, keys, first, by_sym, tops, meta, isin_map, extra, uni_rows, fund, reps), None, "Fondsberichte")
    xrows = safe(lambda: extra_rows(extra, {r["s"] for r in uni_rows}), [], "Zusatztitel") or []
    log(f"  ISIN-Zuordnung Aktien: {len(isin_map)}")
    save(DATA / "holdings.json", {"updated": NOW.isoformat(), "bySymbol": by_sym, "top": tops, "meta": meta, "isin": isin_map,
                                  "extra": xrows, "v": 8, "berichte": sum(1 for k in meta if k.startswith("F:"))})


# ----------------------------------------------------------------- Fondsberichte & Factsheets (echte Bestände)
REPORT_URL = "https://dl.avl-investmentfonds.de/fds/{isin}-{typ}-DE-{d}.pdf"
REPORT_TYP = {"HALBJAHRBER": "Halbjahresbericht", "RECHENSCHAFT": "Rechenschaftsbericht"}
_ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")
_POS_RE = re.compile(r"^(?P<grp>.*?)\s*(?P<isin>[A-Z]{2}[A-Z0-9]{9}[0-9])\s+(?P<rest>.+?)\s+(?P<cur>[A-Z]{3})\s+"
                     r"(?P<nums>[\d\.,\s\-]+?)\s+(?P<pct>-?\d{1,3},\d{2,3})\s*%\s*$")
CUR_SUFFIX = {"USD": "", "CAD": ".TO", "CHF": ".SW", "DKK": ".CO", "GBP": ".L", "NOK": ".OL", "SEK": ".ST", "AUD": ".AX",
              "INR": ".NS", "BRL": ".SA", "PLN": ".WA", "CZK": ".PR", "HUF": ".BD", "SGD": ".SI", "ZAR": ".JO", "MXN": ".MX",
              "ILS": ".TA", "NZD": ".NZ", "IDR": ".JK", "THB": ".BK", "MYR": ".KL", "TRY": ".IS", "SAR": ".SR", "AED": ".AE"}
EUR_SUFFIX = {"DE": ".DE", "FR": ".PA", "IT": ".MI", "ES": ".MC", "NL": ".AS", "BE": ".BR", "FI": ".HE", "AT": ".VI",
              "PT": ".LS", "IE": ".IR", "GR": ".AT", "LU": ".AS"}


def isin_ok(s: str) -> bool:
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", s or ""):
        return False
    d = "".join(str(int(c, 36)) for c in s[:11])
    t = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch) * (2 if i % 2 == 0 else 1)
        t += n // 10 + n % 10
    return (10 - t % 10) % 10 == int(s[11])


def month_ends(n: int) -> list[str]:
    out, y, m = [], NOW.year, NOW.month
    for _ in range(n + 1):
        nxt = datetime(y + (m == 12), m % 12 + 1, 1)
        last = nxt - timedelta(days=1)
        if last.date() < NOW.date():
            out.append(last.strftime("%Y%m%d"))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return out


_POS_EN = re.compile(r"^(?P<grp>.*?)\s*(?P<isin>[A-Z]{2}[A-Z0-9]{9}[0-9])\s+(?P<rest>.+?)\s+(?P<cur>[A-Z]{3})\s+"
                     r"(?P<nums>[\d\.,\s\-]+?)\s+(?P<pct>-?\d{1,3}\.\d{2,3})\s*%\s*$")
_MON_EN = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def parse_fund_report(pdf: bytes) -> dict:
    """Vermögensaufstellung eines Raiffeisen-KAG-Berichts (deutsche oder englische Fassung): alle Wertpapiere mit Anteil am Fondsvermögen."""
    import pypdf
    r = pypdf.PdfReader(io.BytesIO(pdf))
    txt = "\n".join((p.extract_text() or "") for p in r.pages)
    m = re.search(r"Vermögensaufstellung in \w+ per (\d\d\.\d\d\.\d{4})", txt)
    en = None
    if not m:
        me = re.search(r"Portfolio of investments in \w+ as of ([A-Z][a-z]{2})[a-z]* (\d{1,2}), (\d{4})", txt)
        if me and me.group(1).lower() in _MON_EN:
            en = f"{int(me.group(2)):02d}.{_MON_EN[me.group(1).lower()]:02d}.{me.group(3)}"
    rx = _POS_EN if en else _POS_RE
    items, grp, tranchen = [], "", []
    for ln in txt.splitlines():
        ln = ln.strip()
        mt = re.match(r"^([A-Z]{2}[A-Z0-9]{9}[0-9])\s+.+\((?:I|R|RZ|RD|S|SZ)\)\s", ln)
        if mt and isin_ok(mt.group(1)) and mt.group(1) not in tranchen:
            tranchen.append(mt.group(1))
        mm = rx.match(ln)
        if not mm or not isin_ok(mm["isin"]):
            continue
        g = mm["grp"].strip() or grp
        grp = g
        rest = mm["rest"].split()
        tick, code = "", ""
        if en:
            pass  # englische Fassung: keine Ticker-/Kennnummer-Spalte
        elif len(rest) > 1 and re.fullmatch(r"\d{3,6}", rest[-1]):
            code, rest = rest[-1], rest[:-1]
        elif len(rest) > 1 and re.fullmatch(r"[A-Z0-9][A-Z0-9\./\-]{0,7}", rest[-1]):
            tick, rest = rest[-1], rest[:-1]
        items.append({"isin": mm["isin"], "name": " ".join(rest), "tick": tick, "code": code, "cur": mm["cur"],
                      "w": float(mm["pct"] if en else mm["pct"].replace(",", ".")), "grp": g})
    return {"stichtag": m.group(1) if m else en, "items": items, "tranchen": tranchen, "gebuehr": parse_fees(txt)}


def parse_fees(txt: str) -> dict:
    """„effektive Verwaltungsgebühr des Fonds  I-Tranche (EUR): 1,000 % R-Tranche (EUR): 2,000 % …" -> {"I": 1.0, "R": 2.0}"""
    i = txt.find("effektive Verwaltungsgebühr")
    en = i < 0
    if en:
        i = txt.find("Effective management fee")
    if i < 0:
        return {}
    seg = txt[i:i + 600]
    for stop in ("Depotbank", "Custodian bank", "Max. management fee"):
        seg = seg[:seg.find(stop)] if stop in seg else seg
    dec = r"\d{1,2}\.\d{1,3}" if en else r"\d{1,2},\d{1,3}"
    out = {}
    for t, v in re.findall(r"\b([A-Z]{1,3})-Tranche \(\w{3}\):\s*(" + dec + r")\s*%", seg):
        out.setdefault(t, float(v.replace(",", ".")))
    if not out:
        mm = re.search(r"(?:effektive Verwaltungsgebühr|Effective management fee)[^\d]{0,40}(" + dec + r")\s*%", seg)
        if mm:
            out["alle"] = float(mm.group(1).replace(",", "."))
    return out


def pos_kind(it: dict) -> str:
    g = it.get("grp") or ""
    if re.search(r"zertifikat|fonds|ETF|Investmentanteil|fund|investment certificate", g, re.I):
        return "fonds"
    if re.search(r"Aktie|ADR|GDR|REIT|Partizipation|Genuss|Equit|Share", g, re.I):
        return "aktie"
    return "anleihe"


def find_report(isins: list[str], since: str | None) -> tuple[str, str, str] | None:
    """Neuester Halbjahres- bzw. Rechenschaftsbericht (URL, Typ, Datum); prüft nur Stichtage nach `since`."""
    for d in month_ends(16):
        if since and d <= since:
            break
        for typ in REPORT_TYP:
            for isin in isins[:2]:
                url = REPORT_URL.format(isin=isin, typ=typ, d=d)
                try:
                    if HTTP.head(url, timeout=15, allow_redirects=True).status_code == 200:
                        return url, typ, d
                except Exception:
                    pass
    return None


_KEP_END = re.compile(r"\s(-?\d{1,3},\d{2})([A-Z]{2}[A-Z0-9]{9}[0-9])$")
_KEP_NEW = re.compile(r"^([A-Z]{2}[A-Z0-9]{9}[0-9])\s+(.+?)\s+[\d\.,\s\-]+?\s(-?\d{1,3},\d{2})$")
def parse_kepler_text(txt: str) -> dict:
    """Vermögensaufstellung eines KEPLER-Berichts (Prozent und ISIN stehen am Zeilenende)."""
    m = re.search(r"Vermögensaufstellung zum (\d\d)\s?\.(\d\d)\s?\.\s?(\d{4})", txt)
    if not m:
        mo = {"jänner": 1, "januar": 1, "februar": 2, "märz": 3, "april": 4, "mai": 5, "juni": 6, "juli": 7, "august": 8,
              "september": 9, "oktober": 10, "november": 11, "dezember": 12}
        m2 = re.search(r"Vermögensaufstellung zum (\d{1,2})\. ?([A-Za-zäÄ]+) (\d{4})", txt)
        if not m2 or m2.group(2).lower() not in mo:
            return {}
        m = re.search(re.escape(m2.group(0)), txt)
        dd, mm_, yy = f"{int(m2.group(1)):02d}", f"{mo[m2.group(2).lower()]:02d}", m2.group(3)
    else:
        dd, mm_, yy = m.group(1), m.group(2), m.group(3)
    body = txt[m.start():]
    cut = min([i for i in (body.find("Zusammensetzung des Fondsvermögens"), body.find("Devisenkurse"), body.find("Während des Berichtszeitraumes"),
                           body.find("Summe Wertpapiervermögen"))
               if i > 0] or [len(body)])
    body = body[:cut]
    items, buf, grp = [], "", "Aktien"
    tranchen = []
    for mt in re.finditer(r"\b(AT[A-Z0-9]{9}[0-9])\b", txt[:m.start()]):
        if isin_ok(mt.group(1)) and mt.group(1) not in tranchen:
            tranchen.append(mt.group(1))
    for ln in body.splitlines():
        s = ln.rstrip()
        st = s.strip()
        if not st:
            continue
        if not _KEP_END.search(st) and len(st) < 90 and not re.search(r"\d{2,}", st):
            if re.search(r"Investmentzertifikat|OGAW|Investmentfonds|Investmentanteil", st):
                grp, buf = "Investmentzertifikate", ""
                continue
            if re.search(r"Anleihe|Schuldverschreibung|Pfandbrief|Obligation", st):
                grp, buf = "Anleihen", ""
                continue
            if re.match(r"^(Aktien|Genussscheine|Partizipationsscheine|REITs?)\b", st):
                grp, buf = "Aktien", ""
                continue
        if st.startswith(("Summe", "lautend auf", "Bankguthaben", "Sonstiges", "Fondsvermögen", "Wertpapiervermögen", "Zum amtlichen", "Seite ", "KEPLER-FONDS", "Kapitalanlagegesellschaft")):
            buf = ""
            continue
        cand = (buf + ("" if buf and buf[-1].isdigit() and st[:1].isdigit() else " ") + st).strip() if buf else st
        me = _KEP_END.search(cand)
        if me and isin_ok(me.group(2)):
            pre = cand[:me.start()].rstrip()
            name = re.split(r"\s{2,}\d|\s+\d[\d\.]*\s", pre)[0]
            name = re.sub(r"\s+(DL|EO|SF|LS|NK|DK|SK|YN|HD|CD|AD|O\.N\.|-,|,\d)[^A-Za-z]*$", "", name).strip()
            items.append({"isin": me.group(2), "name": re.sub(r"\s{2,}", " ", name), "tick": "", "code": "", "cur": "",
                          "w": float(me.group(1).replace(",", ".")), "grp": grp})
            buf = ""
        else:
            buf = cand if len(cand) < 200 else ""
    if len(items) < 5:  # neues Layout (ab 2026): ISIN am Zeilenanfang, Anteil am Zeilenende
        items, grp = [], "Aktien"
        for ln in txt.splitlines():
            st = ln.strip()
            if not _KEP_NEW.match(st) and len(st) < 90 and not re.search(r"\d{2,}", st):
                if re.search(r"Investmentzertifikat|OGAW|Investmentfonds", st):
                    grp = "Investmentzertifikate"
                elif re.search(r"Anleihe|Schuldverschreibung|Pfandbrief", st):
                    grp = "Anleihen"
                elif re.match(r"^(Aktien|Genussscheine|Partizipationsscheine|REITs?)\b", st):
                    grp = "Aktien"
                elif re.search(r"Zertifikat", st):
                    grp = "Zertifikate"
                continue
            mn = _KEP_NEW.match(st)
            if mn and isin_ok(mn.group(1)):
                name = re.split(r"\s+\d[\d\.]*\s", mn.group(2) + " ")[0].strip()
                items.append({"isin": mn.group(1), "name": name, "tick": "", "code": "", "cur": "",
                              "w": float(mn.group(3).replace(",", ".")), "grp": grp})
    return {"stichtag": f"{dd}.{mm_}.{yy}", "items": items, "tranchen": tranchen}


def parse_any_report(pdf: bytes) -> dict:
    """Raiffeisen- oder KEPLER-Format erkennen und die Vermögensaufstellung lesen."""
    import pypdf
    r = pypdf.PdfReader(io.BytesIO(pdf))
    txt = "\n".join((p.extract_text() or "") for p in r.pages)
    if "KEPLER-FONDS" in txt[:6000]:
        rep = parse_kepler_text(txt)
        if rep.get("items"):
            rep["typ"] = "Halbjahresbericht" if "Halbjahresbericht" in txt[:3000] else "Rechenschaftsbericht"
            return rep
    rep = parse_fund_report(pdf)
    rep["typ"] = "Halbjahresbericht" if re.search(r"Halbjahresbericht|semi-annual", txt[:3000], re.I) else "Rechenschaftsbericht"
    return rep


TT_FUND = "https://gewinn2.ttweb.net/funds/fundDetails.asp?WPKNR={isin}"
TT_DOC = "https://markets.easybank.at/markets/dokumente/{doc}"


def tt_report_links(fonds: list[dict]) -> dict[str, list[str]]:
    """Aktuelle Halbjahres-/Rechenschaftsberichte je Fonds über die öffentliche Fonds-Detailseite (Gewinn/TeleTrader).
    Die Dokumentnamen (HB-/RS-<uuid>.pdf) sind dieselben wie im öffentlichen Dokumentenbereich von easybank markets.
    Wöchentlich je Fonds geprüft; nur die Links werden gemerkt, gelesen wird über den PDF-Cache."""
    cache = load(DATA / "report_tt.json") or {}
    left = 160 if MODE == "nightly" or not cache else 15
    for f in sorted(fonds, key=lambda f_: (cache.get(f_["name"]) or {}).get("checked", "")):
        isins = f.get("isins") or []
        if not isins:
            continue
        c = cache.get(f["name"]) or {}
        if c.get("checked") and (NOW - datetime.fromisoformat(c["checked"])).days < 7 and (c.get("docs") or c.get("v") == 2):
            continue
        if left <= 0:
            break
        docs: list[str] = []
        for isin in isins[:8]:  # nicht jede Tranche hat Dokumente hinterlegt -> bis zum ersten Treffer alle prüfen
            left -= 1
            try:
                html = HTTP.get(TT_FUND.format(isin=isin), timeout=30).text
            except Exception as ex:
                log(f"  Berichtslinks {f['name']}: {ex}")
                continue
            docs = sorted(set(re.findall(r"funddocuments/((?:HB|RS)-[0-9a-f\-]{36}\.pdf)", html)))
            if docs:
                break
            time.sleep(0.5)
        cache[f["name"]] = {"checked": NOW.isoformat(), "docs": docs or c.get("docs") or [], "v": 2}
    save(DATA / "report_tt.json", cache, quiet=True)
    out = {n: [TT_DOC.format(doc=d) for d in v.get("docs") or []] for n, v in cache.items()}
    log(f"  Berichtslinks (Fonds-Detailseiten): {sum(1 for v in out.values() if v)} Fonds")
    return out


def load_fund_reports() -> dict:
    """Berichte aller Fonds mit ISIN laden (Cache: nur neue Berichte werden geladen und gelesen)."""
    from concurrent.futures import ThreadPoolExecutor
    fonds = json.loads((ROOT / "manual" / "fonds.json").read_text(encoding="utf-8")).get("fonds", [])
    cache = load(DATA / "fund_reports.json") or {}
    todo = [f for f in fonds if f.get("isins") and f.get("kag") == "Raiffeisen KAG"]

    def one(f):
        old = cache.get(f["name"]) or {}
        if old.get("url", "").startswith("https://dl.avl") and "gebuehr" not in old:  # Gebühren nachtragen (einmalig)
            try:
                old = {**old, "gebuehr": parse_fund_report(HTTP.get(old["url"], timeout=90).content).get("gebuehr") or {}}
            except Exception:
                pass
        if old.get("checked") and (NOW - datetime.fromisoformat(old["checked"])).days < 6 and old.get("items") is not None:
            return f["name"], old
        hit = find_report(f["isins"], old.get("d"))
        if not hit:
            return f["name"], {**old, "checked": NOW.isoformat()}
        url, typ, d = hit
        try:
            rep = parse_fund_report(HTTP.get(url, timeout=90).content)
        except Exception as ex:
            log(f"  Bericht {f['name']}: nicht lesbar ({ex})")
            return f["name"], {**old, "checked": NOW.isoformat()}
        if not rep["items"]:
            return f["name"], {**old, "checked": NOW.isoformat()}
        return f["name"], {"url": url, "typ": REPORT_TYP[typ], "d": d, "stichtag": rep["stichtag"] or f"{d[6:]}.{d[4:6]}.{d[:4]}",
                           "items": rep["items"], "tranchen": rep["tranchen"], "gebuehr": rep.get("gebuehr") or {},
                           "checked": NOW.isoformat()}

    with ThreadPoolExecutor(6) as ex:
        for name, val in ex.map(one, todo):
            if val:
                cache[name] = val
    # Fonds mit hinterlegten Bericht-Links (z. B. KEPLER) und PDFs in manual/berichte/
    pcache = load(DATA / "report_pdfs.json") or {}
    cand: dict[str, list[dict]] = {}
    srcs = [(f["name"], u) for f in fonds for u in (f.get("berichte") or [])]
    srcs += [(n, u) for n, us in tt_report_links(fonds).items() for u in us]
    for p_ in sorted((ROOT / "manual" / "berichte").glob("*.pdf")) if (ROOT / "manual" / "berichte").exists() else []:
        srcs.append((None, "file:" + p_.name))
    for fname, u in srcs:
        if u not in pcache:
            try:
                if u.startswith("file:"):
                    raw = (ROOT / "manual" / "berichte" / u[5:]).read_bytes()
                else:
                    resp = None
                    alts = [u] + ([u.replace("https://markets.easybank.at/", h) for h in ("https://www.easybank.at/", "https://easybank.at/")]
                                  if u.startswith("https://markets.easybank.at/") else [])
                    for au in alts:  # derselbe Dokumentenbereich ist unter mehreren Adressen erreichbar
                        try:
                            resp = HTTP.get(au, timeout=90)
                        except Exception:
                            continue
                        if resp.status_code == 200 and resp.content.startswith(b"%PDF"):
                            break
                    if resp is None or resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
                        log(f"  Bericht {u.rsplit('/', 1)[-1]}: nicht abrufbar (HTTP {resp.status_code if resp is not None else '-'})")
                        continue  # später erneut versuchen
                    raw = resp.content
                rp = parse_any_report(raw)
                pcache[u] = {k: rp.get(k) for k in ("stichtag", "items", "tranchen", "typ", "gebuehr")}
            except Exception as ex:
                log(f"  Bericht {u.rsplit('/', 1)[-1]}: nicht lesbar ({ex})")
                if "Timeout" in type(ex).__name__ or "Connection" in type(ex).__name__:
                    continue  # Netzwerkfehler: im nächsten Lauf erneut versuchen
                pcache[u] = {"items": []}
        rp = pcache[u]
        if not rp.get("items"):
            continue
        if not fname:  # PDF aus dem Ordner: Fonds über Tranchen-ISINs zuordnen
            tr = set(rp.get("tranchen") or [])
            fname = next((f["name"] for f in fonds if tr & set(f.get("isins") or [])), None)
        if fname:
            cand.setdefault(fname, []).append({**rp, "url": u if not u.startswith("file:") else "manual/berichte/" + u[5:]})
    d_ = lambda x: datetime.strptime(x, "%d.%m.%Y") if x else datetime(1900, 1, 1)  # noqa: E731
    for fname, lst in cand.items():
        best = max(lst, key=lambda r_: d_(r_.get("stichtag")))
        cur = cache.get(fname) or {}
        if not cur.get("items") or d_(best["stichtag"]) >= d_(cur.get("stichtag")):
            f = next(x for x in fonds if x["name"] == fname)
            cache[fname] = {"url": best["url"], "typ": best.get("typ") or "Bericht", "stichtag": best["stichtag"], "items": best["items"],
                            "tranchen": sorted(set((best.get("tranchen") or []) + (f.get("isins") or []))),
                            "gebuehr": best.get("gebuehr") or {}, "checked": NOW.isoformat()}
    save(DATA / "report_pdfs.json", pcache, quiet=True)
    save(DATA / "fund_reports.json", cache, quiet=True)
    n = sum(1 for v in cache.values() if v.get("items"))
    log(f"  Fondsberichte: {n} Fonds mit Vermögensaufstellung (Raiffeisen automatisch, {len(cand)} über Links/Ordner)")
    return cache


def guess_symbols(it: dict) -> list[str]:
    cur, t, code, isin = it.get("cur"), (it.get("tick") or "").strip("/"), it.get("code") or "", it["isin"]
    if code:
        if cur == "HKD":
            return [code.zfill(4) + ".HK"]
        if cur == "JPY":
            return [code + ".T"]
        if cur == "KRW":
            return [code.zfill(6) + ".KS", code.zfill(6) + ".KQ"]
        if cur == "TWD":
            return [code + ".TW", code + ".TWO"]
        if cur == "CNY":
            return [code.zfill(6) + (".SS" if code.startswith("6") else ".SZ")]
        if cur == "INR":
            return []
    if not t:
        return []
    t = t.replace("/", "-").replace(".", "-")
    if cur == "EUR":
        first = EUR_SUFFIX.get(isin[:2])
        sufs = ([first] if first else []) + [s for s in (".DE", ".PA", ".MI", ".AS", ".MC") if s != first]
        return [t + s for s in sufs]
    suf = CUR_SUFFIX.get(cur)
    if suf is None:
        return []
    out = [t + suf]
    if cur in ("SEK", "DKK", "NOK") and len(t) > 3 and t[-1] in "AB" and "-" not in t:
        out.append(t[:-1] + "-" + t[-1] + suf)
    if cur == "GBP" and not t.endswith("-"):
        out.append(t.rstrip("-") + ".L")
    return list(dict.fromkeys(out))


_RES_LEFT = [0]


def resolve_positions(items: list[dict], keys: dict, first: dict, isin_map: dict, min_w: float = 0.0) -> dict[str, str]:
    """ISIN → Yahoo-Symbol für Aktienpositionen: ETF-Listen, Firmenname, Börsenkürzel, zuletzt Yahoo-Suche."""
    cache = load(DATA / "fund_symbols.json") or {}
    out: dict[str, str] = {}
    pending: dict[str, dict] = {}
    for it in items:
        isin = it["isin"]
        if isin in out or isin in pending:
            continue
        if isin in isin_map:
            out[isin] = isin_map[isin]
            continue
        k = name_key(it["name"])
        if k and k in keys:
            out[isin] = keys[k]
            continue
        c = cache.get(isin)
        if c and (c[0] or (len(c) > 2 and (NOW - datetime.fromisoformat(c[1])).days < 30)):
            if c[0]:
                out[isin] = c[0]
            continue
        if it.get("w", 1) >= min_w:
            pending[isin] = it
    if pending:
        cands = {isin: guess_symbols(it) for isin, it in pending.items()}
        allc = sorted({s for v in cands.values() for s in v})
        have = history_block(allc, period="5d", interval="1d") if allc else {}
        ok = {s for s, v in have.items() if v is not None and len(v)}
        searched = 0
        for isin, it in sorted(pending.items(), key=lambda x: -(x[1].get("w") or 0)):
            sym = next((s for s in cands[isin] if s in ok), None)
            tried = bool(cands[isin])
            if not sym and _RES_LEFT[0] > 0:
                _RES_LEFT[0] -= 1
                tried = True
                searched += 1
                try:
                    q = [x for x in (yf.Search(it["name"][:40], max_results=5).quotes or []) if x.get("quoteType") == "EQUITY"]
                    sym = q[0]["symbol"] if q else None
                except Exception:
                    sym = None
            if sym or tried:
                cache[isin] = [sym or "", NOW.isoformat(), 1]
            if sym:
                out[isin] = sym
        save(DATA / "fund_symbols.json", cache, quiet=True)
        log(f"  Fondspositionen: {len(pending)} neue ISINs geprüft, {sum(1 for i in pending if i in out)} zugeordnet ({searched} per Suche)")
    return out


def extra_rows(syms: set[str], have: set[str], limit=None) -> list[dict]:
    """Kennzahlen für Fondstitel außerhalb des Universums (Cache, Auffrischung alle 14 Tage)."""
    limit = limit or (400 if MODE == "nightly" else 150)
    cache = load(DATA / "fund_extra3.json") or {}
    need = [s for s in sorted(syms - have) if s not in cache or (NOW - datetime.fromisoformat(cache[s]["upd"])).days >= 14][:limit]
    got, fail = {}, 0
    for sym in need:
        if elapsed_min() > (95 if MODE == "nightly" else 25):
            break
        try:
            inf = yf.Ticker(sym).info or {}
        except Exception:
            inf = {}
        if inf.get("marketCap") or inf.get("trailingPE") or inf.get("sector"):
            got[sym] = inf
        else:
            fail += 1
        time.sleep(0.15)
    fx = fx_to_eur({(i.get("currency") or "") for i in got.values()} | {"USD"})
    for sym, g in got.items():
        px = g.get("currentPrice") or g.get("regularMarketPrice")
        dy = (g["dividendRate"] / px * 100) if g.get("dividendRate") and px else (
            g["trailingAnnualDividendYield"] * 100 if g.get("trailingAnnualDividendYield") is not None else None)
        pct_ = lambda v: rnd(v * 100, 1) if v is not None else None  # noqa: E731
        cache[sym] = {"upd": NOW.isoformat(), "row": {
            "s": sym, "n": g.get("longName") or g.get("shortName") or sym, "sec": sector_de(g.get("sector")), "c": g.get("country"),
            "cur": g.get("currency"), "mcap": rnd(to_eur(g.get("marketCap"), g.get("currency"), fx), 0) if g.get("marketCap") else None,
            "pe": rnd(g.get("trailingPE"), 1), "fpe": rnd(g.get("forwardPE"), 1), "dy": rnd(dy, 2), "roe": pct_(g.get("returnOnEquity")),
            "eg": pct_(g.get("earningsGrowth", g.get("earningsQuarterlyGrowth"))),
            "feg": rnd((g["forwardEps"] / g["trailingEps"] - 1) * 100, 1) if g.get("forwardEps") and g.get("trailingEps") and g["trailingEps"] > 0 else None,
            "up": rnd((g["targetMeanPrice"] / px - 1) * 100, 1) if g.get("targetMeanPrice") and px else None,
            "ni": rnd(to_eur(g.get("netIncomeToCommon"), g.get("currency"), fx), 0) if g.get("netIncomeToCommon") is not None else None, "x": 1}}
    save(DATA / "fund_extra3.json", cache, quiet=True)
    log(f"  Zusatztitel für Fonds: {len(syms - have)} außerhalb des Universums, {len(got)} neu geladen, {fail} ohne Daten, {sum(1 for s in syms - have if s in cache)} verfügbar")
    return [cache[s]["row"] for s in sorted(syms - have) if s in cache]


_UP = {"AG", "SE", "SA", "NV", "PLC", "ASA", "AB", "SPA", "INC", "ADR", "LP", "CO", "KGAA", "A/S", "OYJ", "BV", "ESG", "ETF", "USA", "UK"}


def nice_name(n: str) -> str:
    return " ".join(w if w.upper() in _UP and len(w) <= 4 else w.capitalize() if not any(c.isdigit() for c in w) else w
                    for w in str(n).split()).replace(" Co ", " Co. ")


def factsheet_top10(url: str) -> tuple[str | None, list[tuple[str, float]]]:
    import pypdf
    r = pypdf.PdfReader(io.BytesIO(HTTP.get(url, timeout=60).content))
    txt = "\n".join((p.extract_text() or "") for p in r.pages)
    m = re.search(r"(?:as at|Stand)[:\s]+(\d{1,2}[-\./][A-Za-z0-9]{2,3}[-\./]\d{4})", txt)
    if not m:
        mg = re.search(r"zum (\d{1,2})\. ?([A-Za-zä]{3,4})\.? (\d{4})", txt)
        mon = {"jan": 1, "feb": 2, "mär": 3, "mrz": 3, "apr": 4, "mai": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "okt": 10, "nov": 11, "dez": 12}
        if mg and mg.group(2).lower() in mon:
            return_stand = f"{int(mg.group(1)):02d}.{mon[mg.group(2).lower()]:02d}.{mg.group(3)}"
            m = re.match(r"(.*)", return_stand)
    i = max(txt.find("Top 10 Holdings"), txt.find("Top-10-Positionen"), txt.find("Top 10 Positionen"))
    out = []
    for ln in (txt[i:].splitlines()[1:16] if i >= 0 else []):
        if re.match(r"^(Total|Summe|Gesamt|Portfolio insgesamt)", ln.strip()):
            break
        mm = re.match(r"^(.+?)\s+(\d{1,2}[\.,]\d{2})\s?%\s*$", ln.strip())
        if mm:
            out.append((mm.group(1).strip(), float(mm.group(2).replace(",", "."))))
    stand = m.group(1) if m else None
    if stand:
        for fmt_ in ("%d-%b-%Y", "%d.%m.%Y", "%d/%m/%Y"):
            try:
                stand = datetime.strptime(stand, fmt_).strftime("%d.%m.%Y")
                break
            except ValueError:
                pass
    return stand, out


def search_symbol(name: str, cache: dict) -> str | None:
    k = "N:" + name_key(name)
    if k in cache:
        return cache[k][0] or None
    try:
        q = [x for x in (yf.Search(name, max_results=5).quotes or []) if x.get("quoteType") == "EQUITY"]
        sym = q[0]["symbol"] if q else None
    except Exception:
        sym = None
    cache[k] = [sym or "", NOW.isoformat(), 1]
    return sym


# Index-Gleichheit: (Muster im ETF-Namen, Kandidaten [(ETF mit bekanntem Bestand, Ausschnitt)], Beschriftung) – erste passende Regel
# gilt; innerhalb der Regel der erste Kandidat mit Daten (zuerst ETFs mit genau diesem Index, zuletzt Ausschnitt aus dem MSCI World)
_W = "IE00B4L5Y983"  # iShares Core MSCI World
_WS = lambda sec, x: [(x, None), (_W, {"sec": sec})]  # noqa: E731
ETF_PROXY = [
    (r"MSCI WORLD.*QUALITY", [("IE00BP3QZ601", None), ("IE00BL25JL35", None)], ""),
    (r"MSCI WORLD.*VALUE", [("IE00BP3QZB59", None)], ""),
    (r"MSCI WORLD.*MOMENTUM", [("IE00BP3QZ825", None)], ""), (r"MSCI WORLD.*MIN.*VOL", [("IE00B8FHGS14", None)], ""),
    (r"MSCI WORLD.*(INFORMATION TECH|\bIT\b)", _WS("Technologie", "IE00BM67HT60"), "Sektor Technologie"),
    (r"MSCI WORLD.*FINANCIAL", _WS("Finanzen", "IE00BM67HL84"), "Sektor Finanzen"),
    (r"MSCI WORLD.*HEALTH", _WS("Gesundheit", "IE00BM67HK77"), "Sektor Gesundheit"),
    (r"MSCI WORLD.*CONSUMER DISCR", _WS("Zyklischer Konsum", "IE00BM67HP23"), "Sektor zyklischer Konsum"),
    (r"MSCI WORLD.*CONSUMER STAP", _WS("Basiskonsum", "IE00BM67HN09"), "Sektor Basiskonsum"),
    (r"MSCI WORLD.*UTILIT", _WS("Versorger", "IE00BM67HQ30"), "Sektor Versorger"),
    (r"MSCI WORLD.*ENERGY", _WS("Energie", "IE00BM67HM91"), "Sektor Energie"),
    (r"MSCI WORLD.*INDUSTRIAL", _WS("Industrie", "IE00BM67HV82"), "Sektor Industrie"),
    (r"MSCI WORLD.*MATERIAL", _WS("Rohstoffe", "IE00BM67HS53"), "Sektor Rohstoffe"),
    (r"MSCI WORLD.*COMMUNICATION", _WS("Kommunikation", "IE00BM67HR47"), "Sektor Kommunikation"),
    (r"MSCI WORLD.*REAL ESTATE", [(_W, {"sec": "Immobilien"})], "Sektor Immobilien"),
    (r"MSCI ACWI|MSCI ALL COUNTRY|MSCI AC WORLD", [("IE00B6R52259", None), ("IE00B44Z5B48", None)], ""),
    (r"MSCI WORLD", [(_W, None), ("IE00BJ0KDQ92", None), ("IE00BFY0GT14", None)], ""),
    (r"MSCI USA|MSCI US\b|MSCI NORTH AMERICA", [("IE00B52SFT06", None), ("IE00BJ0KDR00", None), (_W, {"c": {"United States"}})], ""),
    (r"S&P 500", [("IE00B5BMR087", None)], ""),
    (r"MSCI EUROPE|MSCI EUR\b", [("IE00B4K48X80", None), ("IE00B1YZSC51", None)], ""),
    (r"STOXX EUROPE 600", [("DE0002635307", None)], ""), (r"EURO STOXX 50", [("IE00B53L3W79", None)], ""),
    (r"MSCI EM\b|MSCI EMERGING|EMERGING MARKETS", [("IE00B0M63177", None), ("IE00BTJRMP35", None)], ""),
    (r"MSCI JAPAN", [("LU0274209740", None), (_W, {"c": {"Japan"}})], ""),
    (r"MSCI CANADA", [(_W, {"c": {"Canada"}})], "Kanada-Anteil"),
    (r"PACIFIC EX.?JAPAN", [("IE00B52MJY50", None), (_W, {"c": {"Australia", "Hong Kong", "Singapore", "New Zealand"}})], ""),
]
_SUF_C = {".TO": "Canada", ".T": "Japan", ".AX": "Australia", ".HK": "Hong Kong", ".SI": "Singapore", ".NZ": "New Zealand",
          ".L": "United Kingdom", ".SW": "Switzerland", ".DE": "Germany", ".PA": "France", ".AS": "Netherlands"}
_C_ALIAS = {"USA": "United States", "US": "United States", "Vereinigte Staaten": "United States", "Japan": "Japan",
            "Kanada": "Canada", "Australien": "Australia", "Singapur": "Singapore", "Neuseeland": "New Zealand", "Hongkong": "Hong Kong"}


_BOND_RX = re.compile(r"BOND|GOVT|GOVERNMENT|TREASUR|CORP|AGGREGATE|ANLEIHE|RENTEN|HIGH YIELD|MONEY|GELDMARKT|INFLATION|\bOVERNIGHT|\bSOVEREIGN", re.I)
_ROMAN = {"II", "III", "IV", "V", "VI", "VII", "VIII"}
# iShares-ETFs ohne Umbrella im Namen: wahrscheinliche Umbrellas (werden der Reihe nach geprüft)
ISH_PLC_TRY = ["iv", "ii", "iii", "i"]


def auto_etfs(reps: dict, cfg: dict) -> list[dict]:
    """Aktien-ETFs aus den Vermögensaufstellungen der Fonds, die nicht selbst konfiguriert sind, mit eigener Bestandsquelle
    (Xtrackers- und SPDR-Positionslisten, iShares-Sammelberichte). Nur für den Fonds-Durchblick, nicht in der ETF-Liste."""
    have = {e["isin"] for e in cfg.get("etfs", [])}
    tr = {i for r in reps.values() for i in (r.get("tranchen") or [])}
    cand: dict[str, str] = {}
    for r in reps.values():
        for it in r.get("items") or []:
            i = it["isin"]
            if pos_kind(it) != "fonds" or i in have or i in tr or i in cand or not i.startswith(("IE", "LU", "DE")):
                continue
            nm = f"{it.get('name') or ''} {it.get('tick') or ''}".upper()
            if not re.search(r"\bETF\b|UCITS|XTRACKERS|ISHARES|SPDR", nm) or _BOND_RX.search(nm):
                continue
            cand[i] = nm
    figi = load(DATA / "etf_figi.json") or {}
    spdr = [i for i, nm in cand.items() if re.search(r"SPDR|SSGA", nm) and not (figi.get(i) and (NOW - datetime.fromisoformat(figi[i]["_d"])).days < 90)]
    for k in range(0, len(spdr), 10):
        chunk = spdr[k:k + 10]
        try:
            res = HTTP.post("https://api.openfigi.com/v3/mapping", json=[{"idType": "ID_ISIN", "idValue": i} for i in chunk], timeout=40).json()
        except Exception as ex:
            log(f"  OpenFIGI ETF: {ex}")
            break
        for i, x in zip(chunk, res):
            de = [d for d in (x.get("data") or []) if d.get("exchCode") in ("GR", "GY", "GS", "GT", "GF", "GD", "GH", "GM", "GI")]
            figi[i] = {"ticker": de[0]["ticker"] if de else None, "_d": NOW.isoformat()}
        time.sleep(2.6)
    save(DATA / "etf_figi.json", figi, quiet=True)
    out = []
    for i, nm in cand.items():
        e = {"isin": i, "name": etf_name(nm.split(" ETF ")[0] + (" ETF" if " ETF" in nm else "")), "ticker": None, "auto": True}
        if re.search(r"XTRACKERS|X-TRACKERS", nm):
            e["holdings"] = {"anbieter": "xtrackers", "quelle": "Positionsliste Xtrackers"}
        elif re.search(r"SPDR|SSGA", nm):
            t = (figi.get(i) or {}).get("ticker")
            if not t:
                continue
            e["holdings"] = {"anbieter": "spdr", "ticker": t, "quelle": "Positionsliste SPDR"}
        elif re.search(r"I?SHARES", nm):
            m = re.match(r"^I?SHARES(?:\s+([IVX]+))?\s+PLC\s+-\s+(.+)$", nm.strip())
            rest = m.group(2) if m else nm.strip()
            plc = (m.group(1) or "i").lower() if m and (not m.group(1) or m.group(1) in _ROMAN) else None
            ucits = rest.find("UCITS")
            fonds = (rest[:ucits] if ucits > 0 else rest).strip() + " UCITS ETF"
            fonds = re.sub(r"^I?SHARES\s+", "", fonds)
            fonds = "iShares " + fonds
            e["holdings"] = {"bericht": {"plc": plc, "fonds": fonds} if plc else {"plc_try": ISH_PLC_TRY, "fonds": fonds},
                             "quelle": "iShares-Sammelbericht"}
        else:
            continue
        out.append(e)
    if out:
        log(f"  ETFs aus Dachfonds automatisch ergänzt: {len(out)} (" + ", ".join(sorted({(e['holdings'].get('anbieter') or 'ishares') for e in out})) + ")")
    return out


def sym_match(sym: str, d: dict, flt: dict) -> bool:
    if flt.get("sec"):
        return d.get("sec") == flt["sec"]
    if flt.get("c"):
        c = _C_ALIAS.get(d.get("c") or "", d.get("c"))
        if not c:
            suf = "." + sym.rsplit(".", 1)[1] if "." in sym else ""
            c = _SUF_C.get(suf) or ("United States" if not suf else None)
        return c in flt["c"]
    return True


def etf_name(n: str) -> str:
    """Lesbarer ETF-Name ohne Umbrella-Präfix, z. B. „ISHARES III PLC - ISHARES CORE MSCI WORLD …" -> „iShares Core MSCI World …"."""
    n = re.sub(r"^.*?\b(?:PLC|ICAV|SICAV|FUND|SOLUTIONS|EASY|ETFS EUROPE I PLC)\s+-\s+", "", n or "", flags=re.I)
    n = re.sub(r"^XTRACKERS\s+-\s+", "", n, flags=re.I)
    n = re.sub(r"^SHARES\b", "ISHARES", n)
    n = re.sub(r"\(AC?$", "(Acc)", n.strip())
    out = []
    for w in n.split():
        up = w.upper()
        if up in ("MSCI", "ETF", "UCITS", "USD", "EUR", "ACWI", "EM", "USA", "UK", "SPDR", "S&P", "ESG", "SRI", "IMI", "ECPI", "BNP", "UBS", "IE", "DIST", "ACC", "II", "III", "IV", "V", "VI", "VII", "1C"):
            out.append({"DIST": "Dist", "ACC": "Acc"}.get(up, up))
        elif up.startswith("ISHARES"):
            out.append("iShares" + w[7:].lower())
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out).replace("(dist)", "(Dist)").replace("(acc)", "(Acc)").replace("Wisdomtree", "WisdomTree")


def npos_f(its: list[dict]) -> int:
    return sum(1 for it in its if pos_kind(it) == "fonds")


def add_factsheets(cfg: dict, keys: dict, by_sym: dict, tops: dict, meta: dict, extra: set,
                   uni_rows: list | None = None, fund: dict | None = None):
    """Factsheet-Top-10 (iShares), Rest ggf. über Index-Näherung."""
    symc = load(DATA / "fund_symbols.json") or {}
    for e in cfg.get("etfs", []):
        src = e.get("holdings") or {}
        if not src.get("factsheet"):
            continue
        try:
            stand, top = factsheet_top10(src["factsheet"])
        except Exception as ex:
            log(f"  Factsheet {e.get('ticker')}: {ex}")
            continue
        if not top:
            continue
        hit, got = 0, set()
        for n, w in top:
            k = name_key(n)
            sym = keys.get(k) or search_symbol(n, symc)
            if sym and sym not in got:
                by_sym.setdefault(sym, []).append([e["isin"], w])
                hit += 1
                got.add(sym)
                extra.add(sym)
        ts = sum(w for _, w in top)
        rest = []
        if src.get("filter") and uni_rows is not None:
            fh = [(s_, w) for s_, w in (filter_holdings(src["filter"], uni_rows, fund or {}) or []) if s_ not in got]
            tot = sum(w for _, w in fh) or 1
            rest = [(s_, w / tot * (100 - ts)) for s_, w in fh]
            for s_, w in rest:
                by_sym.setdefault(s_, []).append([e["isin"], rnd(w, 3)])
        tops[e["isin"]] = [[nice_name(n), w] for n, w in top]
        meta[e["isin"]] = {"art": "factsheet", "quelle": "Factsheet des Anbieters", "stichtag": stand, "n": len(top), "hit": hit,
                           "wsum": rnd(ts, 1), "wall": 100.0 if not rest else rnd(ts + sum(w for _, w in rest), 1), "url": src["factsheet"],
                           "rest": len(rest)}
        log(f"  Factsheet {e.get('ticker')}: Top {len(top)} ({sum(w for _, w in top):.1f} %), Stand {stand}")
    save(DATA / "fund_symbols.json", symc, quiet=True)


def add_report_holdings(cfg: dict, keys: dict, first: dict, by_sym: dict, tops: dict, meta: dict, isin_map: dict, extra: set,
                        uni_rows: list | None = None, fund: dict | None = None, reps: dict | None = None):
    """Echte Bestände der Fonds aus den Vermögensaufstellungen der Halbjahres-/Rechenschaftsberichte (inkl. Durchblick durch Subfonds/ETFs)."""
    reps = reps if reps is not None else load_fund_reports()
    by_isin = {}  # Tranchen-ISIN → Fondsname (für Dachfonds)
    for name, r in reps.items():
        for i in r.get("tranchen") or []:
            by_isin[i] = name
    allitems = [it for r in reps.values() for it in (r.get("items") or []) if pos_kind(it) == "aktie"]
    symmap = resolve_positions(allitems, keys, first, isin_map)
    for isin, sym in symmap.items():
        isin_map.setdefault(isin, sym)
    etf_w: dict[str, dict[str, float]] = {}
    for sym, lst in by_sym.items():
        for i, w in lst:
            if w is not None:
                etf_w.setdefault(i, {})[sym] = w
    # ETFs in Dachfonds, die nicht selbst im Universum sind: über ein ETF mit gleichem Index (ggf. Länder-/Sektor-Ausschnitt) zerlegen
    info = {r["s"]: {**r, **(fund.get(r["s"]) or {})} for r in (uni_rows or [])}
    etf_names = {e["isin"]: e.get("name") or e.get("ticker") or e["isin"] for e in cfg.get("etfs", [])}
    proxy_cache: dict[str, tuple] = {}

    def proxy(it: dict):
        """(Gewichte {Symbol: % des ETF}, Beschreibung) oder None"""
        if it["isin"] in proxy_cache:
            return proxy_cache[it["isin"]]
        nm = (it.get("name") or "").upper()
        res = None
        if re.search(r"BOND|GOVT|GOVERNMENT|TREASUR|CORP|AGGREGATE|ANLEIHE|RENTEN|HIGH YIELD|MONEY|GELDMARKT|INFLATION|\bEUR HEDG.*\bBD\b", nm):
            proxy_cache[it["isin"]] = None
            return None
        for rx, cands, lab in ETF_PROXY:
            if not re.search(rx, nm):
                continue
            for base, flt in cands:  # erste passende Regel entscheidet – kein Rückfall auf eine allgemeinere
                if base == it["isin"]:
                    continue
                bw = etf_w.get(base)
                if bw and flt:
                    sel = {s_: w for s_, w in bw.items() if sym_match(s_, info.get(s_) or {}, flt)}
                    tot = sum(sel.values())
                    bw = {s_: w / tot * 100 for s_, w in sel.items()} if tot > 0 else None
                if bw:
                    cut = lab or ("Länder-/Sektorausschnitt" if flt else "")
                    res = (bw, f"≈ {etf_names.get(base, base)}" + (f" · {cut}" if cut else ""))
                    break
            break
        proxy_cache[it["isin"]] = res
        return res
    durch: dict[str, dict[str, float]] = {}

    def equity(name, depth=0):
        """(Aktienpositionen als {Schlüssel: [Name, Gewicht %, Symbol]}, Anteile je Art)"""
        r = reps.get(name) or {}
        pos, art = {}, {"aktie": 0.0, "anleihe": 0.0, "fonds": 0.0}
        for it in r.get("items") or []:
            kd = pos_kind(it)
            art[kd] += it["w"]
            if kd == "aktie":
                sm = symmap.get(it["isin"])
                p = pos.setdefault(sm or it["isin"], [nice_name(it["name"]), 0.0, sm])  # gleicher Schlüssel wie Titel aus ETFs
                p[1] += it["w"]
            elif kd == "fonds" and depth < 2:
                sub = by_isin.get(it["isin"])
                if sub and sub != name:
                    sp, sa = equity(sub, depth + 1)
                    tot = sum(sa.values()) or 100
                    for k, (n, w, s) in sp.items():
                        p = pos.setdefault(k, [n, 0.0, s])
                        p[1] += w * it["w"] / tot
                    art["aktie"] += sa["aktie"] * it["w"] / tot
                    art["fonds"] -= it["w"] * (sa["aktie"] / tot)
                else:
                    ew = etf_w.get(it["isin"])
                    px = None if ew else proxy(it)
                    if px:
                        ew = px[0]
                    if depth == 0:
                        d_ = durch.setdefault(name, {"exakt": 0.0, "naeherung": 0.0, "offen": 0.0})
                        d_["exakt" if it["isin"] in etf_w else "naeherung" if px else "offen"] += it["w"]
                    if ew:
                        tot = sum(ew.values()) or 100
                        sc = 100 / tot if tot > 100.5 else 1  # Summe > 100 % (Rundung) auf 100 normieren
                        for s, w in ew.items():
                            p = pos.setdefault(s, [(info.get(s) or {}).get("n") or s, 0.0, s])
                            p[1] += w * sc * it["w"] / 100
                        art["aktie"] += it["w"]
                        art["fonds"] -= it["w"]
        return pos, art

    nf = 0
    for name, r in reps.items():
        if not r.get("items"):
            continue
        key = "F:" + name
        pos, art = equity(name)
        agg: dict[str, float] = {}
        hit = 0
        for n, w, s in pos.values():
            if s:
                agg[s] = agg.get(s, 0) + w
                hit += 1
                extra.add(s)
        for s, w in agg.items():
            by_sym.setdefault(s, []).append([key, rnd(w, 3)])
        grp: dict[str, list] = {}
        for k_, (n, w, s) in pos.items():
            nk = " ".join((name_key(n) or "").split()[:2])
            if nk:
                grp.setdefault(nk, []).append((s or k_, w))
        dups = {k_: v for k_, v in grp.items() if len(v) > 1}
        if dups:
            log(f"  {name}: {len(pos)} Titel, {len(dups)} Namensgruppen mit mehreren Symbolen, z. B. "
                + "; ".join(f"{k_}: " + ", ".join(f"{s_} {w_:.2f}" for s_, w_ in v[:3]) for k_, v in list(sorted(dups.items(), key=lambda x: -sum(w for _, w in x[1])))[:6]))
        tops[key] = [[n, rnd(w, 2)] for n, w, _ in sorted(pos.values(), key=lambda x: -x[1])[:10]]
        its = r.get("items") or []
        def plab(it):
            if pos_kind(it) != "fonds" or by_isin.get(it["isin"]):
                return None
            if it["isin"] in etf_w:
                m_ = meta.get(it["isin"]) or {}
                q_ = m_.get("quelle") or ""
                q_ = "Positionsliste des Anbieters" if q_.startswith("Positionsliste") and "gleicher Index" not in q_ else q_
                st = m_.get("stichtag") or m_.get("rest_stichtag")
                return "Bestand: " + (q_ or "Anbieter") + (f" per {st}" if st else "")
            px = proxy(it)
            return px[1] if px else None
        allpos = [[etf_name(it["name"]) if pos_kind(it) == "fonds" else nice_name(it["name"]), pos_kind(it), rnd(it["w"], 2), by_isin.get(it["isin"]) if pos_kind(it) == "fonds" else None, plab(it), it["isin"]]
                  for it in sorted(its, key=lambda x: -x["w"])[:40 if npos_f(its) > 10 else 15]]
        curs: dict[str, float] = {}
        for it in its:
            curs[it.get("cur") or "?"] = curs.get(it.get("cur") or "?", 0) + it["w"]
        npos = {k_: sum(1 for it in its if pos_kind(it) == k_) for k_ in ("aktie", "anleihe", "fonds")}
        n_direct = sum(1 for it in its if pos_kind(it) == "aktie")
        meta[key] = {"art": "bericht", "quelle": r["typ"], "stichtag": r["stichtag"], "url": r["url"], "n": len(pos), "hit": hit,
                     "ndirekt": n_direct,
                     "pos": allpos, "cur": [[c, rnd(w, 1)] for c, w in sorted(curs.items(), key=lambda x: -x[1])[:6]], "npos": npos,
                     "wsum": rnd(sum(agg.values()), 1), "wall": rnd(art["aktie"], 1), "aktien": rnd(art["aktie"], 1),
                     "anleihen": rnd(art["anleihe"], 1), "fonds": rnd(max(art["fonds"], 0), 1), "tranchen": r.get("tranchen") or [],
                     "durch": {k_: rnd(v_, 1) for k_, v_ in durch.get(name, {}).items()} or None}
        nf += 1
    log(f"  Fondsbestände aus Berichten: {nf} Fonds")


# ----------------------------------------------------------------- iShares (DE): Halbjahres-/Jahresberichte
ISH_BASE = "https://www.ishares.com/gls-download/literature/"
_ISH_POS = re.compile(r"^(?P<pre>.*?)\s*(?P<isin>[A-Z]{2}[A-Z0-9]{9}[0-9])\s+(?:STK|Units|ANT|Shares)\s+[\d\.,\s\-]+?\s(?P<cur>[A-Z]{3})\s+"
                      r"[\d\.,]+\s+[\d\.,]+\s+(?P<pct>-?\d{1,3}[\.,]\d{2})\s*$")
_ISH_CUT = re.compile(r"\s(Registered|Reg\.|Namens-|Inhaber-|Aandelen|Actions|Act\.|Bearer|Shares|Navne|Nam\.-|Inh\.|Vorzugs|Rg\.|Shs|Ordinary|Class|Cl\.|Acções|Azioni|Aktier|Akt\.|Bonus|Port\.|Nom\.|Acciones|Accions|Bons|Andeler|Unit|Partizipationsscheine|Genussscheine|Namn-Aktier|Acç\.)(?=[\s,]|$).*$")


# Geschäftsjahresende (Monat) der irischen iShares-Umbrellas: Berichte heißen
# "{umbrella}-{annual|interim}-report-en-{tt}-{mon}-{jj}.pdf" (Jahresbericht am GJ-Ende, Halbjahresbericht 6 Monate später)
ISH_FY = {"i": 2, "ii": 10, "iii": 6, "iv": 5, "v": 11, "vi": 3}
_MON_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def ish_dated(plc: str) -> list[str]:
    fy = ISH_FY.get(plc)
    n = f"ishares-{plc}-plc" if plc != "i" else "ishares-plc"
    out = []
    if not fy:  # Geschäftsjahr unbekannt: alle Monatsenden der letzten 15 Monate prüfen
        for k in range(15):
            end = (pd.Timestamp(NOW.replace(tzinfo=None)) - pd.offsets.MonthEnd(k + 1)).to_pydatetime()
            for kind in ("annual", "interim"):
                out.append((end, f"{ISH_BASE}{kind}-report/{n}-{kind}-report-en-{end.day:02d}-{_MON_ABBR[end.month - 1]}-{end.year % 100:02d}.pdf"))
        return [u for _, u in sorted(out, reverse=True)]
    for y in (NOW.year, NOW.year - 1, NOW.year - 2):
        for kind, mon in (("annual", fy), ("interim", (fy + 5) % 12 + 1)):
            yy = y if kind == "annual" or mon > fy else y + 1
            end = (pd.Timestamp(year=yy, month=mon, day=1) + pd.offsets.MonthEnd(0)).to_pydatetime()
            if end > NOW.replace(tzinfo=None):
                continue
            out.append((end, f"{ISH_BASE}{kind}-report/{n}-{kind}-report-en-{end.day:02d}-{_MON_ABBR[mon - 1]}-{end.year % 100:02d}.pdf"))
    return [u for _, u in sorted(out, reverse=True)]


def ish_candidates(src: dict) -> list[str]:
    out = ish_dated(src["plc"]) if src.get("plc") else []
    for y in (NOW.year, NOW.year - 1):
        if src.get("plc"):
            n = f"ishares-{src['plc']}-plc" if src["plc"] != "i" else "ishares-plc"
            out += [f"{ISH_BASE}interim-report/{n}-interim-report-{y}.pdf", f"{ISH_BASE}interim-report/{n}-en-interim-report-{y}.pdf",
                    f"{ISH_BASE}annual-report/{n}-en-annual-report-{y}.pdf", f"{ISH_BASE}annual-report/{n}-annual-report-{y}.pdf"]
        elif src.get("slug"):
            sl = src["slug"]
            out += [f"{ISH_BASE}interim-report/{sl}-en-interim-report-{y}.pdf", f"{ISH_BASE}annual-report/{sl}-jahresbericht-{y}.pdf",
                    f"{ISH_BASE}annual-report/{sl}-pc-annual-report-{y}.pdf"]
        else:
            out += [f"{ISH_BASE}interim-report/ishares-invag-de-en-interim-report-{y}.pdf",
                    f"{ISH_BASE}annual-report/ishares-invag-de-en-annual-report-{y}.pdf"]
    return out


_ISH_DOC: dict[str, list[str]] = {}


def ish_pages(url: str) -> list[str]:
    if url not in _ISH_DOC:
        import pypdf
        r = pypdf.PdfReader(io.BytesIO(HTTP.get(url, timeout=180).content))
        _ISH_DOC[url] = [(p.extract_text() or "") for p in r.pages]
    return _ISH_DOC[url]


def parse_ish_report(pages: list[str], fonds: str | None) -> dict:
    start_re = re.compile(r"(Statement of Net Assets as at (\d\d)/(\d\d)/(\d{4})|Vermögensaufstellung zum (\d\d)\.(\d\d)\.(\d{4}))")
    starts = [i for i, t in enumerate(pages) if start_re.search(t)]
    sel = []
    for k, i in enumerate(starts):
        head = pages[i][:400].upper()
        if fonds and fonds.upper() not in head.replace("\n", " "):
            continue
        end = starts[k + 1] if k + 1 < len(starts) else len(pages)
        sel = pages[i:end]
        break
    if not sel:
        return {}
    m = start_re.search(sel[0])
    stichtag = f"{m.group(2)}.{m.group(3)}.{m.group(4)}" if m.group(2) else f"{m.group(5)}.{m.group(6)}.{m.group(7)}"
    items, buf = [], []
    for ln in "\n".join(sel).splitlines():
        ln = ln.strip()
        mm = _ISH_POS.match(ln)
        if mm and isin_ok(mm["isin"]):
            name = " ".join(buf + [mm["pre"]]).strip()
            name = _ISH_CUT.sub("", name).strip(" ,") or name
            items.append({"isin": mm["isin"], "name": name, "tick": "", "code": "", "cur": mm["cur"],
                          "w": float(mm["pct"].replace(".", "").replace(",", ".")) if "," in mm["pct"] else float(mm["pct"]), "grp": "Aktien"})
            buf = []
        elif re.match(r"^(Aktien|Shares|Wertpapiervermögen|Securities|Börsengehandelte Wertpapiere|Exchange-traded securities|Summe|Total)\s*[\d\.,\s]*$", ln) or re.match(r"^(Gattungsbezeichnung|Designation)", ln) or re.search(r"\d[\d,\.]{5,}\s+-?\d{1,3}[\.,]\d{2}$", ln):
            buf = []
        elif len(ln) < 80 and not re.match(r"^\d", ln):
            buf = (buf + [ln])[-2:]
    return {"stichtag": stichtag, "items": items}


_MON_EN = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                            "october", "november", "december"])}
_IRL_POS = re.compile(r"^([A-Z]{3})\s+([\d,]+)\s+(.+)$")
_IRL_END = re.compile(r"^(.*?)\s+\(?[\d,]+\)?\s+(-?\d{1,3}\.\d{2})$")


def parse_irl_report(pages: list[str], fonds: str) -> dict:
    """Schedule of Investments eines irischen iShares-Sammelberichts (Aktienteil eines Teilfonds)."""
    norm = lambda x: re.sub(r"[^A-Z0-9&$€%]", "", x.upper().replace("(CONTINUED)", ""))  # noqa: E731
    want = norm(fonds)
    sel, stichtag = [], None
    heads = {}
    for t in pages:
        m = re.search(r"SCHEDULE OF INVESTMENTS(?: \(UNAUDITED\))?(?: \(continued\))?\n(.+?)\nAs at (\d{1,2}) (\w+) (\d{4})", t)
        if m:
            heads.setdefault(norm(m.group(1)), 0)
    if want not in heads:  # Name leicht abweichend (gekürzt/umbenannt): eindeutigen Teilfonds mit gleichem Namensanfang nehmen
        near = [h for h in heads if len(want) >= 18 and (h.startswith(want) or want.startswith(h))]
        if len(near) == 1:
            want = near[0]
    for t in pages:
        m = re.search(r"SCHEDULE OF INVESTMENTS(?: \(UNAUDITED\))?(?: \(continued\))?\n(.+?)\nAs at (\d{1,2}) (\w+) (\d{4})", t)
        if m and norm(m.group(1)) == want:
            sel.append(t)
            if not stichtag and m.group(3).lower() in _MON_EN:
                stichtag = f"{int(m.group(2)):02d}.{_MON_EN[m.group(3).lower()]:02d}.{m.group(4)}"
    items, eq, cur_item, country = [], False, None, None
    for ln in "\n".join(sel).splitlines():
        ln = ln.strip()
        if re.match(r"^Equities\b", ln):
            eq = True
            continue
        if re.match(r"^(Total equities|Financial derivative|Corporate debt|Government debt|Collective investment|Rights\b|Warrants\b|"
                    r"Total transferable|Total value of investments|Futures|Forward currency|Bonds\b|Supranational)", ln):
            eq = False
            cur_item = None
            continue
        if not eq:
            continue
        mc = re.match(r"^(.+?) \(\d{1,2} \w+ \d{4}: -?[\d\.]+%\)$", ln)
        if mc and not mc.group(1).startswith("Total"):
            country = mc.group(1).replace(" (continued)", "")
            continue
        mp = _IRL_POS.match(ln)
        if mp and not ln.startswith("Total"):
            cur_item = {"cur": mp.group(1), "txt": mp.group(3)}
        elif cur_item is not None and not ln.startswith("Total"):
            cur_item["txt"] += " " + ln
        else:
            continue
        me = _IRL_END.match(cur_item["txt"])
        if me:
            name = re.sub(r"\s*[\^~∞±*†]+\s*$", "", me.group(1)).strip()
            items.append({"isin": "N:" + name_key(name), "name": name, "tick": "", "code": "", "cur": cur_item["cur"],
                          "w": float(me.group(2)), "grp": "Aktien", "land": country})
            cur_item = None
        elif len(cur_item["txt"]) > 200:
            cur_item = None
    return {"stichtag": stichtag, "items": [i for i in items if i["isin"] != "N:"]} if items else {}


VG_BASE = "https://fund-docs.vanguard.com/"
_VG_POS = re.compile(r"^(.*?\D)\s+([\d,]{1,15})\s+\(?[\d,]+\)?\s+(-?\d{1,3}\.\d{2})%$")


def parse_vg_report(pages: list[str], fonds: str) -> dict:
    """Statement of Investments eines Vanguard-Funds-plc-Berichts (Aktienpositionen eines Teilfonds)."""
    sel, stichtag = [], None
    for t in pages:
        head = t[:300].split("\n")
        hs = [h.strip() for h in head[:7]]
        if fonds not in hs or not ("Statement of Investments" in t[:120] or ".." in hs[:4]):
            continue
        sel.append(t)
        if not stichtag:
            m = re.search(r"Statement of Investments\n(\d{1,2}) (\w+) (\d{4})", t)
            if m and m.group(2).lower() in _MON_EN:
                stichtag = f"{int(m.group(1)):02d}.{_MON_EN[m.group(2).lower()]:02d}.{m.group(3)}"
    items, buf, eq, country = [], "", False, None
    for ln in "\n".join(sel).splitlines():
        ln = ln.strip()
        if not ln or ln in ("..", fonds) or ln.startswith(("Financial Statements", "Statement of Investments", "Number of", "Shares", "Fair Value",
                                                            "% of Total", "Net Assets", "US Dollars", "Euro (", "Pound")):
            continue
        if re.match(r"^Equities \d", ln):
            eq = True
            buf = ""
            continue
        if re.match(r"^(Total Equities|Total Transferable|Total Investments|Total Net Assets)", ln) or (
                re.match(r"^(Financial Derivative|Futures|Forward|Investment Funds|Collective Investment|Rights|Warrants|Cash|Other Assets)", ln)
                and re.search(r"\(\d{1,2} \w+ \d{4}:", ln)):
            eq = False
            buf = ""
            continue
        if not eq:
            continue
        mc = re.match(r"^(.+?) \d{1,3}\.\d{2}% \(\d{1,2} \w+ \d{4}: -?[\d\.]+%\)$", ln)
        if mc:
            country, buf = mc.group(1), ""
            continue
        mp = _VG_POS.match((buf + " " + ln).strip() if buf else ln)
        if mp:
            name = re.sub(r"\s+", " ", mp.group(1)).strip()
            items.append({"isin": "N:" + name_key(name), "name": name, "tick": "", "code": "", "cur": "",
                          "w": float(mp.group(3)), "grp": "Aktien", "land": country})
            buf = ""
        elif re.match(r"^\d{1,3}$", ln):
            continue
        else:
            buf = (buf + " " + ln).strip()[-160:]
    return {"stichtag": stichtag, "items": [i for i in items if i["isin"] != "N:"]} if items else {}


def vg_report(src: dict, cache: dict) -> dict | None:
    """Vanguard-ETFs: Jahres- und Halbjahresbericht unter festen Adressen; der neuere Stichtag gewinnt (wöchentlich geprüft)."""
    key = "vg:" + src["fonds"]
    old = cache.get(key) or {}
    if old.get("checked") and (NOW - datetime.fromisoformat(old["checked"])).days < 6 and old.get("items"):
        return old
    best = old if old.get("items") else None
    d = lambda s_: datetime.strptime(s_, "%d.%m.%Y")  # noqa: E731
    for url in (VG_BASE + "etf-annual-report.pdf", VG_BASE + "etf-semiannual-report.pdf"):
        try:
            rep = parse_vg_report(ish_pages(url), src["fonds"])
        except Exception as ex:
            log(f"  Vanguard-Bericht {url.rsplit('/', 1)[-1]}: {ex}")
            continue
        if not rep.get("items") or sum(i["w"] for i in rep["items"]) < 50 or not rep.get("stichtag"):
            continue
        if not best or d(rep["stichtag"]) > d(best["stichtag"]) or best.get("url") == url:
            if not best or d(rep["stichtag"]) >= d(best["stichtag"]):
                best = {**rep, "url": url, "typ": "Halbjahresbericht" if "semiannual" in url else "Jahresbericht"}
    if best:
        best["checked"] = NOW.isoformat()
        cache[key] = best
    return best


def ish_report(src: dict, cache: dict) -> dict | None:
    key = src.get("slug") or (f"{src.get('plc') or 'invag'}:" + src.get("fonds", ""))
    old = cache.get(key) or {}
    if old.get("rv") == 3 and old.get("checked") and (NOW - datetime.fromisoformat(old["checked"])).days < 6 and old.get("items"):
        return old
    if old.get("miss") and (NOW - datetime.fromisoformat(old["checked"])).days < 6:
        return None  # Teilfonds in diesem Umbrella zuletzt nicht gefunden
    best = old if old.get("items") else None
    seen = set(old.get("seen") or [])
    n_dated = 0
    for url in ish_candidates(src):
        try:
            if url in seen or (best and best.get("url") == url):
                continue
            if HTTP.head(url, timeout=20, allow_redirects=False).status_code != 200:
                continue
            seen.add(url)
            rep = parse_irl_report(ish_pages(url), src["fonds"]) if src.get("plc") else parse_ish_report(ish_pages(url), src.get("fonds"))
        except Exception as ex:
            log(f"  iShares-Bericht {url.rsplit('/', 1)[-1]}: {ex}")
            continue
        if "-report-en-" in url and not rep.get("items"):
            n_dated += 1  # lesbar, Teilfonds aber nicht enthalten
            if n_dated >= 2:
                break
        if not rep.get("items") or sum(i["w"] for i in rep["items"]) < 50:
            continue
        d = lambda s: datetime.strptime(s, "%d.%m.%Y")  # noqa: E731
        if not best or d(rep["stichtag"]) > d(best["stichtag"]):
            best = {**rep, "url": url, "typ": "Halbjahresbericht" if "interim" in url else "Jahresbericht"}
        if "-report-en-" in url:  # datierte Namen: das Datum im Namen stimmt nicht immer mit dem Stichtag überein -> zwei lesbare prüfen
            n_dated += 1
            if n_dated >= 2:
                break
    if best:
        best["checked"] = NOW.isoformat()
        best["rv"] = 3  # datierte Umbrella-Berichte berücksichtigt
        best["seen"] = sorted(seen)
        cache[key] = best
    else:
        cache[key] = {"rv": 3, "miss": True, "checked": NOW.isoformat(), "seen": sorted(seen)}
    return best


def add_ish_reports(cfg: dict, keys: dict, first: dict, by_sym: dict, tops: dict, meta: dict, isin_map: dict, extra: set):
    cache = load(DATA / "ish_reports.json") or {}
    for e in cfg.get("etfs", []):
        src = (e.get("holdings") or {}).get("bericht")
        if not src:
            continue
        if src.get("plc_try"):
            hint = (cache.get("plc:" + e["isin"]) or {}).get("plc")
            rep = None
            for p_ in ([hint] if hint else []) + [x for x in src["plc_try"] if x != hint]:
                rep = safe(lambda: ish_report({**src, "plc": p_}, cache), None, f"Bericht {e['isin']}")
                if rep:
                    cache["plc:" + e["isin"]] = {"plc": p_}
                    break
        else:
            rep = safe(lambda: vg_report(src, cache) if src.get("vanguard") else ish_report(src, cache), None, f"Bericht {e.get('ticker') or e['isin']}")
        if not rep:
            continue
        symmap = resolve_positions(rep["items"], keys, first, isin_map, min_w=0.04)
        agg, names = {}, {}
        for it in rep["items"]:
            s_ = symmap.get(it["isin"])
            if s_:
                agg[s_] = agg.get(s_, 0) + it["w"]
                if not it["isin"].startswith("N:"):
                    isin_map.setdefault(it["isin"], s_)
                extra.add(s_)
        keep = {}
        fsm = meta.get(e["isin"]) or {}
        if src.get("rest") and fsm.get("art") == "factsheet":
            # aktuelle Top 10 aus dem Factsheet behalten, nur den Rest aus dem (älteren) Bericht
            keep = {s_: w for s_, lst in by_sym.items() for i, w in lst if i == e["isin"] and w is not None}
            restw = {s_: w for s_, w in agg.items() if s_ not in keep}
            rt = sum(restw.values()) or 1
            target = max(0.0, 100 - fsm.get("wsum", sum(keep.values())))
            agg = {**keep, **{s_: w / rt * target for s_, w in restw.items()}}
        # bisherige Näherung für diesen ETF ersetzen
        for s_ in list(by_sym):
            by_sym[s_] = [x for x in by_sym[s_] if x[0] != e["isin"]]
            if not by_sym[s_]:
                del by_sym[s_]
        for s_, w in agg.items():
            by_sym.setdefault(s_, []).append([e["isin"], rnd(w, 3)])
        wall = sum(i["w"] for i in rep["items"])
        if keep:
            meta[e["isin"]] = {**fsm, "art": "factsheet+bericht", "rest": len(agg) - len(keep), "rest_stichtag": rep["stichtag"],
                               "rest_url": rep["url"], "wall": rnd(sum(agg.values()), 1), "hit": len(agg)}
            log(f"  iShares-Bericht {e.get('ticker') or e.get('name') or e['isin']}: Top 10 laut Factsheet, Rest {len(agg) - len(keep)} Titel laut Bericht per {rep['stichtag']}")
            continue
        top = sorted(rep["items"], key=lambda x: -x["w"])[:10]
        tops[e["isin"]] = [[nice_name(t["name"]), rnd(t["w"], 2)] for t in top]
        meta[e["isin"]] = {"art": "bericht", "quelle": f"{rep['typ']} iShares", "stichtag": rep["stichtag"], "url": rep["url"],
                           "n": len(rep["items"]), "hit": len(agg), "wsum": rnd(sum(agg.values()), 1), "wall": rnd(wall, 1),
                           "aktien": rnd(wall, 1), "anleihen": 0}
        log(f"  iShares-Bericht {e.get('ticker') or e.get('name') or e['isin']}: {len(rep['items'])} Positionen per {rep['stichtag']} ({wall:.1f} %), {len(agg)} zugeordnet")
    save(DATA / "ish_reports.json", cache, quiet=True)


# ----------------------------------------------------------------- Langfrist-Historie (Beratung)
def build_long_history():
    idx = CFG.get("long_indices", {})
    # Eintrag: "SYMBOL": "Name"  oder  "SCHLÜSSEL": {"name": …, "try": [Kandidaten]} – genommen wird die längste Reihe
    cand = {k: (v.get("try") if isinstance(v, dict) else [k]) for k, v in idx.items() if not (isinstance(v, dict) and v.get("csv"))}
    data = history_block([s for c in cand.values() for s in c], period="max", interval="1mo", chunk=20)
    out = {}
    for key, v in idx.items():
        if key not in cand:
            continue
        name = v["name"] if isinstance(v, dict) else v
        best = max((data.get(s) for s in cand[key] if data.get(s) is not None), key=len, default=None)
        if best is None or len(best) < 24:
            log(f"  Langfrist: {name} ohne Daten")
            continue
        src = next(s for s in cand[key] if data.get(s) is best)
        log(f"  Langfrist: {name} aus {src} ab {best.index[0].date()}")
        out[key] = {"name": name, "src": src, "m": series_pairs(best, 2)}
    # Reihen aus einer CSV-Datei (z. B. Gold seit 1970), am aktuellen Rand mit Yahoo-Monatsschlüssen verlängert
    for key, v in idx.items():
        if not (isinstance(v, dict) and v.get("csv")):
            continue
        try:
            df = pd.read_csv(io.StringIO(HTTP.get(v["csv"], timeout=60).text))
            ser = pd.Series(df.iloc[:, 1].astype(float).values, index=pd.to_datetime(df.iloc[:, 0].astype(str) + "-01"))
            ser = ser[ser.index >= pd.Timestamp(v.get("from", "1970-01") + "-01")]
            ext = None
            if v.get("extend"):
                ext = safe(lambda: yf.Ticker(v["extend"]).history(period="2y", interval="1mo", auto_adjust=False)["Close"].dropna(),
                           None, f"Verlängerung {v['extend']}")
                log(f"  Langfrist: Verlängerung {v['extend']}: {0 if ext is None else len(ext)} Monate")
            if ext is not None and len(ext):
                ext.index = pd.to_datetime(ext.index).tz_localize(None).to_period("M").to_timestamp()
                ser = pd.concat([ser, ext[ext.index > ser.index[-1]]])
            out[key] = {"name": v["name"], "src": "csv", "m": series_pairs(ser, 2)}
            log(f"  Langfrist: {v['name']} aus CSV ab {ser.index[0].date()} bis {ser.index[-1].date()}")
        except Exception as ex:
            log(f"WARNUNG Langfrist {v['name']}: {ex}")
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
    hist5 = safe(lambda: t.history(period="5y", interval="1wk", auto_adjust=False)["Close"].dropna(), None, f"5y {sym}")
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
    actions = safe(lambda: analyst_actions(t), [], f"Analystenmeinungen {sym}")
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
        "stats": stats, "analyst": analyst, "actions": actions,
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
    return detail, prof


def analyst_actions(t: yf.Ticker, months=18, limit=40) -> list[dict]:
    """Einzelne Analystenmeinungen: Haus, Datum, Einstufung und Kursziel (soweit Yahoo sie führt)."""
    df = t.upgrades_downgrades
    if df is None or df.empty:
        return []
    df = df.copy()
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    df.index = idx
    df = df[df.index >= pd.Timestamp(NOW.replace(tzinfo=None)) - pd.DateOffset(months=months)].sort_index(ascending=False)
    col = lambda r, *names: next((r[n] for n in names if n in r and pd.notna(r[n]) and r[n] != ""), None)  # noqa: E731
    out = []
    for d, r in df.head(limit).iterrows():
        pt, ppt = col(r, "currentPriceTarget"), col(r, "priorPriceTarget")
        out.append({"d": d.strftime("%Y-%m-%d"), "firm": str(col(r, "Firm") or ""), "to": col(r, "ToGrade"), "from": col(r, "FromGrade"),
                    "act": col(r, "Action"), "pt": rnd(float(pt), 2) if pt not in (None, 0) else None,
                    "ppt": rnd(float(ppt), 2) if ppt not in (None, 0) else None, "pta": col(r, "priceTargetAction")})
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
    _INV_ISIN.clear()
    for _i, _s in ((load(DATA / "holdings.json") or {}).get("isin") or {}).items():
        if not _i.startswith("N:"):
            _INV_ISIN.setdefault(_s, _i)

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
        done = fails = na = npt = 0
        recent_actions: list[dict] = []
        for s in todo[:batch]:
            if elapsed_min() > budget:
                log("Zeitbudget erreicht – Rest folgt in der nächsten Nacht")
                break
            try:
                d, p = build_ticker(s, edgar, closes_uni.get(s), sent, google=False)
                save(TICK / f"{file_safe(s)}.json", d, quiet=True)
                fund[s] = p
                done += 1
                na += bool(d.get("actions"))
                npt += any(a.get("pt") for a in d.get("actions") or [])
                cut = (NOW - timedelta(days=10)).strftime("%Y-%m-%d")
                for a in d.get("actions") or []:
                    if a.get("d", "") >= cut:
                        recent_actions.append({**a, "s": s, "n": p.get("n"), "cur": p.get("cur")})
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
        log(f"  Analystenmeinungen: {na} Titel mit Einzelmeldungen, davon {npt} mit Kurszielen")
        old_ra = [a for a in (load(DATA / "analyst_recent.json", {}) or {}).get("items", [])
                  if a.get("d", "") >= (NOW - timedelta(days=10)).strftime("%Y-%m-%d")]
        seen = {(a["s"], a["d"], a.get("firm")) for a in recent_actions}
        save(DATA / "analyst_recent.json", {"updated": NOW.isoformat(),
             "items": recent_actions + [a for a in old_ra if (a["s"], a["d"], a.get("firm")) not in seen]})

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
            "rmean": rnd(f.get("rmean"), 2), "nan": f.get("nan"),
            "pm": rnd((f.get("pm") or 0) * 100, 1) if f.get("pm") is not None else None,
            "er": f.get("er"), "src": u.get("src", []), "det": det, "wl": s in watch, "i": u.get("i") or _INV_ISIN.get(s),
        })
    for s in wl:
        if (TICK / f"{file_safe(s)}.json").exists():
            meta_map[s] = file_safe(s)
    save(DATA / "universe.json", {"updated": NOW.isoformat(), "n": len(rows), "rows": rows,
                                  "sources": sorted({x for r in rows for x in r["src"]})})
    save(DATA / "fundamentals.json", fund, quiet=True)
    editions = safe(build_marktupdate, [], "Marktupdate") or []
    calendar = build_calendar(fund, watch, editions)
    save(DATA / "calendar.json", calendar)

    # --- Makro (nachts oder wenn noch nicht vorhanden)
    macro_old = load(DATA / "macro.json", None)
    fred_missing = bool(os.environ.get("FRED_API_KEY")) and macro_old is not None and \
        not any(x.get("source") == "FRED" for x in macro_old.get("series", []))
    ids_old = {x.get("id") for x in (macro_old or {}).get("series", [])}
    new_series = bool(os.environ.get("FRED_API_KEY")) and any(f[0] not in ids_old for f in FRED)
    if MODE == "nightly" or macro_old is None or fred_missing or new_series:
        log("Makro laden …")
        safe(build_macro, None, "Makro")
    safe(lambda: merge_manual_macro(editions), None, "PMI-Zusammenführung")

    safe(build_bonds, None, "Bundesanleihen")
    if MODE == "nightly" or not (DATA / "zertifikate.json").exists():
        safe(build_certificates, None, "Zertifikate")
    if MODE == "nightly" or not (DATA / "anleihen.json").exists():
        safe(build_bank_bonds, None, "Bank-/Wohnbauanleihen")
    safe(build_etfs, None, "ETFs")
    if MODE == "nightly" or not set(CFG.get("long_indices", {})) <= set((load(DATA / "long.json") or {}).get("series", {})):
        safe(build_long_history, None, "Langfrist-Historie")
    _h = load(DATA / "holdings.json") or {}
    _need = {e["isin"] for e in json.loads((ROOT / "manual" / "etfs.json").read_text(encoding="utf-8")).get("etfs", []) if e.get("holdings")}
    if MODE == "nightly" or "meta" not in _h or "isin" not in _h or _h.get("v") != 8 or not _need <= set(_h.get("meta", {})):
        safe(lambda: build_holdings(uni_list, fund), None, "ETF-Positionen")
        safe(build_funds, None, "Fonds (Gebühren aus neuen Berichten)")

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
    movers = safe(lambda: mover_news(rows, sent), [], "News der Top-Mover")
    save(DATA / "news.json", {
        "updated": NOW.isoformat(), "model": CFG.get("sentiment_model") if sent.pipe else None,
        "method": {"halfLifeDays": HALF_LIFE_DAYS, "windowDays": WINDOW_DAYS},
        "general": sorted(general, key=by_date, reverse=True),
        "tickers": sorted([n for d in details.values() for n in d["news"]], key=by_date, reverse=True)[:250],
        "summary": summary, "movers": movers,
    })
    safe(lambda: write_briefing(market, movers, calendar, rows), None, "Briefing-Datei")
    save(DATA / "meta.json", {"updated": NOW.isoformat(), "mode": MODE, "tickers": meta_map,
                              "universe": len(rows), "details": len(meta_map), "marktupdate": len(editions)})
    safe(lambda: build_status(len(rows), len(meta_map)), None, "Datenstatus")
    log(f"Fertig nach {elapsed_min():.1f} Min. · Universum {len(rows)} · Detailseiten {len(meta_map)}")


def build_status(uni_n: int, details_n: int):
    """Datenstatus für die Ansicht STATUS: Stand je Datei, Abdeckung der Positionen, Fonds-Kurse/-Kosten, Warnungen dieses Laufs."""
    files = {}
    for k in ("market", "universe", "news", "calendar", "macro", "bonds", "etfs", "fonds", "holdings", "history", "long", "marktupdate",
              "zertifikate", "anleihen"):
        d = load(DATA / f"{k}.json")
        if isinstance(d, dict):
            files[k] = d.get("updated") or d.get("stand")
    h = load(DATA / "holdings.json") or {}
    meta = h.get("meta") or {}
    etfs = (load(DATA / "etfs.json") or {}).get("etfs", [])
    fonds = (load(DATA / "fonds.json") or {}).get("fonds", [])
    old = load(DATA / "status.json") or {}
    runs = [r for r in (old.get("runs") or []) if r.get("mode")][-29:]
    runs.append({"t": NOW.isoformat(), "mode": MODE, "min": rnd(elapsed_min(), 1), "warn": len(WARNINGS)})
    last_n = old.get("nightly") or {}
    if MODE == "nightly":
        last_n = {"t": NOW.isoformat(), "min": rnd(elapsed_min(), 1), "warn": WARNINGS[:40]}
    et = [{"isin": e["isin"], "t": e.get("ticker"), "n": e.get("name"), "px": bool(e.get("px")), "ter": e.get("ter"),
           "melde": e.get("meldefonds"), "art": (meta.get(e["isin"]) or {}).get("art"), "quelle": (meta.get(e["isin"]) or {}).get("quelle"),
           "stichtag": (meta.get(e["isin"]) or {}).get("stichtag") or (meta.get(e["isin"]) or {}).get("rest_stichtag"),
           "pos": (meta.get(e["isin"]) or {}).get("n"), "hit": (meta.get(e["isin"]) or {}).get("hit")} for e in etfs]
    fo = [{"n": f["name"], "kag": f.get("kag"), "nav": f.get("nav_sym"), "pxd": f.get("pxd"), "geb": bool(f.get("gebuehr")),
           "stichtag": (meta.get("F:" + f["name"]) or {}).get("stichtag"), "quelle": (meta.get("F:" + f["name"]) or {}).get("quelle"),
           "isins": len(f.get("isins") or [])} for f in fonds]
    save(DATA / "status.json", {"updated": NOW.isoformat(), "mode": MODE, "min": rnd(elapsed_min(), 1), "files": files,
                                "universe": uni_n, "details": details_n, "extra": len(h.get("extra") or []), "warn": WARNINGS[:40],
                                "nightly": last_n, "runs": runs, "etfs": et, "fonds": fo}, quiet=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

