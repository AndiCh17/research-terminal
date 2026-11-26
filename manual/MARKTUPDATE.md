# Marktupdate → Terminal

Das wöchentliche Marktupdate (Projekt „WP-Research“, Auslöser „Marktupdate“) wird zusätzlich als
Datendatei ins Terminal übernommen.

## Ablauf (wird von Claude erledigt)
1. Ausgabe als JSON nach dem Schema unten erstellen.
2. `python scripts/mkt_encrypt.py ausgabe.json` → schreibt `manual/marktupdate/<JJJJ>-KW<NN>.json.enc`.
   Der Klartext wird **nicht** ins Repository gelegt (Repository ist öffentlich, Raiffeisen-Research-Inhalte
   haben einen Weitergabevorbehalt).
3. Nur die `.enc`-Datei committen und pushen → der Workflow startet automatisch.
4. Der Workflow entschlüsselt mit dem Secret `MKT_PRIVATE_KEY` und veröffentlicht die Ausgabe
   passwortgeschützt im Terminal (Ansicht MKT / F7). PMI-Werte erscheinen unter ECO, Termine im Kalender.

## Schema
```json
{
  "jahr": 2026, "kw": 49, "datum": "2026-11-30", "datenstand": "2026-11-27",
  "titel": "Kurzer Titel der Ausgabe",
  "kpis": [{"label": "S&P 500", "wert": "6.512", "delta": "+1,2 % Woche", "dir": 1}],
  "sections": [
    {"title": "Kernaussage", "type": "text", "text": "Absatz mit **Fett**.\n\nZweiter Absatz."},
    {"title": "Marktdaten", "type": "table", "columns": ["Index", "Stand", "Woche", "YTD"],
     "rows": [["DAX", "24.900", "−1,1 %", "+12,0 %"]], "note": "Quelle, Stichtag"},
    {"title": "Meine Einschätzung", "type": "list", "items": ["…"], "entwurf": true}
  ],
  "erwartungen": [{"text": "…", "bis": "2026-12-11", "status": "offen | eingetreten | nicht eingetreten"}],
  "pmi": [{"reihe": "Euroraum Industrie-PMI", "monat": "2026-11", "wert": 49.5, "quelle": "HCOB"}],
  "termine": [{"d": "2026-12-04", "title": "US-Arbeitsmarktbericht"}],
  "quellen": ["…"]
}
```
- `sections`: beliebig viele, in Berichtsreihenfolge; `type` = `text`, `list` oder `table`.
  Die Kernaussage heißt „Kernaussage…“, damit die Übersicht sie anzeigt.
- PMI-Reihennamen gleich halten, damit Verläufe entstehen („Euroraum Industrie-PMI“, „Euroraum Dienstleister-PMI“,
  „Österreich EMI (UniCredit Bank Austria)“, „USA ISM Industrie“, „USA ISM Dienstleister“).
- Kein wörtliches Zitieren aus Quellpublikationen; RR-Inhalte nur zusammengefasst.
