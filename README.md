# Research Terminal

Ein Bloomberg-artiges Research-Terminal, das komplett kostenlos auf GitHub läuft.
Auf dem Arbeits-PC brauchst du nur den Browser, installieren musst du nichts.

**So funktioniert es:** GitHub startet alle 30 Minuten ein Python-Skript in der Cloud (Modus *fast*):
Indizes, Märkte, Watchlist, Tageskurse aller ca. 2.500–3.000 Titel des Universums, Schlagzeilen.
Einmal pro Nacht (01:15 UTC, Modus *nightly*) werden zusätzlich die Indexlisten aktualisiert, Kennzahlen und
Detailseiten für mehrere hundert Titel erstellt (rotierend, bis alle erfasst sind), Makrodaten und Kalender geladen.

Die Daten liegen verschlüsselt im Branch `data-store` (immer nur der neueste Stand, ohne Verlauf),
die Webseite wird direkt aus dem Workflow veröffentlicht.

---

## Einrichtung (ca. 15 Minuten, einmalig)

### 1. GitHub-Konto
Auf <https://github.com/signup> kostenlos registrieren, falls noch nicht vorhanden.

### 2. Repository anlegen
1. Rechts oben **+** → **New repository**
2. Name: `research-terminal`
3. **Public** auswählen. GitHub Pages ist im Gratis-Tarif nur für öffentliche Repositories verfügbar.
   Sichtbar sind dann Code, Marktdaten und deine Watchlist, aber keine Kundendaten.
   Deshalb gehören **niemals** Kundennamen oder Depotbestände in dieses Repository.
4. **Create repository**

### 3. Dateien hochladen
1. Im neuen Repository auf **uploading an existing file** klicken
2. Den **Inhalt** des entpackten ZIP-Ordners hineinziehen (`index.html`, `config.json`, `requirements.txt`,
   `README.md`, `.nojekyll` sowie die Ordner `scripts`, `data` und `.github`)
3. **Commit changes**

> **Wichtig:** Der Ordner `.github` ist auf Mac und teilweise unter Windows unsichtbar und wird dann nicht mit hochgeladen.
> Prüfe nach dem Upload, ob es im Repository `.github/workflows/update.yml` gibt. Falls nicht:
> **Add file → Create new file** → als Dateiname `.github/workflows/update.yml` eintippen →
> Inhalt der Datei aus dem ZIP hineinkopieren → **Commit changes**.

### 4. Passwort festlegen
**Settings → Secrets and variables → Actions → New repository secret**
- Name: `TERMINAL_PASSWORD`
- Secret: dein Wunschpasswort (mind. 12 Zeichen, nicht dein Bank-Passwort)
- **Add secret**

Ab dem nächsten Lauf werden alle Daten-Dateien verschlüsselt (AES-256). Die Seite fragt beim Öffnen
nach dem Passwort und entschlüsselt die Daten erst dann im Browser. Wer nur den Link hat, sieht weder Kurse
noch Watchlist-Daten, auch nicht über die Dateien direkt.

Passwort ändern: Secret aktualisieren → Workflow manuell starten. Danach gilt nur noch das neue Passwort.

### 5. Schreibrechte für den Workflow
**Settings → Actions → General → Workflow permissions → Read and write permissions → Save**

### 6. Deine Daten eintragen
`config.json` im Repository öffnen → Stift-Symbol (Bearbeiten):
- `sec_user_agent`: Name und **deine E-Mail-Adresse** eintragen. Die SEC verlangt das und blockiert sonst die Abfrage.
- `watchlist`: deine Titel im Yahoo-Format (siehe unten)
- **Commit changes**. Damit startet der Workflow automatisch.

### 7. Ersten Lauf prüfen
Reiter **Actions** → „Daten aktualisieren“. Der erste Lauf dauert ca. 5–10 Minuten, weil FinBERT einmalig
heruntergeladen wird. Danach sind es 2–4 Minuten. Grüner Haken = fertig.
Manuell starten: **Run workflow**.

### 8. Webseite einschalten
**Settings → Pages → Build and deployment → Source: „GitHub Actions“** auswählen (nicht „Deploy from a branch“).

Danach einmal **Actions → Daten aktualisieren → Run workflow → mode: nightly** starten. Der erste Nachtlauf
dauert bis zu ca. 2 Stunden. Die Seite ist aber schon nach dem ersten schnellen Lauf erreichbar.

Nach 1–2 Minuten ist das Terminal erreichbar unter
`https://<dein-github-name>.github.io/research-terminal/`

Als Lesezeichen speichern. Am Handy: „Zum Home-Bildschirm hinzufügen“.

---

## Bedienung

| Eingabe | Wirkung |
|---|---|
| `AAPL` oder `VOE.VI` + GO | Detailseite des Titels |
| `DAX`, `Gold` | großer Chart für Index/Markt |
| `WEI` | Weltindizes, Devisen, Rohstoffe (Zeile anklicken = Chart) |
| `W` | Watchlist (sortierbar) |
| `SCR` | Screener über das ganze Universum |
| `ECO` | Makro USA / Euroraum / Österreich, Frühindikatoren, Staatsanleihen |
| `TOP` | News mit FinBERT-Stimmung (ⓘ erklärt die Berechnung) |
| `CAL` | Notenbanken, Quartalszahlen, Dividendentermine |
| `F1`–`F9` | Funktionsleiste |
| `/` oder `Strg+K` | Befehlszeile |

## Ticker-Format (Yahoo)
| Börse | Endung | Beispiel |
|---|---|---|
| Wien | `.VI` | `EBS.VI`, `VER.VI`, `RBI.VI` |
| Xetra | `.DE` | `SAP.DE`, `ALV.DE` |
| Paris / Amsterdam / Mailand | `.PA` / `.AS` / `.MI` | `MC.PA`, `ASML.AS`, `ENI.MI` |
| Schweiz | `.SW` | `NESN.SW` |
| London | `.L` | `SHEL.L` |
| USA | keine | `AAPL`, `BRK-B` |
| ETFs (Xetra) | `.DE` | `EUNL.DE` (iShares MSCI World) |

Den richtigen Ticker findest du auf finance.yahoo.com: Titel suchen und das Kürzel in der URL übernehmen.

## Universum
In `config.json` → `universe`: iShares-Positionslisten (MSCI ACWI, STOXX Europe 600), Wikipedia-Listen (S&P 500,
Nasdaq 100) und eine Zusatzliste (österreichische Titel). Weitere iShares-ETFs lassen sich mit ihrer CSV-Adresse ergänzen.
Im Protokoll des Nachtlaufs steht je Quelle, wie viele Titel gefunden wurden. Steht dort eine Warnung, hat sich
vermutlich die Adresse geändert.

## Einkaufsmanager-Indizes (PMI)
Die offiziellen PMIs (ISM, HCOB/S&P Global) sind lizenzpflichtig. Automatisch geladen werden freie Alternativen:
EU-Kommission (Wirtschaftsstimmung ESI, Industrie-, Dienstleistungs-, Verbrauchervertrauen für Euroraum, Österreich,
Deutschland), Philly-Fed, Empire State, Chicago Fed. Offizielle PMI-Werte kannst du in `manual/pmi.csv` selbst eintragen.

## Was du wissen solltest
- **Was das Passwort schützt:** alle Kurse, Kennzahlen, Bilanzen, News und Stimmungswerte.
  **Nicht geschützt** sind der Code und `config.json`, also deine Watchlist, weil das Repository öffentlich ist.
  Vor dem ersten Lauf zeigt die Seite unverschlüsselte Beispieldaten ohne Passwortabfrage.
- **„Angemeldet bleiben“** speichert das Passwort im Browser dieses Geräts. Am Arbeits-PC besser nicht anhaken.
  Abmelden über das ⏻-Symbol oben rechts.
- **Detailseiten:** Watchlist-Titel werden alle 30 Minuten aktualisiert, alle übrigen im Nachtlauf (ca. 600 pro Nacht,
  `nightly_batch` in config.json). Bis alle Titel erfasst sind, dauert es einige Nächte. Bis dahin zeigt die Detailseite ein Kurzprofil.
- **Bilanzen:** US-Titel kommen aus SEC EDGAR (10-K, bis zu 10 Jahre), alle anderen aus Yahoo (ca. 4 Jahre).
- **FinBERT** versteht nur Englisch. Deshalb werden englische Schlagzeilen bewertet.
- **Verzögerung:** Kurse sind 15–20 Minuten verzögert und werden alle 30 Minuten aktualisiert (werktags 7–23 Uhr).
- **Yahoo-Sperren:** Yahoo bremst gelegentlich Abfragen aus Rechenzentren. Dann fehlen in einem Lauf einzelne Werte.
  Der nächste Lauf holt sie normalerweise nach. Hilft das nicht: `pip install -U yfinance` im Workflow behält die neueste Version.
- **Rechtliches:** Yahoo-Daten sind laut Nutzungsbedingungen nur für private Zwecke gedacht. Das Terminal ist eine
  persönliche Research-Hilfe. Für Kundenunterlagen und Beratungsprotokolle gelten die offiziellen Bankquellen.
- **Arbeits-PC:** Falls `github.io` im Bankennetz gesperrt ist, kannst du die IT um Freigabe bitten oder das Terminal am Handy nutzen.

## Dateien
```
index.html                    Terminal-Oberfläche (eine Datei, keine externen Bibliotheken)
config.json                   Watchlist, Indizes, Märkte, News-Themen
scripts/fetch_data.py         Datensammler
requirements.txt              Python-Pakete
.github/workflows/update.yml  Zeitplan für GitHub Actions
manual/pmi.csv                selbst eingetragene PMI-Werte
(Branch data-store)           verschlüsselte Daten, wird automatisch befüllt
```

## Ausbauideen
- ETFs mit steuerlichem Vertreter in Österreich + „In welchen ETFs steckt diese Aktie?“
- Österreichische Staatsanleihen als Einzeltitel (Kupon, Laufzeit, Rendite)
- Europäische Bilanzen aus ESEF-Berichten (filings.xbrl.org)
- Portfolio-Ansicht mit Gewichtung und Performance
