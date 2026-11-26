"""Zeigt den Aufbau der Positionslisten (Kopfzeilen + erste Zeilen) – nur Protokoll."""
import io
import requests
import openpyxl

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
for name, url in [
    ("Xtrackers MSCI World", "https://etf.dws.com/etfdata/export/DEU/DEU/excel/product/constituent/IE00BJ0KDQ92/"),
    ("SPDR MSCI ACWI IMI", "https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-spyi-gy.xlsx"),
]:
    wb = openpyxl.load_workbook(io.BytesIO(requests.get(url, headers=UA, timeout=60).content), read_only=True, data_only=True)
    for ws in wb.worksheets[:2]:
        print(f"[TEST] === {name} · Blatt {ws.title}")
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= 12:
                break
            print(f"[TEST] {i}: {[c for c in row][:12]}")
