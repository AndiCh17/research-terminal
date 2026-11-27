import requests
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36","Accept":"text/csv,*/*"}
urls={
 "ish_de":"https://www.ishares.com/de/privatanleger/de/produkte/251882/ishares-msci-world-ucits-etf-acc-fund/1478358465952.ajax?fileType=csv&fileName=EUNL_holdings&dataType=fund",
 "ish_uk":"https://www.ishares.com/uk/individual/en/products/251882/ishares-msci-world-ucits-etf-acc-fund/1506575576011.ajax?fileType=csv&fileName=SWDA_holdings&dataType=fund",
 "ish_us":"https://www.ishares.com/us/products/239600/ishares-msci-acwi-etf/1467271812596.ajax?fileType=csv&fileName=ACWI_holdings&dataType=fund",
 "br_uk":"https://www.blackrock.com/uk/individual/products/251882/ishares-msci-world-ucits-etf-acc-fund/1472631233320.ajax?fileType=csv&fileName=SWDA_holdings&dataType=fund",
 "br_ch":"https://www.blackrock.com/ch/individual/en/products/251882/ishares-msci-world-ucits-etf-acc-fund/1495092304805.ajax?fileType=csv&fileName=SWDA_holdings&dataType=fund",
 "ish_nl":"https://www.ishares.com/nl/particuliere-belegger/nl/producten/251882/ishares-msci-world-ucits-etf-acc-fund/1497735778843.ajax?fileType=csv&fileName=IWDA_holdings&dataType=fund",
 "rcm":"https://www.rcm.at/at/privatkunden/fonds/",
 "kepler":"https://www.kepler.at/de/startseite/fonds.html",
}
s=requests.Session()
for k,u in urls.items():
    try:
        r=s.get(u,headers=UA,timeout=40,allow_redirects=True)
        print(f"== {k} {r.status_code} {r.headers.get('content-type')} len={len(r.content)} final={r.url[:120]}")
        print(repr(r.text[:400]))
    except Exception as e:
        print("==",k,"ERR",e)
