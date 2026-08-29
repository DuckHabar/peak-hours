#!/usr/bin/env python
"""Загрузка индексов РСВ в таблицу aux(d, hour, src, value, available_from):
индекс равновесной цены и плановый объем потребления 2-й ЦЗ из data_aux/rsv_*.xml
(см. fetch_rsv.py). Публикуются накануне после торгов, поэтому af = D-1.
Часы приводятся к нотации АТС 1..24 МСК. Перезапуск безопасен."""
import os
import re
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import get_conn, init_db, load_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUX = os.path.join(ROOT, "data_aux")

def ingest_rsv(conn):
    total = 0
    for f in sorted(os.listdir(AUX)):
        if not f.startswith("rsv_") or not f.endswith(".xml"):
            continue
        x = open(os.path.join(AUX, f), encoding="utf-8", errors="replace").read()
        rows = []
        for m in re.finditer(r"<row num=\"\d+\">(.*?)</row>", x, re.S):
            cols = dict(re.findall(r"<col name=\"([^\"]+)\">([^<]*)</col>", m.group(1)))
            try:
                dd, mm, yy = cols["DAT"].split(".")
                d = date(int(yy), int(mm), int(dd))
                h = int(cols["HOUR"]) + 1
            except (KeyError, ValueError):
                continue
            af = (d - timedelta(days=1)).isoformat()
            p = cols.get("CONSUMER_PRICE")
            v = cols.get("CONSUMER_VOLUME")
            if p:
                rows.append((d.isoformat(), h, "rsv_price", float(p), af))
            if v:
                rows.append((d.isoformat(), h, "rsv_vol", float(v), af))
        conn.executemany("INSERT OR REPLACE INTO aux VALUES (?,?,?,?,?)", rows)
        conn.commit()
        total += len(rows)
        print(f"{f}: {len(rows)} строк")
    print(f"rsv всего: {total}")

def main():
    cfg = load_cfg()
    conn = get_conn(cfg)
    init_db(conn)
    ingest_rsv(conn)
    for r in conn.execute("SELECT src, COUNT(*), MIN(d), MAX(d) FROM aux GROUP BY src"):
        print(r)

if __name__ == "__main__":
    main()
