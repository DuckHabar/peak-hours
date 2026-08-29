#!/usr/bin/env python
"""Догрузка погоды в глубину истории: ERA5-факты до 2013 (питание ветки кривой),
прогнозы previous_day1 до 2021 (глубже история прогнозов не существует).
Чанками по ~3 года, чтобы не упираться в лимиты open-meteo."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date, datetime, timedelta
from src.db import get_conn, load_cfg
from src.ingest import _om_fetch

def pull(conn, cfg, kind, url, hourly_vars, chunks):
    cities = cfg["weather"]["cities"]
    for city, meta in cities.items():
        for d0, d1 in chunks:
            common = dict(latitude=meta["lat"], longitude=meta["lon"],
                          timezone="Europe/Moscow", start_date=d0, end_date=d1,
                          hourly=",".join(hourly_vars))
            try:
                js = _om_fetch(url, common)
            except Exception as e:
                print(f"{city} {kind} {d0}..{d1}: FAIL {e}", flush=True)
                continue
            hh = js.get("hourly", {})
            times = hh.get("time", [])
            rows = []
            for var in hourly_vars:
                base_var = var.replace("_previous_day1", "")
                for t, v in zip(times, hh.get(var, [])):
                    if v is None:
                        continue
                    dt = datetime.fromisoformat(t)
                    d = dt.date()
                    af = (d + timedelta(days=5)) if kind == "fact" else (d - timedelta(days=1))
                    rows.append((d.isoformat(), dt.hour + 1, city, kind, base_var,
                                 float(v), af.isoformat()))
            conn.executemany("INSERT OR REPLACE INTO weather VALUES (?,?,?,?,?,?,?)", rows)
            conn.commit()
            print(f"{city} {kind} {d0}..{d1}: {len(rows)} rows", flush=True)

def main():
    cfg = load_cfg()
    conn = get_conn(cfg)
    w = cfg["weather"]
    fact_chunks = [("2013-01-01", "2015-12-31"), ("2016-01-01", "2018-12-31"),
                   ("2019-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")]
    fc1_chunks = [("2021-01-01", "2022-12-31")]
    pull(conn, cfg, "fact", "https://archive-api.open-meteo.com/v1/archive",
         w["vars_fact"], fact_chunks)
    pull(conn, cfg, "fc1", "https://previous-runs-api.open-meteo.com/v1/forecast",
         w["vars_fc1"], fc1_chunks)
    n = conn.execute("SELECT kind, MIN(d), MAX(d), COUNT(*) FROM weather GROUP BY kind").fetchall()
    for r in n:
        print(r)

if __name__ == "__main__":
    main()
