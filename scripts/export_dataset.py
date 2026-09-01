#!/usr/bin/env python
"""Слепок всех таблиц базы (кроме сырья и ext_load) в dataset/*.parquet.
Несколько мегабайт вместо гигабайта исходных xls - этого достаточно, чтобы
воспроизвести бэктест без пересборки базы из первоисточников."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.db import get_conn, load_cfg, ROOT

OUT = os.path.join(ROOT, "dataset")
TABLES = ["calendar", "so_window", "target", "official_load", "pdem", "weather", "aux"]

def main():
    os.makedirs(OUT, exist_ok=True)
    conn = get_conn(load_cfg())
    total = 0
    for t in TABLES:
        df = pd.read_sql(f"SELECT * FROM {t}", conn)
        path = os.path.join(OUT, f"{t}.parquet")
        df.to_parquet(path, index=False)
        sz = os.path.getsize(path)
        total += sz
        print(f"{t}: {len(df)} строк, {sz/1e6:.1f} МБ")
    print(f"итого {total/1e6:.1f} МБ -> {OUT}")

if __name__ == "__main__":
    main()
