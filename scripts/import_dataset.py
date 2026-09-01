#!/usr/bin/env python
"""Сборка db/peak.sqlite из dataset/*.parquet. Обратная сторона
export_dataset.py: сырые xls для этого не нужны."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.db import get_conn, load_cfg, init_db, ROOT

def main():
    conn = get_conn(load_cfg())
    init_db(conn)
    for path in sorted(glob.glob(os.path.join(ROOT, "dataset", "*.parquet"))):
        t = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_parquet(path)
        conn.execute(f"DELETE FROM {t}")
        df.to_sql(t, conn, if_exists="append", index=False)
        print(f"{t}: {len(df)} строк")
    conn.commit()
    print("готово -> db/peak.sqlite")

if __name__ == "__main__":
    main()
