#!/usr/bin/env python
"""Оценка потолка точности: насколько точным должен быть суточный прогноз
кривой, чтобы угадывать argmax внутри окна СО.

К настоящим официальным кривым добавляется мультипликативный шум заданного
уровня (имитация ошибки прогноза формы), дальше смотрим, как часто argmax
сохраняется. Шум двух видов: независимый по часам и AR(1) с rho=0.7 -
второй реалистичнее, ошибки соседних часов похожи."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from datetime import date
from src.db import get_conn, load_cfg

def main():
    cfg = load_cfg()
    conn = get_conn(cfg)
    win = {}
    for y, m, h in conn.execute("SELECT year, month, hour FROM so_window"):
        win.setdefault((y, m), []).append(h)
    curves = {}
    for d, h, mwh in conn.execute("SELECT d, hour, mwh FROM official_load"):
        curves.setdefault(date.fromisoformat(d), {})[h] = mwh
    days = []
    for d, cv in curves.items():
        cand = sorted(win.get((d.year, d.month), []))
        if not cand or any(h not in cv for h in cand):
            continue
        days.append(np.array([cv[h] for h in cand], float))
    print(f"дней: {len(days)}")
    margins = []
    for arr in days:
        s = np.sort(arr)[::-1]
        margins.append((s[0] - s[1]) / arr.mean())
    margins = np.array(margins)
    print("отрыв top1 от top2, % от уровня: "
          f"медиана {np.median(margins)*100:.3f}%  p25 {np.percentile(margins,25)*100:.3f}%  "
          f"p75 {np.percentile(margins,75)*100:.3f}%")
    rng = np.random.default_rng(0)
    NT = 400
    print(f"\nсигма ошибки формы -> вероятность угадать argmax ({NT} розыгрышей на день):")
    print("sigma%   iid    AR1(rho=0.7)")
    for sigma in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02]:
        hit_iid = hit_ar = 0
        tot = 0
        for arr in days:
            k = len(arr)
            true = np.argmax(arr)
            noise = rng.normal(0, sigma, (NT, k))
            hit_iid += (np.argmax(arr * (1 + noise), axis=1) == true).sum()
            e = rng.normal(0, sigma, (NT, k))
            ar = np.empty_like(e)
            ar[:, 0] = e[:, 0]
            rho = 0.7
            sc = np.sqrt(1 - rho * rho)
            for j in range(1, k):
                ar[:, j] = rho * ar[:, j - 1] + sc * e[:, j]
            hit_ar += (np.argmax(arr * (1 + ar), axis=1) == true).sum()
            tot += NT
        print(f"{sigma*100:5.2f}   {hit_iid/tot:.3f}   {hit_ar/tot:.3f}")

if __name__ == "__main__":
    main()
