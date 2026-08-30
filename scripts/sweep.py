#!/usr/bin/env python
"""Серия бэктестов с общим FeatureBuilder и кэшем признаков.
Результаты дописываются в results/sweep.csv.

  sweep.py clim rf et                   # оценочный год из config.yaml
  sweep.py --tune rf et                 # валидационный год 2024-08..2025-07
  sweep.py --tf=2021-01-01 et           # обучение с 2021 года
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import get_conn, load_cfg
from src.features import FeatureBuilder
from src.backtest import run_backtest

TUNE = ("2024-08-01", "2025-07-31")
TWO = ("2024-08-01", "2026-07-31")

def main():
    args = sys.argv[1:]
    tune = "--tune" in args
    two = "--two" in args
    if tune:
        args.remove("--tune")
    if two:
        args.remove("--two")
    ef, et = TWO if two else (TUNE if tune else (None, None))
    win = "two" if two else ("tune" if tune else "eval")
    tf = None
    for a in list(args):
        if a.startswith("--tf="):
            tf = a.split("=", 1)[1]
            args.remove(a)
            win += f"|tf={tf}"
    cfg = load_cfg()
    fb = FeatureBuilder(get_conn(cfg), cfg)
    cache = {}
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "sweep.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "window", "model", "n", "top1", "top2", "top3", "mrr", "sec"])
        for m in args:
            t0 = time.time()
            try:
                r = run_backtest(m, save=False, verbose=False, fb=fb, cache=cache,
                                 eval_from=ef, eval_to=et, train_from=tf)
            except Exception as e:
                print(f"{m}: FAIL {e}", flush=True)
                continue
            w.writerow([int(time.time()), win, m, r["n"],
                        f"{r['top1']:.4f}", f"{r['top2']:.4f}", f"{r['top3']:.4f}",
                        f"{r['mrr']:.4f}", f"{time.time()-t0:.0f}"])
            f.flush()

if __name__ == "__main__":
    main()
