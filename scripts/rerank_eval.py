#!/usr/bin/env python
"""Итоговая связка: ступень 1 (помесячный бэктест, save=True) + парная поправка
топ-3 (LogReg по дуэлям, перестановка первого места при отрыве > TAU).

  rerank_eval.py --stage1 "rfx:..." --tf 2021-01-01     # прогнать все с нуля
  rerank_eval.py --run-id "<run_id>"                    # по готовому прогону

Дуэли каждого месяца обучаются только на метках, опубликованных к его началу.
TAU и класс парной модели выбраны на валидационном хвосте обучающего периода,
не на оценочном годе.
"""
import argparse
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import get_conn, load_cfg
from src.features import FeatureBuilder
from src.backtest import month_starts

DIFF_FEATS = ["clim_3y", "clim_wd", "off_meanrank", "off_argmax_share", "mc_p",
              "pdem_z", "fc_temp_h", "fc_cloud_h", "h_vs_sunset", "dark_at_h",
              "is_morning", "h", "pos_in_window"]
CTX_FEATS = ["month", "t_anom", "t_ma_diff", "morning_share", "weekday"]
FEATS = ["rank_gap"] + ["d_" + c for c in DIFF_FEATS] + CTX_FEATS
TAU = 0.05

def build_pairs(fb, days_top3, cache, with_label=True):
    """Пары «час A против часа B» из троек; with_label=False - для прогнозного
    дня, у которого метки еще нет."""
    rows = []
    for D, top3 in days_top3.items():
        df = cache.get(D)
        if df is None:
            df = fb.build_day(D)
            cache[D] = df
        if df is None or df.empty:
            continue
        f = df.set_index("hour")
        true_h = fb.label.get(D)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                a, b = top3[i], top3[j]
                if a not in f.index or b not in f.index:
                    continue
                y = 1 if true_h == a else (0 if true_h == b else None)
                if with_label and y is None:
                    continue
                r = {"d": D, "a": a, "b": b, "y": y, "rank_gap": i - j}
                for c in DIFF_FEATS:
                    va, vb = f.at[a, c], f.at[b, c]
                    r["d_" + c] = (va - vb) if pd.notna(va) and pd.notna(vb) else np.nan
                for c in CTX_FEATS:
                    r[c] = f.at[a, c] if c in f.columns else np.nan
                rows.append(r)
    return pd.DataFrame(rows)

def fit_duel(TR):
    # LogReg, а не бустинг: пар всего ~1.5k, деревья тут переобучаются
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(np.nan_to_num(TR[FEATS].to_numpy(dtype=float)))
    m = LogisticRegression(C=0.3, max_iter=2000).fit(X, TR["y"])
    return m, sc

def rerank_day(m, sc, sub, top3):
    """Перестановка первого места топ-3 при уверенной дуэли, хвост не трогаем."""
    if sub.empty:
        return list(top3)
    p = m.predict_proba(sc.transform(np.nan_to_num(sub[FEATS].to_numpy(dtype=float))))[:, 1]
    sub = sub.assign(p=p)
    score = {h: sub[sub["a"] == h]["p"].mean() for h in top3}
    best = max(top3, key=lambda h: (score.get(h, 0), -h))
    if best != top3[0] and score.get(best, 0) - score.get(top3[0], 0) > TAU:
        return [best] + [h for h in top3 if h != best]
    return list(top3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1")
    ap.add_argument("--tf", default="2021-01-01")
    ap.add_argument("--run-id")
    ap.add_argument("--hist-from", default="2023-08-01")
    ap.add_argument("--eval-from", default="2025-08-01")
    ap.add_argument("--eval-to", default="2026-07-31")
    a = ap.parse_args()
    cfg = load_cfg()
    conn = get_conn(cfg)
    fb = FeatureBuilder(conn, cfg)
    run_id = a.run_id
    if not run_id:
        from src.backtest import run_backtest
        r = run_backtest(a.stage1, save=True, verbose=False, train_from=a.tf,
                         fb=fb, eval_from=a.hist_from, eval_to=a.eval_to)
        run_id = r["run_id"]
        print(f"stage1: {run_id} n={r['n']} top1={r['top1']:.4f} top3={r['top3']:.4f}")
    pred = pd.read_sql_query(
        "SELECT d, hour, rank FROM prediction WHERE run_id=? AND rank<=3",
        conn, params=(run_id,))
    pred["d"] = pd.to_datetime(pred["d"]).dt.date
    days_top3 = {d: list(g.sort_values("rank")["hour"])
                 for d, g in pred.groupby("d") if len(g) == 3}
    af = dict(zip(fb.target["d"], fb.target["af"]))
    ef, et = date.fromisoformat(a.eval_from), date.fromisoformat(a.eval_to)
    ev_days = {d: v for d, v in days_top3.items() if ef <= d <= et}
    cache = {}
    ALL = build_pairs(fb, days_top3, cache)
    hb = hr = h2 = h3 = n = 0
    for ms in month_starts(ef, et):
        mdays = {d: v for d, v in ev_days.items()
                 if d.year == ms.year and d.month == ms.month}
        if not mdays:
            continue
        ok = {d for d in days_top3 if af.get(d, date.max) <= ms}
        TR = ALL[[d in ok for d in ALL["d"]]]
        # пар мало - месяц идет без поправки, но из подсчета не выпадает
        m = sc = None
        if len(TR) >= 200:
            m, sc = fit_duel(TR)
        sub_m = ALL[[d in mdays for d in ALL["d"]]]
        for D, top3 in mdays.items():
            true_h = fb.label.get(D)
            if true_h is None:
                continue
            n += 1
            hb += int(top3[0] == true_h)
            order = (rerank_day(m, sc, sub_m[sub_m["d"] == D], top3)
                     if m is not None else list(top3))
            hr += int(order[0] == true_h)
            h2 += int(true_h in order[:2])
            h3 += int(true_h in top3)
    if n == 0:
        raise SystemExit("в оценочном окне нет дней с тройками")
    print(f"n={n}")
    print(f"top1: {hb/n:.4f} -> {hr/n:.4f}   top2={h2/n:.4f}  top3={h3/n:.4f}")

if __name__ == "__main__":
    main()
