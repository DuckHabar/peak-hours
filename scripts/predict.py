#!/usr/bin/env python
"""Прогноз на день D (по умолчанию завтра): ранжированный список часов окна СО.

  predict.py [YYYY-MM-DD] [model] [--no-rerank]

Ступень 1 обучается на всем опубликованном и ранжирует часы окна. Затем парная
модель (LogReg по дуэлям топ-3, см. rerank_eval.py) может переставить тройку,
если уверена: для этого нужен сохраненный исторический прогон ступени 1
в таблице prediction. Перед боевым запуском догрузить: погоду, pdem на D,
отчеты АТС за прошлый месяц (выходят после 10-го), индексы РСВ (fetch_rsv).
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from src.db import get_conn, load_cfg
from src.features import FeatureBuilder
from src.models import make_model
from src.backtest import rank_df

# параметры - из подбора Optuna на валидационном годе (scripts/optuna_rf.py)
DEFAULT_MODEL = "et:n_estimators=600,min_samples_leaf=23,max_features=0.96"
DEFAULT_TF = "2021-01-01"

def _try_rerank(conn, fb, D, ranked, model_name):
    from scripts.rerank_eval import build_pairs, fit_duel, rerank_day
    pref = model_name.split(":")[0]
    runs = [r[0] for r in conn.execute(
        "SELECT DISTINCT run_id FROM prediction WHERE run_id LIKE ?",
        (model_name + "%",))] or [r[0] for r in conn.execute(
        "SELECT DISTINCT run_id FROM prediction WHERE run_id LIKE ?", (pref + "%",))]
    if not runs:
        return None
    run_id = sorted(runs)[-1]
    pred = pd.read_sql_query(
        "SELECT d, hour, rank FROM prediction WHERE run_id=? AND rank<=3",
        conn, params=(run_id,))
    pred["d"] = pd.to_datetime(pred["d"]).dt.date
    days_top3 = {d: list(g.sort_values("rank")["hour"])
                 for d, g in pred.groupby("d") if len(g) == 3 and d < D}
    if len(days_top3) < 100:
        return None
    cache = {}
    TR = build_pairs(fb, days_top3, cache)
    TR = TR[TR["y"].notna()]
    if len(TR) < 200:
        return None
    m, sc = fit_duel(TR)
    top3 = list(ranked.head(3)["hour"].astype(int))
    day_pairs = build_pairs(fb, {D: top3}, {}, with_label=False)
    order = rerank_day(m, sc, day_pairs, top3)
    return order, run_id

def main():
    args = [a for a in sys.argv[1:] if a != "--no-rerank"]
    no_rr = "--no-rerank" in sys.argv
    D = (date.fromisoformat(args[0]) if args
         else date.today() + timedelta(days=1))
    model_name = args[1] if len(args) > 1 else DEFAULT_MODEL
    cfg = load_cfg()
    conn = get_conn(cfg)
    fb = FeatureBuilder(conn, cfg)
    ms = date(D.year, D.month, 1)
    model = make_model(model_name)
    if hasattr(model, "bind"):
        model.bind(fb)
    if hasattr(model, "fit_month"):
        model.fit_month(ms)
    if model.needs_fit:
        af = dict(zip(fb.target["d"], fb.target["af"]))
        tf = date.fromisoformat(DEFAULT_TF)
        train_days = [d for d in sorted(fb.label) if af[d] <= ms and d >= tf]
        tr = fb.build_range(train_days)
        tr = tr.dropna(subset=["label"])
        model.fit(tr, tr["label"].astype(int))
    df = fb.build_day(D)
    if df is None or df.empty:
        print(f"нет окна СО для {D} - не рабочий день или нет данных")
        return
    ranked = rank_df(df, model.score(df))
    rr_note = ""
    if not no_rr:
        rr = _try_rerank(conn, fb, D, ranked, model_name)
        if rr:
            order, run_id = rr
            if order != list(ranked.head(3)["hour"].astype(int)):
                rr_note = f"  [парная модель переставила топ-3; история: {run_id[:40]}]"
            pos = {h: i for i, h in enumerate(order)}
            ranked["_k"] = [pos.get(int(h), 99) for h in ranked["hour"]]
            head = ranked[ranked["_k"] < 99].sort_values("_k")
            tail = ranked[ranked["_k"] == 99]
            ranked = pd.concat([head, tail]).drop(columns="_k")
            ranked["rank"] = range(1, len(ranked) + 1)
    print(f"Прогноз пиковых часов на {D} ({model_name}){rr_note}")
    for row in ranked.itertuples():
        mark = " <-- top" if row.rank == 1 else ""
        print(f"  {row.rank:2d}. час {int(row.hour):2d} "
              f"({int(row.hour)-1:02d}:00-{int(row.hour):02d}:00 МСК){mark}")

if __name__ == "__main__":
    main()
