#!/usr/bin/env python
"""Обучает финальную модель на всем опубликованном и сохраняет веса в
models/final.joblib: лес шага 1 плюс парная модель поправки. predict.py
по умолчанию обучается заново от текущей базы (при ежемесячном пополнении
так правильнее), веса приложены как контрольная точка."""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import pandas as pd

from src.db import get_conn, load_cfg
from src.features import FeatureBuilder, FEATURES
from src.models import make_model
from scripts.rerank_eval import build_pairs, fit_duel, FEATS, TAU

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "et:n_estimators=600,min_samples_leaf=23,max_features=0.96"
TF = date(2021, 1, 1)

def main():
    cfg = load_cfg()
    conn = get_conn(cfg)
    fb = FeatureBuilder(conn, cfg)
    today = date.today()
    ms = date(today.year, today.month, 1)
    af = dict(zip(fb.target["d"], fb.target["af"]))
    days = [d for d in sorted(fb.label) if af[d] <= ms and d >= TF]
    tr = fb.build_range(days).dropna(subset=["label"])
    print(f"обучение шага 1: {len(days)} дней, {len(tr)} строк")
    m = make_model(MODEL)
    m.fit(tr, tr["label"].astype(int))

    # парная модель - на тройках сохраненного прогона (см. rerank_eval.py)
    duel_m = duel_sc = None
    runs = [r[0] for r in conn.execute(
        "SELECT DISTINCT run_id FROM prediction WHERE run_id LIKE ?", (MODEL + "%",))]
    if runs:
        run_id = sorted(runs)[-1]
        pred = pd.read_sql_query(
            "SELECT d, hour, rank FROM prediction WHERE run_id=? AND rank<=3",
            conn, params=(run_id,))
        pred["d"] = pd.to_datetime(pred["d"]).dt.date
        days_top3 = {d: list(g.sort_values("rank")["hour"])
                     for d, g in pred.groupby("d") if len(g) == 3}
        TR = build_pairs(fb, days_top3, {})
        TR = TR[TR["y"].notna()]
        print(f"обучение поправки: {len(TR)} пар")
        if len(TR) >= 200:
            duel_m, duel_sc = fit_duel(TR)

    out_dir = os.path.join(ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final.joblib")
    joblib.dump({
        "spec": MODEL,
        "train_from": TF.isoformat(),
        "trained_until": ms.isoformat(),
        "stage1": m.model,
        "stage1_columns": FEATURES,
        "duel": duel_m,
        "duel_scaler": duel_sc,
        "duel_columns": FEATS,
        "duel_tau": TAU,
    }, path, compress=3)
    print(f"-> {path}, {os.path.getsize(path)/1e6:.1f} МБ")

if __name__ == "__main__":
    main()
