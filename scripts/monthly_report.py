#!/usr/bin/env python
"""Помесячная таблица точности «модель против статистики»: доля дней месяца,
где фактический час пика попал в топ-1..топ-5 прогноза. Формат повторяет
сравнительную таблицу заказчика.

  monthly_report.py --run-id "<stage1 run_id>" [--from 2025-01-01] [--to 2026-07-31]

Ранжирование модели берется из таблицы prediction (сохраненный прогон ступени 1,
см. rerank_eval.py), поверх топ-3 применяется та же парная поправка с помесячным
переобучением. Колонка «Статистика» - базовая климатология за 3 года.
"""
import argparse
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import get_conn, load_cfg
from src.features import FeatureBuilder
from src.backtest import month_starts, rank_df
from src.models import make_model
from scripts.rerank_eval import build_pairs, fit_duel, rerank_day

MONTH_RU = {1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель", 5: "Май",
            6: "Июнь", 7: "Июль", 8: "Август", 9: "Сентябрь", 10: "Октябрь",
            11: "Ноябрь", 12: "Декабрь"}

def model_ranks(conn, fb, run_id, ef, et):
    """{день: [часы в порядке ранга]} с парной поправкой топ-3."""
    pred = pd.read_sql_query(
        "SELECT d, hour, rank FROM prediction WHERE run_id=?", conn, params=(run_id,))
    pred["d"] = pd.to_datetime(pred["d"]).dt.date
    order_all = {d: list(g.sort_values("rank")["hour"].astype(int))
                 for d, g in pred.groupby("d")}
    days_top3 = {d: v[:3] for d, v in order_all.items() if len(v) >= 3}
    af = dict(zip(fb.target["d"], fb.target["af"]))
    cache = {}
    ALL = build_pairs(fb, days_top3, cache)
    out = {}
    for ms in month_starts(ef, et):
        mdays = [d for d in order_all
                 if ef <= d <= et and d.year == ms.year and d.month == ms.month]
        if not mdays:
            continue
        ok = {d for d in days_top3 if af.get(d, date.max) <= ms}
        TR = ALL[[d in ok for d in ALL["d"]]]
        if len(TR) < 200:
            for d in mdays:
                out[d] = order_all[d]
            continue
        m, sc = fit_duel(TR)
        for d in mdays:
            full = order_all[d]
            top3 = rerank_day(m, sc, ALL[ALL["d"] == d], full[:3])
            out[d] = top3 + full[3:]
    return out

def clim_ranks(fb, days):
    model = make_model("clim")
    out = {}
    for d in days:
        df = fb.build_day(d)
        if df is None or df.empty:
            continue
        out[d] = list(rank_df(df, model.score(df))["hour"].astype(int))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--from", dest="d_from", default="2025-01-01")
    ap.add_argument("--to", dest="d_to", default="2026-07-31")
    ap.add_argument("--out", default="results/monthly.xlsx")
    a = ap.parse_args()
    cfg = load_cfg()
    conn = get_conn(cfg)
    fb = FeatureBuilder(conn, cfg)
    ef, et = date.fromisoformat(a.d_from), date.fromisoformat(a.d_to)
    mr = model_ranks(conn, fb, a.run_id, ef, et)
    cr = clim_ranks(fb, sorted(mr))
    rows = []
    sums = {("model", k): [] for k in range(1, 6)}
    sums.update({("clim", k): [] for k in range(1, 6)})
    for ms in month_starts(ef, et):
        mdays = [d for d in sorted(mr)
                 if d.year == ms.year and d.month == ms.month and d in fb.label]
        if not mdays:
            continue
        row = [f"{MONTH_RU[ms.month]} {ms.year}"]
        for k in range(1, 6):
            hm = sum(fb.label[d] in mr[d][:k] for d in mdays) / len(mdays)
            hc = sum(fb.label[d] in cr.get(d, [])[:k] for d in mdays) / len(mdays)
            row += [round(hm, 3), round(hc, 3)]
            sums[("model", k)].append(hm)
            sums[("clim", k)].append(hc)
        rows.append(row)
    avg = ["Средняя точность"]
    for k in range(1, 6):
        avg += [round(sum(sums[("model", k)]) / len(sums[("model", k)]), 3),
                round(sum(sums[("clim", k)]) / len(sums[("clim", k)]), 3)]
    rows.append(avg)
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    head1 = ["Период"]
    for k in range(1, 6):
        head1 += [f"Топ-{k}", None]
    head2 = [None] + ["Модель", "Статистика"] * 5
    ws.append(head1)
    ws.append(head2)
    for k in range(5):
        ws.merge_cells(start_row=1, start_column=2 + 2 * k,
                       end_row=1, end_column=3 + 2 * k)
    for r in rows:
        ws.append(r)
    ws.column_dimensions["A"].width = 18
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    wb.save(a.out)
    for r in rows:
        print("  ".join(f"{x:>6}" if not isinstance(x, str) else f"{x:<18}" for x in r))
    print(f"-> {a.out}")

if __name__ == "__main__":
    main()
