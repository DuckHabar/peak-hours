"""Годовой бэктест с помесячным переобучением и строгим запретом будущего:
- признаки дня D строятся as-of D-1 (внутри FeatureBuilder);
- обучающая выборка месяца M - только дни, чья целевая была ОПУБЛИКОВАНА
  к первому дню M (available_from <= M/01)."""
import time
from datetime import date
from collections import defaultdict

from .db import get_conn, load_cfg
from .features import FeatureBuilder
from .models import make_model

def month_starts(d_from, d_to):
    cur = date(d_from.year, d_from.month, 1)
    out = []
    while cur <= d_to:
        out.append(cur)
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    return out

def rank_df(day_df, scores):
    df = day_df.copy()
    df["score"] = scores
    df = df.sort_values(["score", "hour"], ascending=[False, True])
    df["rank"] = range(1, len(df) + 1)
    return df

def run_backtest(model_name="clim", save=True, verbose=True, train_from=None,
                 fb=None, cache=None, eval_from=None, eval_to=None):
    cfg = load_cfg()
    conn = get_conn(cfg)
    fb = fb or FeatureBuilder(conn, cfg)
    d_from = date.fromisoformat(eval_from or cfg["eval"]["date_from"])
    d_to = date.fromisoformat(eval_to or cfg["eval"]["date_to"])
    eval_days = sorted(d for d in fb.label if d_from <= d <= d_to)
    all_days = sorted(fb.label)
    if train_from:
        tf = date.fromisoformat(train_from) if isinstance(train_from, str) else train_from
        all_days = [d for d in all_days if d >= tf]
    cache = cache if cache is not None else {}
    run_id = f"{model_name}-{int(time.time())}"
    hits = defaultdict(int)
    n = 0
    mrr = 0.0
    by_month = defaultdict(lambda: [0, 0])
    preds = []

    af = dict(zip(fb.target["d"], fb.target["af"]))
    for ms in month_starts(d_from, d_to):
        month_days = [d for d in eval_days if d.year == ms.year and d.month == ms.month]
        if not month_days:
            continue
        model = make_model(model_name)
        if hasattr(model, "bind"):
            model.bind(fb)
        if hasattr(model, "fit_month"):
            model.fit_month(ms)
        if model.needs_fit:
            train_days = [d for d in all_days if af[d] <= ms]
            tr = fb.build_range(train_days, cache)
            tr = tr.dropna(subset=["label"])
            model.fit(tr, tr["label"].astype(int))
            del tr
        for D in month_days:
            df = fb.build_day(D)
            if df is None or df.empty:
                continue
            ranked = rank_df(df, model.score(df))
            true_h = fb.label[D]
            r = int(ranked.loc[ranked["hour"] == true_h, "rank"].iloc[0])
            n += 1
            mrr += 1.0 / r
            for k in (1, 2, 3):
                hits[k] += int(r <= k)
            key = (D.year, D.month)
            by_month[key][0] += int(r == 1)
            by_month[key][1] += 1
            if save:
                for row in ranked.itertuples():
                    preds.append((run_id, D.isoformat(), int(row.hour), int(row.rank), float(row.score)))
        if verbose:
            y, m = ms.year, ms.month
            got = by_month.get((y, m))
            if got:
                print(f"  {y}-{m:02d}: top1 {got[0]}/{got[1]} = {got[0]/got[1]:.2f}", flush=True)
    if save and preds:
        conn.executemany("INSERT OR REPLACE INTO prediction VALUES (?,?,?,?,?)", preds)
        conn.commit()
    if n == 0:
        raise SystemExit(f"в окне {d_from}..{d_to} нет оценочных дней")
    res = {f"top{k}": hits[k] / n for k in (1, 2, 3)}
    res.update(n=n, mrr=mrr / n, run_id=run_id)
    if verbose:
        print(f"\n{model_name}: n={n}  top1={res['top1']:.3f}  top2={res['top2']:.3f}  "
              f"top3={res['top3']:.3f}  MRR={res['mrr']:.3f}  ({run_id})")
    return res

if __name__ == "__main__":
    import sys
    run_backtest(sys.argv[1] if len(sys.argv) > 1 else "clim")
