#!/usr/bin/env python
"""Подбор гиперпараметров леса (rf/et/rfx) Optuna-ой на валидационном годе,
оценочный год в подбор не входит. Кэш признаков общий для всех испытаний,
дорога только посадка самого леса.

  optuna_rf.py [n_trials] [timeout_sec]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna

from src.db import get_conn, load_cfg
from src.features import FeatureBuilder
from src.backtest import run_backtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = ("2024-08-01", "2025-07-31")

def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = load_cfg()
    fb = FeatureBuilder(get_conn(cfg), cfg)
    cache = {}

    def objective(trial):
        kind = trial.suggest_categorical("kind", ["rf", "et", "rfx"])
        name = (f"{kind}:n_estimators={trial.suggest_int('n_estimators', 600, 2400, step=200)}"
                f",min_samples_leaf={trial.suggest_int('min_samples_leaf', 2, 24)}"
                f",max_features={trial.suggest_float('max_features', 0.2, 1.0)}")
        r = run_backtest(name, save=False, verbose=False, fb=fb, cache=cache,
                         eval_from=WIN[0], eval_to=WIN[1], train_from="2021-01-01")
        return r["top1"]

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    study = optuna.create_study(
        study_name="rf_tune",
        storage=f"sqlite:///{ROOT}/results/optuna_rf.db",
        direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print("BEST:", study.best_value, study.best_params)

if __name__ == "__main__":
    main()
