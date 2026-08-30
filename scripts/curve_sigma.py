#!/usr/bin/env python
"""Остаточная ошибка ветки кривой вне обучения: std остатков, посчитанных
по годам публикации (см. CurveBranch._ensure_residuals). Полученную сигму
можно подставить в таблицу ceiling_sim.py и прочитать оттуда потолок top-1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.db import get_conn, load_cfg
from src.features import FeatureBuilder
from src.curve import CurveBranch

def main():
    cfg = load_cfg()
    fb = FeatureBuilder(get_conn(cfg), cfg)
    br = CurveBranch(fb)
    br._ensure_residuals()
    print(f"дней с остатками: {len(br._residuals)}")
    all_e = np.concatenate(list(br._residuals.values()))
    print(f"std остатка формы: {np.std(all_e)*100:.2f}% на час")
    by_year = {}
    for d, e in br._residuals.items():
        by_year.setdefault(d.year, []).append(e)
    for y in sorted(by_year):
        print(f"  {y}: {np.std(np.concatenate(by_year[y]))*100:.2f}%")

if __name__ == "__main__":
    main()
