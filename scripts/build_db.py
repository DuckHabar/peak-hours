#!/usr/bin/env python
"""Полная (пере)сборка базы: python scripts/build_db.py [шаги]
Шаги: so calendar facthour factregion pdem weather fixcal ext validate.
Без аргументов - все, кроме weather (гоняется отдельно, зависит от сети
open-meteo), fixcal (после погоды) и ext (сторонний xlsx, только EDA)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingest import main

if __name__ == "__main__":
    main(sys.argv[1:])
