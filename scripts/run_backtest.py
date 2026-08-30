#!/usr/bin/env python
"""CLI: .venv/bin/python scripts/run_backtest.py [clim|lgbm] [--no-save]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backtest import run_backtest

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_backtest(args[0] if args else "clim", save="--no-save" not in sys.argv)
