#!/usr/bin/env python
"""Массовая выгрузка сырья с Яндекс.Диска задания: pdem по Сибири, глубокая
история calcfacthour (KRASNOEN) и fact_region (_04_ = Красноярский край),
PDF СО ЕЭС с пиковыми часами. Уже скачанные файлы пропускаются, перезапуск
безопасен. Потоков 5 - дальше API начинает резать запросы."""
import concurrent.futures as cf
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
from yadisk import ls, download

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def build_plan():
    plan = []  # (путь на диске, локальный путь)
    for it in ls("/so-ups/peakhours"):
        plan.append((f"/so-ups/peakhours/{it['name']}", f"{BASE}/peakhours/{it['name']}"))
    # pdem по Сибири: все, что есть в дампе
    for y in ["2025", "2026"]:
        try:
            months = [d['name'] for d in ls(f"/ats/pdem/{y}") if d['type'] == 'dir']
        except Exception:
            continue
        for m in months:
            for it in ls(f"/ats/pdem/{y}/{m}"):
                if '_sib_' in it['name']:
                    plan.append((f"/ats/pdem/{y}/{m}/{it['name']}", f"{BASE}/pdem/{it['name']}"))
    # calcfacthour: глубокая история по Красноярску
    for y in range(2011, 2023):
        try:
            items = ls(f"/ats/calcfacthour/{y}")
        except Exception:
            continue
        for it in items:
            if 'KRASNOEN' in it['name']:
                plan.append((f"/ats/calcfacthour/{y}/{it['name']}", f"{BASE}/calcfacthour/{it['name']}"))
    # fact_region: _04_ = Красноярский край
    for y in range(2013, 2025):
        try:
            items = ls(f"/ats/fact_region/{y}")
        except Exception:
            continue
        for it in items:
            parts = it['name'].split('_')
            if len(parts) > 1 and parts[1] == '04':
                plan.append((f"/ats/fact_region/{y}/{it['name']}", f"{BASE}/fact_region/{it['name']}"))
    return plan

def main():
    plan = build_plan()
    todo = [(r, l) for r, l in plan if not os.path.exists(l)]
    print(f"в плане {len(plan)} файлов, качать {len(todo)}", flush=True)
    lock = threading.Lock()
    cnt = {"ok": 0, "fail": 0}
    def job(item):
        remote, local = item
        try:
            download(remote, local)
            with lock:
                cnt["ok"] += 1
                if cnt["ok"] % 20 == 0:
                    print(f'{cnt["ok"]}/{len(todo)}', flush=True)
        except Exception as e:
            with lock:
                cnt["fail"] += 1
                print("FAIL", remote, repr(e)[:100], flush=True)
    with cf.ThreadPoolExecutor(5) as ex:
        list(ex.map(job, todo))
    print(f'готово: ok={cnt["ok"]} fail={cnt["fail"]}', flush=True)

if __name__ == "__main__":
    main()
