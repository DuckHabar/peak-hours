#!/usr/bin/env python
"""Выгрузка индексов РСВ (2-я ЦЗ) с atsenergo.ru годовыми диапазонами
в data_aux/rsv_YYYY.xml. Сайт может отклонять запросы с датацентровых IP,
с обычного домашнего подключения работает."""
import datetime
import os
import ssl
import time
import urllib.request

BASE = ("https://www.atsenergo.ru/market/stats.xml?period=0"
        "&date1={y}0101&date2={y}1231&zone=2&type=graph&mask=0")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_aux")

def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE          # у сайта российский CA
    for y in range(2015, datetime.date.today().year + 1):
        path = os.path.join(OUT, f"rsv_{y}.xml")
        if os.path.exists(path) and os.path.getsize(path) > 100000 and y < datetime.date.today().year:
            print(f"{y}: есть, пропуск", flush=True)
            continue
        req = urllib.request.Request(BASE.format(y=y), headers={"User-Agent": "Mozilla/5.0"})
        try:
            data = urllib.request.urlopen(req, timeout=90, context=ctx).read()
            open(path, "wb").write(data)
            print(f"{y}: {len(data)} байт", flush=True)
        except Exception as e:
            print(f"{y}: FAIL {e}", flush=True)
        time.sleep(2)

if __name__ == "__main__":
    main()
