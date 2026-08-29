"""Парсеры всех источников -> SQLite. Каждая строка фактов получает
available_from - дату, с которой ее можно было знать. Идемпотентно:
файлы, уже разобранные (по sha1), пропускаются через ingest_log."""
import glob, hashlib, os, sys
from datetime import date, timedelta, datetime
import pandas as pd
import requests
import yaml

from .db import ROOT, load_cfg, get_conn, init_db

def _sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _seen(conn, path):
    row = conn.execute("SELECT sha1 FROM ingest_log WHERE src_file=?", (os.path.basename(path),)).fetchone()
    return row is not None and row[0] == _sha1(path)

def _log(conn, path, rows):
    conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,datetime('now'))",
                 (os.path.basename(path), _sha1(path), rows))

def next_month_10th(d, pub_day=10):
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(y, m, pub_day)

def _find_header(df, col0_value="Дата"):
    idx = df.index[df[0].astype(str).str.strip() == col0_value]
    if len(idx) == 0:
        raise ValueError("header row not found")
    return idx[0]

# --- so_window
def ingest_so_windows(conn, cfg):
    with open(os.path.join(ROOT, cfg["paths"]["so_windows"]), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows = []
    for year, months in data.items():
        for month, intervals in months.items():
            for a, b in intervals:
                rows += [(int(year), int(month), h) for h in range(a, b + 1)]
    conn.execute("DELETE FROM so_window")
    conn.executemany("INSERT OR REPLACE INTO so_window VALUES (?,?,?)", rows)
    conn.commit()
    print(f"so_window: {len(rows)} rows, years {sorted(data)}")

# --- calendar
def ingest_calendar(conn, years=range(2011, 2028)):
    ok_years, rows = [], []
    for y in years:
        try:
            r = requests.get(f"https://xmlcalendar.ru/data/ru/{y}/calendar.json", timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"calendar {y}: FETCH FAILED ({e}); using weekday rule")
            data = None
        special = {}  # (m, day) -> code: 0 nonwork, 2 short
        if data:
            for mrec in data.get("months", []):
                m = mrec["month"]
                for tok in str(mrec["days"]).split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    if tok.endswith("*"):
                        special[(m, int(tok[:-1]))] = 2       # сокращенный рабочий
                    elif tok.endswith("+"):
                        special[(m, int(tok[:-1]))] = 1       # рабочий (перенос)
                    else:
                        special[(m, int(tok))] = 0            # нерабочий
            ok_years.append(y)
        d = date(y, 1, 1)
        while d.year == y:
            wd = d.isoweekday()
            code = special.get((d.month, d.day))
            if code == 0:
                workday, short = 0, 0
            elif code == 2:
                workday, short = 1, 1
            elif code == 1:
                workday, short = 1, 0
            else:
                # не в списке: по xmlcalendar все выходные перечислены, значит день рабочий;
                # без данных календаря - правило будних дней
                workday, short = (1, 0) if data else (1 if wd <= 5 else 0, 0)
            rows.append((d.isoformat(), wd, workday, short, "xmlcalendar" if data else "weekday-rule"))
            d += timedelta(days=1)
    conn.executemany("INSERT OR REPLACE INTO calendar VALUES (?,?,?,?,?)", rows)
    conn.commit()
    print(f"calendar: {len(rows)} days, xmlcalendar years: {ok_years[:1]}..{ok_years[-1:]} ({len(ok_years)})")

# --- calcfacthour
def ingest_calcfacthour(conn, cfg):
    pub_day = cfg["assumptions"]["ats_pub_day"]
    n_files = n_rows = 0
    for path in sorted(glob.glob(os.path.join(ROOT, cfg["paths"]["raw_calcfacthour"], "*.xls*"))):
        if _seen(conn, path):
            continue
        df = pd.read_excel(path, header=None)
        head = " ".join(str(x) for x in df.iloc[:12, 1].tolist())
        if "Красноярск" not in head:
            print(f"  skip (не Красноярский край): {os.path.basename(path)}")
            continue
        hdr = _find_header(df)
        rows = []
        for _, r in df.iloc[hdr + 1:].iterrows():
            try:
                d = datetime.strptime(str(r[0]).strip(), "%d.%m.%Y").date()
            except (ValueError, TypeError):
                continue
            # современный формат: час в колонке 1; формат 2011-2012: колонка 1 = код ГТП, час в колонке 2
            hour = None
            for c in (1, 2):
                try:
                    v = int(float(r[c]))
                    if 1 <= v <= 24:
                        hour = v
                        break
                except (ValueError, TypeError):
                    continue
            if hour is None:
                continue
            rows.append((d.isoformat(), hour, next_month_10th(d, pub_day).isoformat()))
        conn.executemany("INSERT OR REPLACE INTO target VALUES (?,?,?)", rows)
        _log(conn, path, len(rows))
        n_files += 1
        n_rows += len(rows)
    conn.commit()
    print(f"calcfacthour: +{n_files} files, +{n_rows} rows; total days:",
          conn.execute("SELECT COUNT(*) FROM target").fetchone()[0])

# --- fact_region
def ingest_fact_region(conn, cfg):
    pub_day = cfg["assumptions"]["ats_pub_day"]
    n_files = n_rows = 0
    for path in sorted(glob.glob(os.path.join(ROOT, cfg["paths"]["raw_fact_region"], "*.xls*"))):
        if _seen(conn, path):
            continue
        df = pd.read_excel(path, header=None)
        head = " ".join(str(x) for x in df.iloc[:8, 1].tolist())
        if "Красноярск" not in head:
            print(f"  skip (не Красноярский край): {os.path.basename(path)}")
            continue
        hdr = _find_header(df)
        rows = []
        for _, r in df.iloc[hdr + 1:].iterrows():
            try:
                d = datetime.strptime(str(r[0]).strip(), "%d.%m.%Y").date()
                rows.append((d.isoformat(), int(float(r[1])), float(r[2]),
                             next_month_10th(d, pub_day).isoformat()))
            except (ValueError, TypeError):
                continue
        conn.executemany("INSERT OR REPLACE INTO official_load VALUES (?,?,?,?)", rows)
        _log(conn, path, len(rows))
        n_files += 1
        n_rows += len(rows)
    conn.commit()
    print(f"fact_region: +{n_files} files, +{n_rows} rows; total days:",
          conn.execute("SELECT COUNT(DISTINCT d) FROM official_load").fetchone()[0])

# --- ext (стороннее)
def ingest_ext(conn, cfg):
    path = os.path.join(ROOT, cfg["paths"]["ext_xlsx"])
    if _seen(conn, path):
        print("ext_load: unchanged, skip")
        return
    lag = cfg["assumptions"]["ext_lag_days"]
    df = pd.read_excel(path, sheet_name="Почасовка", header=0)
    df = df.rename(columns={df.columns[0]: "dt", df.columns[4]: "h0", df.columns[5]: "TOTAL"})
    df["dt"] = pd.to_datetime(df["dt"], format="%d.%m.%Y %H:%M")
    df["d"] = df["dt"].dt.date
    df["hour"] = df["h0"].astype(int) + 1          # 0-23 -> нотация АТС 1..24
    gtp_cols = ["TOTAL"] + list(df.columns[6:])
    long = df.melt(id_vars=["d", "hour"], value_vars=gtp_cols, var_name="gtp", value_name="mwh")
    long = long.dropna(subset=["mwh"])
    long["available_from"] = (pd.to_datetime(long["d"]) + pd.Timedelta(days=lag)).dt.date
    rows = [(r.d.isoformat(), int(r.hour), r.gtp, float(r.mwh), r.available_from.isoformat())
            for r in long.itertuples()]
    conn.execute("DELETE FROM ext_load")
    conn.executemany("INSERT OR REPLACE INTO ext_load VALUES (?,?,?,?,?)", rows)
    _log(conn, path, len(rows))
    conn.commit()
    print(f"ext_load: {len(rows)} rows, days:",
          conn.execute("SELECT COUNT(DISTINCT d) FROM ext_load").fetchone()[0])

# --- pdem
def ingest_pdem(conn, cfg):
    lag = cfg["assumptions"]["pdem_lag_days"]
    files = sorted(glob.glob(os.path.join(ROOT, cfg["paths"]["raw_pdem"], "*_sib_*.xls*")))
    n_files = n_rows = 0
    for path in files:
        if _seen(conn, path):
            continue
        base = os.path.basename(path)
        d = datetime.strptime(base.split("_")[0], "%Y%m%d").date()
        af = (d - timedelta(days=lag)).isoformat()
        try:
            xl = pd.ExcelFile(path)
        except Exception as e:
            print(f"  pdem parse fail {base}: {e}")
            continue
        rows = []
        for sh in xl.sheet_names:
            try:
                hour = int(sh) + 1
            except ValueError:
                continue
            sdf = xl.parse(sh, header=None, skiprows=8)
            if sdf.shape[1] < 6:
                continue
            sdf = sdf[[1, 3, 5]].dropna(subset=[5])
            kr = sdf[sdf[5].astype(str).str.contains("Красноярск")]
            for _, r in kr.iterrows():
                try:
                    rows.append((d.isoformat(), hour, str(r[1]).strip(), float(r[3]), af))
                except (ValueError, TypeError):
                    continue
        conn.executemany("INSERT OR REPLACE INTO pdem VALUES (?,?,?,?,?)", rows)
        _log(conn, path, len(rows))
        n_files += 1
        n_rows += len(rows)
        if n_files % 50 == 0:
            conn.commit()
            print(f"  pdem: {n_files}/{len(files)} files", flush=True)
    conn.commit()
    print(f"pdem: +{n_files} files, +{n_rows} rows; days:",
          conn.execute("SELECT COUNT(DISTINCT d) FROM pdem").fetchone()[0])

# --- weather
def _om_fetch(url, params):
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()

def ingest_weather(conn, cfg, end_date=None):
    w = cfg["weather"]
    end_date = end_date or date.today().isoformat()
    total = 0
    for city, meta in w["cities"].items():
        common = dict(latitude=meta["lat"], longitude=meta["lon"],
                      timezone="Europe/Moscow", start_date=w["date_from"], end_date=end_date)
        jobs = [
            ("fact", "https://archive-api.open-meteo.com/v1/archive", w["vars_fact"], 5),
            ("fc1", "https://previous-runs-api.open-meteo.com/v1/forecast", w["vars_fc1"], -1),
        ]
        for kind, url, hourly_vars, af_shift in jobs:
            try:
                js = _om_fetch(url, {**common, "hourly": ",".join(hourly_vars)})
            except Exception as e:
                print(f"weather {city} {kind}: FAIL {e}")
                continue
            hh = js.get("hourly", {})
            times = hh.get("time", [])
            rows = []
            for var in hourly_vars:
                vals = hh.get(var, [])
                base_var = var.replace("_previous_day1", "")
                for t, v in zip(times, vals):
                    if v is None:
                        continue
                    dt = datetime.fromisoformat(t)
                    d = dt.date()
                    af = (d + timedelta(days=af_shift)) if af_shift > 0 else (d - timedelta(days=1))
                    rows.append((d.isoformat(), dt.hour + 1, city, kind, base_var, float(v), af.isoformat()))
            conn.executemany("INSERT OR REPLACE INTO weather VALUES (?,?,?,?,?,?,?)", rows)
            conn.commit()
            total += len(rows)
            print(f"weather {city} {kind}: {len(rows)} rows")
    print(f"weather total: {total}")

# --- fixcal
def fix_calendar_from_facts(conn, cfg=None):
    """День, присутствующий в отчете АТС, - рабочий по определению рынка,
    что бы ни думал производственный календарь (ковидные 'нерабочие' недели
    2020-2021, рабочие субботы 2012 при отсутствии xmlcalendar за 2011-2012)."""
    n = conn.execute("""
        UPDATE calendar SET is_workday=1, src=src||'+ats'
        WHERE is_workday=0 AND d IN (SELECT d FROM target)""").rowcount
    conn.commit()
    print(f"calendar: {n} дней переведены в рабочие по фактам АТС")

# --- validate
def validate(conn):
    print("=== validation ===")
    bad = conn.execute("""
        SELECT COUNT(*) FROM target t
        WHERE NOT EXISTS (SELECT 1 FROM so_window w
            WHERE w.year=CAST(strftime('%Y',t.d) AS INT)
              AND w.month=CAST(strftime('%m',t.d) AS INT) AND w.hour=t.peak_hour)
          AND EXISTS (SELECT 1 FROM so_window w2
            WHERE w2.year=CAST(strftime('%Y',t.d) AS INT))""").fetchone()[0]
    print(f"факты вне окна СО (для лет, где окно задано): {bad}")
    nonwork = conn.execute("""
        SELECT COUNT(*) FROM target t JOIN calendar c ON c.d=t.d WHERE c.is_workday=0""").fetchone()[0]
    print(f"факты в 'нерабочие' дни по календарю (должно быть 0): {nonwork}")
    rec = conn.execute("""
        WITH win AS (SELECT o.d, o.hour, o.mwh,
               ROW_NUMBER() OVER (PARTITION BY o.d ORDER BY o.mwh DESC, o.hour ASC) rn
            FROM official_load o JOIN so_window w
              ON w.year=CAST(strftime('%Y',o.d) AS INT)
             AND w.month=CAST(strftime('%m',o.d) AS INT) AND w.hour=o.hour)
        SELECT SUM(CASE WHEN win.hour=t.peak_hour THEN 1 ELSE 0 END), COUNT(*)
        FROM win JOIN target t ON t.d=win.d WHERE win.rn=1""").fetchone()
    print(f"реконструкция целевой из official_load: {rec[0]}/{rec[1]}")

# --- main
STEPS = {
    "so": ingest_so_windows,
    "calendar": lambda c, cfg: ingest_calendar(c),
    "facthour": ingest_calcfacthour,
    "factregion": ingest_fact_region,
    "ext": ingest_ext,
    "pdem": ingest_pdem,
    "weather": ingest_weather,
    "fixcal": fix_calendar_from_facts,
}

def main(argv):
    cfg = load_cfg()
    conn = get_conn(cfg)
    init_db(conn)
    # ext не в списке по умолчанию: сторонний xlsx нужен только для EDA
    steps = argv or ["so", "calendar", "facthour", "factregion", "pdem"]
    for s in steps:
        if s == "validate":
            validate(conn)
        else:
            STEPS[s](conn, cfg)
    if "validate" not in steps:
        validate(conn)

if __name__ == "__main__":
    main(sys.argv[1:])
