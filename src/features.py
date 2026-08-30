"""Признаки для пары (день D, час-кандидат h). Все строится as-of: для дня D
используется только то, что было известно к вечеру D-1 (available_from <= D-1).

Таблица ext_load (сторонняя почасовка) в признаки не входит - по условию задачи
она только для ознакомления. Источники признаков: target/official_load (с лагом
публикации АТС), pdem (план на сутки вперед), индексы РСВ, погода (прогноз
as-of D-1), календарь, астрономия.
"""
import math
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

MORNING_MAX = 8          # часы <=8 считаем утренним блоком

class FeatureBuilder:
    def __init__(self, conn, cfg):
        self.conn = conn
        self.cfg = cfg
        # окна СО
        self.windows = {}
        for y, m, h in conn.execute("SELECT year, month, hour FROM so_window"):
            self.windows.setdefault((y, m), []).append(h)
        for k in self.windows:
            self.windows[k].sort()
        # целевая
        rows = conn.execute("SELECT d, peak_hour, available_from FROM target ORDER BY d").fetchall()
        self.target = pd.DataFrame(rows, columns=["d", "peak", "af"])
        self.target["d"] = pd.to_datetime(self.target["d"]).dt.date
        self.target["af"] = pd.to_datetime(self.target["af"]).dt.date
        self.target["month"] = self.target["d"].map(lambda x: x.month)
        self.target["year"] = self.target["d"].map(lambda x: x.year)
        self.label = dict(zip(self.target["d"], self.target["peak"]))
        # календарь
        self.cal = {}
        for d, wd, work, short in conn.execute("SELECT d, weekday, is_workday, is_short FROM calendar"):
            self.cal[date.fromisoformat(d)] = (wd, work, short)
        # официальная почасовка - для «свежей формы» с лагом публикации
        self.off_curve = {}
        self.off_af = {}
        for d, h, mwh, af in conn.execute("SELECT d, hour, mwh, available_from FROM official_load"):
            dd = date.fromisoformat(d)
            self.off_curve.setdefault(dd, {})[h] = mwh
            self.off_af[dd] = date.fromisoformat(af)
        self.off_days = sorted(self.off_curve)
        # pdem: суммарный план и сбытовая часть (PKRASN*)
        self.pdem_tot, self.pdem_sbyt = {}, {}
        for d, h, gtp, mwh in conn.execute("SELECT d, hour, gtp, mwh FROM pdem"):
            dd = date.fromisoformat(d)
            self.pdem_tot.setdefault(dd, Counter())[h] += mwh or 0.0
            if str(gtp).startswith("PKRASN"):
                self.pdem_sbyt.setdefault(dd, Counter())[h] += mwh or 0.0
        # РСВ: цена и плановый объем 2-й ЦЗ, известны в D-1
        self.rsv_price, self.rsv_vol = {}, {}
        for d, h, s, v in conn.execute(
                "SELECT d, hour, src, value FROM aux WHERE src IN ('rsv_price','rsv_vol')"):
            dd = date.fromisoformat(d)
            (self.rsv_price if s == 'rsv_price' else self.rsv_vol)[(dd, h)] = v
        # погода: прогноз as-of D-1, взвешенный по городам
        wts = {c: v["weight"] for c, v in cfg["weather"]["cities"].items()}
        acc = {}
        for d, h, city, var, val in conn.execute(
                "SELECT d, hour, city, var, value FROM weather WHERE kind='fc1'"):
            key = (date.fromisoformat(d), h, var)
            a = acc.setdefault(key, [0.0, 0.0])
            w = wts.get(city, 0)
            a[0] += w * val
            a[1] += w
        self.fc1 = {k: v[0] / v[1] for k, v in acc.items() if v[1] > 0}
        # фактическая среднесуточная температура - для температурных аномалий
        acc3 = {}
        for d, h, city, var, val in conn.execute(
                "SELECT d, hour, city, var, value FROM weather WHERE kind='fact' AND var='temperature_2m'"):
            key = (date.fromisoformat(d), h)
            a = acc3.setdefault(key, [0.0, 0.0])
            a[0] += wts.get(city, 0) * val
            a[1] += wts.get(city, 0)
        by_day = {}
        for (d, h), (s, w) in acc3.items():
            if w > 0:
                by_day.setdefault(d, []).append(s / w)
        self.t_fact_day = {d: float(np.mean(v)) for d, v in by_day.items()}

    def candidates(self, D):
        return self.windows.get((D.year, D.month), [])

    def _gap_before(self, D):
        g, d = 0, D - timedelta(days=1)
        while d in self.cal and self.cal[d][1] == 0 and g < 12:
            g += 1
            d -= timedelta(days=1)
        return g

    def _gap_after(self, D):
        g, d = 0, D + timedelta(days=1)
        while d in self.cal and self.cal[d][1] == 0 and g < 12:
            g += 1
            d += timedelta(days=1)
        return g

    @staticmethod
    def _sun(D, lat=56.01, lon=92.87):
        doy = D.timetuple().tm_yday
        decl = math.radians(23.45) * math.sin(2 * math.pi * (284 + doy) / 365)
        latr = math.radians(lat)
        x = -math.tan(latr) * math.tan(decl)
        x = min(1, max(-1, x))
        ha = math.acos(x)                       # часовой угол
        daylight = ha * 24 / math.pi
        noon_msk = 12 + 3 - lon / 15
        return daylight, noon_msk - daylight / 2, noon_msk + daylight / 2

    def build_day(self, D):
        as_of = D - timedelta(days=1)
        cand = self.candidates(D)
        if not cand:
            return None
        # климатология по опубликованным фактам
        av = self.target[(self.target["af"] <= as_of)]
        sm = av[av["month"] == D.month]
        n_sm = len(sm)
        cnt_all = Counter(sm["peak"])
        recent_years = sorted(sm["year"].unique())[-3:]
        sm3 = sm[sm["year"].isin(recent_years)]
        cnt_3y = Counter(sm3["peak"])
        n_3y = len(sm3)
        morning_share = (sm["peak"] <= MORNING_MAX).mean() if n_sm else np.nan
        # тот же день недели
        wd = self.cal.get(D, (D.isoweekday(), 1, 0))[0]
        sm_wd = sm[[self.cal.get(x, (x.isoweekday(),))[0] == wd for x in sm["d"]]]
        cnt_wd = Counter(sm_wd["peak"])
        n_wd = len(sm_wd)
        # свежая форма: последние 22 опубликованных дня официальной почасовки
        pub_days = [u for u in self.off_days if self.off_af.get(u, date.max) <= as_of][-22:]
        rank_sum, argmax_cnt = Counter(), Counter()
        for u in pub_days:
            curve = self.off_curve[u]
            vals = [(curve.get(h, -1e9), h) for h in cand]
            order = sorted(vals, key=lambda t: (-t[0], t[1]))
            for r, (_, h) in enumerate(order, 1):
                rank_sum[h] += r
            argmax_cnt[order[0][1]] += 1
        n_pub = len(pub_days)
        # Монте-Карло по кривым того же месяца (посл. 3 года): mu/Sigma по
        # часам-кандидатам, сэмплируем и считаем P(h = argmax). В отличие от
        # счетчика побед учитывает и форму, и разброс.
        mc_p = {}
        mc_days = [u for u in self.off_days
                   if u.month == D.month and u.year >= D.year - 3
                   and self.off_af.get(u, date.max) <= as_of
                   and self.cal.get(u, (0, 1, 0))[1] == 1]
        if len(mc_days) >= 8:
            mat = []
            for u in mc_days:
                cv = self.off_curve[u]
                if all(h in cv for h in cand):
                    arr = np.array([cv[h] for h in cand], float)
                    mat.append(arr / arr.mean())
            if len(mat) >= 8:
                mat = np.array(mat)
                mu = mat.mean(axis=0)
                sig = np.cov(mat.T)
                lam = 0.30
                sig = (1 - lam) * sig + lam * np.diag(np.diag(sig))
                sig += np.eye(len(cand)) * 1e-8
                rng = np.random.default_rng(D.toordinal())
                try:
                    samples = rng.multivariate_normal(mu, sig, size=2000, method="cholesky")
                    win = np.bincount(np.argmax(samples, axis=1), minlength=len(cand))
                    for i, h in enumerate(cand):
                        mc_p[h] = win[i] / win.sum()
                except np.linalg.LinAlgError:
                    pass
        p_tot = self.pdem_tot.get(D)
        p_sb = self.pdem_sbyt.get(D)
        def _z(m, h):
            if not m:
                return np.nan, np.nan, np.nan
            arr = np.array([m.get(x, np.nan) for x in cand], float)
            if np.all(np.isnan(arr)) or np.nanstd(arr) == 0:
                return np.nan, np.nan, np.nan
            v = m.get(h, np.nan)
            z = (v - np.nanmean(arr)) / np.nanstd(arr)
            rank = int(np.nansum(arr > v)) + 1 if not np.isnan(v) else np.nan
            margin = (np.nanmax(arr) - v) / (np.nanmean(arr) or 1)
            return z, rank, margin
        # погода
        t_by_h = {h: self.fc1.get((D, h, "temperature_2m")) for h in range(1, 25)}
        t_day = [t for t in t_by_h.values() if t is not None]
        t_mean = float(np.mean(t_day)) if t_day else np.nan
        t_min = float(np.min(t_day)) if t_day else np.nan
        # балансы утро/день - главный водораздел «когда пик»: отопление и темнота
        # утром против дневной жары и освещения
        t_morn_l = [t_by_h[h] for h in range(5, 10) if t_by_h.get(h) is not None]
        t_aft_l = [t_by_h[h] for h in range(12, 18) if t_by_h.get(h) is not None]
        t_morn = float(np.mean(t_morn_l)) if t_morn_l else np.nan
        t_aft = float(np.mean(t_aft_l)) if t_aft_l else np.nan
        c_morn_l = [self.fc1.get((D, h, "cloud_cover")) for h in range(5, 10)]
        c_morn_l = [c for c in c_morn_l if c is not None]
        c_aft_l = [self.fc1.get((D, h, "cloud_cover")) for h in range(12, 18)]
        c_aft_l = [c for c in c_aft_l if c is not None]
        cloud_morn = float(np.mean(c_morn_l)) if c_morn_l else np.nan
        cloud_aft = float(np.mean(c_aft_l)) if c_aft_l else np.nan
        # факты ERA5 публикуются с лагом ~5 дней, поэтому окно от D-6
        hist_t = [self.t_fact_day.get(D - timedelta(days=k)) for k in range(6, 16)]
        hist_t = [t for t in hist_t if t is not None]
        t_anom = (t_mean - float(np.mean(hist_t))) if (hist_t and not np.isnan(t_mean)) else np.nan
        daylight, sunrise, sunset = self._sun(D)
        wd_, work, short = self.cal.get(D, (D.isoweekday(), 1, 0))
        gap_b, gap_a = self._gap_before(D), self._gap_after(D)

        rsv_v = {h: self.rsv_vol.get((D, h)) for h in cand}
        rsv_p = {h: self.rsv_price.get((D, h)) for h in cand}
        rsv_v = {h: v for h, v in rsv_v.items() if v is not None} or None
        rsv_p = {h: v for h, v in rsv_p.items() if v is not None} or None
        rows = []
        for h in cand:
            zt, rt, mt = _z(p_tot, h)
            zs, rs, _ = _z(p_sb, h)
            zv, rv, mv = _z(rsv_v, h)
            zp, rp, _ = _z(rsv_p, h)
            fc_t = self.fc1.get((D, h, "temperature_2m"), np.nan)
            fc_c = self.fc1.get((D, h, "cloud_cover"), np.nan)
            rows.append({
                "d": D, "hour": h,
                "label": int(self.label.get(D) == h) if D in self.label else np.nan,
                # климатология
                "clim_all": cnt_all.get(h, 0) / n_sm if n_sm else np.nan,
                "clim_3y": cnt_3y.get(h, 0) / n_3y if n_3y else np.nan,
                "clim_wd": cnt_wd.get(h, 0) / n_wd if n_wd else np.nan,
                "clim_n": n_sm,
                "morning_share": morning_share,
                # свежая форма из официальной почасовки (месячный лаг)
                "off_meanrank": rank_sum.get(h, np.nan) / n_pub if n_pub else np.nan,
                "off_argmax_share": argmax_cnt.get(h, 0) / n_pub if n_pub else np.nan,
                "mc_p": mc_p.get(h, np.nan),
                # план на сутки вперед
                "pdem_z": zt, "pdem_rank": rt, "pdem_margin": mt,
                "pdem_sbyt_z": zs, "pdem_sbyt_rank": rs,
                "has_pdem": int(p_tot is not None),
                # РСВ: зонный план и цена, известны в D-1, 11 лет глубины
                "rsv_vol_z": zv, "rsv_vol_rank": rv, "rsv_vol_margin": mv,
                "rsv_price_z": zp, "rsv_price_rank": rp,
                # погода (прогноз as-of D-1)
                "fc_temp_h": fc_t, "fc_temp_dev": fc_t - t_mean if not np.isnan(fc_t) and not np.isnan(t_mean) else np.nan,
                "fc_cloud_h": fc_c,
                "t_day_mean": t_mean, "t_day_min": t_min, "t_anom": t_anom,
                "t_morn": t_morn, "t_aft": t_aft,
                "t_ma_diff": (t_morn - t_aft
                              if not (np.isnan(t_morn) or np.isnan(t_aft)) else np.nan),
                "cloud_morn": cloud_morn, "cloud_aft": cloud_aft,
                "heating_deg": max(0.0, 18.0 - t_mean) if not np.isnan(t_mean) else np.nan,
                # астрономия (МСК)
                "daylight": daylight,
                "dark_at_h": int(h - 0.5 < sunrise or h - 0.5 > sunset),
                "h_vs_sunset": h - sunset,
                # календарь
                "month": D.month, "weekday": wd_,
                "yr": D.year + D.timetuple().tm_yday / 366.0,
                "is_work_sat": int(wd_ >= 6 and work == 1),
                "is_short": short, "gap_before": gap_b, "gap_after": gap_a,
                # статика часа
                "h": h, "is_morning": int(h <= MORNING_MAX),
                "pos_in_window": cand.index(h) / max(1, len(cand) - 1),
            })
        return pd.DataFrame(rows)

    def build_range(self, days, cache=None):
        out = []
        for D in days:
            if cache is not None and D in cache:
                out.append(cache[D])
                continue
            df = self.build_day(D)
            if df is not None:
                if cache is not None:
                    cache[D] = df
                out.append(df)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

FEATURES = [
    "clim_all", "clim_3y", "clim_wd", "clim_n", "morning_share",
    "off_meanrank", "off_argmax_share", "mc_p",
    "pdem_z", "pdem_rank", "pdem_margin", "pdem_sbyt_z", "pdem_sbyt_rank", "has_pdem",
    "rsv_vol_z", "rsv_vol_rank", "rsv_vol_margin", "rsv_price_z", "rsv_price_rank",
    "fc_temp_h", "fc_temp_dev", "fc_cloud_h", "t_day_mean", "t_day_min", "t_anom", "heating_deg",
    "t_morn", "t_aft", "t_ma_diff", "cloud_morn", "cloud_aft",
    "daylight", "dark_at_h", "h_vs_sunset",
    "month", "weekday", "yr", "is_work_sat", "is_short", "gap_before", "gap_after",
    "h", "is_morning", "pos_in_window",
]
