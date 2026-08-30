"""Регрессия аномалии формы суточной кривой official_load.

Цель - не сама форма (mwh/среднее дня), а ее отклонение от климатологии формы
(месяц, класс дня, час), посчитанной as-of. Причина: argmax внутри окна СО
решают деления в 0.2-0.5% уровня, а суточный размах формы ~25%; регрессия
полной формы тратит емкость на ночь и вечер и усредняет плато (top-1 падает
до 0.19 против 0.38 у климатологии). Обучение только на дневных часах
TRAIN_HOURS, окно СО всегда внутри.

Дисциплина as-of: признаки дня u берут только кривые с available_from <= u-1
(срезы по батчам публикации), погоду-прогноз as-of u-1 (для глубокой истории,
где архивов прогнозов нет, - факт ERA5: суточный прогноз температуры ошибается
на ~1°C, подмена мало что искажает), pdem as-of u-1. Остатки регрессии считаются
по годам публикации на данных, которых модель не видела, - честная мера
неопределенности формы."""
import bisect
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

HOURS = list(range(1, 25))
TRAIN_HOURS = list(range(4, 20))     # окна СО всех лет лежат внутри 5..18
# сезонные группы: зима (отопление, темнота), плечи, лето
SEASON_GRP = {11: 0, 12: 0, 1: 0, 2: 0, 3: 0, 4: 1, 9: 1, 10: 1,
              5: 2, 6: 2, 7: 2, 8: 2}

CURVE_FEATURES = [
    "h", "h_sin", "h_cos", "is_morning",
    "month", "doy_sin", "doy_cos", "weekday", "is_workday", "is_work_sat",
    "is_short", "gap_before", "gap_after",
    "daylight", "dark_at_h", "h_vs_sunset", "h_vs_sunrise",
    "w_t", "w_dt", "t_morn", "t_aft", "t_ma_diff",
    "w_cloud", "w_swrad", "w_app", "w_heat",
    "t_day_mean", "t_day_min", "t_anom",
    "shape_mh", "shape_mdh", "shape_mh_all", "shape_r22", "shape_r22_raw",
    "shape_r22_sm", "dev_morn", "dev_aft",
    "yr",
    "pdem_rel", "pdem_anom", "has_pdem",
    "rsv_vol_rel", "rsv_price_rel",
]


class CurveBranch:
    def __init__(self, fb, n_estimators=600, learning_rate=0.05, num_leaves=63,
                 min_child_samples=40):
        self.fb = fb
        self.params = dict(
            objective="regression", n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0, verbose=-1, n_jobs=-1,
        )
        self._prep()
        self._residuals = None                # {день: np.array по TRAIN_HOURS}
        self._model_ms = None
        self._model = None

    def _prep(self):
        fb = self.fb
        # полные кривые (все 24 часа) и их формы
        self.days = [u for u in fb.off_days if len(fb.off_curve[u]) == 24]
        self.shape = {}
        for u in self.days:
            arr = np.array([fb.off_curve[u][h] for h in HOURS], float)
            self.shape[u] = arr / arr.mean()
        # батчи публикации: available_from -> дни
        self.af_dates = sorted({fb.off_af[u] for u in self.days})
        by_af = {}
        for u in self.days:
            by_af.setdefault(fb.off_af[u], []).append(u)
        # накопительные срезы климатологии формы после каждого батча публикации;
        # климатология «за 3 года» = срез(cutoff) - срез(cutoff - 3 года)
        mh_sum = np.zeros((13, 24)); mh_cnt = np.zeros(13)
        mdh_sum = np.zeros((13, 3, 24)); mdh_cnt = np.zeros((13, 3))
        recent = []                            # опубликованные дни по порядку
        rec_grp = {0: [], 1: [], 2: []}        # то же по сезонным группам
        self.snap = []
        self.recent_at = []                    # последние 22 дня на момент среза
        self.recent_sm_at = []                 # то же по сезонным группам
        for af in self.af_dates:
            for u in sorted(by_af[af]):
                m = u.month
                dc = self._dclass(u)
                mh_sum[m] += self.shape[u]; mh_cnt[m] += 1
                mdh_sum[m, dc] += self.shape[u]; mdh_cnt[m, dc] += 1
                recent.append(u)
                rec_grp[SEASON_GRP[m]].append(u)
            self.snap.append((mh_sum.copy(), mh_cnt.copy(),
                              mdh_sum.copy(), mdh_cnt.copy()))
            self.recent_at.append(list(recent[-22:]))
            self.recent_sm_at.append({g: list(v[-22:]) for g, v in rec_grp.items()})
        # погода: fc1 приоритетно, факт как замена для глубокой истории
        self._wx = {}
        conn = fb.conn
        wts = {c: v["weight"] for c, v in fb.cfg["weather"]["cities"].items()}
        for kind in ("fact", "fc1"):           # fc1 вторым, перетирает факт
            acc = {}
            for d, h, city, var, val in conn.execute(
                    "SELECT d, hour, city, var, value FROM weather WHERE kind=?", (kind,)):
                key = (d, h, var)
                a = acc.setdefault(key, [0.0, 0.0])
                w = wts.get(city, 0)
                a[0] += w * val; a[1] += w
            for (d, h, var), (s, w) in acc.items():
                if w > 0:
                    self._wx[(date.fromisoformat(d), h, var)] = s / w
        # РСВ: формы планового объема и цены (известны в D-1)
        self._rsv = {}
        for d, h, s, v in conn.execute(
                "SELECT d, hour, src, value FROM aux WHERE src IN ('rsv_price','rsv_vol')"):
            self._rsv.setdefault((date.fromisoformat(d), s), {})[h] = v
        # суточные агрегаты температуры
        self._t_day = {}
        tmp = {}
        for (d, h, var), v in self._wx.items():
            if var == "temperature_2m":
                tmp.setdefault(d, []).append(v)
        for d, vs in tmp.items():
            self._t_day[d] = (float(np.mean(vs)), float(np.min(vs)))
        self._frame_cache = None

    def _dclass(self, u):
        wd, work, _ = self.fb.cal.get(u, (u.isoweekday(), 1, 0))
        if work:
            return 0
        return 1 if wd == 6 else 2

    def _snapshot_asof(self, cutoff):
        """Климатология формы к cutoff: полная (гладкая, но дрейфует) и 3-летняя
        (свежая, но шумная: ~60 дней на ячейку, шум ~0.2% - масштаб отрыва
        победителя). База - их смесь, обе версии идут и в признаки."""
        i = bisect.bisect_right(self.af_dates, cutoff) - 1
        if i < 0:
            return None
        cutoff3 = date(cutoff.year - 3, cutoff.month, min(cutoff.day, 28))
        j = bisect.bisect_right(self.af_dates, cutoff3) - 1
        mhA_s, mhA_c, mdhA_s, mdhA_c = self.snap[i]
        if j >= 0:
            m0, c0, d0, e0 = self.snap[j]
            mh3_s = mhA_s - m0; mh3_c = mhA_c - c0
            mdh3_s = mdhA_s - d0; mdh3_c = mdhA_c - e0
        else:
            mh3_s, mh3_c, mdh3_s, mdh3_c = mhA_s, mhA_c, mdhA_s, mdhA_c
        return (mh3_s, mh3_c, mdh3_s, mdh3_c,
                mhA_s, mhA_c, mdhA_s, mdhA_c,
                self.recent_at[i], self.recent_sm_at[i])

    def day_rows(self, D, with_label=True):
        """24 строки признаков для дня D, as-of D-1. y_anom заполнен, если кривая есть."""
        fb = self.fb
        cutoff = D - timedelta(days=1)
        snap = self._snapshot_asof(cutoff)
        wd, work, short = fb.cal.get(D, (D.isoweekday(), 1, 0))
        dc = 0 if work else (1 if wd == 6 else 2)
        doy = D.timetuple().tm_yday
        daylight, sunrise, sunset = fb._sun(D)
        t_agg = self._t_day.get(D, (np.nan, np.nan))
        # аномалия: прогнозная средняя против факта последних дней (факты с лагом ~5д)
        hist = [self._t_day.get(D - timedelta(days=k), (np.nan,))[0] for k in range(6, 16)]
        hist = [t for t in hist if not (isinstance(t, float) and np.isnan(t))]
        t_anom = t_agg[0] - float(np.mean(hist)) if hist and not np.isnan(t_agg[0]) else np.nan
        def _rel(m):
            if not m:
                return np.full(24, np.nan)
            a = np.array([m.get(h, np.nan) for h in HOURS], float)
            mu = np.nanmean(a)
            return a / mu if mu else np.full(24, np.nan)
        rsv_vol_rel = _rel(self._rsv.get((D, "rsv_vol")))
        rsv_price_rel = _rel(self._rsv.get((D, "rsv_price")))
        p = fb.pdem_tot.get(D)
        if p:
            pv = np.array([p.get(h, np.nan) for h in HOURS], float)
            pmean = np.nanmean(pv)
            pdem_rel = pv / pmean if pmean else np.full(24, np.nan)
        else:
            pdem_rel = np.full(24, np.nan)
        label = self.shape.get(D) if with_label else None
        # r22_anom: средняя девиация последних 22 опубликованных дней от их
        # собственной клим-нормы (сезонно-чистый сигнал); r22_raw - сырое среднее
        # тех же дней: несет уровень текущего режима, но из-за ~40-дневного лага
        # публикации форму чужого сезона
        r22_anom = np.full(24, np.nan)
        r22_raw = np.full(24, np.nan)
        r22_sm = np.full(24, np.nan)
        if snap is not None:
            mh3_s, mh3_c, mdh3_s, mdh3_c, mhA_s, mhA_c, mdhA_s, mdhA_c, rec, rec_sm = snap
            def _dev(u):
                mu, du = u.month, self._dclass(u)
                if mdh3_c[mu, du] >= 5:
                    return self.shape[u] - mdh3_s[mu, du] / mdh3_c[mu, du]
                if mh3_c[mu] >= 5:
                    return self.shape[u] - mh3_s[mu] / mh3_c[mu]
                return None
            devs = [d for d in (_dev(u) for u in rec) if d is not None]
            if devs:
                r22_anom = np.mean(devs, axis=0)
            if rec:
                r22_raw = np.mean([self.shape[u] for u in rec], axis=0)
            # девиации свежих дней той же сезонной группы: через границу сезона
            # r22_anom тащит чужой профиль (сентябрьский полуденный сдвиг
            # не переносится в ноябрь)
            sg = SEASON_GRP[D.month]
            sm_days = rec_sm.get(sg, [])
            devs_sm = [d for d in (_dev(u) for u in sm_days) if d is not None]
            if devs_sm:
                r22_sm = np.mean(devs_sm, axis=0)
        # блочные агрегаты девиаций: баланс «утро против дня» переносится между
        # месяцами лучше, чем почасовой профиль
        dev_morn = float(np.nanmean(r22_anom[4:8])) if not np.all(np.isnan(r22_anom[4:8])) else np.nan
        dev_aft = float(np.nanmean(r22_anom[10:17])) if not np.all(np.isnan(r22_anom[10:17])) else np.nan
        K = 15.0        # псевдонаблюдения полной климатологии в базе
        wt_arr = np.array([self._wx.get((D, h, "temperature_2m"), np.nan) for h in HOURS])
        t_morn = float(np.nanmean(wt_arr[4:9])) if not np.all(np.isnan(wt_arr[4:9])) else np.nan
        t_aft = float(np.nanmean(wt_arr[11:17])) if not np.all(np.isnan(wt_arr[11:17])) else np.nan
        rows = []
        for i, h in enumerate(HOURS):
            wt = wt_arr[i]
            if snap is not None:
                m, mc = D.month, dc
                s_mh = mh3_s[m][i] / mh3_c[m] if mh3_c[m] >= 5 else np.nan
                s_mdh = (mdh3_s[m, mc][i] / mdh3_c[m, mc]
                         if mdh3_c[m, mc] >= 5 else np.nan)
                s_mh_all = mhA_s[m][i] / mhA_c[m] if mhA_c[m] >= 5 else np.nan
                s_mdh_all = (mdhA_s[m, mc][i] / mdhA_c[m, mc]
                             if mdhA_c[m, mc] >= 5 else np.nan)
                s_r22 = r22_anom[i]
                s_r22_raw = r22_raw[i]
                s_r22_sm = r22_sm[i]
            else:
                s_mh = s_mdh = s_mh_all = s_mdh_all = s_r22 = s_r22_raw = s_r22_sm = np.nan
            # база - смесь 3-летней и полной климатологии: 3-летняя ловит
            # текущий режим, полная гасит шум ячейки
            if not np.isnan(s_mdh):
                b3, n3, bA = s_mdh, mdh3_c[D.month, dc], s_mdh_all
            else:
                b3, n3, bA = s_mh, (mh3_c[D.month] if snap is not None else 0), s_mh_all
            if not np.isnan(b3) and not np.isnan(bA):
                clim_base = (n3 * b3 + K * bA) / (n3 + K)
            elif not np.isnan(b3):
                clim_base = b3
            else:
                clim_base = bA
            yv = label[i] if label is not None else np.nan
            rows.append({
                "d": D, "hour": h,
                "y": yv,
                "clim_base": clim_base,
                "y_anom": (yv - clim_base
                           if not (np.isnan(yv) or (isinstance(clim_base, float) and np.isnan(clim_base)))
                           else np.nan),
                "h": h, "h_sin": math.sin(2 * math.pi * h / 24),
                "h_cos": math.cos(2 * math.pi * h / 24), "is_morning": int(h <= 8),
                "month": D.month, "doy_sin": math.sin(2 * math.pi * doy / 365),
                "doy_cos": math.cos(2 * math.pi * doy / 365),
                "weekday": wd, "is_workday": work,
                "is_work_sat": int(wd >= 6 and work == 1), "is_short": short,
                "gap_before": fb._gap_before(D), "gap_after": fb._gap_after(D),
                "daylight": daylight, "dark_at_h": int(h - 0.5 < sunrise or h - 0.5 > sunset),
                "h_vs_sunset": h - sunset, "h_vs_sunrise": h - sunrise,
                "w_t": wt,
                "w_dt": wt - wt_arr[i - 1] if i > 0 else np.nan,
                "t_morn": t_morn, "t_aft": t_aft,
                "t_ma_diff": t_morn - t_aft if not (np.isnan(t_morn) or np.isnan(t_aft)) else np.nan,
                "w_cloud": self._wx.get((D, h, "cloud_cover"), np.nan),
                "w_swrad": self._wx.get((D, h, "shortwave_radiation"), np.nan),
                "w_app": self._wx.get((D, h, "apparent_temperature"), np.nan),
                "w_heat": max(0.0, 18.0 - wt) if not np.isnan(wt) else np.nan,
                "t_day_mean": t_agg[0], "t_day_min": t_agg[1], "t_anom": t_anom,
                "shape_mh": s_mh, "shape_mdh": s_mdh, "shape_mh_all": s_mh_all,
                "shape_r22": s_r22, "shape_r22_raw": s_r22_raw,
                "shape_r22_sm": s_r22_sm, "dev_morn": dev_morn, "dev_aft": dev_aft,
                "yr": D.year + doy / 366.0,
                "rsv_vol_rel": rsv_vol_rel[i], "rsv_price_rel": rsv_price_rel[i],
                "pdem_rel": pdem_rel[i],
                "pdem_anom": (pdem_rel[i] - clim_base
                              if not (np.isnan(pdem_rel[i]) or np.isnan(clim_base)) else np.nan),
                "has_pdem": int(p is not None),
            })
        return rows

    def master_frame(self):
        if self._frame_cache is None:
            rows = []
            for u in self.days:
                rows.extend(self.day_rows(u))
            self._frame_cache = pd.DataFrame(rows)
        return self._frame_cache

    def _ensure_residuals(self):
        """Остатки по годам публикации: модель на всем, что опубликовано до года Y,
        предсказывает дни, опубликованные в Y. Каждый остаток вне обучения."""
        if self._residuals is not None:
            return
        import lightgbm as lgb
        mf = self._train_frame()
        af = {u: self.fb.off_af[u] for u in self.days}
        mf = mf.assign(af_year=[af[d].year for d in mf["d"]])
        self._residuals = {}
        years = sorted(mf["af_year"].unique())
        nh = len(TRAIN_HOURS)
        for y in years[1:]:
            tr = mf[mf["af_year"] < y]
            te = mf[mf["af_year"] == y]
            if len(tr) < nh * 50 or te.empty:
                continue
            mdl = lgb.LGBMRegressor(**self.params)
            mdl.fit(tr[CURVE_FEATURES], tr["y_anom"])
            pred = mdl.predict(te[CURVE_FEATURES])
            resid = te["y_anom"].to_numpy() - pred
            for dd, grp in te.assign(e=resid).groupby("d", sort=False):
                if len(grp) == nh:
                    self._residuals[dd] = grp.sort_values("hour")["e"].to_numpy()

    def _train_frame(self):
        mf = self.master_frame()
        return mf[mf["hour"].isin(TRAIN_HOURS) & mf["y_anom"].notna()]

    def fit_month(self, ms):
        """Модель месяца: все дни, опубликованные к 1-му числу ms."""
        if self._model_ms == ms:
            return
        import lightgbm as lgb
        self._ensure_residuals()
        mf = self._train_frame()
        ok = {u for u in self.days if self.fb.off_af[u] <= ms}
        tr = mf[[d in ok for d in mf["d"]]]
        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(tr[CURVE_FEATURES], tr["y_anom"])
        self._model_ms = ms

    def day_features(self, D):
        """Дневной контекст для стекинга: агрегаты day_rows по утренним (5-8)
        и дневным (11-17) часам."""
        df = pd.DataFrame(self.day_rows(D, with_label=False))
        morn = df[df["hour"].between(5, 8)]
        aft = df[df["hour"].between(11, 17)]
        r0 = df.iloc[0]
        return {
            "dev_morn": r0["dev_morn"], "dev_aft": r0["dev_aft"],
            "t_morn": morn["w_t"].mean(), "t_aft": aft["w_t"].mean(),
            "sm_morn": morn["shape_r22_sm"].mean(), "sm_aft": aft["shape_r22_sm"].mean(),
            "pdem_morn": morn["pdem_rel"].mean(), "pdem_aft": aft["pdem_rel"].mean(),
        }

    def oof_yhat(self, u):
        """Прогноз формы для исторического дня u, восстановленный как
        факт - отложенный остаток. None, если остатка для дня нет."""
        if self._residuals is None or u not in self._residuals or u not in self.shape:
            return None
        e = self._residuals[u]
        return {h: self.shape[u][h - 1] - e[i] for i, h in enumerate(TRAIN_HOURS)}

    def predict_day(self, D):
        """{hour: yhat формы} по часам окна. None, если окна нет."""
        cand = self.fb.candidates(D)
        if not cand or self._model is None:
            return None
        rows = self.day_rows(D, with_label=False)
        df = pd.DataFrame(rows)
        df = df[df["hour"].isin(TRAIN_HOURS)].reset_index(drop=True)
        anom = self._model.predict(df[CURVE_FEATURES])
        yhat_all = df["clim_base"].to_numpy() + anom
        hour_pos = {h: i for i, h in enumerate(df["hour"])}
        base = yhat_all[[hour_pos[h] for h in cand]]
        if np.any(np.isnan(base)):                 # нет среза климатологии
            return {"yhat": dict(zip(cand, [0.0] * len(cand)))}
        return {"yhat": dict(zip(cand, base))}
