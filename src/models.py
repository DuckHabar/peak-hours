"""Модели ранжирования часов. Интерфейс: fit(X, y) / score(X) -> np.array.
Итоговый порядок - сортировка по убыванию score, при равенстве выше меньший час."""
import numpy as np

from .features import FEATURES

class Climatology:
    """Доля побед часа в этом календарном месяце по опубликованным фактам.
    Берутся последние 3 года: распределение часа пика дрейфует, глубокая
    история 2013-2022 роняет top-1 с ~0.38 до ~0.26."""
    needs_fit = False

    def fit(self, X, y):
        return self

    def score(self, X):
        s = X["clim_3y"].to_numpy(dtype=float)
        fallback = np.nan_to_num(X["clim_all"].to_numpy(dtype=float), nan=-1)
        return np.where(np.isnan(s), fallback, s)

class LgbmModel:
    needs_fit = True

    def __init__(self, **kw):
        self.params = dict(
            objective="binary", n_estimators=500, learning_rate=0.04,
            num_leaves=31, min_child_samples=40, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.85, reg_lambda=1.0,
            verbose=-1, n_jobs=-1,
        )
        self.params.update(kw)

    def fit(self, X, y):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X[FEATURES], y)
        return self

    def score(self, X):
        return self.model.predict_proba(X[FEATURES])[:, 1]

class LgbmRanker:
    """lambdarank: оптимизирует порядок часов внутри дня напрямую."""
    needs_fit = True

    def __init__(self, **kw):
        self.params = dict(
            objective="lambdarank", n_estimators=500, learning_rate=0.04,
            num_leaves=31, min_child_samples=30, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.85, reg_lambda=1.0, label_gain=[0, 1],
            verbose=-1, n_jobs=-1,
        )
        self.params.update(kw)

    def fit(self, X, y):
        import lightgbm as lgb
        groups = X.groupby("d", sort=False).size().to_numpy()
        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(X[FEATURES], y, group=groups)
        return self

    def score(self, X):
        return self.model.predict(X[FEATURES])

class SkTrees:
    """RandomForest/ExtraTrees. На этой задаче бэггинг стабильно обыгрывает
    бустинг (мало данных, шумная цель). NaN -> -999, деревьям это не мешает."""
    needs_fit = True

    def __init__(self, kind="rf", **kw):
        self.kind = kind
        self.params = dict(n_estimators=800, min_samples_leaf=8, n_jobs=-1, random_state=0)
        self.params.update(kw)

    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        cls = ExtraTreesClassifier if self.kind == "et" else RandomForestClassifier
        self.model = cls(**self.params)
        self.model.fit(X[FEATURES].fillna(-999), y)
        return self

    def score(self, X):
        return self.model.predict_proba(X[FEATURES].fillna(-999))[:, 1]

class RfX:
    """Лес поверх FEATURES + выходы регрессии суточной кривой (z-скор, отставание
    от максимума и ранг прогноза формы внутри дня) + дневные агрегаты ветки.
    Для обучающих дней прогноз формы восстанавливается из отложенных остатков
    (см. curve.py), так что согласованность признаков train/eval не течет."""
    needs_fit = True

    EXTRA = ["cv_z", "cv_gap", "cv_rank",
             "x_dev_morn", "x_dev_aft", "x_tma", "x_smdiff", "x_pdem_ma"]

    def __init__(self, **kw):
        self.params = dict(n_estimators=800, min_samples_leaf=8, n_jobs=-1, random_state=0)
        self.params.update(kw)

    def bind(self, fb):
        from .curve import CurveBranch
        if not hasattr(fb, "_curve"):
            fb._curve = CurveBranch(fb)
        self.branch = fb._curve

    def fit_month(self, ms):
        self.branch.fit_month(ms)

    def _day_scores(self, X, train):
        out = {}
        for d in X["d"].drop_duplicates():
            if train:
                out[d] = self.branch.oof_yhat(d) or {}
            else:
                r = self.branch.predict_day(d)
                out[d] = r["yhat"] if r else {}
        return out

    def _day_extra(self, d):
        if not hasattr(self.branch.fb, "_dayfeat_cache"):
            self.branch.fb._dayfeat_cache = {}
        c = self.branch.fb._dayfeat_cache
        if d not in c:
            c[d] = self.branch.day_features(d)
        f = c[d]
        return (f["dev_morn"], f["dev_aft"],
                (f["t_morn"] - f["t_aft"]) if f["t_morn"] == f["t_morn"] and f["t_aft"] == f["t_aft"] else np.nan,
                (f["sm_morn"] - f["sm_aft"]) if f["sm_morn"] == f["sm_morn"] and f["sm_aft"] == f["sm_aft"] else np.nan,
                (f["pdem_morn"] - f["pdem_aft"]) if f["pdem_morn"] == f["pdem_morn"] and f["pdem_aft"] == f["pdem_aft"] else np.nan)

    def _augment(self, X, train):
        sc = self._day_scores(X, train)
        z, gap, rank = [], [], []
        dm, da, tma, smd, pma = [], [], [], [], []
        for d, grp in X.groupby("d", sort=False):
            yh = sc.get(d, {})
            n = len(grp)
            e = self._day_extra(d)
            dm += [e[0]] * n; da += [e[1]] * n; tma += [e[2]] * n
            smd += [e[3]] * n; pma += [e[4]] * n
            v = np.array([yh.get(h, np.nan) for h in grp["hour"]], float)
            if np.all(np.isnan(v)) or np.nanstd(v) == 0:
                z += [np.nan] * n; gap += [np.nan] * n; rank += [np.nan] * n
                continue
            zz = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
            z += list(zz)
            gap += list(np.nanmax(v) - v)
            order = (-v).argsort()
            rr = np.empty_like(order); rr[order] = np.arange(1, len(v) + 1)
            rank += list(rr.astype(float))
        X = X.copy()
        X["cv_z"], X["cv_gap"], X["cv_rank"] = z, gap, rank
        X["x_dev_morn"], X["x_dev_aft"], X["x_tma"] = dm, da, tma
        X["x_smdiff"], X["x_pdem_ma"] = smd, pma
        return X

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        Xa = self._augment(X, train=True)
        self.model = RandomForestClassifier(**self.params)
        self.cols = FEATURES + self.EXTRA
        self.model.fit(Xa[self.cols].fillna(-999), y)
        return self

    def score(self, X):
        Xa = self._augment(X, train=False)
        return self.model.predict_proba(Xa[self.cols].fillna(-999))[:, 1]

def _parse_kw(name):
    """'rf:n_estimators=2000,min_samples_leaf=4' -> словарь (int/float автоматом)."""
    kw = {}
    _, _, rest = name.partition(":")
    if rest:
        for tok in rest.split(","):
            k, _, v = tok.partition("=")
            kw[k] = float(v) if "." in v else int(v)
    return kw

def make_model(name):
    base = name.split(":")[0]
    kw = _parse_kw(name)
    if base == "clim":
        return Climatology()
    if base == "lgbm":
        return LgbmModel(**kw)
    if base == "lgbm_rank":
        return LgbmRanker(**kw)
    if base == "et":
        return SkTrees("et", **kw)
    if base == "rf":
        return SkTrees("rf", **kw)
    if base == "rfx":
        return RfX(**kw)
    raise ValueError(name)
