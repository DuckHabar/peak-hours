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
    raise ValueError(name)
