-- База прогноза пикового часа. Часы всюду в нотации АТС: 1..24, московское время.
-- У каждой строки фактов есть available_from - первая дата, когда строку можно
-- было знать. Признаки обязаны фильтровать available_from <= as_of.

CREATE TABLE IF NOT EXISTS calendar (
  d          TEXT PRIMARY KEY,                -- ISO-дата
  weekday    INTEGER NOT NULL,                -- 1=пн .. 7=вс
  is_workday INTEGER NOT NULL,
  is_short   INTEGER NOT NULL DEFAULT 0,
  src        TEXT NOT NULL DEFAULT 'xmlcalendar'
);

CREATE TABLE IF NOT EXISTS so_window (
  year INTEGER NOT NULL, month INTEGER NOT NULL, hour INTEGER NOT NULL,
  PRIMARY KEY (year, month, hour)
);

CREATE TABLE IF NOT EXISTS target (            -- calcfacthour: ответы
  d TEXT PRIMARY KEY,
  peak_hour INTEGER NOT NULL,
  available_from TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS official_load (     -- fact_region: официальная почасовка, МВт*ч
  d TEXT NOT NULL, hour INTEGER NOT NULL, mwh REAL NOT NULL,
  available_from TEXT NOT NULL,
  PRIMARY KEY (d, hour)
);

CREATE TABLE IF NOT EXISTS ext_load (          -- сторонняя почасовка по ГТП; только для EDA
  d TEXT NOT NULL, hour INTEGER NOT NULL, gtp TEXT NOT NULL, mwh REAL,
  available_from TEXT NOT NULL,
  PRIMARY KEY (d, hour, gtp)
);

CREATE TABLE IF NOT EXISTS pdem (              -- план потребления на сутки вперед, ГТП Красноярска
  d TEXT NOT NULL, hour INTEGER NOT NULL, gtp TEXT NOT NULL, mwh REAL,
  available_from TEXT NOT NULL,                -- D-1, публикуется накануне
  PRIMARY KEY (d, hour, gtp)
);

CREATE TABLE IF NOT EXISTS weather (
  d TEXT NOT NULL, hour INTEGER NOT NULL,
  city TEXT NOT NULL,
  kind TEXT NOT NULL,                          -- 'fact' (ERA5) | 'fc1' (прогноз as-of D-1)
  var TEXT NOT NULL,
  value REAL,
  available_from TEXT NOT NULL,
  PRIMARY KEY (d, hour, city, kind, var)
);

CREATE TABLE IF NOT EXISTS prediction (
  run_id TEXT NOT NULL, d TEXT NOT NULL, hour INTEGER NOT NULL,
  rank INTEGER NOT NULL, score REAL,
  PRIMARY KEY (run_id, d, hour)
);

CREATE TABLE IF NOT EXISTS ingest_log (
  src_file TEXT PRIMARY KEY, sha1 TEXT, rows INTEGER, parsed_at TEXT
);

-- вспомогательные публичные ряды: индексы РСВ 2-й ЦЗ (цена, плановый объем)
CREATE TABLE IF NOT EXISTS aux (
  d TEXT NOT NULL, hour INTEGER NOT NULL, src TEXT NOT NULL,
  value REAL, available_from TEXT NOT NULL,
  PRIMARY KEY (d, hour, src));

CREATE INDEX IF NOT EXISTS idx_official_load_d ON official_load(d);
CREATE INDEX IF NOT EXISTS idx_ext_load_d ON ext_load(d, gtp);
CREATE INDEX IF NOT EXISTS idx_pdem_d ON pdem(d);
CREATE INDEX IF NOT EXISTS idx_weather_d ON weather(d, kind, var);
CREATE INDEX IF NOT EXISTS aux_af ON aux(available_from);
