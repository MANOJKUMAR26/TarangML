-- Local metadata store. Plain SQL so it ports to Postgres later.
-- Raw sensor data never goes in here; it stays in the project folder.

CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL,
    spec          TEXT,              -- ProblemSpec JSON
    status        TEXT NOT NULL,     -- described | data_ready | trained | ranked | built | failed
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id),
    files         TEXT NOT NULL,     -- list of stored file names
    sha256        TEXT NOT NULL,
    summary       TEXT NOT NULL,     -- CsvSummary JSON
    roles         TEXT NOT NULL,     -- {column: role}
    label_map     TEXT,              -- {"normal_label": "..."}
    issues        TEXT NOT NULL,     -- list of DataIssue JSON
    stats         TEXT NOT NULL,     -- sample rate, duration, classes
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id),
    rung          INTEGER NOT NULL,
    algorithm     TEXT NOT NULL,
    metrics       TEXT NOT NULL,
    passed        INTEGER NOT NULL,
    selected      INTEGER NOT NULL DEFAULT 0,
    artifact_path TEXT,
    note          TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chip_results (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id),
    candidate_id  TEXT NOT NULL,
    chip_id       TEXT NOT NULL,
    measurement   TEXT NOT NULL,     -- ChipMeasurement JSON
    score         REAL,
    rank          INTEGER,
    reasons       TEXT,
    tradeoffs     TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS builds (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id),
    chip_id       TEXT NOT NULL,
    version       TEXT NOT NULL,
    package_path  TEXT,
    parity        TEXT NOT NULL,
    passed        INTEGER NOT NULL,
    reproducibility TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      TEXT,
    decision_type   TEXT NOT NULL,
    decision_key    TEXT NOT NULL,
    question        TEXT NOT NULL,
    summary         TEXT NOT NULL,   -- what was sent to the assistant; never raw rows
    options         TEXT NOT NULL,
    answer          TEXT,
    confidence      REAL,
    source          TEXT NOT NULL,   -- rules | learned_rule | jev | claude_code | claude_chat | user | default
    provider_version TEXT,
    follow_up       TEXT,
    final_choice    TEXT,
    final_source    TEXT,
    build_passed    INTEGER,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learned_rules (
    decision_type TEXT NOT NULL,
    decision_key  TEXT NOT NULL,
    choice        TEXT NOT NULL,
    support       INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (decision_type, decision_key)
);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,     -- queued | running | done | failed
    step          TEXT,
    progress      REAL NOT NULL DEFAULT 0,
    log           TEXT NOT NULL DEFAULT '',
    error         TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    params        TEXT,
    result        TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS step_cache (
    cache_key     TEXT PRIMARY KEY,
    step          TEXT NOT NULL,
    output        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
