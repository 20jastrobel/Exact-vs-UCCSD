-- 0001_init.sql
-- Initial SQLite schema for run storage.

BEGIN;

CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  status TEXT NOT NULL,
  git_commit TEXT,
  seed INTEGER,
  config_json TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  system_json TEXT,
  ansatz_json TEXT,
  optimizer_json TEXT,
  summary_json TEXT
);

CREATE TABLE IF NOT EXISTS steps(
  run_id TEXT NOT NULL,
  step_idx INTEGER NOT NULL,
  metrics_json TEXT NOT NULL,
  energy REAL,
  grad_norm REAL,
  wall_time_s REAL,
  PRIMARY KEY(run_id, step_idx),
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts(
  artifact_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  bytes INTEGER,
  created_at TEXT NOT NULL,
  extra_json TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);

COMMIT;

