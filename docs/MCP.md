# MCP: Read-Only Run Ledger Server

This repo includes a small MCP (Model Context Protocol) server that exposes **read-only** access to the SQLite run ledger (`runs` / `steps` / `artifacts` tables).

It is designed for analysis agents to query benchmarking runs without needing to scan `runs/` folders or parse JSONL manually.

## Start The Server

The server opens the DB in SQLite **read-only mode** (`mode=ro`) and does **not** expose any “run arbitrary SQL” tool.

### Stdio Transport (Typical MCP Host Integration)

```bash
python mcp_servers/run_ledger_server.py --db data/runs.db --transport stdio
```

### SSE / HTTP Transports (Optional)

```bash
python mcp_servers/run_ledger_server.py --db data/runs.db --transport sse --host 127.0.0.1 --port 8000
python mcp_servers/run_ledger_server.py --db data/runs.db --transport streamable-http --host 127.0.0.1 --port 8000
```

DB path can also be set via `RUNS_DB_PATH`, but `--db` wins.

## MCP Host Configuration (Example)

Most MCP hosts accept a “stdio server” config of the form:

```json
{
  "name": "run-ledger-ro",
  "command": "python",
  "args": ["mcp_servers/run_ledger_server.py", "--db", "data/runs.db", "--transport", "stdio"],
  "env": {"RUNS_DB_PATH": "data/runs.db"}
}
```

Adjust `command`/`args` to match your Python environment.

## Tools Exposed (No Arbitrary SQL)

All tool outputs are JSON-serializable.

1. `list_runs(filters: dict, limit: int=50) -> list[dict]`
   - SQL-level filters supported: `run_id`, `status`, `seed`, `git_commit` (supports prefix via `"abc*"`), `config_hash`
   - Additional dotted filters are applied post-hoc (e.g. `system.sites`, `ansatz.pool`).

2. `get_run(run_id: str) -> dict | None`
   - Returns full run config + summary + basic counts.

3. `get_steps(run_id: str, metric: str|None, start: int|None, end: int|None) -> dict`
   - `start`/`end` are **inclusive** bounds on `step_idx`.
   - If `metric` is provided, returns compact `{step_idx, value}` entries.

4. `list_artifacts(run_id: str, kind: str|None) -> list[dict]`

5. `export_artifact_path(artifact_id: str) -> dict | None`
   - Returns `{path, sha256}` only (no file contents).

