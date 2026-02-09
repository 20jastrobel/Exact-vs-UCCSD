import sqlite3
import subprocess
import sys
from pathlib import Path


def _write_dummy_pngs(out_dir: Path, n: int = 3) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for i in range(int(n)):
        fig, ax = plt.subplots(figsize=(2.5, 2.0))
        ax.plot([0, 1], [0, i])
        ax.set_title(f"plot {i}")
        p = out_dir / f"plot_{i}.png"
        fig.savefig(p, dpi=80)
        plt.close(fig)
        paths.append(p)
    return paths


def test_plots_pdf_collation_creates_pdf(tmp_path: Path) -> None:
    _write_dummy_pngs(tmp_path, n=3)
    out_pdf = tmp_path / "out.pdf"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/plots.py",
            "pdf",
            "--input",
            str(tmp_path),
            "--output",
            str(out_pdf),
            "--no-register",
        ]
    )
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 1000


def test_plots_pdf_run_registers_artifact(tmp_path: Path) -> None:
    from pydephasing.quantum.vqe.run_store import SqliteRunStore, sha256_file

    pngs = _write_dummy_pngs(tmp_path / "imgs", n=2)
    db_path = tmp_path / "runs.db"

    run_id = "test_plots_pdf_run"
    store = SqliteRunStore(db_path)
    store.start_run(
        {
            "run_id": run_id,
            "seed": 0,
            "system": {"sites": 2, "n_up": 1, "n_down": 1},
            "ansatz": {"kind": "test"},
            "optimizer": {"name": "none"},
        }
    )
    for p in pngs:
        store.add_artifact(
            run_id,
            kind="plot_png",
            path=str(p),
            sha256=sha256_file(p),
            bytes=int(p.stat().st_size),
            extra=None,
        )
    store.finish_run(run_id, status="completed", summary_metrics={})
    store.close()

    out_pdf = tmp_path / "run.pdf"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/plots.py",
            "pdf-run",
            "--run-id",
            run_id,
            "--db",
            str(db_path),
            "--output",
            str(out_pdf),
        ]
    )
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 1000

    conn = sqlite3.connect(str(db_path))
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND kind = 'plots_pdf'",
            (run_id,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert int(n) == 1

