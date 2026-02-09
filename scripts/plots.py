#!/usr/bin/env python3
"""Utilities for collating plots into a single PDF.

Primary use-case: take an experiment output directory (or a DB run_id) and produce
one multi-page PDF so you can open everything at once.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Allow running as `python scripts/...py` without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydephasing.quantum.vqe.run_store import SqliteRunStore, sha256_file


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_meta_run_id(input_dir: Path) -> str | None:
    meta_path = input_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_id = meta.get("run_id")
    return str(run_id) if run_id else None


def _discover_files(
    *,
    input_path: Path,
    recursive: bool,
    globs: list[str],
) -> list[Path]:
    base = Path(input_path)
    if base.is_file():
        return [base]

    found: set[Path] = set()
    for pat in globs:
        if recursive:
            for p in base.rglob(pat):
                if p.is_file():
                    found.add(p)
        else:
            for p in base.glob(pat):
                if p.is_file():
                    found.add(p)
    return sorted(found, key=lambda p: str(p.relative_to(base) if p.is_relative_to(base) else p))


def _default_globs() -> list[str]:
    return ["*.png", "*.jpg", "*.jpeg", "*.pdf"]


def _title_for(path: Path, *, base: Path, mode: str) -> str | None:
    mode = str(mode)
    if mode == "none":
        return None
    if mode == "filename":
        return path.name
    if mode == "relativepath":
        try:
            return str(path.relative_to(base))
        except Exception:
            try:
                return str(path.resolve().relative_to(Path.cwd().resolve()))
            except Exception:
                return str(path)
    raise ValueError(f"Unknown title mode: {mode}")


def collate_plots_to_pdf(
    *,
    paths: list[Path],
    output_pdf: Path,
    base_for_titles: Path,
    title_mode: str,
    nrows: int,
    ncols: int,
) -> dict[str, int]:
    """Write a multi-page PDF containing the given plot/image files."""
    if not paths:
        raise ValueError("No input files to collate.")

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if output_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"--output must be a .pdf file; got: {output_pdf}")

    # Prefer minimal dependencies: matplotlib + PdfPages.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    nrows = int(nrows)
    ncols = int(ncols)
    if nrows < 1 or ncols < 1:
        raise ValueError("--nrows/--ncols must be >= 1")
    per_page = nrows * ncols

    counts = {"images_written": 0, "pdf_skipped": 0, "decode_skipped": 0}

    with PdfPages(str(output_pdf)) as pdf:
        page_paths: list[Path] = []
        for p in paths:
            page_paths.append(Path(p))
            if len(page_paths) < per_page:
                continue

            _write_page(
                pdf=pdf,
                page_paths=page_paths,
                base_for_titles=base_for_titles,
                title_mode=title_mode,
                nrows=nrows,
                ncols=ncols,
                counts=counts,
                plt=plt,
            )
            page_paths = []

        if page_paths:
            _write_page(
                pdf=pdf,
                page_paths=page_paths,
                base_for_titles=base_for_titles,
                title_mode=title_mode,
                nrows=nrows,
                ncols=ncols,
                counts=counts,
                plt=plt,
            )

    return counts


def _write_page(
    *,
    pdf,
    page_paths: list[Path],
    base_for_titles: Path,
    title_mode: str,
    nrows: int,
    ncols: int,
    counts: dict[str, int],
    plt,
) -> None:
    # Page size tuned for readability; scale with grid.
    fig_w = max(6.0, float(ncols) * 6.0)
    fig_h = max(4.0, float(nrows) * 4.0)
    fig, axes = plt.subplots(int(nrows), int(ncols), figsize=(fig_w, fig_h))

    # Normalize axes to 2D list for consistent indexing.
    if nrows == 1 and ncols == 1:
        axes_grid = [[axes]]
    elif nrows == 1:
        axes_grid = [list(axes)]
    elif ncols == 1:
        axes_grid = [[ax] for ax in list(axes)]
    else:
        axes_grid = [list(row) for row in axes]

    idx = 0
    for r in range(int(nrows)):
        for c in range(int(ncols)):
            ax = axes_grid[r][c]
            if idx >= len(page_paths):
                ax.axis("off")
                continue
            p = page_paths[idx]
            idx += 1

            if p.suffix.lower() == ".pdf":
                counts["pdf_skipped"] += 1
                ax.axis("off")
                continue

            try:
                img = plt.imread(str(p))
            except Exception:
                counts["decode_skipped"] += 1
                ax.axis("off")
                continue

            ax.imshow(img)
            ax.axis("off")
            title = _title_for(p, base=base_for_titles, mode=title_mode)
            if title:
                ax.set_title(title, fontsize=8)
            counts["images_written"] += 1

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _query_run_artifacts(db_path: Path, *, run_id: str) -> list[Path]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT path FROM artifacts WHERE run_id = ? ORDER BY path ASC",
            (str(run_id),),
        ).fetchall()
    finally:
        conn.close()

    paths: list[Path] = []
    seen: set[str] = set()
    for (p,) in rows:
        if not p:
            continue
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        paths.append(Path(s))
    return paths


def _filter_plot_inputs(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    raster: list[Path] = []
    pdfs: list[Path] = []
    for p in paths:
        ext = p.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            raster.append(p)
        elif ext == ".pdf":
            pdfs.append(p)
    return raster, pdfs


def _maybe_register_pdf_artifact(*, db_path: Path, run_id: str, output_pdf: Path) -> None:
    sha = sha256_file(output_pdf)
    b = int(output_pdf.stat().st_size)
    store = None
    try:
        store = SqliteRunStore(db_path)
        store.add_artifact(
            str(run_id),
            kind="plots_pdf",
            path=str(output_pdf),
            sha256=str(sha),
            bytes=b,
            extra=None,
        )
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pdf = sub.add_parser("pdf", help="Collate plots under a directory into a multi-page PDF.")
    p_pdf.add_argument("--input", type=str, default="runs", help="Directory or single file to collate.")
    p_pdf.add_argument("--output", type=str, required=True, help="Output PDF path.")
    p_pdf.add_argument("--recursive", action="store_true", default=False, help="Recurse into subdirectories.")
    p_pdf.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Glob pattern for discovery (repeatable). Default: *.png/*.jpg/*.jpeg/*.pdf",
    )
    p_pdf.add_argument("--nrows", type=int, default=1, help="Grid rows per page.")
    p_pdf.add_argument("--ncols", type=int, default=1, help="Grid cols per page.")
    p_pdf.add_argument(
        "--title-mode",
        type=str,
        choices=["filename", "relativepath", "none"],
        default="relativepath",
        help="Title text for each plot.",
    )
    p_pdf.add_argument("--run-id", type=str, default=None, help="Optional run_id for DB artifact registration.")
    p_pdf.add_argument(
        "--db",
        type=str,
        default=os.environ.get("RUNS_DB_PATH", "data/runs.db"),
        help="SQLite DB path (env: RUNS_DB_PATH).",
    )
    p_pdf.add_argument("--no-register", action="store_true", default=False, help="Do not register the PDF in SQLite.")

    p_run = sub.add_parser("pdf-run", help="Collate plots for a DB run_id (from artifacts) into a PDF.")
    p_run.add_argument("--run-id", type=str, required=True)
    p_run.add_argument("--db", type=str, default=os.environ.get("RUNS_DB_PATH", "data/runs.db"))
    p_run.add_argument("--output", type=str, required=True)
    p_run.add_argument("--nrows", type=int, default=1)
    p_run.add_argument("--ncols", type=int, default=1)
    p_run.add_argument(
        "--title-mode",
        type=str,
        choices=["filename", "relativepath", "none"],
        default="relativepath",
    )
    p_run.add_argument("--no-register", action="store_true", default=False)

    args = ap.parse_args()

    if args.cmd == "pdf":
        input_path = Path(args.input)
        globs = list(args.glob) if args.glob else _default_globs()
        discovered = _discover_files(input_path=input_path, recursive=bool(args.recursive), globs=globs)

        raster, pdfs = _filter_plot_inputs(discovered)
        if pdfs:
            _eprint(
                f"[plots] Skipping {len(pdfs)} PDF input(s). "
                "If you need merging, install pypdf and extend this tool."
            )
        if not raster:
            raise SystemExit(f"No raster plots found under: {input_path}")

        out_pdf = Path(args.output)
        counts = collate_plots_to_pdf(
            paths=raster,
            output_pdf=out_pdf,
            base_for_titles=input_path.resolve() if input_path.exists() else Path.cwd(),
            title_mode=str(args.title_mode),
            nrows=int(args.nrows),
            ncols=int(args.ncols),
        )
        print(f"Wrote {out_pdf} ({counts['images_written']} images).")

        if args.no_register:
            return

        run_id = str(args.run_id) if args.run_id else _load_meta_run_id(input_path) if input_path.is_dir() else None
        if not run_id:
            _eprint("[plots] No run_id found; not registering PDF artifact.")
            return

        db_path = Path(args.db)
        if not db_path.exists():
            _eprint(f"[plots] DB not found at {db_path}; not registering artifact.")
            return
        # Register only if the run exists (otherwise FK constraints would fail).
        exists = None
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                exists = conn.execute(
                    "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1", (str(run_id),)
                ).fetchone()
            finally:
                conn.close()
        except Exception as exc:
            _eprint(f"[plots] Could not query DB {db_path} ({exc}); not registering artifact.")
            return
        if not exists:
            _eprint(f"[plots] run_id={run_id} not found in DB {db_path}; not registering artifact.")
            return

        _maybe_register_pdf_artifact(db_path=db_path, run_id=str(run_id), output_pdf=out_pdf)
        print(f"Registered plots PDF in DB for run_id={run_id}.")
        return

    if args.cmd == "pdf-run":
        run_id = str(args.run_id)
        db_path = Path(args.db)
        out_pdf = Path(args.output)

        art_paths = _query_run_artifacts(db_path, run_id=run_id)
        raster, pdfs = _filter_plot_inputs(art_paths)
        if pdfs:
            _eprint(
                f"[plots] Skipping {len(pdfs)} PDF artifact(s). "
                "If you need merging, install pypdf and extend this tool."
            )

        raster_existing = [p for p in raster if p.exists()]
        missing = [p for p in raster if not p.exists()]
        if missing:
            _eprint(f"[plots] Warning: {len(missing)} raster artifact path(s) missing on disk; skipping.")
        if not raster_existing:
            raise SystemExit(f"No raster artifacts found for run_id={run_id}")

        # Deterministic order: sort by path string.
        raster_existing.sort(key=lambda p: str(p))

        counts = collate_plots_to_pdf(
            paths=raster_existing,
            output_pdf=out_pdf,
            base_for_titles=Path.cwd(),
            title_mode=str(args.title_mode),
            nrows=int(args.nrows),
            ncols=int(args.ncols),
        )
        print(f"Wrote {out_pdf} ({counts['images_written']} images).")

        if not args.no_register:
            _maybe_register_pdf_artifact(db_path=db_path, run_id=run_id, output_pdf=out_pdf)
            print(f"Registered plots PDF in DB for run_id={run_id}.")
        return

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
