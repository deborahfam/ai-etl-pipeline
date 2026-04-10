"""File loader — export to CSV, Parquet, JSON, or NDJSON."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_to_file(
    df: pl.DataFrame,
    path: str | Path,
    format: str | None = None,
) -> dict[str, str]:
    """Export a Polars DataFrame to a file.

    Format auto-detected from file extension, or specify explicitly.
    Supported: csv, parquet, json, ndjson.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = format or path.suffix.lstrip(".").lower()

    if fmt == "csv":
        df.write_csv(path)
    elif fmt == "parquet":
        df.write_parquet(path)
    elif fmt == "json":
        df.write_json(path)
    elif fmt == "ndjson":
        df.write_ndjson(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use csv, parquet, json, or ndjson.")

    size_kb = path.stat().st_size / 1024
    logger.info(f"Exported {len(df)} rows to {path.name} ({size_kb:.1f} KB)")

    return {"path": str(path), "format": fmt, "rows": len(df), "size_kb": f"{size_kb:.1f}"}
