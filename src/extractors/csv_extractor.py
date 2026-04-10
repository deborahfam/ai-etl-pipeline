"""CSV/Excel/Parquet extractor with automatic type inference."""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_csv(
    path: str | Path,
    infer_schema_length: int = 10000,
    null_values: list[str] | None = None,
    encoding: str = "utf8",
) -> pl.DataFrame:
    """Extract data from CSV, Excel, or Parquet files.

    Auto-detects format from file extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    null_vals = null_values or ["", "NULL", "null", "N/A", "n/a", "NA", "None", "none", "-"]
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv"):
        separator = "\t" if suffix == ".tsv" else ","
        df = pl.read_csv(
            path,
            separator=separator,
            infer_schema_length=infer_schema_length,
            null_values=null_vals,
            encoding=encoding,
            try_parse_dates=True,
        )
    elif suffix in (".xlsx", ".xls"):
        df = pl.read_excel(path, infer_schema_length=infer_schema_length)
    elif suffix == ".parquet":
        df = pl.read_parquet(path)
    elif suffix == ".json":
        df = pl.read_json(path)
    elif suffix == ".ndjson":
        df = pl.read_ndjson(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info(f"Extracted {len(df)} rows x {len(df.columns)} cols from {path.name}")
    return df


def extract_csv_from_url(
    url: str,
    infer_schema_length: int = 10000,
    null_values: list[str] | None = None,
    encoding: str = "utf8",
    timeout: float = 60.0,
) -> pl.DataFrame:
    """Download a CSV from HTTPS and load it with the same rules as ``extract_csv``."""
    null_vals = null_values or ["", "NULL", "null", "N/A", "n/a", "NA", "None", "none", "-"]
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
    df = pl.read_csv(
        buf,
        infer_schema_length=infer_schema_length,
        null_values=null_vals,
        encoding=encoding,
        try_parse_dates=True,
    )
    logger.info(f"Extracted {len(df)} rows x {len(df.columns)} cols from URL")
    return df


def extract_multiple_csv(
    paths: list[str | Path],
    **kwargs,
) -> pl.DataFrame:
    """Extract and concatenate multiple files of the same schema."""
    frames = [extract_csv(p, **kwargs) for p in paths]
    return pl.concat(frames, how="diagonal_relaxed")
