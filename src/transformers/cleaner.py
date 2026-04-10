"""Data cleaning transformer: deduplication, null handling, normalization, type coercion."""

from __future__ import annotations

import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


def clean_dataframe(
    df: pl.DataFrame,
    drop_duplicates: bool = True,
    subset: list[str] | None = None,
    fill_nulls: dict[str, object] | None = None,
    drop_null_threshold: float | None = None,
    normalize_strings: bool = True,
    coerce_dates: list[str] | None = None,
    coerce_numerics: list[str] | None = None,
    remove_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Apply a comprehensive set of cleaning operations.

    Args:
        df: Input DataFrame.
        drop_duplicates: Remove exact duplicate rows.
        subset: Columns to consider for duplicate detection.
        fill_nulls: Dict of column→value for null filling.
        drop_null_threshold: Drop columns with null% above this (0.0-1.0).
        normalize_strings: Strip whitespace, normalize unicode in string cols.
        coerce_dates: Columns to parse as dates.
        coerce_numerics: Columns to parse as floats.
        remove_columns: Columns to drop.
    """
    original_rows = len(df)
    original_cols = len(df.columns)

    # 1. Remove specified columns
    if remove_columns:
        existing = [c for c in remove_columns if c in df.columns]
        if existing:
            df = df.drop(existing)
            logger.info(f"Dropped columns: {existing}")

    # 2. Drop columns with too many nulls
    if drop_null_threshold is not None:
        null_fractions = {
            col: df[col].null_count() / len(df) for col in df.columns
        }
        cols_to_drop = [c for c, frac in null_fractions.items() if frac > drop_null_threshold]
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.info(f"Dropped high-null columns ({drop_null_threshold:.0%} threshold): {cols_to_drop}")

    # 3. Remove duplicates
    if drop_duplicates:
        before = len(df)
        df = df.unique(subset=subset, maintain_order=True)
        removed = before - len(df)
        if removed:
            logger.info(f"Removed {removed} duplicate rows")

    # 4. Normalize strings
    if normalize_strings:
        string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
        for col in string_cols:
            df = df.with_columns(
                pl.col(col).str.strip_chars().alias(col)
            )

    # 5. Fill nulls
    if fill_nulls:
        for col, value in fill_nulls.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(value).alias(col))

    # 6. Coerce date columns
    if coerce_dates:
        for col in coerce_dates:
            if col in df.columns and df[col].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col(col).str.to_date(strict=False).alias(col)
                )

    # 7. Coerce numeric columns
    if coerce_numerics:
        for col in coerce_numerics:
            if col in df.columns and df[col].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                )

    logger.info(
        f"Cleaning complete: {original_rows}→{len(df)} rows, "
        f"{original_cols}→{len(df.columns)} cols"
    )
    return df


def normalize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column names to snake_case."""
    import re

    mapping = {}
    for col in df.columns:
        normalized = col.strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        normalized = normalized.strip("_")
        mapping[col] = normalized

    return df.rename(mapping)


def remove_outliers_iqr(
    df: pl.DataFrame,
    columns: list[str],
    multiplier: float = 1.5,
) -> pl.DataFrame:
    """Remove rows with outliers based on IQR method."""
    mask = pl.lit(True)
    for col in columns:
        if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                mask = mask & (pl.col(col) >= lower) & (pl.col(col) <= upper)

    before = len(df)
    df = df.filter(mask)
    logger.info(f"IQR outlier removal: {before}→{len(df)} rows ({before - len(df)} removed)")
    return df
