"""SQLite loader — portable relational database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_to_sqlite(
    df: pl.DataFrame,
    table_name: str,
    db_path: str | Path = "output/flowai.sqlite",
    mode: str = "replace",
) -> dict[str, int]:
    """Load a Polars DataFrame into a SQLite table.

    Args:
        df: DataFrame to load.
        table_name: Target table name.
        db_path: Path to SQLite database file.
        mode: 'replace' (drop+create) or 'append'.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to pandas for SQLite compatibility
    pdf = df.to_pandas()
    conn = sqlite3.connect(str(db_path))

    try:
        if_exists = "replace" if mode == "replace" else "append"
        pdf.to_sql(table_name, conn, if_exists=if_exists, index=False)

        cursor = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        total = cursor.fetchone()[0]

        logger.info(f"Loaded {len(df)} rows into SQLite table '{table_name}' ({db_path})")
        return {"rows_loaded": len(df), "total_rows": total, "table": table_name}

    finally:
        conn.close()
