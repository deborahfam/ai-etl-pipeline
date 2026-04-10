"""DuckDB loader — high-performance local analytical database."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_to_duckdb(
    df: pl.DataFrame,
    table_name: str,
    db_path: str | Path = "output/flowai.duckdb",
    mode: str = "replace",
) -> dict[str, int]:
    """Load a Polars DataFrame into a DuckDB table.

    Args:
        df: DataFrame to load.
        table_name: Target table name.
        db_path: Path to DuckDB database file.
        mode: 'replace' (drop+create), 'append', or 'create' (fail if exists).

    Returns:
        Dict with row count and table info.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))

    try:
        if mode == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        elif mode == "append":
            # Check if table exists
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]
            if table_name in table_names:
                conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            else:
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        elif mode == "create":
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'replace', 'append', or 'create'.")

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info(f"Loaded {len(df)} rows into DuckDB table '{table_name}' ({db_path})")

        return {"rows_loaded": len(df), "total_rows": row_count, "table": table_name}

    finally:
        conn.close()


def query_duckdb(
    query: str,
    db_path: str | Path = "output/flowai.duckdb",
) -> pl.DataFrame:
    """Run a SQL query against a DuckDB database and return results as Polars DataFrame."""
    conn = duckdb.connect(str(db_path))
    try:
        result = conn.execute(query).pl()
        return result
    finally:
        conn.close()
