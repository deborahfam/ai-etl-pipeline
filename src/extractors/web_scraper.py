"""Basic web scraper extractor — fetches HTML and extracts tables."""

from __future__ import annotations

import httpx
import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_html_tables(
    url: str,
    table_index: int = 0,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Extract tables from an HTML page.

    Uses a simple regex-based approach to parse HTML tables
    without requiring heavy dependencies like BeautifulSoup.
    """
    import re

    resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    html = resp.text

    # Find all tables
    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL | re.IGNORECASE)
    if not tables:
        logger.warning(f"No tables found at {url}")
        return pl.DataFrame()

    if table_index >= len(tables):
        raise IndexError(f"Table index {table_index} out of range (found {len(tables)} tables)")

    table_html = tables[table_index]

    # Extract headers
    header_match = re.findall(r"<th[^>]*>(.*?)</th>", table_html, re.DOTALL | re.IGNORECASE)
    col_names = [re.sub(r"<[^>]+>", "", h).strip() for h in header_match]

    # Extract rows
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)
    data: list[list[str]] = []
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL | re.IGNORECASE)
        if cells:
            data.append([re.sub(r"<[^>]+>", "", c).strip() for c in cells])

    if not data:
        return pl.DataFrame()

    # If no headers found, generate column names
    if not col_names:
        col_names = [f"col_{i}" for i in range(len(data[0]))]

    # Ensure all rows have the same number of columns
    n_cols = len(col_names)
    data = [row[:n_cols] + [""] * max(0, n_cols - len(row)) for row in data]

    logger.info(f"Extracted table with {len(data)} rows x {n_cols} cols from {url}")
    return pl.DataFrame(data, schema=col_names, orient="row")
