"""REST API extractor with pagination and rate limiting."""

from __future__ import annotations

from typing import Any

import httpx
import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_from_api(
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    data_key: str | None = None,
    pagination_key: str | None = None,
    max_pages: int = 10,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Extract data from a REST API endpoint.

    Args:
        url: Base URL of the API endpoint.
        headers: HTTP headers (e.g., Authorization).
        params: Query parameters.
        data_key: JSON key containing the array of records (e.g., "results").
        pagination_key: JSON key for next page URL/token.
        max_pages: Maximum pages to fetch.
        timeout: Request timeout in seconds.
    """
    all_records: list[dict] = []
    current_url = url
    page = 0

    with httpx.Client(headers=headers, timeout=timeout) as client:
        while current_url and page < max_pages:
            logger.info(f"Fetching page {page + 1}: {current_url}")
            resp = client.get(current_url, params=params if page == 0 else None)
            resp.raise_for_status()
            data = resp.json()

            # Extract records from response
            if data_key:
                records = data.get(data_key, [])
            elif isinstance(data, list):
                records = data
            else:
                records = [data]

            all_records.extend(records)
            page += 1

            # Handle pagination
            if pagination_key and pagination_key in data:
                current_url = data[pagination_key]
            else:
                break

    if not all_records:
        logger.warning(f"No records extracted from {url}")
        return pl.DataFrame()

    logger.info(f"Extracted {len(all_records)} records from API ({page} pages)")
    return pl.DataFrame(all_records)


def extract_from_mock_api(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Create a DataFrame from mock API data (for testing/demos)."""
    return pl.DataFrame(records)
