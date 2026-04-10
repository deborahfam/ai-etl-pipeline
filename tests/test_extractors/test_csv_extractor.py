"""CSV extractor tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.extractors.csv_extractor import extract_csv_from_url


def test_extract_csv_from_url(monkeypatch):
    csv_bytes = b"id,name\n1,Ada\n2,Bob\n"

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url: str):
            r = MagicMock()
            r.content = csv_bytes
            r.raise_for_status = MagicMock()
            return r

    monkeypatch.setattr("src.extractors.csv_extractor.httpx.Client", lambda **kw: FakeClient())

    df = extract_csv_from_url("https://example.com/data.csv")
    assert len(df) == 2
    assert df.columns == ["id", "name"]
