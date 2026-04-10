"""Tests for auto data profiler."""

import json

import polars as pl

from src.intelligence.profiler import auto_profile


class TestAutoProfile:
    def test_profile_without_llm(self):
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", None, "Diana", "Eve"],
            "score": [95.0, 80.0, 70.0, 85.0, 90.0],
        })
        profile = auto_profile(df, dataset_name="test")

        assert profile.dataset_name == "test"
        assert profile.row_count == 5
        assert profile.column_count == 3
        assert len(profile.columns) == 3

        name_col = next(c for c in profile.columns if c.name == "name")
        assert name_col.null_count == 1
        assert name_col.null_percentage == 20.0

    def test_profile_with_mock_llm(self, mock_gateway, mock_adapter):
        mock_adapter.responses["data profiling"] = json.dumps({
            "id": {"semantic_type": "identifier", "description": "Unique ID", "issues": []},
            "value": {"semantic_type": "currency", "description": "Amount in USD", "issues": ["Some negative values"]},
            "_summary": "Test dataset with IDs and values",
            "_relationships": ["id is primary key"],
            "_recommendations": ["Validate value >= 0"],
        })

        df = pl.DataFrame({"id": [1, 2, 3], "value": [100, -50, 200]})
        profile = auto_profile(df, dataset_name="test", llm=mock_gateway)

        assert profile.summary != ""

    def test_quality_score_calculation(self):
        df = pl.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad": [None, None, None, None, 5],  # 80% null
        })
        profile = auto_profile(df, dataset_name="test")

        good_col = next(c for c in profile.columns if c.name == "good")
        bad_col = next(c for c in profile.columns if c.name == "bad")
        assert good_col.quality_score > bad_col.quality_score

    def test_numeric_stats(self):
        df = pl.DataFrame({"val": [10, 20, 30, 40, 50]})
        profile = auto_profile(df)

        col = profile.columns[0]
        assert col.min_value == "10"
        assert col.max_value == "50"
        assert col.mean_value == 30.0
