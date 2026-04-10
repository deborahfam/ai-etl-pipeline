"""Tests for data cleaning transformer."""

import polars as pl
import pytest

from src.transformers.cleaner import (
    clean_dataframe,
    normalize_column_names,
    remove_outliers_iqr,
)


class TestCleanDataframe:
    def test_drop_duplicates(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = clean_dataframe(df, drop_duplicates=True)
        assert len(result) == 2

    def test_drop_duplicates_subset(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        result = clean_dataframe(df, drop_duplicates=True, subset=["a"])
        assert len(result) == 2

    def test_fill_nulls(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
        result = clean_dataframe(df, fill_nulls={"a": 0, "b": "missing"}, drop_duplicates=False)
        assert result["a"].null_count() == 0
        assert result["b"].null_count() == 0

    def test_normalize_strings(self):
        df = pl.DataFrame({"a": ["  hello  ", "world  ", "  test"]})
        result = clean_dataframe(df, normalize_strings=True, drop_duplicates=False)
        assert result["a"].to_list() == ["hello", "world", "test"]

    def test_drop_null_threshold(self):
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [None, None, None, None, 5],  # 80% null
            "c": [1, 2, None, 4, 5],  # 20% null
        })
        result = clean_dataframe(df, drop_null_threshold=0.5, drop_duplicates=False)
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_remove_columns(self):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = clean_dataframe(df, remove_columns=["b", "c"], drop_duplicates=False)
        assert result.columns == ["a"]

    def test_coerce_numerics(self):
        df = pl.DataFrame({"price": ["10.5", "20.0", "invalid", "30.0"]})
        result = clean_dataframe(df, coerce_numerics=["price"], drop_duplicates=False)
        assert result["price"].dtype == pl.Float64


class TestNormalizeColumnNames:
    def test_basic_normalization(self):
        df = pl.DataFrame({"First Name": [1], "Last-Name": [2], "email_address": [3]})
        result = normalize_column_names(df)
        assert result.columns == ["first_name", "last_name", "email_address"]

    def test_special_characters(self):
        df = pl.DataFrame({"Price ($)": [1], "Qty.": [2]})
        result = normalize_column_names(df)
        assert all(c.isalnum() or c == "_" for col in result.columns for c in col)


class TestRemoveOutliersIQR:
    def test_removes_outliers(self):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        result = remove_outliers_iqr(df, columns=["val"])
        assert 100 not in result["val"].to_list()

    def test_preserves_normal_data(self):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = remove_outliers_iqr(df, columns=["val"])
        assert len(result) == 5
