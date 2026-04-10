"""Tests for data validation."""

import polars as pl

from src.transformers.validator import validate_dataframe


class TestValidateDataframe:
    def test_not_null_check(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result_df, result = validate_dataframe(df, not_null=["a"])
        assert not result.is_valid
        assert any("null" in e.lower() for e in result.errors)

    def test_unique_check(self):
        df = pl.DataFrame({"id": [1, 2, 2, 3]})
        _, result = validate_dataframe(df, unique=["id"])
        assert not result.is_valid
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_range_check(self):
        df = pl.DataFrame({"age": [25, 150, 30, -5]})
        _, result = validate_dataframe(df, value_ranges={"age": (0, 120)})
        assert not result.is_valid
        assert result.rows_failed > 0

    def test_custom_check(self):
        df = pl.DataFrame({"qty": [1, 2, 3], "price": [10, 20, 30], "total": [10, 50, 90]})

        def check_totals(df):
            bad = df.filter((pl.col("total") - pl.col("qty") * pl.col("price")).abs() > 0.01)
            return f"{len(bad)} bad totals" if len(bad) > 0 else None

        _, result = validate_dataframe(df, custom_checks=[check_totals])
        assert not result.is_valid

    def test_valid_data_passes(self):
        df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        _, result = validate_dataframe(df, not_null=["id"], unique=["id"], value_ranges={"value": (0, 100)})
        assert result.is_valid

    def test_validation_flag_column(self):
        df = pl.DataFrame({"a": [1, None, 3]})
        result_df, _ = validate_dataframe(df, not_null=["a"])
        assert "_validation_failed" in result_df.columns
