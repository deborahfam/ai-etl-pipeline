"""Tests for pipeline step decorators."""

from __future__ import annotations

import polars as pl

from src.engine.decorators import extract, transform, load, ai_transform
from src.engine.models import StepType
from src.engine.step import Step


class TestDecorators:
    def test_extract_decorator(self):
        @extract(name="my_extract", description="Test extractor")
        def my_func(ctx=None) -> pl.DataFrame:
            return pl.DataFrame({"a": [1]})

        assert isinstance(my_func, Step)
        assert my_func.name == "my_extract"
        assert my_func.step_type == StepType.EXTRACT
        assert my_func.description == "Test extractor"

    def test_transform_decorator(self):
        @transform(name="my_transform", depends_on=["step_a"])
        def my_func(df: pl.DataFrame) -> pl.DataFrame:
            return df

        assert my_func.step_type == StepType.TRANSFORM
        assert my_func.depends_on == ["step_a"]

    def test_load_decorator(self):
        @load(name="my_loader")
        def my_func(df: pl.DataFrame) -> pl.DataFrame:
            return df

        assert my_func.step_type == StepType.LOAD

    def test_ai_transform_decorator(self):
        @ai_transform(name="my_ai", retries=3)
        def my_func(df: pl.DataFrame, llm=None) -> pl.DataFrame:
            return df

        assert my_func.step_type == StepType.AI_TRANSFORM
        assert my_func.retries == 3

    def test_default_name_from_function(self):
        @extract()
        def load_data(ctx=None) -> pl.DataFrame:
            return pl.DataFrame()

        assert load_data.name == "load_data"

    def test_step_execution(self):
        @extract(name="exec_test")
        def my_func(ctx=None) -> pl.DataFrame:
            return pl.DataFrame({"x": [42]})

        output, result = my_func.execute()
        assert isinstance(output, pl.DataFrame)
        assert output["x"][0] == 42
        assert result.step_name == "exec_test"
