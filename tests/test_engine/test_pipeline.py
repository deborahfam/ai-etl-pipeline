"""Tests for the pipeline engine: DAG resolution, execution, context."""

from __future__ import annotations

import polars as pl
import pytest

from src.engine.pipeline import Pipeline
from src.engine.context import PipelineContext
from src.engine.decorators import extract, transform, load
from src.engine.models import PipelineStatus, StepStatus


class TestPipelineDAG:
    def test_simple_pipeline(self):
        @extract(name="step_a")
        def step_a(ctx: PipelineContext) -> pl.DataFrame:
            return pl.DataFrame({"x": [1, 2, 3]})

        @transform(name="step_b", depends_on=["step_a"])
        def step_b(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns((pl.col("x") * 2).alias("y"))

        pipeline = Pipeline("test_simple")
        pipeline.add_steps([step_a, step_b])
        ctx = pipeline.run(show_progress=False)

        assert ctx.run.status == PipelineStatus.COMPLETED
        assert len(ctx.run.steps) == 2
        assert all(s.status == StepStatus.COMPLETED for s in ctx.run.steps)

    def test_dag_ordering(self):
        """Steps added out of order should execute in dependency order."""
        execution_order = []

        @extract(name="first")
        def first(ctx: PipelineContext) -> pl.DataFrame:
            execution_order.append("first")
            return pl.DataFrame({"a": [1]})

        @transform(name="second", depends_on=["first"])
        def second(df: pl.DataFrame) -> pl.DataFrame:
            execution_order.append("second")
            return df

        @transform(name="third", depends_on=["second"])
        def third(df: pl.DataFrame) -> pl.DataFrame:
            execution_order.append("third")
            return df

        pipeline = Pipeline("test_ordering")
        # Add in reverse order
        pipeline.add_steps([third, first, second])
        pipeline.run(show_progress=False)

        assert execution_order == ["first", "second", "third"]

    def test_missing_dependency_raises(self):
        @transform(name="orphan", depends_on=["nonexistent"])
        def orphan(df: pl.DataFrame) -> pl.DataFrame:
            return df

        pipeline = Pipeline("test_missing")
        pipeline.add_step(orphan)

        with pytest.raises(ValueError, match="depends on 'nonexistent'"):
            pipeline.run(show_progress=False)

    def test_step_failure_propagates(self):
        @extract(name="fail_step")
        def fail_step(ctx: PipelineContext) -> pl.DataFrame:
            raise ValueError("intentional failure")

        pipeline = Pipeline("test_failure")
        pipeline.add_step(fail_step)

        with pytest.raises(ValueError, match="intentional failure"):
            pipeline.run(show_progress=False)

    def test_context_snapshots(self):
        @extract(name="source")
        def source(ctx: PipelineContext) -> pl.DataFrame:
            return pl.DataFrame({"val": [10, 20, 30]})

        @transform(name="double", depends_on=["source"])
        def double(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns((pl.col("val") * 2).alias("val"))

        pipeline = Pipeline("test_snapshots")
        pipeline.add_steps([source, double])
        ctx = pipeline.run(show_progress=False)

        snap = ctx.get_snapshot("source")
        assert snap is not None
        assert snap["val"].to_list() == [10, 20, 30]

    def test_lineage_tracking(self):
        @extract(name="base")
        def base(ctx: PipelineContext) -> pl.DataFrame:
            return pl.DataFrame({"a": [1], "b": [2]})

        @transform(name="add_col", depends_on=["base"])
        def add_col(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit(3).alias("c"))

        pipeline = Pipeline("test_lineage")
        pipeline.add_steps([base, add_col])
        ctx = pipeline.run(show_progress=False)

        assert "c" in ctx.lineage
        assert "add_col" in ctx.lineage["c"]


class TestPipelineContext:
    def test_llm_metrics(self):
        ctx = PipelineContext("test")
        ctx.record_llm_usage(tokens=1000, cost=0.01)
        ctx.record_llm_usage(tokens=500, cost=0.005)

        assert ctx.run.total_llm_calls == 2
        assert ctx.run.total_tokens_used == 1500
        assert ctx.run.total_llm_cost_usd == pytest.approx(0.015)

    def test_store(self):
        ctx = PipelineContext("test")
        ctx.store["key"] = "value"
        assert ctx.store["key"] == "value"
