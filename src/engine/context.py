"""Pipeline execution context — shared state, metrics, and lineage."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl

from src.engine.events import EventBus
from src.engine.models import PipelineRun, PipelineStatus


class PipelineContext:
    """Shared context passed through every pipeline step.

    Carries configuration, the event bus, run metrics, data snapshots
    for debugging, and column-level lineage tracking.
    """

    def __init__(
        self,
        pipeline_name: str,
        config: dict[str, Any] | None = None,
        event_bus: EventBus | None = None,
        output_dir: str | Path = "output",
    ) -> None:
        self.pipeline_name = pipeline_name
        self.config = config or {}
        self.event_bus = event_bus or EventBus()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Run tracking
        self.run = PipelineRun(pipeline_name=pipeline_name)

        # Data snapshots (step_name → DataFrame copy)
        self._snapshots: dict[str, pl.DataFrame] = {}

        # Column lineage: column_name → list of steps that touched it
        self._lineage: dict[str, list[str]] = {}

        # Arbitrary shared store between steps
        self.store: dict[str, Any] = {}

        # LLM metrics accumulator
        self._llm_calls = 0
        self._llm_cost = 0.0
        self._llm_tokens = 0

    # -- snapshots -----------------------------------------------------------

    def take_snapshot(self, step_name: str, df: pl.DataFrame) -> None:
        self._snapshots[step_name] = df.clone()

    def get_snapshot(self, step_name: str) -> pl.DataFrame | None:
        return self._snapshots.get(step_name)

    # -- lineage -------------------------------------------------------------

    def record_lineage(self, step_name: str, columns: list[str]) -> None:
        for col in columns:
            self._lineage.setdefault(col, []).append(step_name)

    @property
    def lineage(self) -> dict[str, list[str]]:
        return dict(self._lineage)

    # -- LLM metrics ---------------------------------------------------------

    def record_llm_usage(self, tokens: int, cost: float) -> None:
        self._llm_calls += 1
        self._llm_tokens += tokens
        self._llm_cost += cost
        self.run.total_llm_calls = self._llm_calls
        self.run.total_llm_cost_usd = self._llm_cost
        self.run.total_tokens_used = self._llm_tokens

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.run.status = PipelineStatus.RUNNING
        self.run.started_at = dt.datetime.now(dt.timezone.utc)
        self.event_bus.emit("on_pipeline_start", pipeline_name=self.pipeline_name)

    def complete(self) -> None:
        self.run.status = PipelineStatus.COMPLETED
        self.run.finished_at = dt.datetime.now(dt.timezone.utc)
        self.event_bus.emit(
            "on_pipeline_complete", pipeline_name=self.pipeline_name, run=self.run
        )

    def fail(self, error: Exception) -> None:
        self.run.status = PipelineStatus.FAILED
        self.run.finished_at = dt.datetime.now(dt.timezone.utc)
        self.event_bus.emit(
            "on_pipeline_error", pipeline_name=self.pipeline_name, error=error
        )
