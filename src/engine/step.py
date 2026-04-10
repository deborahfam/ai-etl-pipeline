"""Step definition: the building block of a pipeline."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable

from src.engine.models import StepResult, StepStatus, StepType


@dataclass
class Step:
    """Wraps a user function as an executable pipeline step."""

    name: str
    func: Callable[..., Any]
    step_type: StepType
    depends_on: list[str] = field(default_factory=list)
    description: str = ""
    retries: int = 0
    retry_delay: float = 1.0

    def execute(self, **kwargs: Any) -> tuple[Any, StepResult]:
        """Run the wrapped function, returning (output, StepResult)."""
        result = StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.RUNNING,
            started_at=dt.datetime.now(dt.timezone.utc),
        )

        attempts = 0
        last_error: Exception | None = None

        while attempts <= self.retries:
            try:
                output = self.func(**kwargs)
                result.status = StepStatus.COMPLETED
                result.finished_at = dt.datetime.now(dt.timezone.utc)
                result.duration_seconds = (
                    result.finished_at - result.started_at
                ).total_seconds()
                return output, result
            except Exception as exc:
                last_error = exc
                attempts += 1
                if attempts <= self.retries:
                    import time
                    time.sleep(self.retry_delay * attempts)

        result.status = StepStatus.FAILED
        result.finished_at = dt.datetime.now(dt.timezone.utc)
        result.duration_seconds = (
            result.finished_at - result.started_at
        ).total_seconds()
        result.error = str(last_error)
        raise last_error  # type: ignore[misc]
