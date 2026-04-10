"""Decorators for defining pipeline steps declaratively."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from src.engine.models import StepType
from src.engine.step import Step


def _make_step_decorator(step_type: StepType):
    """Factory that creates a step decorator for a given StepType."""

    def decorator(
        name: str | None = None,
        depends_on: list[str] | None = None,
        description: str = "",
        retries: int = 0,
        retry_delay: float = 1.0,
    ) -> Callable:
        def wrapper(func: Callable) -> Step:
            step_name = name or func.__name__

            @wraps(func)
            def inner(**kwargs: Any) -> Any:
                return func(**kwargs)

            step = Step(
                name=step_name,
                func=inner,
                step_type=step_type,
                depends_on=depends_on or [],
                description=description or func.__doc__ or "",
                retries=retries,
                retry_delay=retry_delay,
            )
            # Preserve original function reference for testing
            step.func.__wrapped__ = func  # type: ignore[attr-defined]
            return step

        return wrapper

    return decorator


extract = _make_step_decorator(StepType.EXTRACT)
transform = _make_step_decorator(StepType.TRANSFORM)
load = _make_step_decorator(StepType.LOAD)
ai_transform = _make_step_decorator(StepType.AI_TRANSFORM)
