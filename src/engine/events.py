"""Event bus for pipeline lifecycle hooks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

EventHandler = Callable[..., None]


class EventBus:
    """Simple synchronous event bus for pipeline lifecycle events.

    Supported events:
        - on_pipeline_start(pipeline_name)
        - on_pipeline_complete(pipeline_name, run)
        - on_pipeline_error(pipeline_name, error)
        - on_step_start(step_name, step_type)
        - on_step_complete(step_name, result)
        - on_step_error(step_name, error)
        - on_anomaly_detected(anomaly)
        - on_llm_call(provider, model, tokens, cost)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        self._handlers[event].append(handler)

    def off(self, event: str, handler: EventHandler) -> None:
        self._handlers[event] = [h for h in self._handlers[event] if h is not handler]

    def emit(self, event: str, **kwargs: Any) -> None:
        for handler in self._handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception:
                pass  # Handlers should not break the pipeline

    def clear(self) -> None:
        self._handlers.clear()
