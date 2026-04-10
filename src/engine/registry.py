"""Global registry for pipeline and step auto-discovery."""

from __future__ import annotations

from src.engine.pipeline import Pipeline
from src.engine.step import Step


class Registry:
    """Singleton registry that catalogues pipelines and steps."""

    _instance: Registry | None = None

    def __new__(cls) -> Registry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipelines = {}
            cls._instance._steps = {}
        return cls._instance

    # -- pipelines -----------------------------------------------------------

    def register_pipeline(self, pipeline: Pipeline) -> None:
        self._pipelines[pipeline.name] = pipeline

    def get_pipeline(self, name: str) -> Pipeline | None:
        return self._pipelines.get(name)

    def list_pipelines(self) -> list[str]:
        return list(self._pipelines.keys())

    # -- steps ---------------------------------------------------------------

    def register_step(self, step: Step) -> None:
        self._steps[step.name] = step

    def get_step(self, name: str) -> Step | None:
        return self._steps.get(name)

    def list_steps(self) -> list[str]:
        return list(self._steps.keys())

    # -- reset (for testing) -------------------------------------------------

    def clear(self) -> None:
        self._pipelines.clear()
        self._steps.clear()


registry = Registry()
