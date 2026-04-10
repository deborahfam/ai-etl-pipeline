"""Abstract base class for LLM provider adapters."""

from __future__ import annotations

import abc
from typing import Any, Type

from pydantic import BaseModel


class LLMResponse:
    """Normalized response from any LLM provider."""

    def __init__(
        self,
        content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        provider: str = "",
    ) -> None:
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.model = model
        self.provider = provider


class LLMAdapter(abc.ABC):
    """Interface that every LLM provider must implement."""

    provider_name: str = "base"
    supports_vision: bool = False
    supports_structured_output: bool = False

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Simple text completion."""

    @abc.abstractmethod
    def complete_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[BaseModel, LLMResponse]:
        """Completion with structured (Pydantic) output."""

    @abc.abstractmethod
    def complete_vision(
        self,
        prompt: str,
        images: list[bytes],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Completion with image inputs."""

    def complete_vision_structured(
        self,
        prompt: str,
        images: list[bytes],
        response_model: Type[BaseModel],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[BaseModel, LLMResponse]:
        """Vision + structured output. Default: parse JSON from vision response."""
        import json
        resp = self.complete_vision(
            prompt=prompt + "\n\nRespond ONLY with valid JSON matching the schema.",
            images=images,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = response_model.model_validate(json.loads(text))
        return parsed, resp

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD. Override per provider."""
        return 0.0
