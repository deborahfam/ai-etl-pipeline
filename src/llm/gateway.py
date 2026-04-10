"""LLM Gateway: unified interface with auto-detection, routing, fallback, caching, and cost tracking."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Type

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.adapters.base import LLMAdapter, LLMResponse
from src.llm.adapters.anthropic_adapter import AnthropicAdapter
from src.llm.adapters.openai_adapter import OpenAIAdapter
from src.llm.adapters.lmstudio_adapter import LMStudioAdapter
from src.llm.adapters.openrouter_adapter import OpenRouterAdapter
from src.llm.cache import LLMCache
from src.llm.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

# Default priority order for provider selection
PROVIDER_PRIORITY = ["anthropic", "openai", "openrouter", "lmstudio"]


class LLMGateway:
    """Unified LLM interface with multi-provider support.

    Features:
        - Auto-detects available providers from environment variables
        - Fallback chain: tries providers in priority order on failure
        - Response caching (SQLite-backed, content-hash keyed)
        - Cost and token tracking per call
        - Smart routing: vision tasks only go to vision-capable providers
    """

    def __init__(
        self,
        adapters: dict[str, LLMAdapter] | None = None,
        primary_provider: str | None = None,
        cache_enabled: bool = True,
        cache_path: str = ".llm_cache.db",
    ) -> None:
        self.adapters = adapters or {}
        self.primary_provider = primary_provider
        self.cost_tracker = CostTracker()
        self.cache = LLMCache(cache_path) if cache_enabled else None

    @classmethod
    def auto_detect(cls, cache_enabled: bool | None = None) -> "LLMGateway":
        """Create a gateway by auto-detecting available providers."""
        if cache_enabled is None:
            cache_enabled = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"

        adapter_classes: dict[str, type[LLMAdapter]] = {
            "anthropic": AnthropicAdapter,
            "openai": OpenAIAdapter,
            "lmstudio": LMStudioAdapter,
            "openrouter": OpenRouterAdapter,
        }

        adapters: dict[str, LLMAdapter] = {}
        for name, cls_type in adapter_classes.items():
            adapter = cls_type()
            if adapter.is_available():
                adapters[name] = adapter
                logger.info(f"LLM provider detected: {name}")

        if not adapters:
            logger.warning("No LLM providers detected. Set API keys in .env file.")

        primary = os.getenv("LLM_PROVIDER")
        if primary and primary not in adapters:
            primary = None

        return cls(
            adapters=adapters,
            primary_provider=primary,
            cache_enabled=cache_enabled,
        )

    # -- provider selection --------------------------------------------------

    def _select_adapter(self, require_vision: bool = False) -> LLMAdapter:
        """Select the best available adapter."""
        candidates = list(self.adapters.values())
        if require_vision:
            candidates = [a for a in candidates if a.supports_vision]

        if not candidates:
            cap = "vision-capable " if require_vision else ""
            available = list(self.adapters.keys()) or ["none"]
            raise RuntimeError(
                f"No {cap}LLM provider available. "
                f"Available: {available}. Configure API keys in .env"
            )

        # Prefer primary provider
        if self.primary_provider:
            for a in candidates:
                if a.provider_name == self.primary_provider:
                    return a

        # Follow priority order
        for name in PROVIDER_PRIORITY:
            for a in candidates:
                if a.provider_name == name:
                    return a

        return candidates[0]

    def _get_fallback_chain(self, require_vision: bool = False) -> list[LLMAdapter]:
        """Get ordered list of adapters for fallback."""
        candidates = list(self.adapters.values())
        if require_vision:
            candidates = [a for a in candidates if a.supports_vision]

        primary = self._select_adapter(require_vision)
        others = [a for a in candidates if a is not primary]
        return [primary] + others

    # -- core methods --------------------------------------------------------

    def _track(self, resp: LLMResponse, latency_ms: float) -> None:
        adapter = self.adapters.get(resp.provider)
        cost = adapter.estimate_cost(resp.input_tokens, resp.output_tokens) if adapter else 0.0
        self.cost_tracker.record(
            provider=resp.provider,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Text completion with fallback and caching."""
        adapter = self._select_adapter()

        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(adapter.provider_name, adapter.model, prompt, system, temperature)
            if cached:
                return LLMResponse(
                    content=cached[0],
                    input_tokens=cached[1],
                    output_tokens=cached[2],
                    model=adapter.model,
                    provider=adapter.provider_name + " (cached)",
                )

        # Try with fallback
        chain = self._get_fallback_chain()
        last_error = None
        for adapter in chain:
            try:
                start = time.time()
                resp = adapter.complete(prompt, system, temperature, max_tokens)
                latency = (time.time() - start) * 1000
                self._track(resp, latency)

                if use_cache and self.cache:
                    self.cache.put(
                        adapter.provider_name, adapter.model, prompt, system,
                        temperature, resp.content, resp.input_tokens, resp.output_tokens,
                    )
                return resp
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {adapter.provider_name} failed: {e}")

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[BaseModel, LLMResponse]:
        """Structured output with fallback."""
        chain = self._get_fallback_chain()
        last_error = None
        for adapter in chain:
            try:
                start = time.time()
                parsed, resp = adapter.complete_structured(
                    prompt, response_model, system, temperature, max_tokens
                )
                latency = (time.time() - start) * 1000
                self._track(resp, latency)
                return parsed, resp
            except Exception as e:
                last_error = e
                logger.warning(f"Structured output failed on {adapter.provider_name}: {e}")

        raise RuntimeError(f"All providers failed for structured output. Last error: {last_error}")

    def complete_vision(
        self,
        prompt: str,
        images: list[bytes],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Vision completion with fallback (only vision-capable providers)."""
        chain = self._get_fallback_chain(require_vision=True)
        last_error = None
        for adapter in chain:
            try:
                start = time.time()
                resp = adapter.complete_vision(prompt, images, system, temperature, max_tokens)
                latency = (time.time() - start) * 1000
                self._track(resp, latency)
                return resp
            except Exception as e:
                last_error = e
                logger.warning(f"Vision failed on {adapter.provider_name}: {e}")

        raise RuntimeError(f"All vision providers failed. Last error: {last_error}")

    def complete_vision_structured(
        self,
        prompt: str,
        images: list[bytes],
        response_model: Type[BaseModel],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[BaseModel, LLMResponse]:
        """Vision + structured output with fallback."""
        chain = self._get_fallback_chain(require_vision=True)
        last_error = None
        for adapter in chain:
            try:
                start = time.time()
                parsed, resp = adapter.complete_vision_structured(
                    prompt, images, response_model, system, temperature, max_tokens
                )
                latency = (time.time() - start) * 1000
                self._track(resp, latency)
                return parsed, resp
            except Exception as e:
                last_error = e
                logger.warning(f"Vision structured failed on {adapter.provider_name}: {e}")

        raise RuntimeError(f"All providers failed for vision structured output. Last error: {last_error}")

    # -- info ----------------------------------------------------------------

    @property
    def available_providers(self) -> list[str]:
        return list(self.adapters.keys())

    @property
    def has_vision(self) -> bool:
        return any(a.supports_vision for a in self.adapters.values())

    def __repr__(self) -> str:
        return (
            f"LLMGateway(providers={self.available_providers}, "
            f"primary={self.primary_provider})"
        )
