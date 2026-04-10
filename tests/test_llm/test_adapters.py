"""Tests for LLM adapter interface compliance."""

import os

import pytest

from src.llm.adapters.base import LLMAdapter, LLMResponse
from src.llm.adapters.anthropic_adapter import AnthropicAdapter
from src.llm.adapters.openai_adapter import OpenAIAdapter
from src.llm.adapters.lmstudio_adapter import LMStudioAdapter
from src.llm.adapters.openrouter_adapter import OpenRouterAdapter


class TestAdapterInterface:
    """Verify all adapters implement the required interface."""

    adapters = [AnthropicAdapter, OpenAIAdapter, LMStudioAdapter, OpenRouterAdapter]

    @pytest.mark.parametrize("adapter_cls", adapters)
    def test_has_provider_name(self, adapter_cls):
        adapter = adapter_cls()
        assert hasattr(adapter, "provider_name")
        assert isinstance(adapter.provider_name, str)

    @pytest.mark.parametrize("adapter_cls", adapters)
    def test_has_vision_flag(self, adapter_cls):
        adapter = adapter_cls()
        assert isinstance(adapter.supports_vision, bool)

    @pytest.mark.parametrize("adapter_cls", adapters)
    def test_has_structured_output_flag(self, adapter_cls):
        adapter = adapter_cls()
        assert isinstance(adapter.supports_structured_output, bool)

    @pytest.mark.parametrize("adapter_cls", adapters)
    def test_is_available_returns_bool(self, adapter_cls, monkeypatch):
        # Clear all keys to ensure predictable behavior
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "LM_STUDIO_URL"]:
            monkeypatch.delenv(key, raising=False)
        adapter = adapter_cls()
        assert isinstance(adapter.is_available(), bool)

    @pytest.mark.parametrize("adapter_cls", adapters)
    def test_estimate_cost_returns_float(self, adapter_cls):
        adapter = adapter_cls()
        cost = adapter.estimate_cost(1000, 500)
        assert isinstance(cost, float)
        assert cost >= 0.0


class TestAnthropicAdapter:
    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert AnthropicAdapter().is_available()

    def test_is_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert not AnthropicAdapter().is_available()

    def test_cost_estimation(self):
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
        cost = adapter.estimate_cost(1000, 500)
        assert cost > 0


class TestLMStudioAdapter:
    def test_cost_is_zero(self):
        adapter = LMStudioAdapter()
        assert adapter.estimate_cost(10000, 5000) == 0.0

    def test_vision_not_supported(self):
        adapter = LMStudioAdapter()
        with pytest.raises(NotImplementedError):
            adapter.complete_vision("test", [b"image"])


class TestOpenRouterAdapter:
    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        assert OpenRouterAdapter().is_available()
