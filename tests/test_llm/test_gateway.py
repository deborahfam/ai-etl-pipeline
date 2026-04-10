"""Tests for LLM Gateway."""

import pytest

from src.llm.gateway import LLMGateway
from src.llm.adapters.base import LLMResponse


class TestLLMGateway:
    def test_auto_detect_with_no_keys(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("LM_STUDIO_URL", raising=False)

        gateway = LLMGateway.auto_detect(cache_enabled=False)
        assert len(gateway.available_providers) == 0

    def test_complete_with_mock(self, mock_gateway, mock_adapter):
        mock_adapter.responses["hello"] = "world"
        resp = mock_gateway.complete("hello")
        assert resp.content == "world"
        assert len(mock_adapter.calls) == 1

    def test_fallback_on_failure(self):
        from tests.conftest import MockAdapter

        bad = MockAdapter()
        bad.complete = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail"))
        bad.provider_name = "bad"

        good = MockAdapter(responses={"test": "success"})
        good.provider_name = "good"

        gw = LLMGateway(
            adapters={"bad": bad, "good": good},
            primary_provider="bad",
            cache_enabled=False,
        )
        resp = gw.complete("test")
        assert resp.content == "success"

    def test_all_fail_raises(self):
        from tests.conftest import MockAdapter

        bad = MockAdapter()
        bad.complete = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail"))

        gw = LLMGateway(adapters={"bad": bad}, cache_enabled=False)
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            gw.complete("test")

    def test_vision_requires_capable_provider(self, mock_gateway, mock_adapter):
        mock_adapter.supports_vision = False
        with pytest.raises(RuntimeError, match="No vision-capable"):
            mock_gateway.complete_vision("test", images=[b"fake"])

    def test_cost_tracking(self, mock_gateway, mock_adapter):
        mock_adapter.responses["test"] = "response"
        mock_gateway.complete("test")
        assert mock_gateway.cost_tracker.total_calls == 1
        assert mock_gateway.cost_tracker.total_tokens > 0


class TestLLMCache:
    def test_cache_hit(self, mock_gateway, mock_adapter):
        mock_gateway.cache_enabled = True
        from src.llm.cache import LLMCache
        mock_gateway.cache = LLMCache(":memory:")

        mock_adapter.responses["cached_prompt"] = "cached_response"
        resp1 = mock_gateway.complete("cached_prompt")
        assert resp1.content == "cached_response"
        assert len(mock_adapter.calls) == 1

        resp2 = mock_gateway.complete("cached_prompt")
        assert resp2.content == "cached_response"
        # Second call should hit cache, not the adapter
        assert len(mock_adapter.calls) == 1
