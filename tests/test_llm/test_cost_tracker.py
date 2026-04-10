"""Tests for cost tracking."""

from src.llm.cost_tracker import CostTracker


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0

    def test_record_and_totals(self):
        tracker = CostTracker()
        tracker.record("anthropic", "claude-sonnet", 1000, 500, 0.01, 200)
        tracker.record("openai", "gpt-4o", 2000, 1000, 0.02, 300)

        assert tracker.total_calls == 2
        assert tracker.total_tokens == 4500
        assert tracker.total_cost == pytest.approx(0.03)
        assert tracker.avg_latency_ms == pytest.approx(250)

    def test_by_provider(self):
        tracker = CostTracker()
        tracker.record("anthropic", "model", 100, 50, 0.001, 100)
        tracker.record("anthropic", "model", 200, 100, 0.002, 150)
        tracker.record("openai", "model", 300, 150, 0.003, 200)

        groups = tracker.by_provider()
        assert len(groups["anthropic"]) == 2
        assert len(groups["openai"]) == 1


import pytest
