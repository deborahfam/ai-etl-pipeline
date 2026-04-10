"""Tests for the event bus."""

from src.engine.events import EventBus


class TestEventBus:
    def test_emit_and_handle(self):
        bus = EventBus()
        received = []
        bus.on("test_event", lambda **kw: received.append(kw))
        bus.emit("test_event", key="value")
        assert len(received) == 1
        assert received[0] == {"key": "value"}

    def test_multiple_handlers(self):
        bus = EventBus()
        count = [0]
        bus.on("evt", lambda **kw: count.__setitem__(0, count[0] + 1))
        bus.on("evt", lambda **kw: count.__setitem__(0, count[0] + 1))
        bus.emit("evt")
        assert count[0] == 2

    def test_off_removes_handler(self):
        bus = EventBus()
        calls = []
        handler = lambda **kw: calls.append(1)
        bus.on("evt", handler)
        bus.off("evt", handler)
        bus.emit("evt")
        assert len(calls) == 0

    def test_handler_exception_doesnt_break(self):
        bus = EventBus()
        results = []

        def bad_handler(**kw):
            raise RuntimeError("boom")

        def good_handler(**kw):
            results.append("ok")

        bus.on("evt", bad_handler)
        bus.on("evt", good_handler)
        bus.emit("evt")
        assert results == ["ok"]

    def test_clear(self):
        bus = EventBus()
        bus.on("a", lambda **kw: None)
        bus.on("b", lambda **kw: None)
        bus.clear()
        assert len(bus._handlers) == 0
