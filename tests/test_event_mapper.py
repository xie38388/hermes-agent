"""Tests for gateway/event_mapper.py — DomainEvent + EventMapper.

Covers all factory methods, transport mappers, and edge cases.
No mocks needed — everything is pure data transformation.
"""
import json
import time

import pytest

from gateway.event_mapper import (
    DomainEvent,
    EventKind,
    EventMapper,
    _safe_json_clone,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DomainEvent factory methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainEventFactories:
    def test_tool_started_basic(self):
        evt = DomainEvent.tool_started("web_search")
        assert evt.kind == EventKind.TOOL_STARTED
        assert evt.tool_name == "web_search"
        assert evt.preview is None
        assert evt.args is None
        assert isinstance(evt.timestamp, float)

    def test_tool_started_with_preview_and_args(self):
        evt = DomainEvent.tool_started("write_file", preview="writing main.py", args={"path": "/tmp/main.py"})
        assert evt.tool_name == "write_file"
        assert evt.preview == "writing main.py"
        assert evt.args == {"path": "/tmp/main.py"}

    def test_tool_completed_basic(self):
        evt = DomainEvent.tool_completed("web_search", duration=1.234, is_error=False)
        assert evt.kind == EventKind.TOOL_COMPLETED
        assert evt.tool_name == "web_search"
        assert evt.duration == 1.234
        assert evt.is_error is False

    def test_tool_completed_with_error(self):
        evt = DomainEvent.tool_completed("write_file", duration=0.5, is_error=True, result="Permission denied")
        assert evt.is_error is True
        assert evt.result == "Permission denied"

    def test_reasoning_available(self):
        evt = DomainEvent.reasoning_available("Let me think about this...")
        assert evt.kind == EventKind.REASONING_AVAILABLE
        assert evt.text == "Let me think about this..."

    def test_stream_chunk(self):
        evt = DomainEvent.stream_chunk("Hello, ")
        assert evt.kind == EventKind.STREAM_CHUNK
        assert evt.content == "Hello, "

    def test_stream_done(self):
        evt = DomainEvent.stream_done()
        assert evt.kind == EventKind.STREAM_DONE

    def test_frozen_immutability(self):
        evt = DomainEvent.stream_chunk("test")
        with pytest.raises(AttributeError):
            evt.content = "modified"


# ═══════════════════════════════════════════════════════════════════════════════
# EventMapper.to_run_event
# ═══════════════════════════════════════════════════════════════════════════════

class TestToRunEvent:
    def test_tool_started_event(self):
        evt = DomainEvent.tool_started("web_search", preview="searching...")
        payload = EventMapper.to_run_event(evt, run_id="run_abc")
        assert payload["event"] == "tool.started"
        assert payload["run_id"] == "run_abc"
        assert payload["tool"] == "web_search"
        assert payload["preview"] == "searching..."
        assert "timestamp" in payload

    def test_tool_completed_event(self):
        evt = DomainEvent.tool_completed("write_file", duration=2.3456, is_error=False, result="OK")
        payload = EventMapper.to_run_event(evt, run_id="run_xyz")
        assert payload["event"] == "tool.completed"
        assert payload["tool"] == "write_file"
        assert payload["duration"] == 2.346  # rounded to 3 decimals
        assert payload["error"] is False
        assert payload["result"] == "OK"

    def test_tool_completed_truncates_large_result(self):
        large_result = "x" * 60_000
        evt = DomainEvent.tool_completed("read_file", result=large_result)
        payload = EventMapper.to_run_event(evt, run_id="run_1")
        assert len(payload["result"]) == 50_000  # _MAX_RESULT_BYTES

    def test_reasoning_available_event(self):
        evt = DomainEvent.reasoning_available("Thinking step 1")
        payload = EventMapper.to_run_event(evt, run_id="run_r")
        assert payload["event"] == "reasoning.available"
        assert payload["text"] == "Thinking step 1"

    def test_stream_chunk_event(self):
        evt = DomainEvent.stream_chunk("Hello")
        payload = EventMapper.to_run_event(evt, run_id="run_s")
        assert payload["event"] == "stream.chunk"
        assert payload["content"] == "Hello"

    def test_stream_done_event(self):
        evt = DomainEvent.stream_done()
        payload = EventMapper.to_run_event(evt, run_id="run_d")
        assert payload["event"] == "stream.done"
        assert "content" not in payload

    def test_metadata_passthrough(self):
        evt = DomainEvent(kind=EventKind.RUN_COMPLETED, metadata={"tokens": 150})
        payload = EventMapper.to_run_event(evt, run_id="run_m")
        assert payload["metadata"] == {"tokens": 150}

    def test_no_metadata_when_none(self):
        evt = DomainEvent.stream_done()
        payload = EventMapper.to_run_event(evt, run_id="run_n")
        assert "metadata" not in payload

    def test_tool_started_with_args(self):
        evt = DomainEvent.tool_started("write_file", args={"path": "/tmp/test.py", "content": "print('hi')"})
        payload = EventMapper.to_run_event(evt, run_id="run_a")
        assert payload["args"] == {"path": "/tmp/test.py", "content": "print('hi')"}

    def test_tool_completed_with_args(self):
        evt = DomainEvent.tool_completed("web_search", args={"query": "test"}, result="found")
        payload = EventMapper.to_run_event(evt, run_id="run_b")
        assert payload["args"] == {"query": "test"}


# ═══════════════════════════════════════════════════════════════════════════════
# EventMapper.to_chat_chunk
# ═══════════════════════════════════════════════════════════════════════════════

class TestToChatChunk:
    def test_stream_chunk_produces_delta(self):
        evt = DomainEvent.stream_chunk("Hello")
        chunk = EventMapper.to_chat_chunk(evt, completion_id="chatcmpl-1", model="gpt-4o", created=1700000000)
        assert chunk["id"] == "chatcmpl-1"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "gpt-4o"
        assert chunk["created"] == 1700000000
        assert chunk["choices"][0]["delta"]["content"] == "Hello"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_stream_done_produces_stop(self):
        evt = DomainEvent.stream_done()
        chunk = EventMapper.to_chat_chunk(evt, completion_id="chatcmpl-2", model="gpt-4o")
        assert chunk["choices"][0]["delta"] == {}
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_tool_started_returns_none(self):
        evt = DomainEvent.tool_started("web_search")
        assert EventMapper.to_chat_chunk(evt, completion_id="x", model="m") is None

    def test_reasoning_returns_none(self):
        evt = DomainEvent.reasoning_available("thinking")
        assert EventMapper.to_chat_chunk(evt, completion_id="x", model="m") is None

    def test_created_defaults_to_now(self):
        evt = DomainEvent.stream_chunk("test")
        before = int(time.time())
        chunk = EventMapper.to_chat_chunk(evt, completion_id="c", model="m")
        after = int(time.time())
        assert before <= chunk["created"] <= after

    def test_chunk_is_json_serializable(self):
        evt = DomainEvent.stream_chunk("Hello world")
        chunk = EventMapper.to_chat_chunk(evt, completion_id="c", model="m")
        serialized = json.dumps(chunk)
        assert '"content": "Hello world"' in serialized


# ═══════════════════════════════════════════════════════════════════════════════
# EventMapper.to_tool_progress_sse
# ═══════════════════════════════════════════════════════════════════════════════

class TestToToolProgressSse:
    def test_tool_started(self):
        evt = DomainEvent.tool_started("web_search", preview="searching...")
        payload = EventMapper.to_tool_progress_sse(evt)
        assert payload == {
            "type": "tool.started",
            "tool": "web_search",
            "preview": "searching...",
        }

    def test_tool_completed(self):
        evt = DomainEvent.tool_completed("write_file", duration=1.2345, is_error=False)
        payload = EventMapper.to_tool_progress_sse(evt)
        assert payload == {
            "type": "tool.completed",
            "tool": "write_file",
            "duration": 1.234,
            "error": False,
        }

    def test_tool_completed_with_error(self):
        evt = DomainEvent.tool_completed("write_file", duration=0.1, is_error=True)
        payload = EventMapper.to_tool_progress_sse(evt)
        assert payload["error"] is True

    def test_stream_chunk_returns_none(self):
        evt = DomainEvent.stream_chunk("text")
        assert EventMapper.to_tool_progress_sse(evt) is None

    def test_stream_done_returns_none(self):
        evt = DomainEvent.stream_done()
        assert EventMapper.to_tool_progress_sse(evt) is None


# ═══════════════════════════════════════════════════════════════════════════════
# _safe_json_clone helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeJsonClone:
    def test_dict_roundtrip(self):
        obj = {"key": "value", "nested": {"a": 1}}
        assert _safe_json_clone(obj) == obj

    def test_list_roundtrip(self):
        obj = [1, "two", {"three": 3}]
        assert _safe_json_clone(obj) == obj

    def test_non_serializable_falls_back_to_str(self):
        obj = object()
        result = _safe_json_clone(obj)
        assert isinstance(result, str)
        assert "object" in result

    def test_set_falls_back_to_str(self):
        obj = {1, 2, 3}
        result = _safe_json_clone(obj)
        assert isinstance(result, str)
