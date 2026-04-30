"""
gateway/event_mapper.py — Anti-corruption layer between domain events and SSE transport.

Separates WHAT happened (domain events) from HOW it is transmitted (SSE format).
All event construction logic is centralized here instead of scattered across callbacks.

Usage:
    from gateway.event_mapper import DomainEvent, EventMapper

    mapper = EventMapper()
    domain_evt = DomainEvent.tool_started("web_search", preview="searching...")
    sse_payload = mapper.to_run_event(domain_evt, run_id="run_123")
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Domain Event Types
# ---------------------------------------------------------------------------

class EventKind(str, Enum):
    """Exhaustive enumeration of domain-level agent lifecycle events."""
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    REASONING_AVAILABLE = "reasoning.available"
    STREAM_CHUNK = "stream.chunk"
    STREAM_DONE = "stream.done"
    RUN_CREATED = "run.created"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"


@dataclass(frozen=True)
class DomainEvent:
    """Immutable domain event — represents something that happened in the agent lifecycle.

    Domain events carry NO transport-specific formatting.  They are pure data.
    """
    kind: EventKind
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    preview: Optional[str] = None
    args: Optional[Any] = None
    result: Optional[str] = None
    duration: float = 0.0
    is_error: bool = False
    text: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # ---- Factory methods for readability ----

    @classmethod
    def tool_started(
        cls, tool_name: str, preview: str = None, args: Any = None
    ) -> "DomainEvent":
        return cls(kind=EventKind.TOOL_STARTED, tool_name=tool_name, preview=preview, args=args)

    @classmethod
    def tool_completed(
        cls,
        tool_name: str,
        duration: float = 0.0,
        is_error: bool = False,
        result: str = None,
        args: Any = None,
    ) -> "DomainEvent":
        return cls(
            kind=EventKind.TOOL_COMPLETED,
            tool_name=tool_name,
            duration=duration,
            is_error=is_error,
            result=result,
            args=args,
        )

    @classmethod
    def reasoning_available(cls, text: str) -> "DomainEvent":
        return cls(kind=EventKind.REASONING_AVAILABLE, text=text)

    @classmethod
    def stream_chunk(cls, content: str) -> "DomainEvent":
        return cls(kind=EventKind.STREAM_CHUNK, content=content)

    @classmethod
    def stream_done(cls) -> "DomainEvent":
        return cls(kind=EventKind.STREAM_DONE)


# ---------------------------------------------------------------------------
# Transport Mappers
# ---------------------------------------------------------------------------

# Maximum result payload size to prevent oversized SSE frames (50 KB).
_MAX_RESULT_BYTES = 50_000


class EventMapper:
    """Maps domain events to transport-specific payloads.

    Each ``to_*`` method produces a dict ready for ``json.dumps`` + SSE write.
    The mapper is stateless and safe to share across threads/coroutines.
    """

    # ---- Run Events SSE (/v1/runs/{run_id}/events) ----

    @staticmethod
    def to_run_event(event: DomainEvent, run_id: str) -> Dict[str, Any]:
        """Convert a domain event to the structured JSON payload used by the
        ``/v1/runs/{run_id}/events`` SSE endpoint."""
        base = {
            "event": event.kind.value,
            "run_id": run_id,
            "timestamp": event.timestamp,
        }

        if event.kind == EventKind.TOOL_STARTED:
            base["tool"] = event.tool_name
            base["preview"] = event.preview
            if event.args is not None:
                base["args"] = _safe_json_clone(event.args)

        elif event.kind == EventKind.TOOL_COMPLETED:
            base["tool"] = event.tool_name
            base["duration"] = round(event.duration, 3)
            base["error"] = event.is_error
            if event.result is not None:
                result_str = str(event.result)
                base["result"] = (
                    result_str[:_MAX_RESULT_BYTES]
                    if len(result_str) > _MAX_RESULT_BYTES
                    else result_str
                )
            if event.args is not None:
                base["args"] = _safe_json_clone(event.args)

        elif event.kind == EventKind.REASONING_AVAILABLE:
            base["text"] = event.text or ""

        elif event.kind == EventKind.STREAM_CHUNK:
            base["content"] = event.content

        # RUN_CREATED / RUN_COMPLETED / RUN_FAILED / RUN_CANCELLED / STREAM_DONE
        # carry only the base fields.

        if event.metadata:
            base["metadata"] = event.metadata

        return base

    # ---- Chat Completions SSE (/v1/chat/completions stream) ----

    @staticmethod
    def to_chat_chunk(
        event: DomainEvent,
        completion_id: str,
        model: str,
        created: int = None,
    ) -> Optional[Dict[str, Any]]:
        """Convert a domain event to an OpenAI-compatible chat completion chunk.

        Returns ``None`` for event kinds that have no chat-completion
        representation (e.g. ``REASONING_AVAILABLE`` is sent as a custom
        SSE event, not a delta chunk).
        """
        if created is None:
            created = int(time.time())

        if event.kind == EventKind.STREAM_CHUNK:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": event.content}, "finish_reason": None}
                ],
            }

        if event.kind == EventKind.STREAM_DONE:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        # Tool progress events are sent as custom SSE events, not delta chunks.
        return None

    @staticmethod
    def to_tool_progress_sse(event: DomainEvent) -> Optional[Dict[str, Any]]:
        """Convert a tool event to the ``hermes.tool.progress`` custom SSE payload.

        Used by the chat completions stream for inline tool status updates.
        """
        if event.kind == EventKind.TOOL_STARTED:
            return {
                "type": "tool.started",
                "tool": event.tool_name,
                "preview": event.preview,
            }
        if event.kind == EventKind.TOOL_COMPLETED:
            return {
                "type": "tool.completed",
                "tool": event.tool_name,
                "duration": round(event.duration, 3),
                "error": event.is_error,
            }
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_clone(obj: Any) -> Any:
    """Round-trip through JSON to ensure the object is serializable.
    Falls back to ``str(obj)`` on failure."""
    try:
        return json.loads(json.dumps(obj))
    except (TypeError, ValueError):
        return str(obj)
