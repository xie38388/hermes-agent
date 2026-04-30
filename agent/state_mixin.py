"""Auto-generated mixin for AIAgent."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from datetime import datetime
import json
import re
import time
from agent.trajectory import convert_scratchpad_to_think
from utils import atomic_json_write
logger = logging.getLogger(__name__)

class StateMixin:
    """AIAgent mixin: state methods."""

    def reset_session_state(self):
        """Reset all session-scoped token counters to 0 for a fresh session.
        
        This method encapsulates the reset logic for all session-level metrics
        including:
        - Token usage counters (input, output, total, prompt, completion)
        - Cache read/write tokens
        - API call count
        - Reasoning tokens
        - Estimated cost tracking
        - Context compressor internal counters
        
        The method safely handles optional attributes (e.g., context compressor)
        using ``hasattr`` checks.
        
        This keeps the counter reset logic DRY and maintainable in one place
        rather than scattering it across multiple methods.
        """
        # Token usage counters
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0

        # Context engine reset (works for both built-in compressor and plugins)
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.on_session_reset()
    

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message before persistence/return.

        Some call paths need an API-only user-message variant without letting
        that synthetic text leak into persisted transcripts or resumed session
        history. When an override is configured for the active turn, mutate the
        in-memory messages list in place so both persistence and returned
        history stay clean.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override


    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite on any exit path.

        Ensures conversations are never lost, even on errors or early returns.
        Skipped when ``persist_session=False`` (ephemeral helper flows).
        """
        if not self.persist_session:
            return
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)


    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Persist any un-flushed messages to the SQLite session store.

        Uses _last_flushed_db_idx to track which messages have already been
        written, so repeated calls (from multiple exit paths) only write
        truly new messages — preventing the duplicate-write bug (#860).
        """
        if not self._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # If create_session() failed at startup (e.g. transient lock), the
            # session row may not exist yet.  ensure_session() uses INSERT OR
            # IGNORE so it is a no-op when the row is already there.
            self._session_db.ensure_session(
                self.session_id,
                source=self.platform or "cli",
                model=self.model,
            )
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)


    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()


    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        Save the full raw session to a JSON file.

        Stores every message exactly as the agent sees it: user messages,
        assistant messages (with reasoning, finish_reason, tool_calls),
        tool responses (with tool_call_id, tool_name), and injected system
        messages (compression summaries, todo snapshots, etc.).

        REASONING_SCRATCHPAD tags are converted to <think> blocks for consistency.
        Overwritten after each turn so it always reflects the latest state.
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # Clean assistant content for session logs
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            # Guard: never overwrite a larger session log with fewer messages.
            # This protects against data loss when --resume loads a session whose
            # messages weren't fully written to SQLite — the resumed agent starts
            # with partial history and would otherwise clobber the full JSON log.
            if self.session_log_file.exists():
                try:
                    existing = json.loads(self.session_log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # corrupted existing file — allow the overwrite

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
    

    def get_rate_limit_state(self):
        """Return the last captured RateLimitState, or None."""
        return self._rate_limit_state


    def flush_memories(self, messages: list = None, min_turns: int = None):
        """Give the model one turn to persist memories before context is lost.

        Called before compression, session reset, or CLI exit. Injects a flush
        message, makes one API call, executes any memory tool calls, then
        strips all flush artifacts from the message list.

        Args:
            messages: The current conversation messages. If None, uses
                      self._session_messages (last run_conversation state).
            min_turns: Minimum user turns required to trigger the flush.
                       None = use config value (flush_min_turns).
                       0 = always flush (used for compression).
        """
        if self._memory_flush_min_turns == 0 and min_turns is None:
            return
        if "memory" not in self.valid_tool_names or not self._memory_store:
            return
        effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, '_session_messages', None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Save anything worth remembering — prioritize user preferences, "
            "corrections, and recurring patterns over task-specific details.]"
        )
        _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
        flush_msg = {"role": "user", "content": flush_content, "_flush_sentinel": _sentinel}
        messages.append(flush_msg)

        try:
            # Build API messages for the flush call
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                if msg.get("role") == "assistant":
                    reasoning = msg.get("reasoning")
                    if reasoning:
                        api_msg["reasoning_content"] = reasoning
                api_msg.pop("reasoning", None)
                api_msg.pop("finish_reason", None)
                api_msg.pop("_flush_sentinel", None)
                api_msg.pop("_thinking_prefill", None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            if self._cached_system_prompt:
                api_messages = [{"role": "system", "content": self._cached_system_prompt}] + api_messages

            # Make one API call with only the memory tool available
            memory_tool_def = None
            for t in (self.tools or []):
                if t.get("function", {}).get("name") == "memory":
                    memory_tool_def = t
                    break

            if not memory_tool_def:
                messages.pop()  # remove flush msg
                return

            # Use auxiliary client for the flush call when available --
            # it's cheaper and avoids Codex Responses API incompatibility.
            from agent.auxiliary_client import call_llm as _call_llm
            _aux_available = True
            try:
                response = _call_llm(
                    task="flush_memories",
                    messages=api_messages,
                    tools=[memory_tool_def],
                    temperature=0.3,
                    max_tokens=5120,
                    # timeout resolved from auxiliary.flush_memories.timeout config
                )
            except RuntimeError:
                _aux_available = False
                response = None

            if not _aux_available and self.api_mode == "codex_responses":
                # No auxiliary client -- use the Codex Responses path directly
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs["tools"] = self._responses_tools([memory_tool_def])
                codex_kwargs["temperature"] = 0.3
                if "max_output_tokens" in codex_kwargs:
                    codex_kwargs["max_output_tokens"] = 5120
                response = self._run_codex_stream(codex_kwargs)
            elif not _aux_available and self.api_mode == "anthropic_messages":
                # Native Anthropic — use the Anthropic client directly
                from agent.anthropic_adapter import build_anthropic_kwargs as _build_ant_kwargs
                ant_kwargs = _build_ant_kwargs(
                    model=self.model, messages=api_messages,
                    tools=[memory_tool_def], max_tokens=5120,
                    reasoning_config=None,
                    preserve_dots=self._anthropic_preserve_dots(),
                )
                response = self._anthropic_messages_create(ant_kwargs)
            elif not _aux_available:
                api_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": [memory_tool_def],
                    "temperature": 0.3,
                    **self._max_tokens_param(5120),
                }
                from agent.auxiliary_client import _get_task_timeout
                response = self._ensure_primary_openai_client(reason="flush_memories").chat.completions.create(
                    **api_kwargs, timeout=_get_task_timeout("flush_memories")
                )

            # Extract tool calls from the response, handling all API formats
            tool_calls = []
            if self.api_mode == "codex_responses" and not _aux_available:
                assistant_msg, _ = self._normalize_codex_response(response)
                if assistant_msg and assistant_msg.tool_calls:
                    tool_calls = assistant_msg.tool_calls
            elif self.api_mode == "anthropic_messages" and not _aux_available:
                from agent.anthropic_adapter import normalize_anthropic_response as _nar_flush
                _flush_msg, _ = _nar_flush(response, strip_tool_prefix=self._is_anthropic_oauth)
                if _flush_msg and _flush_msg.tool_calls:
                    tool_calls = _flush_msg.tool_calls
            elif hasattr(response, "choices") and response.choices:
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    tool_calls = assistant_message.tool_calls

            for tc in tool_calls:
                if tc.function.name == "memory":
                    try:
                        args = json.loads(tc.function.arguments)
                        flush_target = args.get("target", "memory")
                        from tools.memory_tool import memory_tool as _memory_tool
                        _memory_tool(
                            action=args.get("action"),
                            target=flush_target,
                            content=args.get("content"),
                            old_text=args.get("old_text"),
                            store=self._memory_store,
                        )
                        if not self.quiet_mode:
                            print(f"  🧠 Memory flush: saved to {args.get('target', 'memory')}")
                    except Exception as e:
                        logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # Strip flush artifacts: remove everything from the flush message onward.
            # Use sentinel marker instead of identity check for robustness.
            while messages and messages[-1].get("_flush_sentinel") != _sentinel:
                messages.pop()
                if not messages:
                    break
            if messages and messages[-1].get("_flush_sentinel") == _sentinel:
                messages.pop()


