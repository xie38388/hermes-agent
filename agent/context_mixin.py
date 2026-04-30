"""Auto-generated mixin for AIAgent."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from datetime import datetime
import os
import re
import time
import uuid
from agent.model_metadata import estimate_messages_tokens_rough
from agent.model_metadata import estimate_tokens_rough
logger = logging.getLogger(__name__)

class ContextMixin:
    """AIAgent mixin: context methods."""
    _context_pressure_last_warned: dict = {}

    def _check_compression_model_feasibility(self) -> None:
        """Warn at session start if the auxiliary compression model's context
        window is smaller than the main model's compression threshold.

        When the auxiliary model cannot fit the content that needs summarising,
        compression will either fail outright (the LLM call errors) or produce
        a severely truncated summary.

        Called during ``__init__`` so CLI users see the warning immediately
        (via ``_vprint``).  The gateway sets ``status_callback`` *after*
        construction, so ``_replay_compression_warning()`` re-sends the
        stored warning through the callback on the first
        ``run_conversation()`` call.
        """
        if not self.compression_enabled:
            return
        try:
            from agent.auxiliary_client import get_text_auxiliary_client
            from agent.model_metadata import get_model_context_length

            client, aux_model = get_text_auxiliary_client(
                "compression",
                main_runtime=self._current_main_runtime(),
            )
            if client is None or not aux_model:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "No auxiliary LLM provider for compression — "
                    "summaries will be unavailable."
                )
                return

            aux_base_url = str(getattr(client, "base_url", ""))
            aux_api_key = str(getattr(client, "api_key", ""))

            # Read user-configured context_length for the compression model.
            # Custom endpoints often don't support /models API queries so
            # get_model_context_length() falls through to the 128K default,
            # ignoring the explicit config value.  Pass it as the highest-
            # priority hint so the configured value is always respected.
            _aux_cfg = (self.config or {}).get("auxiliary", {}).get("compression", {})
            _aux_context_config = _aux_cfg.get("context_length") if isinstance(_aux_cfg, dict) else None
            if _aux_context_config is not None:
                try:
                    _aux_context_config = int(_aux_context_config)
                except (TypeError, ValueError):
                    _aux_context_config = None

            aux_context = get_model_context_length(
                aux_model,
                base_url=aux_base_url,
                api_key=aux_api_key,
                config_context_length=_aux_context_config,
            )

            threshold = self.context_compressor.threshold_tokens
            if aux_context < threshold:
                # Suggest a threshold that would fit the aux model,
                # rounded down to a clean percentage.
                safe_pct = int((aux_context / self.context_compressor.context_length) * 100)
                msg = (
                    f"⚠ Compression model ({aux_model}) context "
                    f"is {aux_context:,} tokens, but the main model's "
                    f"compression threshold is {threshold:,} tokens. "
                    f"Context compression will not be possible — the "
                    f"content to summarise will exceed the auxiliary "
                    f"model's context window.\n"
                    f"  Fix options (config.yaml):\n"
                    f"  1. Use a larger compression model:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{threshold:,}+-context>\n"
                    f"  2. Lower the compression threshold to fit "
                    f"the current model:\n"
                    f"       compression:\n"
                    f"         threshold: 0.{safe_pct:02d}"
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "Auxiliary compression model %s has %d token context, "
                    "below the main model's compression threshold of %d "
                    "tokens — compression summaries will fail or be "
                    "severely truncated.",
                    aux_model,
                    aux_context,
                    threshold,
                )
        except Exception as exc:
            logger.debug(
                "Compression feasibility check failed (non-fatal): %s", exc
            )


    def _max_tokens_param(self, value: int) -> dict:
        """Return the correct max tokens kwarg for the current provider.
        
        OpenAI's newer models (gpt-4o, o-series, gpt-5+) require
        'max_completion_tokens'. OpenRouter, local models, and older
        OpenAI models use 'max_tokens'.
        """
        if self._is_direct_openai_url():
            return {"max_completion_tokens": value}
        return {"max_tokens": value}


    @staticmethod
    def _extract_api_error_context(error: Exception) -> Dict[str, Any]:
        """Extract structured rate-limit details from provider errors."""
        context: Dict[str, Any] = {}

        body = getattr(error, "body", None)
        payload = None
        if isinstance(body, dict):
            payload = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(payload, dict):
            reason = payload.get("code") or payload.get("error")
            if isinstance(reason, str) and reason.strip():
                context["reason"] = reason.strip()
            message = payload.get("message") or payload.get("error_description")
            if isinstance(message, str) and message.strip():
                context["message"] = message.strip()
            for key in ("resets_at", "reset_at"):
                value = payload.get(key)
                if value not in (None, ""):
                    context["reset_at"] = value
                    break
            retry_after = payload.get("retry_after")
            if retry_after not in (None, "") and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass
            ratelimit_reset = headers.get("x-ratelimit-reset")
            if ratelimit_reset and "reset_at" not in context:
                context["reset_at"] = ratelimit_reset

        if "message" not in context:
            raw_message = str(error).strip()
            if raw_message:
                context["message"] = raw_message[:500]

        if "reset_at" not in context:
            message = context.get("message") or ""
            if isinstance(message, str):
                delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
                if delay_match:
                    value = float(delay_match.group(1))
                    seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                    context["reset_at"] = time.time() + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

        return context


    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )


    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default", focus_topic: str = None) -> tuple:
        """Compress conversation context and split the session in SQLite.

        Args:
            focus_topic: Optional focus string for guided compression — the
                summariser will prioritise preserving information related to
                this topic.  Inspired by Claude Code's ``/compact <focus>``.

        Returns:
            (compressed_messages, new_system_prompt) tuple
        """
        _pre_msg_count = len(messages)
        logger.info(
            "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
            self.session_id or "none", _pre_msg_count,
            f"{approx_tokens:,}" if approx_tokens else "unknown", self.model,
            focus_topic,
        )
        # Pre-compression memory flush: let the model save memories before they're lost
        self.flush_memories(messages, min_turns=0)

        # Notify external memory provider before compression discards context
        if self._memory_manager:
            try:
                self._memory_manager.on_pre_compress(messages)
            except Exception:
                pass

        compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic)

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # Propagate title to the new session with auto-numbering
                old_title = self._session_db.get_session_title(self.session_id)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                # Update session_log_file to point to the new session's JSON file
                self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    parent_session_id=old_session_id,
                )
                # Auto-number the title for the continuation session
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(old_title)
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(self.session_id, new_system_prompt)
                # Reset flush cursor — new session starts with no messages written
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # Warn on repeated compressions (quality degrades with each pass)
        _cc = self.context_compressor.compression_count
        if _cc >= 2:
            self._vprint(
                f"{self.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh.",
                force=True,
            )

        # Update token estimate after compaction so pressure calculations
        # use the post-compression count, not the stale pre-compression one.
        _compressed_est = (
            estimate_tokens_rough(new_system_prompt)
            + estimate_messages_tokens_rough(compressed)
        )
        self.context_compressor.last_prompt_tokens = _compressed_est
        self.context_compressor.last_completion_tokens = 0

        # Only reset the pressure warning if compression actually brought
        # us below the warning level (85% of threshold).  When compression
        # can't reduce enough (e.g. threshold is very low, or system prompt
        # alone exceeds the warning level), keep the tier set to prevent
        # spamming the user with repeated warnings every loop iteration.
        if self.context_compressor.threshold_tokens > 0:
            _post_progress = _compressed_est / self.context_compressor.threshold_tokens
            if _post_progress < 0.85:
                self._context_pressure_warned_at = 0.0
                # Clear class-level dedup for this session so a fresh
                # warning cycle can start if context grows again.
                _sid = self.session_id or "default"
                type(self)._context_pressure_last_warned.pop(_sid, None)

        # Clear the file-read dedup cache.  After compression the original
        # read content is summarised away — if the model re-reads the same
        # file it needs the full content, not a "file unchanged" stub.
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d tokens=~%s",
            self.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt


