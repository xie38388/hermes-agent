"""Auto-generated mixin for AIAgent."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from agent.shared_utils import _install_safe_stdio, _sanitize_messages_non_ascii, _sanitize_messages_surrogates, _sanitize_surrogates, IterationBudget

from openai import OpenAI
from tools.interrupt import set_interrupt as _set_interrupt
from agent.memory_manager import build_memory_context_block
import copy
from agent.retry_utils import jittered_backoff
import json
import random
import re
import threading
import time
import uuid
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason
from agent.error_classifier import classify_api_error
from agent.model_metadata import estimate_messages_tokens_rough
from agent.model_metadata import estimate_request_tokens_rough
from agent.model_metadata import get_next_probe_tier
from agent.model_metadata import parse_available_output_tokens_from_error
from agent.model_metadata import parse_context_limit_from_error
from agent.model_metadata import save_context_length
from agent.prompt_caching import apply_anthropic_cache_control
from agent.trajectory import has_incomplete_scratchpad
from agent.usage_pricing import estimate_usage_cost
from agent.usage_pricing import normalize_usage
from utils import env_var_enabled
logger = logging.getLogger(__name__)

class ConversationMixin:
    """AIAgent mixin: conversation methods."""

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """Spawn a background thread to review the conversation for memory/skill saves.

        Creates a full AIAgent fork with the same model, tools, and context as the
        main session. The review prompt is appended as the next user turn in the
        forked conversation. Writes directly to the shared memory/skill stores.
        Never modifies the main conversation history or produces user-visible output.
        """
        import threading

        # Pick the right prompt based on which triggers fired
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib, os as _os
            review_agent = None
            try:
                with open(_os.devnull, "w") as _devnull, \
                     contextlib.redirect_stdout(_devnull), \
                     contextlib.redirect_stderr(_devnull):
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=8,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                    )
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # Scan the review agent's messages for successful tool actions
                # and surface a compact summary to the user.
                actions = []
                for msg in getattr(review_agent, "_session_messages", []):
                    if not isinstance(msg, dict) or msg.get("role") != "tool":
                        continue
                    try:
                        data = json.loads(msg.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not data.get("success"):
                        continue
                    message = data.get("message", "")
                    target = data.get("target", "")
                    if "created" in message.lower():
                        actions.append(message)
                    elif "updated" in message.lower():
                        actions.append(message)
                    elif "added" in message.lower() or (target and "add" in message.lower()):
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "Entry added" in message:
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "removed" in message.lower() or "replaced" in message.lower():
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(f"  💾 {summary}")
                    _bg_cb = self.background_review_callback
                    if _bg_cb:
                        try:
                            _bg_cb(f"💾 {summary}")
                        except Exception:
                            pass

            except Exception as e:
                logger.debug("Background memory/skill review failed: %s", e)
            finally:
                # Close all resources (httpx client, subprocesses, etc.) so
                # GC doesn't try to clean them up on a dead asyncio event
                # loop (which produces "Event loop is closed" errors).
                if review_agent is not None:
                    try:
                        review_agent.close()
                    except Exception:
                        pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()


    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        Recover todo state from conversation history.
        
        The gateway creates a fresh AIAgent per message, so the in-memory
        TodoStore is empty. We scan the history for the most recent todo
        tool response and replay it to reconstruct the state.
        """
        # Walk history backwards to find the most recent todo tool response
        last_todo_response = None
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Quick check: todo responses contain "todos" key
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if last_todo_response:
            # Replay the items into the store (replace mode)
            self._todo_store.write(last_todo_response, merge=False)
            if not self.quiet_mode:
                self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)
    

    def _chat_messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal chat-style messages to Responses input items."""
        items: List[Dict[str, Any]] = []
        seen_item_ids: set = set()

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                continue

            if role in {"user", "assistant"}:
                content = msg.get("content", "")
                content_text = str(content) if content is not None else ""

                if role == "assistant":
                    # Replay encrypted reasoning items from previous turns
                    # so the API can maintain coherent reasoning chains.
                    codex_reasoning = msg.get("codex_reasoning_items")
                    has_codex_reasoning = False
                    if isinstance(codex_reasoning, list):
                        for ri in codex_reasoning:
                            if isinstance(ri, dict) and ri.get("encrypted_content"):
                                item_id = ri.get("id")
                                if item_id and item_id in seen_item_ids:
                                    continue
                                items.append(ri)
                                if item_id:
                                    seen_item_ids.add(item_id)
                                has_codex_reasoning = True

                    if content_text.strip():
                        items.append({"role": "assistant", "content": content_text})
                    elif has_codex_reasoning:
                        # The Responses API requires a following item after each
                        # reasoning item (otherwise: missing_following_item error).
                        # When the assistant produced only reasoning with no visible
                        # content, emit an empty assistant message as the required
                        # following item.
                        items.append({"role": "assistant", "content": ""})

                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function", {})
                            fn_name = fn.get("name")
                            if not isinstance(fn_name, str) or not fn_name.strip():
                                continue

                            embedded_call_id, embedded_response_item_id = self._split_responses_tool_id(
                                tc.get("id")
                            )
                            call_id = tc.get("call_id")
                            if not isinstance(call_id, str) or not call_id.strip():
                                call_id = embedded_call_id
                            if not isinstance(call_id, str) or not call_id.strip():
                                if (
                                    isinstance(embedded_response_item_id, str)
                                    and embedded_response_item_id.startswith("fc_")
                                    and len(embedded_response_item_id) > len("fc_")
                                ):
                                    call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                                else:
                                    _raw_args = str(fn.get("arguments", "{}"))
                                    call_id = self._deterministic_call_id(fn_name, _raw_args, len(items))
                            call_id = call_id.strip()

                            arguments = fn.get("arguments", "{}")
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            elif not isinstance(arguments, str):
                                arguments = str(arguments)
                            arguments = arguments.strip() or "{}"

                            items.append({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": fn_name,
                                "arguments": arguments,
                            })
                    continue

                items.append({"role": role, "content": content_text})
                continue

            if role == "tool":
                raw_tool_call_id = msg.get("tool_call_id")
                call_id, _ = self._split_responses_tool_id(raw_tool_call_id)
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                        call_id = raw_tool_call_id.strip()
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(msg.get("content", "") or ""),
                })

        return items


    def _qwen_prepare_chat_messages(self, api_messages: list) -> list:
        prepared = copy.deepcopy(api_messages)
        if not prepared:
            return prepared

        for msg in prepared:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # Normalize: convert bare strings to text dicts, keep dicts as-is.
                # deepcopy already created independent copies, no need for dict().
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        # Inject cache_control on the last part of the system message.
        for msg in prepared:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return prepared


    def _qwen_prepare_chat_messages_inplace(self, messages: list) -> None:
        """In-place variant — mutates an already-copied message list."""
        if not messages:
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break


    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """Request a summary when max iterations are reached. Returns the final response text."""
        print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # Build API messages, stripping internal-only fields
            # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                    api_msg.pop(internal_field, None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            summary_extra_body = {}
            _is_nous = "nousresearch" in self._base_url_lower
            if self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                assistant_message, _ = self._normalize_codex_response(summary_response)
                final_response = (assistant_message.content or "").strip() if assistant_message else ""
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))

                # Include provider routing preferences
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences:
                    summary_extra_body["provider"] = provider_preferences

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak, normalize_anthropic_response as _nar
                    _ant_kw = _bak(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=self._is_anthropic_oauth,
                                   preserve_dots=self._anthropic_preserve_dots())
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _msg, _ = _nar(summary_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_msg.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                # Retry summary generation
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    retry_msg, _ = self._normalize_codex_response(retry_response)
                    final_response = (retry_msg.content or "").strip() if retry_msg else ""
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak2, normalize_anthropic_response as _nar2
                    _ant_kw2 = _bak2(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=self._is_anthropic_oauth,
                                    max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                    preserve_dots=self._anthropic_preserve_dots())
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_msg, _ = _nar2(retry_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_retry_msg.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    if final_response:
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response


    def _execute_api_with_retry(self, api_messages, messages, api_kwargs,
                                api_start_time, retry_count, max_retries,
                                primary_recovery_attempted, max_compression_attempts,
                                thinking_sig_retry_attempted, has_retried_429,
                                restart_with_compressed_messages, restart_with_length_continuation,
                                finish_reason, response, thinking_spinner,
                                approx_tokens, total_chars, api_call_count,
                                conversation_history, active_system_prompt):
        """Execute the API call with retry/failover/compression logic.
        
        Returns dict with: response, finish_reason, retry_count,
        restart_with_compressed_messages, restart_with_length_continuation,
        interrupted, thinking_spinner, api_kwargs
        """
        codex_auth_retry_attempted = False
        anthropic_auth_retry_attempted = False
        nous_auth_retry_attempted = False
        while retry_count < max_retries:
            try:
                self._reset_stream_delivery_tracking()
                api_kwargs = self._build_api_kwargs(api_messages)
                if self.api_mode == "codex_responses":
                    api_kwargs = self._preflight_codex_api_kwargs(api_kwargs, allow_stream=False)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "pre_api_request",
                        task_id=effective_task_id,
                        session_id=self.session_id or "",
                        platform=self.platform or "",
                        model=self.model,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_mode=self.api_mode,
                        api_call_count=api_call_count,
                        message_count=len(api_messages),
                        tool_count=len(self.tools or []),
                        approx_input_tokens=approx_tokens,
                        request_char_count=total_chars,
                        max_tokens=self.max_tokens,
                    )
                except Exception:
                    pass

                if env_var_enabled("HERMES_DUMP_REQUESTS"):
                    self._dump_api_request_debug(api_kwargs, reason="preflight")

                # Always prefer the streaming path — even without stream
                # consumers.  Streaming gives us fine-grained health
                # checking (90s stale-stream detection, 60s read timeout)
                # that the non-streaming path lacks.  Without this,
                # subagents and other quiet-mode callers can hang
                # indefinitely when the provider keeps the connection
                # alive with SSE pings but never delivers a response.
                # The streaming path is a no-op for callbacks when no
                # consumers are registered, and falls back to non-
                # streaming automatically if the provider doesn't
                # support it.
                def _stop_spinner():
                    nonlocal thinking_spinner
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                _use_streaming = True
                if not self._has_stream_consumers():
                    # No display/TTS consumer. Still prefer streaming for
                    # health checking, but skip for Mock clients in tests
                    # (mocks return SimpleNamespace, not stream iterators).
                    from unittest.mock import Mock
                    if isinstance(getattr(self, "client", None), Mock):
                        _use_streaming = False

                if _use_streaming:
                    response = self._interruptible_streaming_api_call(
                        api_kwargs, on_first_delta=_stop_spinner
                    )
                else:
                    response = self._interruptible_api_call(api_kwargs)

                api_duration = time.time() - api_start_time

                # Stop thinking spinner silently -- the response box or tool
                # execution messages that follow are more informative.
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if self.thinking_callback:
                    self.thinking_callback("")

                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")

                if self.verbose_logging:
                    # Log response with provider info if available
                    resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                    logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")

                # Validate response shape
                _resp_valid, error_details = self._validate_api_response(response)
                response_invalid = not _resp_valid
                if response_invalid:
                    # Stop spinner before printing error messages
                    if thinking_spinner:
                        thinking_spinner.stop("(´;ω;`) oops, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    # Invalid response — could be rate limiting, provider timeout,
                    # upstream server error, or malformed response.
                    retry_count += 1

                    # Eager fallback: empty/malformed responses are a common
                    # rate-limit symptom.  Switch to fallback immediately
                    # rather than retrying with extended backoff.
                    if self._fallback_index < len(self._fallback_chain):
                        self._emit_status("⚠️ Empty/malformed response — switching to fallback...")
                    if self._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue

                    error_msg, provider_name, _failure_hint = self._diagnose_invalid_response(response, api_duration)
                    self._vprint(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                    self._vprint(f"{self.log_prefix}   🏢 Provider: {provider_name}", force=True)
                    cleaned_provider_error = self._clean_error_message(error_msg)
                    self._vprint(f"{self.log_prefix}   📝 Provider message: {cleaned_provider_error}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  {_failure_hint}", force=True)

                    if retry_count >= max_retries:
                        # Try fallback before giving up
                        self._emit_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        self._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                        logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                        self._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                            "failed": True  # Mark as failure for filtering
                        }

                    # Backoff before retry — jittered exponential: 5s base, 120s cap
                    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                    self._vprint(f"{self.log_prefix}⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...", force=True)
                    logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")

                    # Sleep in small increments to stay responsive to interrupts
                    sleep_end = time.time() + wait_time
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)
                    continue  # Retry the API call

                # Check finish_reason before proceeding
                if self.api_mode == "codex_responses":
                    status = getattr(response, "status", None)
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = None
                    if isinstance(incomplete_details, dict):
                        incomplete_reason = incomplete_details.get("reason")
                    else:
                        incomplete_reason = getattr(incomplete_details, "reason", None)
                    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                elif self.api_mode == "anthropic_messages":
                    stop_reason_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}
                    finish_reason = stop_reason_map.get(response.stop_reason, "stop")
                else:
                    finish_reason = response.choices[0].finish_reason

                _length_result = self._handle_finish_reason_length(
                    response, finish_reason, messages,
                    conversation_history, api_call_count,
                    effective_task_id,
                    length_continue_retries,
                    truncated_response_prefix,
                    truncated_tool_call_retries,
                    restart_with_length_continuation,
                )
                if _length_result is not None:
                    if _length_result["action"] == "return":
                        return _length_result["payload"]
                    elif _length_result["action"] == "break":
                        restart_with_length_continuation = _length_result.get("restart_with_length_continuation", False)
                        length_continue_retries = _length_result.get("length_continue_retries", length_continue_retries)
                        truncated_response_prefix = _length_result.get("truncated_response_prefix", truncated_response_prefix)
                        break
                    elif _length_result["action"] == "continue":
                        truncated_tool_call_retries = _length_result.get("truncated_tool_call_retries", truncated_tool_call_retries)
                        continue

                # Track token usage and cost
                self._track_usage_and_cost(response, api_duration, api_call_count)
                has_retried_429 = False  # Reset on success
                self._touch_activity(f"API call #{api_call_count} completed")
                break  # Success, exit retry loop

            except InterruptedError:
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if self.thinking_callback:
                    self.thinking_callback("")
                api_elapsed = time.time() - api_start_time
                self._vprint(f"{self.log_prefix}⚡ Interrupted during API call.", force=True)
                self._persist_session(messages, conversation_history)
                interrupted = True
                final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                break

            except Exception as api_error:
                # Stop spinner before printing error messages
                if thinking_spinner:
                    thinking_spinner.stop("(╥_╥) error, retrying...")
                    thinking_spinner = None
                if self.thinking_callback:
                    self.thinking_callback("")

                # -----------------------------------------------------------
                # UnicodeEncodeError recovery.  Two common causes:
                #   1. Lone surrogates (U+D800..U+DFFF) from clipboard paste
                #      (Google Docs, rich-text editors) — sanitize and retry.
                #   2. ASCII codec on systems with LANG=C or non-UTF-8 locale
                #      (e.g. Chromebooks) — any non-ASCII character fails.
                #      Detect via the error message mentioning 'ascii' codec.
                # We sanitize messages in-place and may retry twice:
                # first to strip surrogates, then once more for pure
                # ASCII-only locale sanitization if needed.
                # -----------------------------------------------------------
                if isinstance(api_error, UnicodeEncodeError) and getattr(self, '_unicode_sanitization_passes', 0) < 2:
                    _err_str = str(api_error).lower()
                    _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                    _surrogates_found = _sanitize_messages_surrogates(messages)
                    if _surrogates_found:
                        self._unicode_sanitization_passes += 1
                        self._vprint(
                            f"{self.log_prefix}⚠️  Stripped invalid surrogate characters from messages. Retrying...",
                            force=True,
                        )
                        continue
                    if _is_ascii_codec:
                        # ASCII codec: the system encoding can't handle
                        # non-ASCII characters at all. Sanitize all
                        # non-ASCII content from messages and retry.
                        if _sanitize_messages_non_ascii(messages):
                            self._unicode_sanitization_passes += 1
                            self._vprint(
                                f"{self.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from messages. Retrying...",
                                force=True,
                            )
                            continue
                    # Nothing to sanitize in messages — might be in system
                    # prompt or prefill. Fall through to normal error path.

                status_code = getattr(api_error, "status_code", None)
                error_context = self._extract_api_error_context(api_error)

                # ── Classify the error for structured recovery decisions ──
                _compressor = getattr(self, "context_compressor", None)
                _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                classified = classify_api_error(
                    api_error,
                    provider=getattr(self, "provider", "") or "",
                    model=getattr(self, "model", "") or "",
                    approx_tokens=approx_tokens,
                    context_length=_ctx_len,
                    num_messages=len(api_messages) if api_messages else 0,
                )
                logger.debug(
                    "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                    classified.reason.value, classified.status_code,
                    classified.retryable, classified.should_compress,
                    classified.should_rotate_credential, classified.should_fallback,
                )

                _cred_result = self._handle_credential_recovery(
                    status_code, has_retried_429, classified,
                    error_context, codex_auth_retry_attempted,
                    nous_auth_retry_attempted, anthropic_auth_retry_attempted,
                )
                has_retried_429 = _cred_result["has_retried_429"]
                codex_auth_retry_attempted = _cred_result["codex_auth_retry_attempted"]
                nous_auth_retry_attempted = _cred_result["nous_auth_retry_attempted"]
                anthropic_auth_retry_attempted = _cred_result["anthropic_auth_retry_attempted"]
                if _cred_result["should_continue"]:
                    continue

                # ── Thinking block signature recovery ─────────────────
                # Anthropic signs thinking blocks against the full turn
                # content.  Any upstream mutation (context compression,
                # session truncation, message merging) invalidates the
                # signature → HTTP 400.  Recovery: strip reasoning_details
                # from all messages so the next retry sends no thinking
                # blocks at all.  One-shot — don't retry infinitely.
                if (
                    classified.reason == FailoverReason.thinking_signature
                    and not thinking_sig_retry_attempted
                ):
                    thinking_sig_retry_attempted = True
                    for _m in messages:
                        if isinstance(_m, dict):
                            _m.pop("reasoning_details", None)
                    self._vprint(
                        f"{self.log_prefix}⚠️  Thinking block signature invalid — "
                        f"stripped all thinking blocks, retrying...",
                        force=True,
                    )
                    logging.warning(
                        "%sThinking block signature recovery: stripped "
                        "reasoning_details from %d messages",
                        self.log_prefix, len(messages),
                    )
                    continue

                error_type, error_msg, _error_summary = self._log_api_error_diagnostics(
                    api_error, retry_count, max_retries,
                    api_start_time, status_code, api_messages,
                    approx_tokens,
                )

                # Check for interrupt before deciding to retry
                if self._interrupt_requested:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                    self._persist_session(messages, conversation_history)
                    self.clear_interrupt()
                    return {
                        "final_response": f"Operation interrupted: handling API error ({error_type}: {self._clean_error_message(str(api_error))}).",
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "interrupted": True,
                    }

                # Check for 413 payload-too-large BEFORE generic 4xx handler.
                # A 413 is a payload-size error — the correct response is to
                # compress history and retry, not abort immediately.
                status_code = getattr(api_error, "status_code", None)

                _lct_result = self._handle_long_context_tier(
                    classified, messages, conversation_history,
                    compression_attempts, max_compression_attempts,
                    system_message, effective_task_id, approx_tokens,
                )
                if _lct_result is not None:
                    if _lct_result["action"] == "break":
                        restart_with_compressed_messages = _lct_result.get("restart_with_compressed_messages", False)
                        compression_attempts = _lct_result.get("compression_attempts", compression_attempts)
                        conversation_history = _lct_result.get("conversation_history", conversation_history)
                        break

                # Eager fallback for rate-limit errors (429 or quota exhaustion).
                # When a fallback model is configured, switch immediately instead
                # of burning through retries with exponential backoff -- the
                # primary provider won't recover within the retry window.
                is_rate_limited = classified.reason in (
                    FailoverReason.rate_limit,
                    FailoverReason.billing,
                )
                if is_rate_limited and self._fallback_index < len(self._fallback_chain):
                    # Don't eagerly fallback if credential pool rotation may
                    # still recover.  The pool's retry-then-rotate cycle needs
                    # at least one more attempt to fire — jumping to a fallback
                    # provider here short-circuits it.
                    pool = self._credential_pool
                    pool_may_recover = pool is not None and pool.has_available()
                    if not pool_may_recover:
                        self._emit_status("⚠️ Rate limited — switching to fallback provider...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                _ptl_result = self._handle_payload_too_large(
                    classified, messages, conversation_history,
                    api_call_count, approx_tokens,
                    compression_attempts, max_compression_attempts,
                    system_message, effective_task_id,
                )
                if _ptl_result is not None:
                    if _ptl_result["action"] == "return":
                        return _ptl_result["payload"]
                    elif _ptl_result["action"] == "break":
                        restart_with_compressed_messages = _ptl_result.get("restart_with_compressed_messages", False)
                        compression_attempts = _ptl_result.get("compression_attempts", compression_attempts)
                        conversation_history = _ptl_result.get("conversation_history", conversation_history)
                        break

                # Handle context overflow errors
                _ctx_result = self._handle_context_overflow_error(
                    classified, error_msg, messages,
                    conversation_history, api_call_count,
                    approx_tokens, compression_attempts,
                    max_compression_attempts, system_message,
                    effective_task_id,
                )
                if _ctx_result is not None:
                    if _ctx_result["action"] == "return":
                        return _ctx_result["payload"]
                    elif _ctx_result["action"] == "break":
                        restart_with_compressed_messages = _ctx_result.get("restart_with_compressed_messages", False)
                        compression_attempts = _ctx_result.get("compression_attempts", compression_attempts)
                        conversation_history = _ctx_result.get("conversation_history", conversation_history)
                        break
                    elif _ctx_result["action"] == "continue":
                        compression_attempts = _ctx_result.get("compression_attempts", compression_attempts)
                        conversation_history = _ctx_result.get("conversation_history", conversation_history)
                        continue

                is_context_length_error = (
                    classified.reason == FailoverReason.context_overflow
                )
                _nr_result = self._handle_non_retryable_error(
                    api_error, classified, status_code, messages,
                    conversation_history, api_call_count,
                    api_messages, approx_tokens, api_kwargs,
                    is_context_length_error, primary_recovery_attempted,
                    compression_attempts,
                )
                if _nr_result is not None:
                    if _nr_result["action"] == "return":
                        return _nr_result["payload"]
                    elif _nr_result["action"] == "continue":
                        retry_count = _nr_result.get("retry_count", retry_count)
                        compression_attempts = _nr_result.get("compression_attempts", compression_attempts)
                        primary_recovery_attempted = _nr_result.get("primary_recovery_attempted", primary_recovery_attempted)
                        continue

                # Handle retries exhausted
                _exhaust_result = self._handle_retries_exhausted(
                    api_error, error_msg, retry_count, max_retries,
                    messages, conversation_history, api_call_count,
                    api_messages, approx_tokens, api_kwargs,
                    is_rate_limited, primary_recovery_attempted,
                    compression_attempts,
                )
                if _exhaust_result is not None:
                    if _exhaust_result["action"] == "return":
                        return _exhaust_result["payload"]
                    elif _exhaust_result["action"] == "continue":
                        retry_count = _exhaust_result.get("retry_count", retry_count)
                        compression_attempts = _exhaust_result.get("compression_attempts", compression_attempts)
                        primary_recovery_attempted = _exhaust_result.get("primary_recovery_attempted", primary_recovery_attempted)
                        continue

                # For rate limits, respect the Retry-After header if present
                _retry_after = None
                if is_rate_limited:
                    _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                    if _resp_headers and hasattr(_resp_headers, "get"):
                        _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                        if _ra_raw:
                            try:
                                _retry_after = min(int(_ra_raw), 120)  # Cap at 2 minutes
                            except (TypeError, ValueError):
                                pass
                wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                if is_rate_limited:
                    self._emit_status(f"⏱️ Rate limit reached. Waiting {wait_time}s before retry (attempt {retry_count + 1}/{max_retries})...")
                else:
                    self._emit_status(f"⏳ Retrying in {wait_time}s (attempt {retry_count}/{max_retries})...")
                logger.warning(
                    "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                    wait_time,
                    retry_count,
                    max_retries,
                    self._client_log_context(),
                    api_error,
                )
                # Sleep in small increments so we can respond to interrupts quickly
                # instead of blocking the entire wait_time in one sleep() call
                sleep_end = time.time() + wait_time
                while time.time() < sleep_end:
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    time.sleep(0.2)  # Check interrupt every 200ms

        # If the API call was interrupted, skip response processing
        return {
            'response': response,
            'finish_reason': finish_reason,
            'retry_count': retry_count,
            'restart_with_compressed_messages': restart_with_compressed_messages,
            'restart_with_length_continuation': restart_with_length_continuation,
            'interrupted': interrupted if 'interrupted' in dir() else False,
            'thinking_spinner': thinking_spinner,
            'api_kwargs': api_kwargs,
        }

    def _process_response(self, response, finish_reason, messages, api_messages,
                          api_call_count, approx_tokens, conversation_history,
                          _depth_nudge_count, _turn_exit_reason):
        """Process the API response: normalize, handle tool calls, or return final text.
        
        Returns:
            tuple: (action, result, _depth_nudge_count, _turn_exit_reason)
                action: 'continue' | 'break'
                result: final_response text (for break) or None (for continue)
        """
        assistant_message, finish_reason = self._normalize_api_response(response)
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
            _assistant_text = assistant_message.content or ""
            _invoke_hook(
                "post_api_request",
                task_id=effective_task_id,
                session_id=self.session_id or "",
                platform=self.platform or "",
                model=self.model,
                provider=self.provider,
                base_url=self.base_url,
                api_mode=self.api_mode,
                api_call_count=api_call_count,
                api_duration=api_duration,
                finish_reason=finish_reason,
                message_count=len(api_messages),
                response_model=getattr(response, "model", None),
                usage=self._usage_summary_for_api_request_hook(response),
                assistant_content_chars=len(_assistant_text),
                assistant_tool_call_count=len(_assistant_tool_calls),
            )
        except Exception:
            pass

        # Handle assistant response
        if assistant_message.content and not self.quiet_mode:
            if self.verbose_logging:
                self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
            else:
                self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

        # Notify progress callback of model's thinking (used by subagent
        # delegation to relay the child's reasoning to the parent display).
        if (assistant_message.content and self.tool_progress_callback):
            _think_text = assistant_message.content.strip()
            # Strip reasoning XML tags that shouldn't leak to parent display
            _think_text = re.sub(
                r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
            ).strip()
            # For subagents: relay first line to parent display (existing behaviour).
            # For all agents with a structured callback: emit reasoning.available event.
            first_line = _think_text.split('\n')[0][:80] if _think_text else ""
            if first_line and getattr(self, '_delegate_depth', 0) > 0:
                try:
                    self.tool_progress_callback("_thinking", first_line)
                except Exception:
                    pass
            elif _think_text:
                try:
                    self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                except Exception:
                    pass

        # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
        # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
        if has_incomplete_scratchpad(assistant_message.content or ""):
            self._incomplete_scratchpad_retries += 1

            self._vprint(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")

            if self._incomplete_scratchpad_retries <= 2:
                self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                # Don't add the broken message, just retry
                return ('continue', None, _depth_nudge_count, _turn_exit_reason)
            else:
                # Max retries - discard this turn and save as partial
                self._vprint(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                self._incomplete_scratchpad_retries = 0

                rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                self._cleanup_task_resources(effective_task_id)
                self._persist_session(messages, conversation_history)

                return {
                    "final_response": None,
                    "messages": rolled_back_messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                }

        # Reset incomplete scratchpad counter on clean response
        self._incomplete_scratchpad_retries = 0

        _codex_result = self._handle_codex_incomplete(
            assistant_message, finish_reason, messages,
            conversation_history, api_call_count,
            _depth_nudge_count, _turn_exit_reason,
        )
        if _codex_result is not None:
            return _codex_result

        # Check for tool calls
        if assistant_message.tool_calls:
            if not self.quiet_mode:
                self._vprint(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")

            if self.verbose_logging:
                for tc in assistant_message.tool_calls:
                    logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")

            # Validate tool calls (names and JSON arguments)
            _tool_val_result = self._validate_tool_calls(
                assistant_message, finish_reason, messages,
                conversation_history, api_call_count,
                _depth_nudge_count, _turn_exit_reason,
            )
            if _tool_val_result is not None:
                return _tool_val_result

            return self._execute_tool_call_turn(
                assistant_message, finish_reason, messages,
                conversation_history, api_call_count,
                effective_task_id, system_message,
                _depth_nudge_count, _turn_exit_reason,
            )

        else:
            # Handle final text response (no tool calls)
            return self._handle_final_text_response(
                assistant_message, finish_reason, messages,
                conversation_history, api_call_count,
                _depth_nudge_count, _turn_exit_reason,
            )
        # Default: continue the loop
        return ('continue', None, _depth_nudge_count, _turn_exit_reason)

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling until completion.

        Args:
            user_message (str): The user's message/question
            system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
            conversation_history (List[Dict]): Previous conversation messages (optional)
            task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)
            stream_callback: Optional callback invoked with each text delta during streaming.
                Used by the TTS pipeline to start audio generation before the full response.
                When None (default), API calls use the standard non-streaming path.
            persist_user_message: Optional clean user message to store in
                transcripts/history when user_message contains API-only
                synthetic prefixes.
                    or queuing follow-up prefetch work.

        Returns:
            Dict: Complete conversation result with final response and message history
        """
        # ── Pre-loop setup ──
        (
            messages, original_user_message, current_turn_user_idx,
            effective_task_id, _should_review_memory, conversation_history,
        ) = self._prepare_conversation_turn(
            user_message, system_message, conversation_history,
            task_id, stream_callback, persist_user_message,
        )

        active_system_prompt = self._resolve_system_prompt(
            system_message, conversation_history,
        )


        messages, active_system_prompt, conversation_history = self._preflight_compress(
            messages, active_system_prompt, conversation_history,
            system_message, effective_task_id,
        )

        # ── Plugin hook: pre_llm_call ──
        _plugin_user_context = self._invoke_pre_llm_plugins(
            original_user_message, messages, conversation_history,
        )

        # Main conversation loop
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_tool_call_retries = 0
        truncated_response_prefix = ""
        compression_attempts = 0
        _turn_exit_reason = "unknown"  # Diagnostic: why the loop ended
        
        # Record the execution thread so interrupt()/clear_interrupt() can
        # scope the tool-level interrupt signal to THIS agent's thread only.
        # Must be set before clear_interrupt() which uses it.
        self._execution_thread_id = threading.current_thread().ident

        # Clear any stale interrupt state at start
        self.clear_interrupt()

        # External memory provider: prefetch once before the tool loop.
        # Reuse the cached result on every iteration to avoid re-calling
        # prefetch_all() on each tool call (10 tool calls = 10x latency + cost).
        # Use original_user_message (clean input) — user_message may contain
        # injected skill content that bloats / breaks provider queries.
        _ext_prefetch_cache = ""
        if self._memory_manager:
            try:
                _query = original_user_message if isinstance(original_user_message, str) else ""
                _ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""
            except Exception:
                pass

        while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) or self._budget_grace_call:
            # Reset per-turn checkpoint dedup so each iteration can take one snapshot
            if api_call_count == 0:
                _depth_nudge_count = 0
            self._checkpoint_mgr.new_turn()

            # Check for interrupt request (e.g., user sent new message)
            if self._interrupt_requested:
                interrupted = True
                _turn_exit_reason = "interrupted_by_user"
                if not self.quiet_mode:
                    self._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1
            self._api_call_count = api_call_count
            self._touch_activity(f"starting API call #{api_call_count}")

            # Grace call: the budget is exhausted but we gave the model one
            # more chance.  Consume the grace flag so the loop exits after
            # this iteration regardless of outcome.
            if self._budget_grace_call:
                self._budget_grace_call = False
            elif not self.iteration_budget.consume():
                _turn_exit_reason = "budget_exhausted"
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
                break

            # Fire step_callback for gateway hooks (agent:step event)
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _idx, _m in enumerate(reversed(messages)):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            _fwd_start = len(messages) - _idx
                            _results_by_id = {}
                            for _tm in messages[_fwd_start:]:
                                if _tm.get("role") != "tool":
                                    break
                                _tcid = _tm.get("tool_call_id")
                                if _tcid:
                                    _results_by_id[_tcid] = _tm.get("content", "")
                            prev_tools = [
                                {
                                    "name": tc["function"]["name"],
                                    "result": _results_by_id.get(tc.get("id")),
                                }
                                for tc in _m["tool_calls"]
                                if isinstance(tc, dict)
                            ]
                            break
                    self.step_callback(api_call_count, prev_tools)
                except Exception as _step_err:
                    logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

            # Track tool-calling iterations for skill nudge.
            # Counter resets whenever skill_manage is actually used.
            if (self._skill_nudge_interval > 0
                    and "skill_manage" in self.valid_tool_names):
                self._iters_since_skill += 1
            
            api_messages, approx_tokens, total_chars = self._prepare_api_messages(
                messages, current_turn_user_idx,
                _ext_prefetch_cache, _plugin_user_context,
                active_system_prompt,
            )

            
            # Thinking spinner for quiet mode (animated during API call)
            thinking_spinner = None
            
            if not self.quiet_mode:
                self._vprint(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
                self._vprint(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
                self._vprint(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            else:
                # Animated thinking spinner in quiet mode
                face = random.choice(KawaiiSpinner.KAWAII_THINKING)
                verb = random.choice(KawaiiSpinner.THINKING_VERBS)
                if self.thinking_callback:
                    # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                    # (works in both streaming and non-streaming modes)
                    self.thinking_callback(f"{face} {verb}...")
                elif not self._has_stream_consumers() and self._should_start_quiet_spinner():
                    # Raw KawaiiSpinner only when no streaming consumers and the
                    # spinner output has a safe sink.
                    spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                    thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=self._print_fn)
                    thinking_spinner.start()
            
            # Log request details if verbose
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            primary_recovery_attempted = False
            max_compression_attempts = 3
            codex_auth_retry_attempted=False
            anthropic_auth_retry_attempted=False
            nous_auth_retry_attempted=False
            thinking_sig_retry_attempted = False
            has_retried_429 = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # Guard against UnboundLocalError if all retries fail
            api_kwargs = None  # Guard against UnboundLocalError in except handler

            _retry_result = self._execute_api_with_retry(
                api_messages=api_messages, messages=messages, api_kwargs=api_kwargs,
                api_start_time=api_start_time, retry_count=retry_count, max_retries=max_retries,
                primary_recovery_attempted=primary_recovery_attempted,
                max_compression_attempts=max_compression_attempts,
                thinking_sig_retry_attempted=thinking_sig_retry_attempted,
                has_retried_429=has_retried_429,
                restart_with_compressed_messages=restart_with_compressed_messages,
                restart_with_length_continuation=restart_with_length_continuation,
                finish_reason=finish_reason, response=response,
                thinking_spinner=thinking_spinner, approx_tokens=approx_tokens,
                total_chars=total_chars, api_call_count=api_call_count,
                conversation_history=conversation_history,
                active_system_prompt=active_system_prompt
            )
            response = _retry_result['response']
            finish_reason = _retry_result['finish_reason']
            retry_count = _retry_result['retry_count']
            restart_with_compressed_messages = _retry_result['restart_with_compressed_messages']
            restart_with_length_continuation = _retry_result['restart_with_length_continuation']
            interrupted = _retry_result.get('interrupted', False)
            thinking_spinner = _retry_result['thinking_spinner']
            api_kwargs = _retry_result['api_kwargs']

            if interrupted:
                _turn_exit_reason = "interrupted_during_api_call"
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                # Count compression restarts toward the retry limit to prevent
                # infinite loops when compression reduces messages but not enough
                # to fit the context window.
                retry_count += 1
                restart_with_compressed_messages = False
                continue

            if restart_with_length_continuation:
                continue

            # Guard: if all retries exhausted without a successful response
            # (e.g. repeated context-length errors that exhausted retry_count),
            # the `response` variable is still None. Break out cleanly.
            if response is None:
                _turn_exit_reason = "all_retries_exhausted_no_response"
                print(f"{self.log_prefix}❌ All API retries exhausted with no successful response.")
                self._persist_session(messages, conversation_history)
                break

            try:
                _action, _result, _depth_nudge_count, _turn_exit_reason = self._process_response(
                    response, finish_reason, messages, api_messages,
                    api_call_count, approx_tokens, conversation_history,
                    _depth_nudge_count, _turn_exit_reason
                )
                if _action == 'break':
                    final_response = _result
                    break
                # else: continue to next iteration
                continue
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    logger.error(error_msg)
                
                logger.debug("Outer loop error in API call #%d", api_call_count, exc_info=True)
                
                # If an assistant message with tool_calls was already appended,
                # the API expects a role="tool" result for every tool_call_id.
                # Fill in error results for any that weren't answered yet.
                for idx in range(len(messages) - 1, -1, -1):
                    msg = messages[idx]
                    if not isinstance(msg, dict):
                        break
                    if msg.get("role") == "tool":
                        continue
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        answered_ids = {
                            m["tool_call_id"]
                            for m in messages[idx + 1:]
                            if isinstance(m, dict) and m.get("role") == "tool"
                        }
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict): continue
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                messages.append(err_msg)
                    break
                
                # Non-tool errors don't need a synthetic message injected.
                # The error is already printed to the user (line above), and
                # the retry loop continues.  Injecting a fake user/assistant
                # message pollutes history, burns tokens, and risks violating
                # role-alternation invariants.

                # If we're near the limit, break to avoid infinite loops
                if api_call_count >= self.max_iterations - 1:
                    _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    # Append as assistant so the history stays valid for
                    # session resume (avoids consecutive user messages).
                    messages.append({"role": "assistant", "content": final_response})
                    break
        
        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ) and not self._budget_exhausted_injected:
            # Budget exhausted but we haven't tried asking the model to
            # summarise yet.  Inject a user message and give it one grace
            # API call to produce a text response.
            self._budget_exhausted_injected = True
            self._budget_grace_call = True
            _grace_msg = (
                "Your tool budget ran out. Please give me the information "
                "or actions you've completed so far."
            )
            messages.append({"role": "user", "content": _grace_msg})
            self._emit_status(
                f"⚠️ Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                "— asking model to summarise"
            )
            if not self.quiet_mode:
                self._safe_print(
                    f"\n⚠️  Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                    "— requesting summary..."
                )

        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ) and not self._budget_grace_call:
            _turn_exit_reason = f"max_iterations_reached({api_call_count}/{self.max_iterations})"
            if self.iteration_budget.remaining <= 0 and not self.quiet_mode:
                print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
            final_response = self._handle_max_iterations(messages, api_call_count)
        
        # Determine if conversation completed successfully
        completed = final_response is not None and api_call_count < self.max_iterations

        # Save trajectory if enabled
        self._save_trajectory(messages, user_message, completed)

        # Clean up VM and browser for this task after conversation completes
        self._cleanup_task_resources(effective_task_id)

        # Persist session to both JSON log and SQLite
        self._persist_session(messages, conversation_history)

        # ── Post-loop finalization ──
        return self._finalize_conversation_turn(
            messages, final_response, api_call_count, interrupted,
            original_user_message, _should_review_memory,
            conversation_history,
        )


    # ── Extracted sub-methods (Phase 4 Round 6) ──────────────


    def _log_api_error_diagnostics(self, api_error, retry_count,
                                    max_retries, api_start_time,
                                    status_code, api_messages,
                                    approx_tokens):
        """Log detailed error diagnostics for a failed API call.
        
        Logs error type, provider info, endpoint, error summary,
        and actionable hints (e.g. OpenRouter tool support).
        Pure logging — no control flow side effects.
        
        Returns:
            tuple: (error_type, error_msg, _error_summary)
        """
        retry_count += 1
        elapsed_time = time.time() - api_start_time

        error_type = type(api_error).__name__
        error_msg = str(api_error).lower()
        _error_summary = self._summarize_api_error(api_error)
        logger.warning(
            "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
            retry_count,
            max_retries,
            error_type,
            self._client_log_context(),
            _error_summary,
        )

        _provider = getattr(self, "provider", "unknown")
        _base = getattr(self, "base_url", "unknown")
        _model = getattr(self, "model", "unknown")
        _status_code_str = f" [HTTP {status_code}]" if status_code else ""
        self._vprint(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}", force=True)
        self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
        self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
        self._vprint(f"{self.log_prefix}   📝 Error: {_error_summary}", force=True)
        if status_code and status_code < 500:
            _err_body = getattr(api_error, "body", None)
            _err_body_str = str(_err_body)[:300] if _err_body else None
            if _err_body_str:
                self._vprint(f"{self.log_prefix}   📋 Details: {_err_body_str}", force=True)
        self._vprint(f"{self.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

        # Actionable hint for OpenRouter "no tool endpoints" error.
        # This fires regardless of whether fallback succeeds — the
        # user needs to know WHY their model failed so they can fix
        # their provider routing, not just silently fall back.
        if (
            self._is_openrouter_url()
            and "support tool use" in error_msg
        ):
            self._vprint(
                f"{self.log_prefix}   💡 No OpenRouter providers for {_model} support tool calling with your current settings.",
                force=True,
            )
            if self.providers_allowed:
                self._vprint(
                    f"{self.log_prefix}      Your provider_routing.only restriction is filtering out tool-capable providers.",
                    force=True,
                )
                self._vprint(
                    f"{self.log_prefix}      Try removing the restriction or adding providers that support tools for this model.",
                    force=True,
                )
            self._vprint(
                f"{self.log_prefix}      Check which providers support tools: https://openrouter.ai/models/{_model}",
                force=True,
            )

        return error_type, error_msg, _error_summary


    def _handle_long_context_tier(self, classified, messages,
                                   conversation_history,
                                   compression_attempts,
                                   max_compression_attempts,
                                   system_message, effective_task_id,
                                   approx_tokens):
        """Handle Anthropic long-context tier gate errors.
        
        When Anthropic returns 429 for long-context tier, reduce context
        to 200k and compress. This is a subscription limitation, not
        a transient rate limit.
        
        Returns:
            dict with action="break" and state, or None to fall through.
        """
        # ── Anthropic Sonnet long-context tier gate ───────────
        # Anthropic returns HTTP 429 "Extra usage is required for
        # long context requests" when a Claude Max (or similar)
        # subscription doesn't include the 1M-context tier.  This
        # is NOT a transient rate limit — retrying or switching
        # credentials won't help.  Reduce context to 200k (the
        # standard tier) and compress.
        if classified.reason == FailoverReason.long_context_tier:
            _reduced_ctx = 200000
            compressor = self.context_compressor
            old_ctx = compressor.context_length
            if old_ctx > _reduced_ctx:
                compressor.update_model(
                    model=self.model,
                    context_length=_reduced_ctx,
                    base_url=self.base_url,
                    api_key=getattr(self, "api_key", ""),
                    provider=self.provider,
                )
                # Context probing flags — only set on built-in
                # compressor (plugin engines manage their own).
                if hasattr(compressor, "_context_probed"):
                    compressor._context_probed = True
                    # Don't persist — this is a subscription-tier
                    # limitation, not a model capability.  If the
                    # user later enables extra usage the 1M limit
                    # should come back automatically.
                    compressor._context_probe_persistable = False
                self._vprint(
                    f"{self.log_prefix}⚠️  Anthropic long-context tier "
                    f"requires extra usage — reducing context: "
                    f"{old_ctx:,} → {_reduced_ctx:,} tokens",
                    force=True,
                )

            compression_attempts += 1
            if compression_attempts <= max_compression_attempts:
                original_len = len(messages)
                messages, active_system_prompt = self._compress_context(
                    messages, system_message,
                    approx_tokens=approx_tokens,
                    task_id=effective_task_id,
                )
                # Compression created a new session — clear history
                # so _flush_messages_to_session_db writes compressed
                # messages to the new session, not skipping them.
                conversation_history = None
                if len(messages) < original_len or old_ctx > _reduced_ctx:
                    self._emit_status(
                        f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                        f"(was {old_ctx:,}), retrying..."
                    )
                    time.sleep(2)
                    restart_with_compressed_messages = True
                    return {
                        "action": "break",
                        "restart_with_compressed_messages": True,
                        "compression_attempts": compression_attempts,
                        "conversation_history": conversation_history,
                    }
            # Fall through to normal error handling if compression
            # is exhausted or didn't help.

        return None  # Not a long-context tier error, or compression exhausted

    # ── Extracted sub-methods (Phase 4 Round 5) ──────────────


    def _handle_credential_recovery(self, status_code, has_retried_429,
                                     classified, error_context,
                                     codex_auth_retry_attempted,
                                     nous_auth_retry_attempted,
                                     anthropic_auth_retry_attempted):
        """Attempt credential recovery for authentication errors.
        
        Tries credential pool rotation, then provider-specific 401
        recovery (Codex, Nous, Anthropic).
        
        Returns:
            dict with:
                should_continue (bool): True if recovery succeeded, retry the request
                has_retried_429 (bool): Updated 429 retry flag
                codex_auth_retry_attempted (bool): Updated flag
                nous_auth_retry_attempted (bool): Updated flag
                anthropic_auth_retry_attempted (bool): Updated flag
        """
        recovered_with_pool, has_retried_429 = self._recover_with_credential_pool(
            status_code=status_code,
            has_retried_429=has_retried_429,
            classified_reason=classified.reason,
            error_context=error_context,
        )
        if recovered_with_pool:
            return {
                "should_continue": True,
                "has_retried_429": has_retried_429,
                "codex_auth_retry_attempted": codex_auth_retry_attempted,
                "nous_auth_retry_attempted": nous_auth_retry_attempted,
                "anthropic_auth_retry_attempted": anthropic_auth_retry_attempted,
            }
        if (
            self.api_mode == "codex_responses"
            and self.provider == "openai-codex"
            and status_code == 401
            and not codex_auth_retry_attempted
        ):
            codex_auth_retry_attempted = True
            if self._try_refresh_codex_client_credentials(force=True):
                self._vprint(f"{self.log_prefix}🔐 Codex auth refreshed after 401. Retrying request...")
                return {
                    "should_continue": True,
                    "has_retried_429": has_retried_429,
                    "codex_auth_retry_attempted": codex_auth_retry_attempted,
                    "nous_auth_retry_attempted": nous_auth_retry_attempted,
                    "anthropic_auth_retry_attempted": anthropic_auth_retry_attempted,
                }
        if (
            self.api_mode == "chat_completions"
            and self.provider == "nous"
            and status_code == 401
            and not nous_auth_retry_attempted
        ):
            nous_auth_retry_attempted = True
            if self._try_refresh_nous_client_credentials(force=True):
                print(f"{self.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                return {
                    "should_continue": True,
                    "has_retried_429": has_retried_429,
                    "codex_auth_retry_attempted": codex_auth_retry_attempted,
                    "nous_auth_retry_attempted": nous_auth_retry_attempted,
                    "anthropic_auth_retry_attempted": anthropic_auth_retry_attempted,
                }
        if (
            self.api_mode == "anthropic_messages"
            and status_code == 401
            and hasattr(self, '_anthropic_api_key')
            and not anthropic_auth_retry_attempted
        ):
            anthropic_auth_retry_attempted = True
            from agent.anthropic_adapter import _is_oauth_token
            if self._try_refresh_anthropic_client_credentials():
                print(f"{self.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                return {
                    "should_continue": True,
                    "has_retried_429": has_retried_429,
                    "codex_auth_retry_attempted": codex_auth_retry_attempted,
                    "nous_auth_retry_attempted": nous_auth_retry_attempted,
                    "anthropic_auth_retry_attempted": anthropic_auth_retry_attempted,
                }
            # Credential refresh didn't help — show diagnostic info
            key = self._anthropic_api_key
            auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
            print(f"{self.log_prefix}🔐 Anthropic 401 — authentication failed.")
            print(f"{self.log_prefix}   Auth method: {auth_method}")
            print(f"{self.log_prefix}   Token prefix: {key[:12]}..." if key and len(key) > 12 else f"{self.log_prefix}   Token: (empty or short)")
            print(f"{self.log_prefix}   Troubleshooting:")
            from hermes_constants import display_hermes_home as _dhh_fn
            _dhh = _dhh_fn()
            print(f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
            print(f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
            print(f"{self.log_prefix}     • For API keys: verify at https://console.anthropic.com/settings/keys")
            print(f"{self.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
            print(f"{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
            print(f"{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

        return {
            "should_continue": False,
            "has_retried_429": has_retried_429,
            "codex_auth_retry_attempted": codex_auth_retry_attempted,
            "nous_auth_retry_attempted": nous_auth_retry_attempted,
            "anthropic_auth_retry_attempted": anthropic_auth_retry_attempted,
        }


    def _handle_non_retryable_error(self, api_error, classified,
                                     status_code, messages,
                                     conversation_history, api_call_count,
                                     api_messages, approx_tokens, api_kwargs,
                                     is_context_length_error,
                                     primary_recovery_attempted,
                                     compression_attempts):
        """Handle non-retryable client errors (4xx, validation errors).
        
        Returns:
            dict with action="return"/"continue" and payload, or None to fall through.
        """
        # Check for non-retryable client errors.  The classifier
        # already accounts for 413, 429, 529 (transient), context
        # overflow, and generic-400 heuristics.  Local validation
        # errors (ValueError, TypeError) are programming bugs.
        is_local_validation_error = (
            isinstance(api_error, (ValueError, TypeError))
            and not isinstance(api_error, UnicodeEncodeError)
        )
        is_client_error = (
            is_local_validation_error
            or (
                not classified.retryable
                and not classified.should_compress
                and classified.reason not in (
                    FailoverReason.rate_limit,
                    FailoverReason.billing,
                    FailoverReason.overloaded,
                    FailoverReason.context_overflow,
                    FailoverReason.payload_too_large,
                    FailoverReason.long_context_tier,
                    FailoverReason.thinking_signature,
                )
            )
        ) and not is_context_length_error

        if is_client_error:
            # Try fallback before aborting — a different provider
            # may not have the same issue (rate limit, auth, etc.)
            self._emit_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
            if self._try_activate_fallback():
                retry_count = 0
                compression_attempts = 0
                primary_recovery_attempted = False
                return {
                    "action": "continue",
                    "retry_count": 0,
                    "compression_attempts": 0,
                    "primary_recovery_attempted": False,
                }
            if api_kwargs is not None:
                self._dump_api_request_debug(
                    api_kwargs, reason="non_retryable_client_error", error=api_error,
                )
            self._emit_status(
                f"❌ Non-retryable error (HTTP {status_code}): "
                f"{self._summarize_api_error(api_error)}"
            )
            _provider = getattr(self, "provider", "unknown")
            _base = getattr(self, "base_url", "unknown")
            _model = getattr(self, "model", "unknown")
            self._vprint(f"{self.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
            self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
            self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
            # Actionable guidance for common auth errors
            if classified.is_auth or classified.reason == FailoverReason.billing:
                if _provider == "openai-codex" and status_code == 401:
                    self._vprint(f"{self.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                    self._vprint(f"{self.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                    self._vprint(f"{self.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                    self._vprint(f"{self.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                else:
                    self._vprint(f"{self.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                    self._vprint(f"{self.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                    self._vprint(f"{self.log_prefix}      • Does your account have access to {_model}?", force=True)
                    if "openrouter" in str(_base).lower():
                        self._vprint(f"{self.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
            else:
                self._vprint(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
            logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")
            # Skip session persistence when the error is likely
            # context-overflow related (status 400 + large session).
            # Persisting the failed user message would make the
            # session even larger, causing the same failure on the
            # next attempt. (#1630)
            if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                self._vprint(
                    f"{self.log_prefix}⚠️  Skipping session persistence "
                    f"for large failed session to prevent growth loop.",
                    force=True,
                )
            else:
                self._persist_session(messages, conversation_history)
            return {
                "final_response": None,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "failed": True,
                "error": str(api_error),
            }

        return None  # Not a non-retryable error


    def _handle_payload_too_large(self, classified, messages,
                                   conversation_history, api_call_count,
                                   approx_tokens, compression_attempts,
                                   max_compression_attempts, system_message,
                                   effective_task_id):
        """Handle 413 payload-too-large errors with compression.
        
        Returns:
            dict with action="return"/"break" and payload, or None to fall through.
        """
        is_payload_too_large = (
            classified.reason == FailoverReason.payload_too_large
        )

        if is_payload_too_large:
            compression_attempts += 1
            if compression_attempts > max_compression_attempts:
                self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                logging.error(f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                self._persist_session(messages, conversation_history)
                return {"action": "return", "payload": {
                    "messages": messages,
                    "completed": False,
                    "api_calls": api_call_count,
                    "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                    "partial": True
                }}
            self._emit_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

            original_len = len(messages)
            messages, active_system_prompt = self._compress_context(
                messages, system_message, approx_tokens=approx_tokens,
                task_id=effective_task_id,
            )
            # Compression created a new session — clear history
            # so _flush_messages_to_session_db writes compressed
            # messages to the new session, not skipping them.
            conversation_history = None

            if len(messages) < original_len:
                self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                time.sleep(2)  # Brief pause between compression retries
                restart_with_compressed_messages = True
                return {
                    "action": "break",
                    "restart_with_compressed_messages": True,
                    "compression_attempts": compression_attempts,
                    "conversation_history": conversation_history,
                }
            else:
                self._vprint(f"{self.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                logging.error(f"{self.log_prefix}413 payload too large. Cannot compress further.")
                self._persist_session(messages, conversation_history)
                return {"action": "return", "payload": {
                    "messages": messages,
                    "completed": False,
                    "api_calls": api_call_count,
                    "error": "Request payload too large (413). Cannot compress further.",
                    "partial": True
                }}

        return None  # Not payload-too-large

    # ── Extracted sub-methods (Phase 4 Round 4) ──────────────


    def _resolve_system_prompt(self, system_message, conversation_history):
        """Resolve the system prompt for this turn.
        
        Loads from session DB for continuing sessions (preserves
        Anthropic prefix cache), or builds fresh for new sessions.
        Fires on_session_start plugin hook for new sessions.
        
        Returns:
            str: The resolved system prompt.
        """
        # ── System prompt (cached per session for prefix caching) ──
        # Built once on first call, reused for all subsequent calls.
        # Only rebuilt after context compression events (which invalidate
        # the cache and reload memory from disk).
        #
        # For continuing sessions (gateway creates a fresh AIAgent per
        # message), we load the stored system prompt from the session DB
        # instead of rebuilding.  Rebuilding would pick up memory changes
        # from disk that the model already knows about (it wrote them!),
        # producing a different system prompt and breaking the Anthropic
        # prefix cache.
        if self._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and self._session_db:
                try:
                    session_row = self._session_db.get_session(self.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass  # Fall through to build fresh

            if stored_prompt:
                # Continuing session — reuse the exact system prompt from
                # the previous turn so the Anthropic cache prefix matches.
                self._cached_system_prompt = stored_prompt
            else:
                # First turn of a new session — build from scratch.
                self._cached_system_prompt = self._build_system_prompt(system_message)
                # Plugin hook: on_session_start
                # Fired once when a brand-new session is created (not on
                # continuation).  Plugins can use this to initialise
                # session-scoped state (e.g. warm a memory cache).
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_start",
                        session_id=self.session_id,
                        model=self.model,
                        platform=getattr(self, "platform", None) or "",
                    )
                except Exception as exc:
                    logger.warning("on_session_start hook failed: %s", exc)

                # Store the system prompt snapshot in SQLite
                if self._session_db:
                    try:
                        self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                    except Exception as e:
                        logger.debug("Session DB update_system_prompt failed: %s", e)

        active_system_prompt = self._cached_system_prompt
        return self._cached_system_prompt


    def _preflight_compress(self, messages, active_system_prompt,
                            conversation_history, system_message,
                            effective_task_id):
        """Check if loaded history exceeds context threshold and compress.
        
        Handles model switches to smaller context windows by compressing
        proactively rather than waiting for API errors.
        
        Returns:
            tuple: (messages, active_system_prompt, conversation_history)
        """
        # ── Preflight context compression ──
        # Before entering the main loop, check if the loaded conversation
        # history already exceeds the model's context threshold.  This handles
        # cases where a user switches to a model with a smaller context window
        # while having a large existing session — compress proactively rather
        # than waiting for an API error (which might be caught as a non-retryable
        # 4xx and abort the request entirely).
        if (
            self.compression_enabled
            and len(messages) > self.context_compressor.protect_first_n
                                + self.context_compressor.protect_last_n + 1
        ):
            # Include tool schema tokens — with many tools these can add
            # 20-30K+ tokens that the old sys+msg estimate missed entirely.
            _preflight_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=self.tools or None,
            )

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                if not self.quiet_mode:
                    self._safe_print(
                        f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                        f">= {self.context_compressor.threshold_tokens:,} threshold"
                    )
                # May need multiple passes for very large sessions with small
                # context windows (each pass summarises the middle N turns).
                for _pass in range(3):
                    _orig_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=_preflight_tokens,
                        task_id=effective_task_id,
                    )
                    if len(messages) >= _orig_len:
                        break  # Cannot compress further
                    # Compression created a new session — clear the history
                    # reference so _flush_messages_to_session_db writes ALL
                    # compressed messages to the new session's SQLite, not
                    # skipping them because conversation_history is still the
                    # pre-compression length.
                    conversation_history = None
                    # Re-estimate after compression
                    _preflight_tokens = estimate_request_tokens_rough(
                        messages,
                        system_prompt=active_system_prompt or "",
                        tools=self.tools or None,
                    )
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # Under threshold

        return messages, active_system_prompt, conversation_history


    def _prepare_api_messages(self, messages, current_turn_user_idx,
                              _ext_prefetch_cache, _plugin_user_context,
                              active_system_prompt):
        """Build API-ready messages from conversation messages.
        
        Injects ephemeral context (memory, plugins) into user message,
        adds reasoning_content for API compatibility, strips internal
        fields, applies prompt caching and sanitization.
        
        Returns:
            tuple: (api_messages, approx_tokens, total_chars)
        """
        # Prepare messages for API call
        # If we have an ephemeral system prompt, prepend it to the messages
        # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
        # However, providers like Moonshot AI require a separate 'reasoning_content' field
        # on assistant messages with tool_calls. We handle both cases here.
        api_messages = []
        for idx, msg in enumerate(messages):
            api_msg = msg.copy()

            # Inject ephemeral context into the current turn's user message.
            # Sources: memory manager prefetch + plugin pre_llm_call hooks
            # with target="user_message" (the default).  Both are
            # API-call-time only — the original message in `messages` is
            # never mutated, so nothing leaks into session persistence.
            if idx == current_turn_user_idx and msg.get("role") == "user":
                _injections = []
                if _ext_prefetch_cache:
                    _fenced = build_memory_context_block(_ext_prefetch_cache)
                    if _fenced:
                        _injections.append(_fenced)
                if _plugin_user_context:
                    _injections.append(_plugin_user_context)
                if _injections:
                    _base = api_msg.get("content", "")
                    if isinstance(_base, str):
                        api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

            # For ALL assistant messages, pass reasoning back to the API
            # This ensures multi-turn reasoning context is preserved
            if msg.get("role") == "assistant":
                reasoning_text = msg.get("reasoning")
                if reasoning_text:
                    # Add reasoning_content for API compatibility (Moonshot AI, Novita, OpenRouter)
                    api_msg["reasoning_content"] = reasoning_text

            # Remove 'reasoning' field - it's for trajectory storage only
            # We've copied it to 'reasoning_content' for the API above
            if "reasoning" in api_msg:
                api_msg.pop("reasoning")
            # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
            if "finish_reason" in api_msg:
                api_msg.pop("finish_reason")
            # Strip internal thinking-prefill marker
            api_msg.pop("_thinking_prefill", None)
            # Strip Codex Responses API fields (call_id, response_item_id) for
            # strict providers like Mistral, Fireworks, etc. that reject unknown fields.
            # Uses new dicts so the internal messages list retains the fields
            # for Codex Responses compatibility.
            if self._should_sanitize_tool_calls():
                self._sanitize_tool_calls_for_strict_api(api_msg)
            # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
            # The signature field helps maintain reasoning continuity
            api_messages.append(api_msg)

        # Build the final system message: cached prompt + ephemeral system prompt.
        # Ephemeral additions are API-call-time only (not persisted to session DB).
        # External recall context is injected into the user message, not the system
        # prompt, so the stable cache prefix remains unchanged.
        effective_system = active_system_prompt or ""
        if self.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
        # NOTE: Plugin context from pre_llm_call hooks is injected into the
        # user message (see injection block above), NOT the system prompt.
        # This is intentional — system prompt modifications break the prompt
        # cache prefix.  The system prompt is reserved for Hermes internals.
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages

        # Inject ephemeral prefill messages right after the system prompt
        # but before conversation history. Same API-call-time-only pattern.
        if self.prefill_messages:
            sys_offset = 1 if effective_system else 0
            for idx, pfm in enumerate(self.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # Apply Anthropic prompt caching for Claude models via OpenRouter.
        # Auto-detected: if model name contains "claude" and base_url is OpenRouter,
        # inject cache_control breakpoints (system + last 3 messages) to reduce
        # input token costs by ~75% on multi-turn conversations.
        if self._use_prompt_caching:
            api_messages = apply_anthropic_cache_control(api_messages, cache_ttl=self._cache_ttl, native_anthropic=(self.api_mode == 'anthropic_messages'))

        # Safety net: strip orphaned tool results / add stubs for missing
        # results before sending to the API.  Runs unconditionally — not
        # gated on context_compressor — so orphans from session loading or
        # manual message manipulation are always caught.
        api_messages = self._sanitize_api_messages(api_messages)

        # Normalize message whitespace and tool-call JSON for consistent
        # prefix matching.  Ensures bit-perfect prefixes across turns,
        # which enables KV cache reuse on local inference servers
        # (llama.cpp, vLLM, Ollama) and improves cache hit rates for
        # cloud providers.  Operates on api_messages (the API copy) so
        # the original conversation history in `messages` is untouched.
        for am in api_messages:
            if isinstance(am.get("content"), str):
                am["content"] = am["content"].strip()
        for am in api_messages:
            tcs = am.get("tool_calls")
            if not tcs:
                continue
            new_tcs = []
            for tc in tcs:
                if isinstance(tc, dict) and "function" in tc:
                    try:
                        args_obj = json.loads(tc["function"]["arguments"])
                        tc = {**tc, "function": {
                            **tc["function"],
                            "arguments": json.dumps(
                                args_obj, separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }}
                    except Exception:
                        pass
                new_tcs.append(tc)
            am["tool_calls"] = new_tcs

        # Calculate approximate request size for logging
        total_chars = sum(len(str(msg)) for msg in api_messages)
        approx_tokens = estimate_messages_tokens_rough(api_messages)
        return api_messages, approx_tokens, total_chars

    # ── Extracted sub-methods (Phase 4 Round 3) ──────────────


    def _handle_codex_incomplete(self, assistant_message, finish_reason,
                                  messages, conversation_history,
                                  api_call_count, _depth_nudge_count,
                                  _turn_exit_reason):
        """Handle Codex incomplete response with continuation retries.
        
        Returns:
            None to fall through (not codex incomplete or retries reset),
            or tuple/dict to return from _process_response.
        """
        if self.api_mode == "codex_responses" and finish_reason == "incomplete":
            self._codex_incomplete_retries += 1

            interim_msg = self._build_assistant_message(assistant_message, finish_reason)
            interim_has_content = bool((interim_msg.get("content") or "").strip())
            interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
            interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))

            if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning:
                last_msg = messages[-1] if messages else None
                # Duplicate detection: two consecutive incomplete assistant
                # messages with identical content AND reasoning are collapsed.
                # For reasoning-only messages (codex_reasoning_items differ but
                # visible content/reasoning are both empty), we also compare
                # the encrypted items to avoid silently dropping new state.
                last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                interim_codex_items = interim_msg.get("codex_reasoning_items")
                duplicate_interim = (
                    isinstance(last_msg, dict)
                    and last_msg.get("role") == "assistant"
                    and last_msg.get("finish_reason") == "incomplete"
                    and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                    and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                    and last_codex_items == interim_codex_items
                )
                if not duplicate_interim:
                    messages.append(interim_msg)
                    self._emit_interim_assistant_message(interim_msg)

            if self._codex_incomplete_retries < 3:
                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)")
                self._session_messages = messages
                self._save_session_log(messages)
                return ('continue', None, _depth_nudge_count, _turn_exit_reason)

            self._codex_incomplete_retries = 0
            self._persist_session(messages, conversation_history)
            return {
                "final_response": None,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "partial": True,
                "error": "Codex response remained incomplete after 3 continuation attempts",
            }
        elif hasattr(self, "_codex_incomplete_retries"):
            self._codex_incomplete_retries = 0

        return None  # Not codex incomplete or retries reset


    def _execute_tool_call_turn(self, assistant_message, finish_reason,
                                messages, conversation_history,
                                api_call_count, effective_task_id,
                                system_message, _depth_nudge_count,
                                _turn_exit_reason):
        """Execute tool calls and handle post-execution logic.
        
        Includes guardrails, message building, tool execution,
        context pressure warnings, and compression checks.
        
        Returns:
            tuple: ('continue', None, _depth_nudge_count, _turn_exit_reason)
        """
        # ── Post-call guardrails ──────────────────────────
        assistant_message.tool_calls = self._cap_delegate_task_calls(
            assistant_message.tool_calls
        )
        assistant_message.tool_calls = self._deduplicate_tool_calls(
            assistant_message.tool_calls
        )

        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)

        # If this turn has both content AND tool_calls, capture the content
        # as a fallback final response. Common pattern: model delivers its
        # answer and calls memory/skill tools as a side-effect in the same
        # turn. If the follow-up turn after tools is empty, we use this.
        turn_content = assistant_message.content or ""
        if turn_content and self._has_content_after_think_block(turn_content):
            self._last_content_with_tools = turn_content
            # Only mute subsequent output when EVERY tool call in
            # this turn is post-response housekeeping (memory, todo,
            # skill_manage, etc.).  If any substantive tool is present
            # (search_files, read_file, write_file, terminal, ...),
            # keep output visible so the user sees progress.
            _HOUSEKEEPING_TOOLS = frozenset({
                "memory", "todo", "skill_manage", "session_search",
            })
            _all_housekeeping = all(
                tc.function.name in _HOUSEKEEPING_TOOLS
                for tc in assistant_message.tool_calls
            )
            if _all_housekeeping and self._has_stream_consumers():
                self._mute_post_response = True
            elif self.quiet_mode:
                clean = self._strip_think_blocks(turn_content).strip()
                if clean:
                    self._vprint(f"  ┊ 💬 {clean}")

        # Pop thinking-only prefill message(s) before appending
        # (tool-call path — same rationale as the final-response path).
        _had_prefill = False
        while (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("_thinking_prefill")
        ):
            messages.pop()
            _had_prefill = True

        # Reset prefill counter when tool calls follow a prefill
        # recovery.  Without this, the counter accumulates across
        # the whole conversation — a model that intermittently
        # empties (empty → prefill → tools → empty → prefill →
        # tools) burns both prefill attempts and the third empty
        # gets zero recovery.  Resetting here treats each tool-
        # call success as a fresh start.
        if _had_prefill:
            self._thinking_prefill_retries = 0
            self._empty_content_retries = 0

        messages.append(assistant_msg)
        self._emit_interim_assistant_message(assistant_msg)

        # Close any open streaming display (response box, reasoning
        # box) before tool execution begins.  Intermediate turns may
        # have streamed early content that opened the response box;
        # flushing here prevents it from wrapping tool feed lines.
        # Only signal the display callback — TTS (_stream_callback)
        # should NOT receive None (it uses None as end-of-stream).
        if self.stream_delta_callback:
            try:
                self.stream_delta_callback(None)
            except Exception:
                pass

        self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

        # Reset per-turn retry counters after successful tool
        # execution so a single truncation doesn't poison the
        # entire conversation.
        truncated_tool_call_retries = 0

        # Signal that a paragraph break is needed before the next
        # streamed text.  We don't emit it immediately because
        # multiple consecutive tool iterations would stack up
        # redundant blank lines.  Instead, _fire_stream_delta()
        # will prepend a single "\n\n" the next time real text
        # arrives.
        self._stream_needs_break = True

        # Refund the iteration if the ONLY tool(s) called were
        # execute_code (programmatic tool calling).  These are
        # cheap RPC-style calls that shouldn't eat the budget.
        _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
        if _tc_names == {"execute_code"}:
            self.iteration_budget.refund()

        # Use real token counts from the API response to decide
        # compression.  prompt_tokens + completion_tokens is the
        # actual context size the provider reported plus the
        # assistant turn — a tight lower bound for the next prompt.
        # Tool results appended above aren't counted yet, but the
        # threshold (default 50%) leaves ample headroom; if tool
        # results push past it, the next API call will report the
        # real total and trigger compression then.
        #
        # If last_prompt_tokens is 0 (stale after API disconnect
        # or provider returned no usage data), fall back to rough
        # estimate to avoid missing compression.  Without this,
        # a session can grow unbounded after disconnects because
        # should_compress(0) never fires.  (#2153)
        _compressor = self.context_compressor
        if _compressor.last_prompt_tokens > 0:
            _real_tokens = (
                _compressor.last_prompt_tokens
                + _compressor.last_completion_tokens
            )
        else:
            _real_tokens = estimate_messages_tokens_rough(messages)

        # ── Context pressure warnings (user-facing only) ──────────
        # Notify the user (NOT the LLM) as context approaches the
        # compaction threshold.  Thresholds are relative to where
        # compaction fires, not the raw context window.
        # Does not inject into messages — just prints to CLI output
        # and fires status_callback for gateway platforms.
        # Tiered: 85% (orange) and 95% (red/critical).
        if _compressor.threshold_tokens > 0:
            _compaction_progress = _real_tokens / _compressor.threshold_tokens
            # Determine the warning tier for this progress level
            _warn_tier = 0.0
            if _compaction_progress >= 0.95:
                _warn_tier = 0.95
            elif _compaction_progress >= 0.85:
                _warn_tier = 0.85
            if _warn_tier > self._context_pressure_warned_at:
                # Class-level dedup: check if this session was already
                # warned at this tier within the cooldown window.
                _sid = self.session_id or "default"
                _last = type(self)._context_pressure_last_warned.get(_sid)
                _now = time.time()
                if _last is None or _last[0] < _warn_tier or (_now - _last[1]) >= self._CONTEXT_PRESSURE_COOLDOWN:
                    self._context_pressure_warned_at = _warn_tier
                    type(self)._context_pressure_last_warned[_sid] = (_warn_tier, _now)
                    self._emit_context_pressure(_compaction_progress, _compressor)
                    # Evict stale entries (older than 2x cooldown)
                    _cutoff = _now - self._CONTEXT_PRESSURE_COOLDOWN * 2
                    type(self)._context_pressure_last_warned = {
                        k: v for k, v in type(self)._context_pressure_last_warned.items()
                        if v[1] > _cutoff
                    }

        if self.compression_enabled and _compressor.should_compress(_real_tokens):
            self._safe_print("  ⟳ compacting context…")
            messages, active_system_prompt = self._compress_context(
                messages, system_message,
                approx_tokens=self.context_compressor.last_prompt_tokens,
                task_id=effective_task_id,
            )
            # Compression created a new session — clear history so
            # _flush_messages_to_session_db writes compressed messages
            # to the new session (see preflight compression comment).
            conversation_history = None

        # Save session log incrementally (so progress is visible even if interrupted)
        self._session_messages = messages
        self._save_session_log(messages)

        # Continue loop for next response
        return ('continue', None, _depth_nudge_count, _turn_exit_reason)

    # ── Extracted sub-methods (Phase 4 Round 2) ──────────────


    def _diagnose_invalid_response(self, response, api_duration):
        """Extract diagnostic info from an invalid API response.
        
        Returns:
            tuple: (error_msg, provider_name, failure_hint)
        """
        # Check for error field in response (some providers include this)
        error_msg = "Unknown"
        provider_name = "Unknown"
        if response and hasattr(response, 'error') and response.error:
            error_msg = str(response.error)
            # Try to extract provider from error metadata
            if hasattr(response.error, 'metadata') and response.error.metadata:
                provider_name = response.error.metadata.get('provider_name', 'Unknown')
        elif response and hasattr(response, 'message') and response.message:
            error_msg = str(response.message)

        # Try to get provider from model field (OpenRouter often returns actual model used)
        if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
            provider_name = f"model={response.model}"

        # Check for x-openrouter-provider or similar metadata
        if provider_name == "Unknown" and response:
            # Log all response attributes for debugging
            resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
            if self.verbose_logging:
                logging.debug(f"Response attributes for invalid response: {resp_attrs}")

        # Extract error code from response for contextual diagnostics
        _resp_error_code = None
        if response and hasattr(response, 'error') and response.error:
            _code_raw = getattr(response.error, 'code', None)
            if _code_raw is None and isinstance(response.error, dict):
                _code_raw = response.error.get('code')
            if _code_raw is not None:
                try:
                    _resp_error_code = int(_code_raw)
                except (TypeError, ValueError):
                    pass

        # Build a human-readable failure hint from the error code
        # and response time, instead of always assuming rate limiting.
        if _resp_error_code == 524:
            _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
        elif _resp_error_code == 504:
            _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
        elif _resp_error_code == 429:
            _failure_hint = f"rate limited by upstream provider (429)"
        elif _resp_error_code in (500, 502):
            _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
        elif _resp_error_code in (503, 529):
            _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
        elif _resp_error_code is not None:
            _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
        elif api_duration < 10:
            _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
        elif api_duration > 60:
            _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
        else:
            _failure_hint = f"response time {api_duration:.1f}s"

        return error_msg, provider_name, _failure_hint


    def _handle_finish_reason_length(self, response, finish_reason, messages,
                                      conversation_history, api_call_count,
                                      effective_task_id,
                                      length_continue_retries,
                                      truncated_response_prefix,
                                      truncated_tool_call_retries,
                                      restart_with_length_continuation):
        """Handle finish_reason="length" (truncated response).
        
        Returns:
            None if finish_reason != "length",
            or dict with "action" key and state updates.
        """
        if finish_reason == "length":
            self._vprint(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

            # ── Detect thinking-budget exhaustion ──────────────
            # When the model spends ALL output tokens on reasoning
            # and has none left for the response, continuation
            # retries are pointless.  Detect this early and give a
            # targeted error instead of wasting 3 API calls.
            _trunc_content = None
            _trunc_has_tool_calls = False
            if self.api_mode == "chat_completions":
                _trunc_msg = response.choices[0].message if (hasattr(response, "choices") and response.choices) else None
                _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False
            elif self.api_mode == "anthropic_messages":
                # Anthropic response.content is a list of blocks
                _text_parts = []
                for _blk in getattr(response, "content", []):
                    if getattr(_blk, "type", None) == "text":
                        _text_parts.append(getattr(_blk, "text", ""))
                _trunc_content = "\n".join(_text_parts) if _text_parts else None

            # A response is "thinking exhausted" only when the model
            # actually produced reasoning blocks but no visible text after
            # them.  Models that do not use <think> tags (e.g. GLM-4.7 on
            # NVIDIA Build, minimax) may return content=None or an empty
            # string for unrelated reasons — treat those as normal
            # truncations that deserve continuation retries, not as
            # thinking-budget exhaustion.
            _has_think_tags = bool(
                _trunc_content and re.search(
                    r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                    _trunc_content,
                    re.IGNORECASE,
                )
            )
            _thinking_exhausted = (
                not _trunc_has_tool_calls
                and _has_think_tags
                and (
                    (_trunc_content is not None and not self._has_content_after_think_block(_trunc_content))
                    or _trunc_content is None
                )
            )

            if _thinking_exhausted:
                _exhaust_error = (
                    "Model used all output tokens on reasoning with none left "
                    "for the response. Try lowering reasoning effort or "
                    "increasing max_tokens."
                )
                self._vprint(
                    f"{self.log_prefix}💭 Reasoning exhausted the output token budget — "
                    f"no visible response was produced.",
                    force=True,
                )
                # Return a user-friendly message as the response so
                # CLI (response box) and gateway (chat message) both
                # display it naturally instead of a suppressed error.
                _exhaust_response = (
                    "⚠️ **Thinking Budget Exhausted**\n\n"
                    "The model used all its output tokens on reasoning "
                    "and had none left for the actual response.\n\n"
                    "To fix this:\n"
                    "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                    "→ Increase the output token limit: "
                    "set `model.max_tokens` in config.yaml"
                )
                self._cleanup_task_resources(effective_task_id)
                self._persist_session(messages, conversation_history)
                return {"action": "return", "payload": {
                    "final_response": _exhaust_response,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": _exhaust_error,
                }}

            if self.api_mode == "chat_completions":
                assistant_message = response.choices[0].message
                if not assistant_message.tool_calls:
                    length_continue_retries += 1
                    interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                    messages.append(interim_msg)
                    if assistant_message.content:
                        truncated_response_prefix += assistant_message.content

                    if length_continue_retries < 3:
                        self._vprint(
                            f"{self.log_prefix}↻ Requesting continuation "
                            f"({length_continue_retries}/3)..."
                        )
                        continue_msg = {
                            "role": "user",
                            "content": (
                                "[System: Your previous response was truncated by the output "
                                "length limit. Continue exactly where you left off. Do not "
                                "restart or repeat prior text. Finish the answer directly.]"
                            ),
                        }
                        messages.append(continue_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        restart_with_length_continuation = True
                        return {"action": "break",
                                "restart_with_length_continuation": restart_with_length_continuation,
                                "length_continue_retries": length_continue_retries,
                                "truncated_response_prefix": truncated_response_prefix}

                    partial_response = self._strip_think_blocks(truncated_response_prefix).strip()
                    self._cleanup_task_resources(effective_task_id)
                    self._persist_session(messages, conversation_history)
                    return {"action": "return", "payload": {
                        "final_response": partial_response or None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Response remained truncated after 3 continuation attempts",
                    }}

            if self.api_mode == "chat_completions":
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    if truncated_tool_call_retries < 1:
                        truncated_tool_call_retries += 1
                        self._vprint(
                            f"{self.log_prefix}⚠️  Truncated tool call detected — retrying API call...",
                            force=True,
                        )
                        # Don't append the broken response to messages;
                        # just re-run the same API call from the current
                        # message state, giving the model another chance.
                        return {"action": "continue",
                                "truncated_tool_call_retries": truncated_tool_call_retries}
                    self._vprint(
                        f"{self.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                        force=True,
                    )
                    self._cleanup_task_resources(effective_task_id)
                    self._persist_session(messages, conversation_history)
                    return {"action": "return", "payload": {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Response truncated due to output length limit",
                    }}

            # If we have prior messages, roll back to last complete state
            if len(messages) > 1:
                self._vprint(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                rolled_back_messages = self._get_messages_up_to_last_assistant(messages)

                self._cleanup_task_resources(effective_task_id)
                self._persist_session(messages, conversation_history)

                return {"action": "return", "payload": {
                    "final_response": None,
                    "messages": rolled_back_messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Response truncated due to output length limit"
                }}
            else:
                # First message was truncated - mark as failed
                self._vprint(f"{self.log_prefix}❌ First response truncated - cannot recover", force=True)
                self._persist_session(messages, conversation_history)
                return {"action": "return", "payload": {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "failed": True,
                    "error": "First response truncated due to output length limit"
                }}

        return None  # Not a length finish_reason


    def _handle_context_overflow_error(self, classified, error_msg, messages,
                                        conversation_history, api_call_count,
                                        approx_tokens, compression_attempts,
                                        max_compression_attempts, system_message,
                                        effective_task_id):
        """Handle context-length overflow errors.
        
        Returns:
            None if not a context overflow error,
            or dict with action and updated state.
        """
        # Check for context-length errors BEFORE generic 4xx handler.
        # The classifier detects context overflow from: explicit error
        # messages, generic 400 + large session heuristic (#1630), and
        # server disconnect + large session pattern (#2153).
        is_context_length_error = (
            classified.reason == FailoverReason.context_overflow
        )

        if is_context_length_error:
            compressor = self.context_compressor
            old_ctx = compressor.context_length

            # ── Distinguish two very different errors ───────────
            # 1. "Prompt too long": the INPUT exceeds the context window.
            #    Fix: reduce context_length + compress history.
            # 2. "max_tokens too large": input is fine, but
            #    input_tokens + requested max_tokens > context_window.
            #    Fix: reduce max_tokens (the OUTPUT cap) for this call.
            #    Do NOT shrink context_length — the window is unchanged.
            #
            # Note: max_tokens = output token cap (one response).
            #       context_length = total window (input + output combined).
            available_out = parse_available_output_tokens_from_error(error_msg)
            if available_out is not None:
                # Error is purely about the output cap being too large.
                # Cap output to the available space and retry without
                # touching context_length or triggering compression.
                safe_out = max(1, available_out - 64)  # small safety margin
                self._ephemeral_max_output_tokens = safe_out
                self._vprint(
                    f"{self.log_prefix}⚠️  Output cap too large for current prompt — "
                    f"retrying with max_tokens={safe_out:,} "
                    f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})",
                    force=True,
                )
                # Still count against compression_attempts so we don't
                # loop forever if the error keeps recurring.
                compression_attempts += 1
                if compression_attempts > max_compression_attempts:
                    self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                    self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                    logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                    self._persist_session(messages, conversation_history)
                    return {"action": "return", "payload": {
                        "messages": messages,
                        "completed": False,
                        "api_calls": api_call_count,
                        "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                        "partial": True
                    }}
                restart_with_compressed_messages = True
                return {"action": "break",
                        "restart_with_compressed_messages": True,
                        "compression_attempts": compression_attempts,
                        "conversation_history": conversation_history}

            # Error is about the INPUT being too large — reduce context_length.
            # Try to parse the actual limit from the error message
            parsed_limit = parse_context_limit_from_error(error_msg)
            if parsed_limit and parsed_limit < old_ctx:
                new_ctx = parsed_limit
                self._vprint(f"{self.log_prefix}⚠️  Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
            else:
                # Step down to the next probe tier
                new_ctx = get_next_probe_tier(old_ctx)

            if new_ctx and new_ctx < old_ctx:
                compressor.update_model(
                    model=self.model,
                    context_length=new_ctx,
                    base_url=self.base_url,
                    api_key=getattr(self, "api_key", ""),
                    provider=self.provider,
                )
                # Context probing flags — only set on built-in
                # compressor (plugin engines manage their own).
                if hasattr(compressor, "_context_probed"):
                    compressor._context_probed = True
                    # Only persist limits parsed from the provider's
                    # error message (a real number).  Guessed fallback
                    # tiers from get_next_probe_tier() should stay
                    # in-memory only — persisting them pollutes the
                    # cache with wrong values.
                    compressor._context_probe_persistable = bool(
                        parsed_limit and parsed_limit == new_ctx
                    )
                self._vprint(f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
            else:
                self._vprint(f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

            compression_attempts += 1
            if compression_attempts > max_compression_attempts:
                self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                self._persist_session(messages, conversation_history)
                return {"action": "return", "payload": {
                    "messages": messages,
                    "completed": False,
                    "api_calls": api_call_count,
                    "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                    "partial": True
                }}
            self._emit_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

            original_len = len(messages)
            messages, active_system_prompt = self._compress_context(
                messages, system_message, approx_tokens=approx_tokens,
                task_id=effective_task_id,
            )
            # Compression created a new session — clear history
            # so _flush_messages_to_session_db writes compressed
            # messages to the new session, not skipping them.
            conversation_history = None

            if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                if len(messages) < original_len:
                    self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                time.sleep(2)  # Brief pause between compression retries
                restart_with_compressed_messages = True
                return {"action": "break",
                        "restart_with_compressed_messages": True,
                        "compression_attempts": compression_attempts,
                        "conversation_history": conversation_history}
            else:
                # Can't compress further and already at minimum tier
                self._vprint(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                self._vprint(f"{self.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                self._persist_session(messages, conversation_history)
                return {"action": "return", "payload": {
                    "messages": messages,
                    "completed": False,
                    "api_calls": api_call_count,
                    "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                    "partial": True
                }}

        return None


    def _handle_retries_exhausted(self, api_error, error_msg, retry_count,
                                   max_retries, messages, conversation_history,
                                   api_call_count, api_messages, approx_tokens,
                                   api_kwargs, is_rate_limited,
                                   primary_recovery_attempted, compression_attempts):
        """Handle the case when max retries are exhausted.
        
        Returns:
            None if retry_count < max_retries,
            or dict with action and payload.
        """
        if retry_count >= max_retries:
            # Before falling back, try rebuilding the primary
            # client once for transient transport errors (stale
            # connection pool, TCP reset).  Only attempted once
            # per API call block.
            if not primary_recovery_attempted and self._try_recover_primary_transport(
                api_error, retry_count=retry_count, max_retries=max_retries,
            ):
                primary_recovery_attempted = True
                retry_count = 0
                return {"action": "continue",
                        "retry_count": 0,
                        "compression_attempts": 0,
                        "primary_recovery_attempted": False}
            # Try fallback before giving up entirely
            self._emit_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
            if self._try_activate_fallback():
                retry_count = 0
                compression_attempts = 0
                primary_recovery_attempted = False
                return {"action": "continue",
                        "retry_count": 0,
                        "compression_attempts": 0,
                        "primary_recovery_attempted": False}
            _final_summary = self._summarize_api_error(api_error)
            if is_rate_limited:
                self._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
            else:
                self._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
            self._vprint(f"{self.log_prefix}   💀 Final error: {_final_summary}", force=True)

            # Detect SSE stream-drop pattern (e.g. "Network
            # connection lost") and surface actionable guidance.
            # This typically happens when the model generates a
            # very large tool call (write_file with huge content)
            # and the proxy/CDN drops the stream mid-response.
            _is_stream_drop = (
                not getattr(api_error, "status_code", None)
                and any(p in error_msg for p in (
                    "connection lost", "connection reset",
                    "connection closed", "network connection",
                    "network error", "terminated",
                ))
            )
            if _is_stream_drop:
                self._vprint(
                    f"{self.log_prefix}   💡 The provider's stream "
                    f"connection keeps dropping. This often happens "
                    f"when the model tries to write a very large "
                    f"file in a single tool call.",
                    force=True,
                )
                self._vprint(
                    f"{self.log_prefix}      Try asking the model "
                    f"to use execute_code with Python's open() for "
                    f"large files, or to write the file in smaller "
                    f"sections.",
                    force=True,
                )

            _provider = getattr(self, "provider", "unknown")
            _model = getattr(self, "model", "unknown")
            logging.error(
                "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                self.log_prefix, max_retries, _final_summary,
                _provider, _model, len(api_messages), f"{approx_tokens:,}",
            )
            if api_kwargs is not None:
                self._dump_api_request_debug(
                    api_kwargs, reason="max_retries_exhausted", error=api_error,
                )
            self._persist_session(messages, conversation_history)
            _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
            if _is_stream_drop:
                _final_response += (
                    "\n\nThe provider's stream connection keeps "
                    "dropping — this often happens when generating "
                    "very large tool call responses (e.g. write_file "
                    "with long content). Try asking me to use "
                    "execute_code with Python's open() for large "
                    "files, or to write in smaller sections."
                )
            return {"action": "return", "payload": {
                "final_response": _final_response,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "failed": True,
                "error": _final_summary,
            }}

        return None

    # ── Extracted sub-methods (Phase 4 refactoring) ──────────────

    def _validate_api_response(self, response):
        """Validate API response shape before processing.
        
        Returns:
            tuple: (is_valid: bool, error_details: list[str])
        """
        error_details = []
        if self.api_mode == "codex_responses":
            output_items = getattr(response, "output", None) if response is not None else None
            if response is None:
                error_details.append("response is None")
            elif not isinstance(output_items, list):
                error_details.append("response.output is not a list")
            elif not output_items:
                # Stream backfill may have failed, but
                # _normalize_codex_response can still recover
                # from response.output_text. Only mark invalid
                # when that fallback is also absent.
                _out_text = getattr(response, "output_text", None)
                _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                if _out_text_stripped:
                    logger.debug(
                        "Codex response.output is empty but output_text is present "
                        "(%d chars); deferring to normalization.",
                        len(_out_text_stripped),
                    )
                else:
                    _resp_status = getattr(response, "status", None)
                    _resp_incomplete = getattr(response, "incomplete_details", None)
                    logger.warning(
                        "Codex response.output is empty after stream backfill "
                        "(status=%s, incomplete_details=%s, model=%s). %s",
                        _resp_status, _resp_incomplete,
                        getattr(response, "model", None),
                        f"api_mode={self.api_mode} provider={self.provider}",
                    )
                    error_details.append("response.output is empty")
        elif self.api_mode == "anthropic_messages":
            content_blocks = getattr(response, "content", None) if response is not None else None
            if response is None:
                error_details.append("response is None")
            elif not isinstance(content_blocks, list):
                error_details.append("response.content is not a list")
            elif not content_blocks:
                error_details.append("response.content is empty")
        else:
            if response is None or not hasattr(response, 'choices') or response.choices is None or not response.choices:
                if response is None:
                    error_details.append("response is None")
                elif not hasattr(response, 'choices'):
                    error_details.append("response has no 'choices' attribute")
                elif response.choices is None:
                    error_details.append("response.choices is None")
                else:
                    error_details.append("response.choices is empty")

        return (len(error_details) == 0, error_details)

    def _track_usage_and_cost(self, response, api_duration, api_call_count):
        """Track token usage, cost estimation, and cache stats from API response.
        
        Updates session-level counters, persists to session DB, and logs cache stats.
        """
        if hasattr(response, 'usage') and response.usage:
            canonical_usage = normalize_usage(
                response.usage,
                provider=self.provider,
                api_mode=self.api_mode,
            )
            prompt_tokens = canonical_usage.prompt_tokens
            completion_tokens = canonical_usage.output_tokens
            total_tokens = canonical_usage.total_tokens
            usage_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            self.context_compressor.update_from_response(usage_dict)

            # Cache discovered context length after successful call.
            # Only persist limits confirmed by the provider (parsed
            # from the error message), not guessed probe tiers.
            if getattr(self.context_compressor, "_context_probed", False):
                ctx = self.context_compressor.context_length
                if getattr(self.context_compressor, "_context_probe_persistable", False):
                    save_context_length(self.model, self.base_url, ctx)
                    self._safe_print(f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}")
                self.context_compressor._context_probed = False
                self.context_compressor._context_probe_persistable = False

            self.session_prompt_tokens += prompt_tokens
            self.session_completion_tokens += completion_tokens
            self.session_total_tokens += total_tokens
            self.session_api_calls += 1
            self.session_input_tokens += canonical_usage.input_tokens
            self.session_output_tokens += canonical_usage.output_tokens
            self.session_cache_read_tokens += canonical_usage.cache_read_tokens
            self.session_cache_write_tokens += canonical_usage.cache_write_tokens
            self.session_reasoning_tokens += canonical_usage.reasoning_tokens

            # Log API call details for debugging/observability
            _cache_pct = ""
            if canonical_usage.cache_read_tokens and prompt_tokens:
                _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
            logger.info(
                "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                self.session_api_calls, self.model, self.provider or "unknown",
                prompt_tokens, completion_tokens, total_tokens,
                api_duration, _cache_pct,
            )

            cost_result = estimate_usage_cost(
                self.model,
                canonical_usage,
                provider=self.provider,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
            )
            if cost_result.amount_usd is not None:
                self.session_estimated_cost_usd += float(cost_result.amount_usd)
            self.session_cost_status = cost_result.status
            self.session_cost_source = cost_result.source

            # Persist token counts to session DB for /insights.
            # Do this for every platform with a session_id so non-CLI
            # sessions (gateway, cron, delegated runs) cannot lose
            # token/accounting data if a higher-level persistence path
            # is skipped or fails. Gateway/session-store writes use
            # absolute totals, so they safely overwrite these per-call
            # deltas instead of double-counting them.
            if self._session_db and self.session_id:
                try:
                    self._session_db.update_token_counts(
                        self.session_id,
                        input_tokens=canonical_usage.input_tokens,
                        output_tokens=canonical_usage.output_tokens,
                        cache_read_tokens=canonical_usage.cache_read_tokens,
                        cache_write_tokens=canonical_usage.cache_write_tokens,
                        reasoning_tokens=canonical_usage.reasoning_tokens,
                        estimated_cost_usd=float(cost_result.amount_usd)
                        if cost_result.amount_usd is not None else None,
                        cost_status=cost_result.status,
                        cost_source=cost_result.source,
                        billing_provider=self.provider,
                        billing_base_url=self.base_url,
                        billing_mode="subscription_included"
                        if cost_result.status == "included" else None,
                        model=self.model,
                    )
                except Exception:
                    pass  # never block the agent loop

            if self.verbose_logging:
                logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")

            # Log cache hit stats when prompt caching is active
            if self._use_prompt_caching:
                if self.api_mode == "anthropic_messages":
                    # Anthropic uses cache_read_input_tokens / cache_creation_input_tokens
                    cached = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
                    written = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                else:
                    # OpenRouter uses prompt_tokens_details.cached_tokens
                    details = getattr(response.usage, 'prompt_tokens_details', None)
                    cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
                    written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
                prompt = usage_dict["prompt_tokens"]
                hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}   💾 Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)")


    def _normalize_api_response(self, response):
        """Normalize API response across different API modes.
        
        Handles codex_responses, anthropic_messages, and standard chat_completions.
        Also normalizes content type (dict/list -> string).
        
        Returns:
            tuple: (assistant_message, finish_reason)
        """
        if self.api_mode == "codex_responses":
            assistant_message, finish_reason = self._normalize_codex_response(response)
        elif self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import normalize_anthropic_response
            assistant_message, finish_reason = normalize_anthropic_response(
                response, strip_tool_prefix=self._is_anthropic_oauth
            )
        else:
            assistant_message = response.choices[0].message

        # Normalize content to string — some OpenAI-compatible servers
        # (llama-server, etc.) return content as a dict or list instead
        # of a plain string, which crashes downstream .strip() calls.
        if assistant_message.content is not None and not isinstance(assistant_message.content, str):
            raw = assistant_message.content
            if isinstance(raw, dict):
                assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
            elif isinstance(raw, list):
                # Multimodal content list — extract text parts
                parts = []
                for part in raw:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, dict) and "text" in part:
                        parts.append(str(part["text"]))
                assistant_message.content = "\n".join(parts)
            else:
                assistant_message.content = str(raw)

        return assistant_message, finish_reason

    def _validate_tool_calls(self, assistant_message, finish_reason, messages,
                             conversation_history, api_call_count,
                             _depth_nudge_count, _turn_exit_reason):
        """Validate tool call names and JSON arguments.
        
        Returns:
            None if validation passes, or a tuple/dict to return from _process_response.
        """
        # Repair mismatched tool names before validating
        for tc in assistant_message.tool_calls:
            if tc.function.name not in self.valid_tool_names:
                repaired = self._repair_tool_call(tc.function.name)
                if repaired:
                    print(f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                    tc.function.name = repaired
        invalid_tool_calls = [
            tc.function.name for tc in assistant_message.tool_calls
            if tc.function.name not in self.valid_tool_names
        ]
        if invalid_tool_calls:
            # Track retries for invalid tool calls
            self._invalid_tool_retries += 1

            # Return helpful error to model — model can self-correct next turn
            available = ", ".join(sorted(self.valid_tool_names))
            invalid_name = invalid_tool_calls[0]
            invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
            self._vprint(f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)")

            if self._invalid_tool_retries >= 3:
                self._vprint(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                self._invalid_tool_retries = 0
                self._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": f"Model generated invalid tool call: {invalid_preview}"
                }

            assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
            messages.append(assistant_msg)
            for tc in assistant_message.tool_calls:
                if tc.function.name not in self.valid_tool_names:
                    content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                else:
                    content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })
            return ('continue', None, _depth_nudge_count, _turn_exit_reason)
        # Reset retry counter on successful tool call validation
        self._invalid_tool_retries = 0

        # Validate tool call arguments are valid JSON
        # Handle empty strings as empty objects (common model quirk)
        invalid_json_args = []
        for tc in assistant_message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, (dict, list)):
                tc.function.arguments = json.dumps(args)
                continue
            if args is not None and not isinstance(args, str):
                tc.function.arguments = str(args)
                args = tc.function.arguments
            # Treat empty/whitespace strings as empty object
            if not args or not args.strip():
                tc.function.arguments = "{}"
                continue
            try:
                json.loads(args)
            except json.JSONDecodeError as e:
                invalid_json_args.append((tc.function.name, str(e)))

        if invalid_json_args:
            # Check if the invalid JSON is due to truncation rather
            # than a model formatting mistake.  Routers sometimes
            # rewrite finish_reason from "length" to "tool_calls",
            # hiding the truncation from the length handler above.
            # Detect truncation: args that don't end with } or ]
            # (after stripping whitespace) are cut off mid-stream.
            _truncated = any(
                not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                for tc in assistant_message.tool_calls
                if tc.function.name in {n for n, _ in invalid_json_args}
            )
            if _truncated:
                self._vprint(
                    f"{self.log_prefix}⚠️  Truncated tool call arguments detected "
                    f"(finish_reason={finish_reason!r}) — refusing to execute.",
                    force=True,
                )
                self._invalid_json_retries = 0
                self._cleanup_task_resources(effective_task_id)
                self._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Response truncated due to output length limit",
                }

            # Track retries for invalid JSON arguments
            self._invalid_json_retries += 1

            tool_name, error_msg = invalid_json_args[0]
            self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

            if self._invalid_json_retries < 3:
                self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                # Don't add anything to messages, just retry the API call
                return ('continue', None, _depth_nudge_count, _turn_exit_reason)
            else:
                # Instead of returning partial, inject tool error results so the model can recover.
                # Using tool results (not user messages) preserves role alternation.
                self._vprint(f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                self._invalid_json_retries = 0  # Reset for next attempt

                # Append the assistant message with its (broken) tool_calls
                recovery_assistant = self._build_assistant_message(assistant_message, finish_reason)
                messages.append(recovery_assistant)

                # Respond with tool error results for each tool call
                invalid_names = {name for name, _ in invalid_json_args}
                for tc in assistant_message.tool_calls:
                    if tc.function.name in invalid_names:
                        err = next(e for n, e in invalid_json_args if n == tc.function.name)
                        tool_result = (
                            f"Error: Invalid JSON arguments. {err}. "
                            f"For tools with no required parameters, use an empty object: {{}}. "
                            f"Please retry with valid JSON."
                        )
                    else:
                        tool_result = "Skipped: other tool call in this response had invalid JSON."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                return ('continue', None, _depth_nudge_count, _turn_exit_reason)

        # Reset retry counter on successful JSON validation
        self._invalid_json_retries = 0
        return None  # Validation passed

    def _handle_final_text_response(self, assistant_message, finish_reason,
                                     messages, conversation_history,
                                     api_call_count, _depth_nudge_count,
                                     _turn_exit_reason):
        """Handle the case where the model returns text without tool calls.
        
        Covers: empty response retry, fallback provider switch, thinking-only
        prefill continuation, codex ack continuation, depth nudge, and final
        response delivery.
        
        Returns:
            tuple: (action, result, _depth_nudge_count, _turn_exit_reason)
        """
        final_response = assistant_message.content or ""

        # Check if response only has think block with no actual content after it
        if not self._has_content_after_think_block(final_response):
            # If the previous turn already delivered real content alongside
            # tool calls (e.g. "You're welcome!" + memory save), the model
            # has nothing more to say. Use the earlier content immediately
            # instead of wasting API calls on retries that won't help.
            fallback = getattr(self, '_last_content_with_tools', None)
            if fallback:
                _turn_exit_reason = "fallback_prior_turn_content"
                logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                self._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                self._last_content_with_tools = None
                self._empty_content_retries = 0
                for i in range(len(messages) - 1, -1, -1):
                    msg = messages[i]
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        tool_names = []
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict): continue
                            fn = tc.get("function", {})
                            tool_names.append(fn.get("name", "unknown"))
                        msg["content"] = f"Calling the {', '.join(tool_names)} tool{'s' if len(tool_names) > 1 else ''}..."
                        break
                final_response = self._strip_think_blocks(fallback).strip()
                self._response_was_previewed = True
                return ('break', final_response, _depth_nudge_count, _turn_exit_reason)

            # ── Thinking-only prefill continuation ──────────
            # The model produced structured reasoning (via API
            # fields) but no visible text content.  Rather than
            # giving up, append the assistant message as-is and
            # continue — the model will see its own reasoning
            # on the next turn and produce the text portion.
            # Inspired by clawdbot's "incomplete-text" recovery.
            _has_structured = bool(
                getattr(assistant_message, "reasoning", None)
                or getattr(assistant_message, "reasoning_content", None)
                or getattr(assistant_message, "reasoning_details", None)
            )
            if _has_structured and self._thinking_prefill_retries < 2:
                self._thinking_prefill_retries += 1
                logger.info(
                    "Thinking-only response (no visible content) — "
                    "prefilling to continue (%d/2)",
                    self._thinking_prefill_retries,
                )
                self._emit_status(
                    f"↻ Thinking-only response — prefilling to continue "
                    f"({self._thinking_prefill_retries}/2)"
                )
                interim_msg = self._build_assistant_message(
                    assistant_message, "incomplete"
                )
                interim_msg["_thinking_prefill"] = True
                messages.append(interim_msg)
                self._session_messages = messages
                self._save_session_log(messages)
                return ('continue', None, _depth_nudge_count, _turn_exit_reason)

            # ── Empty response retry ──────────────────────
            # Model returned nothing usable.  Retry up to 3
            # times before attempting fallback.  This covers
            # both truly empty responses (no content, no
            # reasoning) AND reasoning-only responses after
            # prefill exhaustion — models like mimo-v2-pro
            # always populate reasoning fields via OpenRouter,
            # so the old `not _has_structured` guard blocked
            # retries for every reasoning model after prefill.
            _truly_empty = not self._strip_think_blocks(
                final_response
            ).strip()
            _prefill_exhausted = (
                _has_structured
                and self._thinking_prefill_retries >= 2
            )
            if _truly_empty and (not _has_structured or _prefill_exhausted) and self._empty_content_retries < 3:
                self._empty_content_retries += 1
                logger.warning(
                    "Empty response (no content or reasoning) — "
                    "retry %d/3 (model=%s)",
                    self._empty_content_retries, self.model,
                )
                self._emit_status(
                    f"⚠️ Empty response from model — retrying "
                    f"({self._empty_content_retries}/3)"
                )
                return ('continue', None, _depth_nudge_count, _turn_exit_reason)

            # ── Exhausted retries — try fallback provider ──
            # Before giving up with "(empty)", attempt to
            # switch to the next provider in the fallback
            # chain.  This covers the case where a model
            # (e.g. GLM-4.5-Air) consistently returns empty
            # due to context degradation or provider issues.
            if _truly_empty and self._fallback_chain:
                logger.warning(
                    "Empty response after %d retries — "
                    "attempting fallback (model=%s, provider=%s)",
                    self._empty_content_retries, self.model,
                    self.provider,
                )
                self._emit_status(
                    "⚠️ Model returning empty responses — "
                    "switching to fallback provider..."
                )
                if self._try_activate_fallback():
                    self._empty_content_retries = 0
                    self._emit_status(
                        f"↻ Switched to fallback: {self.model} "
                        f"({self.provider})"
                    )
                    logger.info(
                        "Fallback activated after empty responses: "
                        "now using %s on %s",
                        self.model, self.provider,
                    )
                    return ('continue', None, _depth_nudge_count, _turn_exit_reason)

            # Exhausted retries and fallback chain (or no
            # fallback configured).  Fall through to the
            # "(empty)" terminal.
            _turn_exit_reason = "empty_response_exhausted"
            reasoning_text = self._extract_reasoning(assistant_message)
            assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
            assistant_msg["content"] = "(empty)"
            messages.append(assistant_msg)

            if reasoning_text:
                reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                logger.warning(
                    "Reasoning-only response (no visible content) "
                    "after exhausting retries and fallback. "
                    "Reasoning: %s", reasoning_preview,
                )
                self._emit_status(
                    "⚠️ Model produced reasoning but no visible "
                    "response after all retries. Returning empty."
                )
            else:
                logger.warning(
                    "Empty response (no content or reasoning) "
                    "after %d retries. No fallback available. "
                    "model=%s provider=%s",
                    self._empty_content_retries, self.model,
                    self.provider,
                )
                self._emit_status(
                    "❌ Model returned no content after all retries"
                    + (" and fallback attempts." if self._fallback_chain else
                       ". No fallback providers configured.")
                )

            final_response = "(empty)"
            return ('break', final_response, _depth_nudge_count, _turn_exit_reason)

        # Reset retry counter/signature on successful content
        self._empty_content_retries = 0
        self._thinking_prefill_retries = 0

        if (
            self.api_mode == "codex_responses"
            and self.valid_tool_names
            and codex_ack_continuations < 2
            and self._looks_like_codex_intermediate_ack(
                user_message=user_message,
                assistant_content=final_response,
                messages=messages,
            )
        ):
            codex_ack_continuations += 1
            interim_msg = self._build_assistant_message(assistant_message, "incomplete")
            messages.append(interim_msg)
            self._emit_interim_assistant_message(interim_msg)

            continue_msg = {
                "role": "user",
                "content": (
                    "[System: Continue now. Execute the required tool calls and only "
                    "send your final answer after completing the task.]"
                ),
            }
            messages.append(continue_msg)
            self._session_messages = messages
            self._save_session_log(messages)
            return ('continue', None, _depth_nudge_count, _turn_exit_reason)

        codex_ack_continuations = 0

        if truncated_response_prefix:
            final_response = truncated_response_prefix + final_response
            truncated_response_prefix = ""
            length_continue_retries = 0

        # Strip <think> blocks from user-facing response (keep raw in messages for trajectory)
        final_response = self._strip_think_blocks(final_response).strip()

        final_msg = self._build_assistant_message(assistant_message, finish_reason)

        # Pop thinking-only prefill message(s) before appending
        # the final response.  This avoids consecutive assistant
        # messages which break strict-alternation providers
        # (Anthropic Messages API) and keeps history clean.
        while (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("_thinking_prefill")
        ):
            messages.pop()

        messages.append(final_msg)

        # --- Continuation nudge: prevent premature stopping ---
        # If the agent stops very early (< 8 calls) and has tools available,
        # nudge it to reconsider before accepting the text response as final.
        if (
            api_call_count < 8
            and self.valid_tool_names
            and _depth_nudge_count < 2
            and final_response
            and final_response != "(empty)"
        ):
            _depth_nudge_count += 1
            nudge_msg = {
                "role": "user",
                "content": (
                    f"[System: You responded with text after only {api_call_count} tool calls. "
                    "This seems premature. Please verify: is the task truly complete? "
                    "Have you gathered all necessary information, executed all required actions, "
                    "and verified the results? If there is more work to do, continue using tools. "
                    "If you are genuinely finished, re-output your final answer.]"
                ),
            }
            messages.append(final_msg)
            messages.append(nudge_msg)
            self._session_messages = messages
            self._save_session_log(messages)
            if not self.quiet_mode:
                self._safe_print(f"🔄 Depth nudge #{_depth_nudge_count}: agent stopped after {api_call_count} calls, nudging to continue")
            return ('continue', None, _depth_nudge_count, _turn_exit_reason)
        # --- End continuation nudge ---

        _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
        if not self.quiet_mode:
            self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
        return ('break', final_response if 'final_response' in dir() else None, _depth_nudge_count, _turn_exit_reason)


    def _prepare_conversation_turn(
        self,
        user_message: str,
        system_message,
        conversation_history,
        task_id,
        stream_callback,
        persist_user_message,
    ):
        """Prepare state for a new conversation turn: reset counters, health check, build messages."""
        # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
        # Installed once, transparent when streams are healthy, prevents crash on write.
        _install_safe_stdio()

        # Tag all log records on this thread with the session ID so
        # ``hermes logs --session <id>`` can filter a single conversation.
        from hermes_logging import set_session_context
        set_session_context(self.session_id)

        # If the previous turn activated fallback, restore the primary
        # runtime so this turn gets a fresh attempt with the preferred model.
        # No-op when _fallback_activated is False (gateway, first turn, etc.).
        self._restore_primary_runtime()

        # Sanitize surrogate characters from user input.  Clipboard paste from
        # rich-text editors (Google Docs, Word, etc.) can inject lone surrogates
        # that are invalid UTF-8 and crash JSON serialization in the OpenAI SDK.
        if isinstance(user_message, str):
            user_message = _sanitize_surrogates(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = _sanitize_surrogates(persist_user_message)

        # Store stream callback for _interruptible_api_call to pick up
        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message
        # Generate unique task_id if not provided to isolate VMs between concurrent tasks
        effective_task_id = task_id or str(uuid.uuid4())
        
        # Reset retry counters and iteration budget at the start of each turn
        # so subagent usage from a previous turn doesn't eat into the next one.
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._thinking_prefill_retries = 0
        self._last_content_with_tools = None
        self._mute_post_response = False
        self._unicode_sanitization_passes = 0

        # Pre-turn connection health check: detect and clean up dead TCP
        # connections left over from provider outages or dropped streams.
        # This prevents the next API call from hanging on a zombie socket.
        if self.api_mode != "anthropic_messages":
            try:
                if self._cleanup_dead_connections():
                    self._emit_status(
                        "🔌 Detected stale connections from a previous provider "
                        "issue — cleaned up automatically. Proceeding with fresh "
                        "connection."
                    )
            except Exception:
                pass
        # Replay compression warning through status_callback for gateway
        # platforms (the callback was not wired during __init__).
        if self._compression_warning:
            self._replay_compression_warning()
            self._compression_warning = None  # send once

        # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
        # They are initialized in __init__ and must persist across run_conversation
        # calls so that nudge logic accumulates correctly in CLI mode.
        self.iteration_budget = IterationBudget(self.max_iterations)

        # Log conversation turn start for debugging/observability
        _msg_preview = (user_message[:80] + "...") if len(user_message) > 80 else user_message
        _msg_preview = _msg_preview.replace("\n", " ")
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            self.session_id or "none", self.model, self.provider or "unknown",
            self.platform or "unknown", len(conversation_history or []),
            _msg_preview,
        )

        # Initialize conversation (copy to avoid mutating the caller's list)
        messages = list(conversation_history) if conversation_history else []

        # Hydrate todo store from conversation history (gateway creates a fresh
        # AIAgent per message, so the in-memory store is empty -- we need to
        # recover the todo state from the most recent todo tool response in history)
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)
        
        # Prefill messages (few-shot priming) are injected at API-call time only,
        # never stored in the messages list. This keeps them ephemeral: they won't
        # be saved to session DB, session logs, or batch trajectories, but they're
        # automatically re-applied on every API call (including session continuations).
        
        # Track user turns for memory flush and periodic nudge logic
        self._user_turn_count += 1

        # Preserve the original user message (no nudge injection).
        original_user_message = persist_user_message if persist_user_message is not None else user_message

        # Track memory nudge trigger (turn-based, checked here).
        # Skill trigger is checked AFTER the agent loop completes, based on
        # how many tool iterations THIS turn used.
        _should_review_memory = False
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                _should_review_memory = True
                self._turns_since_memory = 0

        # Add user message
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx
        
        if not self.quiet_mode:
            self._safe_print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
        return (
            messages, original_user_message, current_turn_user_idx,
            effective_task_id, _should_review_memory, conversation_history,
        )

    def _invoke_pre_llm_plugins(
        self,
        original_user_message: str,
        messages: list,
        conversation_history,
    ) -> str:
        """Invoke pre_llm_call plugin hook. Returns plugin context string."""
        # Plugin hook: pre_llm_call
        # Fired once per turn before the tool-calling loop.  Plugins can
        # return a dict with a ``context`` key (or a plain string) whose
        # value is appended to the current turn's user message.
        #
        # Context is ALWAYS injected into the user message, never the
        # system prompt.  This preserves the prompt cache prefix — the
        # system prompt stays identical across turns so cached tokens
        # are reused.  The system prompt is Hermes's territory; plugins
        # contribute context alongside the user's input.
        #
        # All injected context is ephemeral (not persisted to session DB).
        _plugin_user_context = ""
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _pre_results = _invoke_hook(
                "pre_llm_call",
                session_id=self.session_id,
                user_message=original_user_message,
                conversation_history=list(messages),
                is_first_turn=(not bool(conversation_history)),
                model=self.model,
                platform=getattr(self, "platform", None) or "",
                sender_id=getattr(self, "_user_id", None) or "",
            )
            _ctx_parts: list[str] = []
            for r in _pre_results:
                if isinstance(r, dict) and r.get("context"):
                    _ctx_parts.append(str(r["context"]))
                elif isinstance(r, str) and r.strip():
                    _ctx_parts.append(r)
            if _ctx_parts:
                _plugin_user_context = "\n\n".join(_ctx_parts)
        except Exception as exc:
            logger.warning("pre_llm_call hook failed: %s", exc)

        return _plugin_user_context

    def _finalize_conversation_turn(
        self,
        messages: list,
        final_response,
        api_call_count: int,
        interrupted: bool,
        original_user_message: str,
        _should_review_memory: bool,
        conversation_history,
    ) -> dict:
        """Post-loop finalization: build result dict, run plugins, cleanup."""
        completed = final_response is not None and api_call_count < self.max_iterations
        # ── Turn-exit diagnostic log ─────────────────────────────────────
        # Always logged at INFO so agent.log captures WHY every turn ended.
        # When the last message is a tool result (agent was mid-work), log
        # at WARNING — this is the "just stops" scenario users report.
        _last_msg_role = messages[-1].get("role") if messages else None
        _last_tool_name = None
        if _last_msg_role == "tool":
            # Walk back to find the assistant message with the tool call
            for _m in reversed(messages):
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    _tcs = _m["tool_calls"]
                    if _tcs and isinstance(_tcs[0], dict):
                        _last_tool_name = _tcs[-1].get("function", {}).get("name")
                    break

        _turn_tool_count = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
        )
        _resp_len = len(final_response) if final_response else 0
        _budget_used = self.iteration_budget.used if self.iteration_budget else 0
        _budget_max = self.iteration_budget.max_total if self.iteration_budget else 0

        _diag_msg = (
            "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
            "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
        )
        _diag_args = (
            _turn_exit_reason, self.model, api_call_count, self.max_iterations,
            _budget_used, _budget_max,
            _turn_tool_count, _last_msg_role, _resp_len,
            self.session_id or "none",
        )

        if _last_msg_role == "tool" and not interrupted:
            # Agent was mid-work — this is the "just stops" case.
            logger.warning(
                "Turn ended with pending tool result (agent may appear stuck). "
                + _diag_msg + " last_tool=%s",
                *_diag_args, _last_tool_name,
            )
        else:
            logger.info(_diag_msg, *_diag_args)

        # Plugin hook: post_llm_call
        # Fired once per turn after the tool-calling loop completes.
        # Plugins can use this to persist conversation data (e.g. sync
        # to an external memory system).
        if final_response and not interrupted:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "post_llm_call",
                    session_id=self.session_id,
                    user_message=original_user_message,
                    assistant_response=final_response,
                    conversation_history=list(messages),
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("post_llm_call hook failed: %s", exc)

        # Extract reasoning from the last assistant message (if any)
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        # Build result with interrupt info if applicable
        result = {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "partial": False,  # True only when stopped due to invalid tool calls
            "interrupted": interrupted,
            "response_previewed": getattr(self, "_response_was_previewed", False),
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "input_tokens": self.session_input_tokens,
            "output_tokens": self.session_output_tokens,
            "cache_read_tokens": self.session_cache_read_tokens,
            "cache_write_tokens": self.session_cache_write_tokens,
            "reasoning_tokens": self.session_reasoning_tokens,
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "last_prompt_tokens": getattr(self.context_compressor, "last_prompt_tokens", 0) or 0,
            "estimated_cost_usd": self.session_estimated_cost_usd,
            "cost_status": self.session_cost_status,
            "cost_source": self.session_cost_source,
        }
        self._response_was_previewed = False
        
        # Include interrupt message if one triggered the interrupt
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        # Clear interrupt state after handling
        self.clear_interrupt()

        # Clear stream callback so it doesn't leak into future calls
        self._stream_callback = None

        # Check skill trigger NOW — based on how many tool iterations THIS turn used.
        _should_review_skills = False
        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            _should_review_skills = True
            self._iters_since_skill = 0

        # External memory provider: sync the completed turn + queue next prefetch.
        # Use original_user_message (clean input) — user_message may contain
        # injected skill content that bloats / breaks provider queries.
        if self._memory_manager and final_response and original_user_message:
            try:
                self._memory_manager.sync_all(original_user_message, final_response)
                self._memory_manager.queue_prefetch_all(original_user_message)
            except Exception:
                pass

        # Background memory/skill review — runs AFTER the response is delivered
        # so it never competes with the user's task for model attention.
        if final_response and not interrupted and (_should_review_memory or _should_review_skills):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                )
            except Exception:
                pass  # Background review is best-effort

        # Note: Memory provider on_session_end() + shutdown_all() are NOT
        # called here — run_conversation() is called once per user message in
        # multi-turn sessions. Shutting down after every turn would kill the
        # provider before the second message. Actual session-end cleanup is
        # handled by the CLI (atexit / /reset) and gateway (session expiry /
        # _reset_session).

        # Plugin hook: on_session_end
        # Fired at the very end of every run_conversation call.
        # Plugins can use this for cleanup, flushing buffers, etc.
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_end",
                session_id=self.session_id,
                completed=completed,
                interrupted=interrupted,
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("on_session_end hook failed: %s", exc)

        return result



    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """
        Simple chat interface that returns just the final response.

        Args:
            message (str): User message
            stream_callback: Optional callback invoked with each text delta during streaming.

        Returns:
            str: Final assistant response
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]



