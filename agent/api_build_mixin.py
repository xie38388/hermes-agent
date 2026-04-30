"""Auto-generated mixin for AIAgent."""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pathlib import Path
import copy
from datetime import datetime
import hashlib
import json
import os
import tempfile
import uuid
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
from agent.prompt_builder import DEVELOPER_ROLE_MODELS
from agent.usage_pricing import normalize_usage
from utils import env_var_enabled
import base64
logger = logging.getLogger(__name__)


class ApiBuildMixin:
    """AIAgent mixin: api_build methods."""
    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        if isinstance(tc, dict):
            return tc.get("id", "")
        return getattr(tc, "id", "")
    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        if isinstance(tc, dict):
            return tc.get("id", "")
        return getattr(tc, "id", "")

    def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[Dict[str, Any]]:
        """Token buckets for ``post_api_request`` plugins (no raw ``response`` object)."""
        if response is None:
            return None
        raw_usage = getattr(response, "usage", None)
        if not raw_usage:
            return None
        from dataclasses import asdict

        cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
        summary = asdict(cu)
        summary.pop("raw_usage", None)
        summary["prompt_tokens"] = cu.prompt_tokens
        summary["total_tokens"] = cu.total_tokens
        return summary



    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        Dump a debug-friendly HTTP request record for the active inference API.

        Captures the request body from api_kwargs (excluding transport-only keys
        like timeout). Intended for debugging provider-side 4xx failures where
        retries are not useful.
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(response_obj, "status_code", None)
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None



    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs before every LLM call.

        Runs unconditionally — not gated on whether the context compressor
        is present — so orphans from session loading or manual message
        manipulation are always caught.
        """
        # --- Role allowlist: drop messages with roles the API won't accept ---
        filtered = []
        for msg in messages:
            role = msg.get("role")
            if role not in ApiBuildMixin._VALID_API_ROLES:
                logger.debug(
                    "Pre-call sanitizer: dropping message with invalid role %r",
                    role,
                )
                continue
            filtered.append(msg)
        messages = filtered

        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = ApiBuildMixin._get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Drop tool results with no matching assistant call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            logger.debug(
                "Pre-call sanitizer: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. Inject stub results for calls whose result was dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = ApiBuildMixin._get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result unavailable — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            logger.debug(
                "Pre-call sanitizer: added %d stub tool result(s)",
                len(missing_results),
            )
        return messages



    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """Truncate excess delegate_task calls to max_concurrent_children.

        The delegate_tool caps the task list inside a single call, but the
        model can emit multiple separate delegate_task tool_calls in one
        turn.  This truncates the excess, preserving all non-delegate calls.

        Returns the original list if no truncation was needed.
        """
        from tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < max_children:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "Truncated %d excess delegate_task call(s) to enforce "
            "max_concurrent_children=%d limit",
            delegate_count - max_children, max_children,
        )
        return truncated



    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        """Generate a deterministic call_id from tool call content.

        Used as a fallback when the API doesn't provide a call_id.
        Deterministic IDs prevent cache invalidation — random UUIDs would
        make every API call's prefix unique, breaking OpenAI's prompt cache.
        """
        import hashlib
        seed = f"{fn_name}:{arguments}:{index}"
        digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"call_{digest}"



    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()



    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        with tmp:
            tmp.write(base64.b64decode(data))
        path = Path(tmp.name)
        return str(path), path



    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed



    def _anthropic_preserve_dots(self) -> bool:
        """True when using an anthropic-compatible endpoint that preserves dots in model names.
        Alibaba/DashScope keeps dots (e.g. qwen3.5-plus).
        MiniMax keeps dots (e.g. MiniMax-M2.7).
        OpenCode Go keeps dots (e.g. minimax-m2.7)."""
        if (getattr(self, "provider", "") or "").lower() in {"alibaba", "minimax", "minimax-cn", "opencode-go"}:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return "dashscope" in base or "aliyuncs" in base or "minimax" in base or "opencode.ai/zen/go" in base



    def _is_qwen_portal(self) -> bool:
        """Return True when the base URL targets Qwen Portal."""
        return "portal.qwen.ai" in self._base_url_lower



    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the active API mode."""
        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_kwargs
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            # Pass context_length (total input+output window) so the adapter can
            # clamp max_tokens (output cap) when the user configured a smaller
            # context window than the model's native output limit.
            ctx_len = getattr(self, "context_compressor", None)
            ctx_len = ctx_len.context_length if ctx_len else None
            # _ephemeral_max_output_tokens is set for one call when the API
            # returns "max_tokens too large given prompt" — it caps output to
            # the available window space without touching context_length.
            ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
            if ephemeral_out is not None:
                self._ephemeral_max_output_tokens = None  # consume immediately
            return build_anthropic_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=ephemeral_out if ephemeral_out is not None else self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=self._is_anthropic_oauth,
                preserve_dots=self._anthropic_preserve_dots(),
                context_length=ctx_len,
                base_url=getattr(self, "_anthropic_base_url", None),
                fast_mode=(self.request_overrides or {}).get("speed") == "fast",
            )

        if self.api_mode == "codex_responses":
            instructions = ""
            payload_messages = api_messages
            if api_messages and api_messages[0].get("role") == "system":
                instructions = str(api_messages[0].get("content") or "").strip()
                payload_messages = api_messages[1:]
            if not instructions:
                instructions = DEFAULT_AGENT_IDENTITY

            is_github_responses = (
                "models.github.ai" in self.base_url.lower()
                or "api.githubcopilot.com" in self.base_url.lower()
            )
            is_codex_backend = (
                self.provider == "openai-codex"
                or "chatgpt.com/backend-api/codex" in self.base_url.lower()
            )

            # Resolve reasoning effort: config > default (medium)
            reasoning_effort = "medium"
            reasoning_enabled = True
            if self.reasoning_config and isinstance(self.reasoning_config, dict):
                if self.reasoning_config.get("enabled") is False:
                    reasoning_enabled = False
                elif self.reasoning_config.get("effort"):
                    reasoning_effort = self.reasoning_config["effort"]

            kwargs = {
                "model": self.model,
                "instructions": instructions,
                "input": self._chat_messages_to_responses_input(payload_messages),
                "tools": self._responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "store": False,
            }

            if not is_github_responses:
                kwargs["prompt_cache_key"] = self.session_id

            if reasoning_enabled:
                if is_github_responses:
                    # Copilot's Responses route advertises reasoning-effort support,
                    # but not OpenAI-specific prompt cache or encrypted reasoning
                    # fields. Keep the payload to the documented subset.
                    github_reasoning = self._github_models_reasoning_extra_body()
                    if github_reasoning is not None:
                        kwargs["reasoning"] = github_reasoning
                else:
                    kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                    kwargs["include"] = ["reasoning.encrypted_content"]
            elif not is_github_responses:
                kwargs["include"] = []

            if self.request_overrides:
                kwargs.update(self.request_overrides)

            if self.max_tokens is not None and not is_codex_backend:
                kwargs["max_output_tokens"] = self.max_tokens

            return kwargs

        sanitized_messages = api_messages
        needs_sanitization = False
        for msg in api_messages:
            if not isinstance(msg, dict):
                continue
            if "codex_reasoning_items" in msg:
                needs_sanitization = True
                break

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    if "call_id" in tool_call or "response_item_id" in tool_call:
                        needs_sanitization = True
                        break
                if needs_sanitization:
                    break

        if needs_sanitization:
            sanitized_messages = copy.deepcopy(api_messages)
            for msg in sanitized_messages:
                if not isinstance(msg, dict):
                    continue

                # Codex-only replay state must not leak into strict chat-completions APIs.
                msg.pop("codex_reasoning_items", None)

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call.pop("call_id", None)
                            tool_call.pop("response_item_id", None)

        # Qwen portal: normalize content to list-of-dicts, inject cache_control.
        # Must run AFTER codex sanitization so we transform the final messages.
        # If sanitization already deepcopied, reuse that copy (in-place).
        if self._is_qwen_portal():
            if sanitized_messages is api_messages:
                # No sanitization was done — we need our own copy.
                sanitized_messages = self._qwen_prepare_chat_messages(sanitized_messages)
            else:
                # Already a deepcopy — transform in place to avoid a second deepcopy.
                self._qwen_prepare_chat_messages_inplace(sanitized_messages)

        # GPT-5 and Codex models respond better to 'developer' than 'system'
        # for instruction-following.  Swap the role at the API boundary so
        # internal message representation stays uniform ("system").
        _model_lower = (self.model or "").lower()
        if (
            sanitized_messages
            and sanitized_messages[0].get("role") == "system"
            and any(p in _model_lower for p in DEVELOPER_ROLE_MODELS)
        ):
            # Shallow-copy the list + first message only — rest stays shared.
            sanitized_messages = list(sanitized_messages)
            sanitized_messages[0] = {**sanitized_messages[0], "role": "developer"}

        provider_preferences = {}
        if self.providers_allowed:
            provider_preferences["only"] = self.providers_allowed
        if self.providers_ignored:
            provider_preferences["ignore"] = self.providers_ignored
        if self.providers_order:
            provider_preferences["order"] = self.providers_order
        if self.provider_sort:
            provider_preferences["sort"] = self.provider_sort
        if self.provider_require_parameters:
            provider_preferences["require_parameters"] = True
        if self.provider_data_collection:
            provider_preferences["data_collection"] = self.provider_data_collection

        api_kwargs = {
            "model": self.model,
            "messages": sanitized_messages,
            "timeout": float(os.getenv("HERMES_API_TIMEOUT", 1800.0)),
        }
        if self._is_qwen_portal():
            api_kwargs["metadata"] = {
                "sessionId": self.session_id or "hermes",
                "promptId": str(uuid.uuid4()),
            }
        if self.tools:
            api_kwargs["tools"] = self.tools

        if self.max_tokens is not None:
            api_kwargs.update(self._max_tokens_param(self.max_tokens))
        elif self._is_qwen_portal():
            # Qwen Portal defaults to a very low max_tokens when omitted.
            # Reasoning models (qwen3-coder-plus) exhaust that budget on
            # thinking tokens alone, causing the portal to return
            # finish_reason="stop" with truncated output — the agent sees
            # this as an intentional stop and exits the loop.  Send 65536
            # (the documented max output for qwen3-coder models) so the
            # model has adequate output budget for tool calls.
            api_kwargs.update(self._max_tokens_param(65536))
        elif (self._is_openrouter_url() or "nousresearch" in self._base_url_lower) and "claude" in (self.model or "").lower():
            # OpenRouter and Nous Portal translate requests to Anthropic's
            # Messages API, which requires max_tokens as a mandatory field.
            # When we omit it, the proxy picks a default that can be too
            # low — the model spends its output budget on thinking and has
            # almost nothing left for the actual response (especially large
            # tool calls like write_file).  Sending the model's real output
            # limit ensures full capacity.
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                _model_output_limit = _get_anthropic_max_output(self.model)
                api_kwargs["max_tokens"] = _model_output_limit
            except Exception:
                pass  # fail open — let the proxy pick its default

        extra_body = {}

        _is_openrouter = self._is_openrouter_url()
        _is_github_models = (
            "models.github.ai" in self._base_url_lower
            or "api.githubcopilot.com" in self._base_url_lower
        )

        # Provider preferences (only, ignore, order, sort) are OpenRouter-
        # specific.  Only send to OpenRouter-compatible endpoints.
        # TODO: Nous Portal will add transparent proxy support — re-enable
        # for _is_nous when their backend is updated.
        if provider_preferences and _is_openrouter:
            extra_body["provider"] = provider_preferences
        _is_nous = "nousresearch" in self._base_url_lower

        if self._supports_reasoning_extra_body():
            if _is_github_models:
                github_reasoning = self._github_models_reasoning_extra_body()
                if github_reasoning is not None:
                    extra_body["reasoning"] = github_reasoning
            else:
                if self.reasoning_config is not None:
                    rc = dict(self.reasoning_config)
                    # Nous Portal requires reasoning enabled — don't send
                    # enabled=false to it (would cause 400).
                    if _is_nous and rc.get("enabled") is False:
                        pass  # omit reasoning entirely for Nous when disabled
                    else:
                        extra_body["reasoning"] = rc
                else:
                    extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }

        # Nous Portal product attribution
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        # Ollama num_ctx: override the 2048 default so the model actually
        # uses the context window it was trained for.  Passed via the OpenAI
        # SDK's extra_body → options.num_ctx, which Ollama's OpenAI-compat
        # endpoint forwards to the runner as --ctx-size.
        if self._ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = self._ollama_num_ctx
            extra_body["options"] = options

        if self._is_qwen_portal():
            extra_body["vl_high_resolution_images"] = True

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # xAI prompt caching: send x-grok-conv-id header to route requests
        # to the same server, maximizing automatic cache hits.
        # https://docs.x.ai/developers/advanced-api-usage/prompt-caching
        if "x.ai" in self._base_url_lower and hasattr(self, "session_id") and self.session_id:
            api_kwargs["extra_headers"] = {"x-grok-conv-id": self.session_id}

        # Priority Processing / generic request overrides (e.g. service_tier).
        # Applied last so overrides win over any defaults set above.
        if self.request_overrides:
            api_kwargs.update(self.request_overrides)

        return api_kwargs



    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        reasoning_text = self._extract_reasoning(assistant_message)
        _from_structured = bool(reasoning_text)

        # Fallback: extract inline <think> blocks from content when no structured
        # reasoning fields are present (some models/providers embed thinking
        # directly in the content rather than returning separate API fields).
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

        if reasoning_text and self.reasoning_callback:
            # Skip callback when streaming is active — reasoning was already
            # displayed during the stream via one of two paths:
            #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
            #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
            # When streaming is NOT active, always fire so non-streaming modes
            # (gateway, batch, quiet) still get reasoning.
            # Any reasoning that wasn't shown during streaming is caught by the
            # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
            if not self.stream_delta_callback:
                try:
                    self.reasoning_callback(reasoning_text)
                except Exception:
                    pass

        msg = {
            "role": "assistant",
            "content": assistant_message.content or "",
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            # Pass reasoning_details back unmodified so providers (OpenRouter,
            # Anthropic, OpenAI) can maintain reasoning continuity across turns.
            # Each provider may include opaque fields (signature, encrypted_content)
            # that must be preserved exactly.
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API: preserve encrypted reasoning items for
        # multi-turn continuity. These get replayed as input on the next turn.
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        _fn = getattr(tool_call, "function", None)
                        _fn_name = getattr(_fn, "name", "") if _fn else ""
                        _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                        call_id = self._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if not isinstance(response_item_id, str) or not response_item_id.strip():
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                }
                # Preserve extra_content (e.g. Gemini thought_signature) so it
                # is sent back on subsequent API calls.  Without this, Gemini 3
                # thinking models reject the request with a 400 error.
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg



