"""ResponsesAPIMixin – Codex/Responses API format conversion methods.

Extracted from run_agent.py Step 5.
Methods handle conversion between Chat Completions format and Responses API format.

Mixin contract (attributes provided by AIAgent):
    self.tools: list          – registered tool definitions
    self._cached_system_prompt: str – cached system prompt
    self._memory_store: Any   – memory store reference
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import re
import uuid
from types import SimpleNamespace
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

logger = logging.getLogger(__name__)


class ResponsesAPIMixin:
    """Codex/Responses API format conversion methods."""

    def _get_tool_call_id_static(tc) -> str:
        """Extract call ID from a tool_call entry (dict or object)."""
        if isinstance(tc, dict):
            return tc.get("id", "") or ""
        return getattr(tc, "id", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

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
            from run_agent import AIAgent as _AIAgent
            if role not in _AIAgent._VALID_API_ROLES:
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
                    cid = _AIAgent._get_tool_call_id_static(tc)
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
                        cid = _AIAgent._get_tool_call_id_static(tc)
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

    def _invalidate_system_prompt(self):
        """
        Invalidate the cached system prompt, forcing a rebuild on the next turn.
        
        Called after context compression events. Also reloads memory from disk
        so the rebuilt prompt captures any writes from this session.
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()

    def _responses_tools(self, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """Convert chat-completions tool schemas to Responses function-tool schemas."""
        source_tools = tools if tools is not None else self.tools
        if not source_tools:
            return None

        converted: List[Dict[str, Any]] = []
        for item in source_tools:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append({
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted or None

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
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """Split a stored tool id into (call_id, response_item_id)."""
        if not isinstance(raw_id, str):
            return None, None
        value = raw_id.strip()
        if not value:
            return None, None
        if "|" in value:
            call_id, response_item_id = value.split("|", 1)
            call_id = call_id.strip() or None
            response_item_id = response_item_id.strip() or None
            return call_id, response_item_id
        if value.startswith("fc_"):
            return None, value
        return value, None

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """Build a valid Responses `function_call.id` (must start with `fc_`)."""
        if isinstance(response_item_id, str):
            candidate = response_item_id.strip()
            if candidate.startswith("fc_"):
                return candidate

        source = (call_id or "").strip()
        if source.startswith("fc_"):
            return source
        if source.startswith("call_") and len(source) > len("call_"):
            return f"fc_{source[len('call_'):]}"

        sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
        if sanitized.startswith("fc_"):
            return sanitized
        if sanitized.startswith("call_") and len(sanitized) > len("call_"):
            return f"fc_{sanitized[len('call_'):]}"
        if sanitized:
            return f"fc_{sanitized[:48]}"

        seed = source or str(response_item_id or "") or uuid.uuid4().hex
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
        return f"fc_{digest}"

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

    def _preflight_codex_input_items(self, raw_items: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            raise ValueError("Codex Responses input must be a list of input items.")

        normalized: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Codex Responses input[{idx}] must be an object.")

            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                arguments = arguments.strip() or "{}"

                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": call_id.strip(),
                        "name": name.strip(),
                        "arguments": arguments,
                    }
                )
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
                output = item.get("output", "")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    output = str(output)

                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": output,
                    }
                )
                continue

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id:
                        if item_id in seen_ids:
                            continue
                        seen_ids.add(item_id)
                    reasoning_item = {"type": "reasoning", "encrypted_content": encrypted}
                    if isinstance(item_id, str) and item_id:
                        reasoning_item["id"] = item_id
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        reasoning_item["summary"] = summary
                    else:
                        reasoning_item["summary"] = []
                    normalized.append(reasoning_item)
                continue

            role = item.get("role")
            if role in {"user", "assistant"}:
                content = item.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)

                normalized.append({"role": role, "content": content})
                continue

            raise ValueError(
                f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
            )

        return normalized

    def _preflight_codex_api_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(api_kwargs, dict):
            raise ValueError("Codex Responses request must be a dict.")

        required = {"model", "instructions", "input"}
        missing = [key for key in required if key not in api_kwargs]
        if missing:
            raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")

        model = api_kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Codex Responses request 'model' must be a non-empty string.")
        model = model.strip()

        instructions = api_kwargs.get("instructions")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            instructions = str(instructions)
        instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY

        normalized_input = self._preflight_codex_input_items(api_kwargs.get("input"))

        tools = api_kwargs.get("tools")
        normalized_tools = None
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("Codex Responses request 'tools' must be a list when provided.")
            normalized_tools = []
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] must be an object.")
                if tool.get("type") != "function":
                    raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")

                name = tool.get("name")
                parameters = tool.get("parameters")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses tools[{idx}] is missing a valid name.")
                if not isinstance(parameters, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] is missing valid parameters.")

                description = tool.get("description", "")
                if description is None:
                    description = ""
                if not isinstance(description, str):
                    description = str(description)

                strict = tool.get("strict", False)
                if not isinstance(strict, bool):
                    strict = bool(strict)

                normalized_tools.append(
                    {
                        "type": "function",
                        "name": name.strip(),
                        "description": description,
                        "strict": strict,
                        "parameters": parameters,
                    }
                )

        store = api_kwargs.get("store", False)
        if store is not False:
            raise ValueError("Codex Responses contract requires 'store' to be false.")

        allowed_keys = {
            "model", "instructions", "input", "tools", "store",
            "reasoning", "include", "max_output_tokens", "temperature",
            "tool_choice", "parallel_tool_calls", "prompt_cache_key", "service_tier",
        }
        normalized: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": normalized_input,
            "store": False,
        }
        if normalized_tools is not None:
            normalized["tools"] = normalized_tools

        # Pass through reasoning config
        reasoning = api_kwargs.get("reasoning")
        if isinstance(reasoning, dict):
            normalized["reasoning"] = reasoning
        include = api_kwargs.get("include")
        if isinstance(include, list):
            normalized["include"] = include
        service_tier = api_kwargs.get("service_tier")
        if isinstance(service_tier, str) and service_tier.strip():
            normalized["service_tier"] = service_tier.strip()

        # Pass through max_output_tokens and temperature
        max_output_tokens = api_kwargs.get("max_output_tokens")
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            normalized["max_output_tokens"] = int(max_output_tokens)
        temperature = api_kwargs.get("temperature")
        if isinstance(temperature, (int, float)):
            normalized["temperature"] = float(temperature)

        # Pass through tool_choice, parallel_tool_calls, prompt_cache_key
        for passthrough_key in ("tool_choice", "parallel_tool_calls", "prompt_cache_key"):
            val = api_kwargs.get(passthrough_key)
            if val is not None:
                normalized[passthrough_key] = val

        if allow_stream:
            stream = api_kwargs.get("stream")
            if stream is not None and stream is not True:
                raise ValueError("Codex Responses 'stream' must be true when set.")
            if stream is True:
                normalized["stream"] = True
            allowed_keys.add("stream")
        elif "stream" in api_kwargs:
            raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")

        unexpected = sorted(key for key in api_kwargs if key not in allowed_keys)
        if unexpected:
            raise ValueError(
                f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
            )

        return normalized

    def _extract_responses_message_text(self, item: Any) -> str:
        """Extract assistant text from a Responses message output item."""
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            return ""

        chunks: List[str] = []
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype not in {"output_text", "text"}:
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks).strip()

    def _extract_responses_reasoning_text(self, item: Any) -> str:
        """Extract a compact reasoning text from a Responses reasoning item."""
        summary = getattr(item, "summary", None)
        if isinstance(summary, list):
            chunks: List[str] = []
            for part in summary:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return text.strip()
        return ""

    def _normalize_codex_response(self, response: Any) -> tuple[Any, str]:
        """Normalize a Responses API object to an assistant_message-like object."""
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            # The Codex backend can return empty output when the answer was
            # delivered entirely via stream events. Check output_text as a
            # last-resort fallback before raising.
            out_text = getattr(response, "output_text", None)
            if isinstance(out_text, str) and out_text.strip():
                logger.debug(
                    "Codex response has empty output but output_text is present (%d chars); "
                    "synthesizing output item.", len(out_text.strip()),
                )
                output = [SimpleNamespace(
                    type="message", role="assistant", status="completed",
                    content=[SimpleNamespace(type="output_text", text=out_text.strip())],
                )]
                response.output = output
            else:
                raise RuntimeError("Responses API returned no output items")

        response_status = getattr(response, "status", None)
        if isinstance(response_status, str):
            response_status = response_status.strip().lower()
        else:
            response_status = None

        if response_status in {"failed", "cancelled"}:
            error_obj = getattr(response, "error", None)
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = str(error_obj) if error_obj else f"Responses API returned status '{response_status}'"
            raise RuntimeError(error_msg)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items_raw: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
        saw_commentary_phase = False
        saw_final_answer_phase = False

        for item in output:
            item_type = getattr(item, "type", None)
            item_status = getattr(item, "status", None)
            if isinstance(item_status, str):
                item_status = item_status.strip().lower()
            else:
                item_status = None

            if item_status in {"queued", "in_progress", "incomplete"}:
                has_incomplete_items = True

            if item_type == "message":
                item_phase = getattr(item, "phase", None)
                if isinstance(item_phase, str):
                    normalized_phase = item_phase.strip().lower()
                    if normalized_phase in {"commentary", "analysis"}:
                        saw_commentary_phase = True
                    elif normalized_phase in {"final_answer", "final"}:
                        saw_final_answer_phase = True
                message_text = self._extract_responses_message_text(item)
                if message_text:
                    content_parts.append(message_text)
            elif item_type == "reasoning":
                reasoning_text = self._extract_responses_reasoning_text(item)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                # Capture the full reasoning item for multi-turn continuity.
                # encrypted_content is an opaque blob the API needs back on
                # subsequent turns to maintain coherent reasoning chains.
                encrypted = getattr(item, "encrypted_content", None)
                if isinstance(encrypted, str) and encrypted:
                    raw_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = getattr(item, "id", None)
                    if isinstance(item_id, str) and item_id:
                        raw_item["id"] = item_id
                    # Capture summary — required by the API when replaying reasoning items
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        raw_summary = []
                        for part in summary:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                raw_summary.append({"type": "summary_text", "text": text})
                        raw_item["summary"] = raw_summary
                    reasoning_items_raw.append(raw_item)
            elif item_type == "function_call":
                if item_status in {"queued", "in_progress", "incomplete"}:
                    continue
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))
            elif item_type == "custom_tool_call":
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "input", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))

        final_text = "\n".join([p for p in content_parts if p]).strip()
        if not final_text and hasattr(response, "output_text"):
            out_text = getattr(response, "output_text", "")
            if isinstance(out_text, str):
                final_text = out_text.strip()

        assistant_message = SimpleNamespace(
            content=final_text,
            tool_calls=tool_calls,
            reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=reasoning_items_raw or None,
        )

        if tool_calls:
            finish_reason = "tool_calls"
        elif has_incomplete_items or (saw_commentary_phase and not saw_final_answer_phase):
            finish_reason = "incomplete"
        elif reasoning_items_raw and not final_text:
            # Response contains only reasoning (encrypted thinking state) with
            # no visible content or tool calls.  The model is still thinking and
            # needs another turn to produce the actual answer.  Marking this as
            # "stop" would send it into the empty-content retry loop which burns
            # 3 retries then fails — treat it as incomplete instead so the Codex
            # continuation path handles it correctly.
            finish_reason = "incomplete"
        else:
            finish_reason = "stop"
        return assistant_message, finish_reason


