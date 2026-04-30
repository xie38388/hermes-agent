"""Auto-generated mixin for AIAgent."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from agent.shared_utils import _is_destructive_command, _should_parallelize_tool_batch

from openai import OpenAI
import concurrent.futures
import copy
from tools.tool_result_storage import enforce_turn_budget
from tools.terminal_tool import get_active_env
import json
from tools.tool_result_storage import maybe_persist_tool_result
import os
import random
import time
from agent.display import KawaiiSpinner
from agent.display import build_tool_preview as _build_tool_preview
from agent.display import _detect_tool_failure
from agent.display import get_cute_tool_message as _get_cute_tool_message_impl
from agent.display import get_tool_emoji as _get_tool_emoji
from agent.serialization_variation import apply_serialization_variation
from model_tools import handle_function_call
logger = logging.getLogger(__name__)

class ToolMixin:
    """AIAgent mixin: tool methods."""

    def _should_emit_quiet_tool_messages(self) -> bool:
        """Return True when quiet-mode tool summaries should print directly.

        When the caller provides ``tool_progress_callback`` (for example the CLI
        TUI or a gateway progress renderer), that callback owns progress display.
        Emitting quiet-mode summary lines here duplicates progress and leaks tool
        previews into flows that are expected to stay silent, such as
        ``hermes chat -q``.
        """
        return self.quiet_mode and not self.tool_progress_callback


    def _format_tools_for_system_message(self) -> str:
        """
        Format tool definitions for the system message in the trajectory format.
        
        Returns:
            str: JSON string representation of tool definitions
        """
        if not self.tools:
            return "[]"
        
        # Convert tool definitions to the format expected in trajectories
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools, ensure_ascii=False)
    

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """Extract call ID from a tool_call entry (dict or object)."""
        if isinstance(tc, dict):
            return tc.get("id", "") or ""
        return getattr(tc, "id", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})


    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """Remove duplicate (tool_name, arguments) pairs within a single turn.

        Only the first occurrence of each unique pair is kept.
        Returns the original list if no duplicates were found.
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls


    def _repair_tool_call(self, tool_name: str) -> str | None:
        """Attempt to repair a mismatched tool name before aborting.

        1. Try lowercase
        2. Try normalized (lowercase + hyphens/spaces -> underscores)
        3. Try fuzzy match (difflib, cutoff=0.7)

        Returns the repaired name if found in valid_tool_names, else None.
        """
        from difflib import get_close_matches

        # 1. Lowercase
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered

        # 2. Normalize
        normalized = lowered.replace("-", "_").replace(" ", "_")
        if normalized in self.valid_tool_names:
            return normalized

        # 3. Fuzzy match
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None


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


    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """Notify display layer that the model is generating tool call arguments.

        Fires once per tool name when the streaming response begins producing
        tool_call / tool_use tokens.  Gives the TUI a chance to show a spinner
        or status line so the user isn't staring at a frozen screen while a
        large tool payload (e.g. a 45 KB write_file) is being generated.
        """
        cb = self.tool_gen_callback
        if cb is not None:
            try:
                cb(tool_name)
            except Exception:
                pass


    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
        validate the Chat Completions schema and reject unknown fields (call_id,
        response_item_id) with 400 or 422 errors. These fields are preserved in
        the internal message history — this method only modifies the outgoing
        API copy.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).

        Fields stripped: call_id, response_item_id
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg


    def _should_sanitize_tool_calls(self) -> bool:
        """Determine if tool_calls need sanitization for strict APIs.

        Codex Responses API uses fields like call_id and response_item_id
        that are not part of the standard Chat Completions schema. These
        fields must be stripped when calling any other API to avoid
        validation errors (400 Bad Request).

        Returns:
            bool: True if sanitization is needed (non-Codex API), False otherwise.
        """
        return self.api_mode != "codex_responses"


    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        Dispatches to concurrent execution only for batches that look
        independent: read-only tools may always share the parallel path, while
        file reads/writes may do so only when their target paths do not overlap.
        """
        tool_calls = assistant_message.tool_calls

        # Allow _vprint during tool execution even with stream consumers
        self._executing_tools = True
        try:
            if not _should_parallelize_tool_batch(tool_calls):
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            return self._execute_tool_calls_concurrent(
                assistant_message, messages, effective_task_id, api_call_count
            )
        finally:
            self._executing_tools = False


    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None) -> str:
        """Invoke a single tool and return the result string. No display logic.

        Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
        tools. Used by the concurrent execution path; the sequential path retains
        its own inline invocation for backward-compatible display handling.
        """
        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            if not self._session_db:
                return json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=self._session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # Bridge: notify external memory provider of built-in memory writes
            if self._memory_manager and function_args.get("action") in ("add", "replace"):
                try:
                    self._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                    )
                except Exception:
                    pass
            return result
        elif self._memory_manager and self._memory_manager.has_tool(function_name):
            return self._memory_manager.handle_tool_call(function_name, function_args)
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            from tools.delegate_tool import delegate_task as _delegate_task
            return _delegate_task(
                goal=function_args.get("goal"),
                context=function_args.get("context"),
                toolsets=function_args.get("toolsets"),
                tasks=function_args.get("tasks"),
                max_iterations=function_args.get("max_iterations"),
                parent_agent=self,
            )
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                tool_call_id=tool_call_id,
                session_id=self.session_id or "",
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
            )


    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages so the API sees them in the expected sequence.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── Pre-flight: interrupt check ──────────────────────────────────
        if self._interrupt_requested:
            print(f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                    "tool_call_id": tc.id,
                })
            return

        # ── Parse args + pre-execution bookkeeping ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Reset nudge counters
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Checkpoint for file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

            parsed_calls.append((tool_call, function_name, function_args))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for tc, name, args in parsed_calls:
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args in parsed_calls:
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # Each slot holds (function_name, function_args, function_result, duration, error_flag)
        results = [None] * num_tools

        def _run_tool(index, tool_call, function_name, function_args):
            """Worker function executed in a thread."""
            start = time.time()
            try:
                result = self._invoke_tool(function_name, function_args, effective_task_id, tool_call.id)
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error)

        # Start spinner for CLI mode (skip when TUI handles tool progress)
        spinner = None
        if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=self._print_fn)
            spinner.start()

        try:
            max_workers = min(num_tools, _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, (tc, name, args) in enumerate(parsed_calls):
                    f = executor.submit(_run_tool, i, tc, name, args)
                    futures.append(f)

                # Wait for all to complete (exceptions are captured inside _run_tool)
                concurrent.futures.wait(futures)
        finally:
            if spinner:
                # Build a summary message for the spinner stop
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args) in enumerate(parsed_calls):
            r = results[i]
            if r is None:
                # Shouldn't happen, but safety fallback
                function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error = r

                if is_error:
                    result_preview = function_result[:200] if len(function_result) > 200 else function_result
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                if self.tool_progress_callback:
                    try:
                        self.tool_progress_callback(
                            "tool.completed", function_name, None,
                            duration=tool_duration, is_error=is_error,
                            result=function_result, args=function_args,
                        )
                    except Exception as cb_err:
                        logging.debug(f"Tool progress callback error: {cb_err}")

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # Print cute message per tool
            if self._should_emit_quiet_tool_messages():
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=name,
                tool_use_id=tc.id,
                env=get_active_env(effective_task_id),
            )

            subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
            if subdir_hints:
                function_result += subdir_hints

            # CE-P6: Apply serialization variation to break few-shot patterns
            function_result = apply_serialization_variation(
                function_result, api_call_count, tool_name=name
            )
            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools = len(parsed_calls)
        if num_tools > 0:
            turn_tool_msgs = messages[-num_tools:]
            enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id))


    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls sequentially (original behavior). Used for single calls or interactive tools."""
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # SAFETY: check interrupt BEFORE starting each tool.
            # If the user sent "stop" during a previous tool's execution,
            # do NOT start any more tools -- skip them all immediately.
            if self._interrupt_requested:
                remaining_calls = assistant_message.tool_calls[i-1:]
                if remaining_calls:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            # Reset nudge counters when the relevant tool is actually used
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            self._current_tool = function_name
            self._touch_activity(f"executing tool: {function_name}")

            # Set activity callback for long-running tool execution (terminal
            # commands, etc.) so the gateway's inactivity monitor doesn't kill
            # the agent while a command is running.
            try:
                from tools.environments.base import set_activity_callback
                set_activity_callback(self._touch_activity)
            except Exception:
                pass

            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback("tool.started", function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # Checkpoint: snapshot working dir before file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(
                            work_dir, f"before {function_name}"
                        )
                except Exception:
                    pass  # never block tool execution

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass  # never block tool execution

            tool_start_time = time.time()

            if function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                if not self._session_db:
                    function_result = json.dumps({"success": False, "error": "Session database not available."})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=self._session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            elif function_name == "memory":
                target = function_args.get("target", "memory")
                from tools.memory_tool import memory_tool as _memory_tool
                function_result = _memory_tool(
                    action=function_args.get("action"),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                from tools.delegate_tool import delegate_task as _delegate_task
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = _delegate_task(
                        goal=function_args.get("goal"),
                        context=function_args.get("context"),
                        toolsets=function_args.get("toolsets"),
                        tasks=tasks_arg,
                        max_iterations=function_args.get("max_iterations"),
                        parent_agent=self,
                    )
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._context_engine_tool_names and function_name in self._context_engine_tool_names:
                # Context engine tools (lcm_grep, lcm_describe, lcm_expand, etc.)
                spinner = None
                if self.quiet_mode and not self.tool_progress_callback:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _ce_result = None
                try:
                    function_result = self.context_compressor.handle_tool_call(function_name, function_args, messages=messages)
                    _ce_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                    logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self.quiet_mode:
                        self._vprint(f"  {cute_msg}")
            elif self._memory_manager and self._memory_manager.has_tool(function_name):
                # Memory provider tools (hindsight_retain, honcho_search, etc.)
                # These are not in the tool registry — route through MemoryManager.
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _mem_result = None
                try:
                    function_result = self._memory_manager.handle_tool_call(function_name, function_args)
                    _mem_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                    logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode:
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            result_preview = function_result if self.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )

            # Log tool errors to the persistent error log so [error] tags
            # in the UI always have a corresponding detailed entry on disk.
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, len(function_result))

            if self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed", function_name, None,
                        duration=tool_duration, is_error=_is_error_result,
                        result=function_result, args=function_args,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=function_name,
                tool_use_id=tool_call.id,
                env=get_active_env(effective_task_id),
            )

            # Discover subdirectory context files from tool arguments
            subdir_hints = self._subdirectory_hints.check_tool_call(function_name, function_args)
            if subdir_hints:
                function_result += subdir_hints

            # CE-P6: Apply serialization variation to break few-shot patterns
            function_result = apply_serialization_variation(
                function_result, api_call_count, tool_name=function_name
            )
            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools_seq = len(assistant_message.tool_calls)
        if num_tools_seq > 0:
            enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id))




