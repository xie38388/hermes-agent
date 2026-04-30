"""Slash command handlers: reset, model, status, help, personality, retry, undo, etc."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import copy
import os
logger = logging.getLogger(__name__)

from gateway._helpers import _hermes_home, _load_gateway_config, _platform_config_key, _resolve_gateway_model


from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType

class GwCommandsAdvancedMixin:
    """Advanced slash command handlers: rollback, reasoning, fast, verbose, compress, branch, usage, approve, deny, debug."""

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from tools.checkpoint_manager import CheckpointManager, format_checkpoint_list

        # Read checkpoint config from config.yaml
        cp_cfg = {}
        try:
            import yaml as _y
            _cfg_path = _hermes_home / "config.yaml"
            if _cfg_path.exists():
                with open(_cfg_path, encoding="utf-8") as _f:
                    _data = _y.safe_load(_f) or {}
                cp_cfg = _data.get("checkpoints", {})
                if isinstance(cp_cfg, bool):
                    cp_cfg = {"enabled": cp_cfg}
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._handle_rollback_command", e, exc_info=True)
            pass

        if not cp_cfg.get("enabled", False):
            return (
                "Checkpoints are not enabled.\n"
                "Enable in config.yaml:\n```\ncheckpoints:\n  enabled: true\n```"
            )

        mgr = CheckpointManager(
            enabled=True,
            max_snapshots=cp_cfg.get("max_snapshots", 50),
        )

        cwd = os.getenv("MESSAGING_CWD", str(Path.home()))
        arg = event.get_command_args().strip()

        if not arg:
            checkpoints = mgr.list_checkpoints(cwd)
            return format_checkpoint_list(checkpoints, cwd)

        # Restore by number or hash
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            return f"No checkpoints found for {cwd}"

        target_hash = None
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(checkpoints):
                target_hash = checkpoints[idx]["hash"]
            else:
                return f"Invalid checkpoint number. Use 1-{len(checkpoints)}."
        except ValueError:
            target_hash = arg

        result = mgr.restore(cwd, target_hash)
        if result["success"]:
            return (
                f"✅ Restored to checkpoint {result['restored_to']}: {result['reason']}\n"
                f"A pre-rollback snapshot was saved automatically."
            )
        return f"❌ {result['error']}"

    async def _handle_reasoning_command(self, event: MessageEvent) -> str:
        """Handle /reasoning command — manage reasoning effort and display toggle.

        Usage:
            /reasoning              Show current effort level and display state
            /reasoning <level>      Set reasoning effort (none, minimal, low, medium, high, xhigh)
            /reasoning show|on      Show model reasoning in responses
            /reasoning hide|off     Hide model reasoning from responses
        """
        import yaml

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / "config.yaml"
        self._reasoning_config = self._load_reasoning_config()
        self._show_reasoning = self._load_show_reasoning()

        def _save_config_key(key_path: str, value):
            """Save a dot-separated key to config.yaml."""
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                keys = key_path.split(".")
                current = user_config
                for k in keys[:-1]:
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                atomic_yaml_write(config_path, user_config)
                return True
            except Exception as e:
                logger.error("Failed to save config key %s: %s", key_path, e)
                return False

        if not args:
            # Show current state
            rc = self._reasoning_config
            if rc is None:
                level = "medium (default)"
            elif rc.get("enabled") is False:
                level = "none (disabled)"
            else:
                level = rc.get("effort", "medium")
            display_state = "on ✓" if self._show_reasoning else "off"
            return (
                "🧠 **Reasoning Settings**\n\n"
                f"**Effort:** `{level}`\n"
                f"**Display:** {display_state}\n\n"
                "_Usage:_ `/reasoning <none|minimal|low|medium|high|xhigh|show|hide>`"
            )

        # Display toggle (per-platform)
        platform_key = _platform_config_key(event.source.platform)
        if args in ("show", "on"):
            self._show_reasoning = True
            _save_config_key(f"display.platforms.{platform_key}.show_reasoning", True)
            return (
                "🧠 ✓ Reasoning display: **ON**\n"
                f"Model thinking will be shown before each response on **{platform_key}**."
            )

        if args in ("hide", "off"):
            self._show_reasoning = False
            _save_config_key(f"display.platforms.{platform_key}.show_reasoning", False)
            return f"🧠 ✓ Reasoning display: **OFF** for **{platform_key}**"

        # Effort level change
        effort = args.strip()
        if effort == "none":
            parsed = {"enabled": False}
        elif effort in ("minimal", "low", "medium", "high", "xhigh"):
            parsed = {"enabled": True, "effort": effort}
        else:
            return (
                f"⚠️ Unknown argument: `{effort}`\n\n"
                "**Valid levels:** none, minimal, low, medium, high, xhigh\n"
                "**Display:** show, hide"
            )

        self._reasoning_config = parsed
        if _save_config_key("agent.reasoning_effort", effort):
            return f"🧠 ✓ Reasoning effort set to `{effort}` (saved to config)\n_(takes effect on next message)_"
        else:
            return f"🧠 ✓ Reasoning effort set to `{effort}` (this session only)"

    async def _handle_fast_command(self, event: MessageEvent) -> str:
        """Handle /fast — mirror the CLI Priority Processing toggle in gateway chats."""
        import yaml
        from hermes_cli.models import model_supports_fast_mode

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / "config.yaml"
        self._service_tier = self._load_service_tier()

        user_config = _load_gateway_config()
        model = _resolve_gateway_model(user_config)
        if not model_supports_fast_mode(model):
            return "⚡ /fast is only available for OpenAI models that support Priority Processing."

        def _save_config_key(key_path: str, value):
            """Save a dot-separated key to config.yaml."""
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                keys = key_path.split(".")
                current = user_config
                for k in keys[:-1]:
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                atomic_yaml_write(config_path, user_config)
                return True
            except Exception as e:
                logger.error("Failed to save config key %s: %s", key_path, e)
                return False

        if not args or args == "status":
            status = "fast" if self._service_tier == "priority" else "normal"
            return (
                "⚡ Priority Processing\n\n"
                f"Current mode: `{status}`\n\n"
                "_Usage:_ `/fast <normal|fast|status>`"
            )

        if args in {"fast", "on"}:
            self._service_tier = "priority"
            saved_value = "fast"
            label = "FAST"
        elif args in {"normal", "off"}:
            self._service_tier = None
            saved_value = "normal"
            label = "NORMAL"
        else:
            return (
                f"⚠️ Unknown argument: `{args}`\n\n"
                "**Valid options:** normal, fast, status"
            )

        if _save_config_key("agent.service_tier", saved_value):
            return f"⚡ ✓ Priority Processing: **{label}** (saved to config)\n_(takes effect on next message)_"
        return f"⚡ ✓ Priority Processing: **{label}** (this session only)"

    async def _handle_yolo_command(self, event: MessageEvent) -> str:
        """Handle /yolo — toggle dangerous command approval bypass for this session only."""
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        session_key = self._session_key_for_source(event.source)
        current = is_session_yolo_enabled(session_key)
        if current:
            disable_session_yolo(session_key)
            return "⚠️ YOLO mode **OFF** for this session — dangerous commands will require approval."
        else:
            enable_session_yolo(session_key)
            return "⚡ YOLO mode **ON** for this session — all commands auto-approved. Use with caution."

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose command — cycle tool progress display mode.

        Gated by ``display.tool_progress_command`` in config.yaml (default off).
        When enabled, cycles the tool progress mode through off → new → all →
        verbose → off for the *current platform*.  The setting is saved to
        ``display.platforms.<platform>.tool_progress`` so each channel can
        have its own verbosity level independently.
        """
        import yaml

        config_path = _hermes_home / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        # --- check config gate ------------------------------------------------
        try:
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            gate_enabled = user_config.get("display", {}).get("tool_progress_command", False)
        except Exception:
            gate_enabled = False

        if not gate_enabled:
            return (
                "The `/verbose` command is not enabled for messaging platforms.\n\n"
                "Enable it in `config.yaml`:\n```yaml\n"
                "display:\n  tool_progress_command: true\n```"
            )

        # --- cycle mode (per-platform) ----------------------------------------
        cycle = ["off", "new", "all", "verbose"]
        descriptions = {
            "off": "⚙️ Tool progress: **OFF** — no tool activity shown.",
            "new": "⚙️ Tool progress: **NEW** — shown when tool changes (preview length: `display.tool_preview_length`, default 40).",
            "all": "⚙️ Tool progress: **ALL** — every tool call shown (preview length: `display.tool_preview_length`, default 40).",
            "verbose": "⚙️ Tool progress: **VERBOSE** — every tool call with full arguments.",
        }

        # Read current effective mode for this platform via the resolver
        from gateway.display_config import resolve_display_setting
        current = resolve_display_setting(user_config, platform_key, "tool_progress", "all")
        if current not in cycle:
            current = "all"
        idx = (cycle.index(current) + 1) % len(cycle)
        new_mode = cycle[idx]

        # Save to display.platforms.<platform>.tool_progress
        try:
            if "display" not in user_config or not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            display = user_config["display"]
            if "platforms" not in display or not isinstance(display.get("platforms"), dict):
                display["platforms"] = {}
            if platform_key not in display["platforms"] or not isinstance(display["platforms"].get(platform_key), dict):
                display["platforms"][platform_key] = {}
            display["platforms"][platform_key]["tool_progress"] = new_mode
            atomic_yaml_write(config_path, user_config)
            return (
                f"{descriptions[new_mode]}\n"
                f"_(saved for **{platform_key}** — takes effect on next message)_"
            )
        except Exception as e:
            logger.warning("Failed to save tool_progress mode: %s", e)
            return f"{descriptions[new_mode]}\n_(could not save to config: {e})_"

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context.

        Accepts an optional focus topic: ``/compress <focus>`` guides the
        summariser to preserve information related to *focus* while being
        more aggressive about discarding everything else.
        """
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)

        if not history or len(history) < 4:
            return "Not enough conversation to compress (need at least 4 messages)."

        # Extract optional focus topic from command args
        focus_topic = (event.get_command_args() or "").strip() or None

        try:
            from run_agent import AIAgent
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_messages_tokens_rough

            session_key = self._session_key_for_source(source)
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
            )
            if not runtime_kwargs.get("api_key"):
                return "No provider configured -- cannot compress."

            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            original_count = len(msgs)
            approx_tokens = estimate_messages_tokens_rough(msgs)

            tmp_agent = AIAgent(
                **runtime_kwargs,
                model=model,
                max_iterations=4,
                quiet_mode=True,
                enabled_toolsets=["memory"],
                session_id=session_entry.session_id,
            )
            tmp_agent._print_fn = lambda *a, **kw: None

            compressor = tmp_agent.context_compressor
            compress_start = compressor.protect_first_n
            compress_start = compressor._align_boundary_forward(msgs, compress_start)
            compress_end = compressor._find_tail_cut_by_tokens(msgs, compress_start)
            if compress_start >= compress_end:
                return "Nothing to compress yet (the transcript is still all protected context)."

            loop = asyncio.get_event_loop()
            compressed, _ = await loop.run_in_executor(
                None,
                lambda: tmp_agent._compress_context(msgs, "", approx_tokens=approx_tokens, focus_topic=focus_topic)
            )

            # _compress_context already calls end_session() on the old session
            # (preserving its full transcript in SQLite) and creates a new
            # session_id for the continuation.  Write the compressed messages
            # into the NEW session so the original history stays searchable.
            new_session_id = tmp_agent.session_id
            if new_session_id != session_entry.session_id:
                session_entry.session_id = new_session_id
                self.session_store._save()

            self.session_store.rewrite_transcript(new_session_id, compressed)
            # Reset stored token count — transcript changed, old value is stale
            self.session_store.update_session(
                session_entry.session_key, last_prompt_tokens=0
            )
            new_tokens = estimate_messages_tokens_rough(compressed)
            summary = summarize_manual_compression(
                msgs,
                compressed,
                approx_tokens,
                new_tokens,
            )
            lines = [f"🗜️ {summary['headline']}"]
            if focus_topic:
                lines.append(f"Focus: \"{focus_topic}\"")
            lines.append(summary["token_line"])
            if summary["note"]:
                lines.append(summary["note"])
            return "\n".join(lines)
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return f"Compression failed: {e}"

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_id = session_entry.session_id

        if not self._session_db:
            return "Session database not available."

        # Ensure session exists in SQLite DB (it may only exist in session_store
        # if this is the first command in a new session)
        existing_title = self._session_db.get_session_title(session_id)
        if existing_title is None:
            # Session doesn't exist in DB yet — create it
            try:
                self._session_db.create_session(
                    session_id=session_id,
                    source=source.platform.value if source.platform else "unknown",
                    user_id=source.user_id,
                )
            except Exception:
                pass  # Session might already exist, ignore errors

        title_arg = event.get_command_args().strip()
        if title_arg:
            # Sanitize the title before setting
            try:
                sanitized = self._session_db.sanitize_title(title_arg)
            except ValueError as e:
                return f"⚠️ {e}"
            if not sanitized:
                return "⚠️ Title is empty after cleanup. Please use printable characters."
            # Set the title
            try:
                if self._session_db.set_session_title(session_id, sanitized):
                    return f"✏️ Session title set: **{sanitized}**"
                else:
                    return "Session not found in database."
            except ValueError as e:
                return f"⚠️ {e}"
        else:
            # Show the current title and session ID
            title = self._session_db.get_session_title(session_id)
            if title:
                return f"📌 Session: `{session_id}`\nTitle: **{title}**"
            else:
                return f"📌 Session: `{session_id}`\nNo title set. Usage: `/title My Session Name`"

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — switch to a previously-named session."""
        if not self._session_db:
            return "Session database not available."

        source = event.source
        session_key = self._session_key_for_source(source)
        name = event.get_command_args().strip()

        if not name:
            # List recent titled sessions for this user/platform
            try:
                user_source = source.platform.value if source.platform else None
                sessions = self._session_db.list_sessions_rich(
                    source=user_source, limit=10
                )
                titled = [s for s in sessions if s.get("title")]
                if not titled:
                    return (
                        "No named sessions found.\n"
                        "Use `/title My Session` to name your current session, "
                        "then `/resume My Session` to return to it later."
                    )
                lines = ["📋 **Named Sessions**\n"]
                for s in titled[:10]:
                    title = s["title"]
                    preview = s.get("preview", "")[:40]
                    preview_part = f" — _{preview}_" if preview else ""
                    lines.append(f"• **{title}**{preview_part}")
                lines.append("\nUsage: `/resume <session name>`")
                return "\n".join(lines)
            except Exception as e:
                logger.debug("Failed to list titled sessions: %s", e)
                return f"Could not list sessions: {e}"

        # Resolve the name to a session ID
        target_id = self._session_db.resolve_session_by_title(name)
        if not target_id:
            return (
                f"No session found matching '**{name}**'.\n"
                "Use `/resume` with no arguments to see available sessions."
            )

        # Check if already on that session
        current_entry = self.session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return f"📌 Already on session **{name}**."

        # Flush memories for current session before switching
        try:
            _flush_task = asyncio.create_task(
                self._async_flush_memories(current_entry.session_id, session_key)
            )
            self._background_tasks.add(_flush_task)
            _flush_task.add_done_callback(self._background_tasks.discard)
        except Exception as e:
            logger.debug("Memory flush on resume failed: %s", e)

        # Clear any running agent for this session key
        if session_key in self._running_agents:
            del self._running_agents[session_key]

        # Switch the session entry to point at the old session
        new_entry = self.session_store.switch_session(session_key, target_id)
        if not new_entry:
            return "Failed to switch session."

        # Get the title for confirmation
        title = self._session_db.get_session_title(target_id) or name

        # Count messages for context
        history = self.session_store.load_transcript(target_id)
        msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
        msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""

        return f"↻ Resumed session **{title}**{msg_part}. Conversation restored."

    async def _handle_branch_command(self, event: MessageEvent) -> str:
        """Handle /branch [name] — fork the current session into a new independent copy.

        Copies conversation history to a new session so the user can explore
        a different approach without losing the original.
        Inspired by Claude Code's /branch command.
        """
        import uuid as _uuid

        if not self._session_db:
            return "Session database not available."

        source = event.source
        session_key = self._session_key_for_source(source)

        # Load the current session and its transcript
        current_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(current_entry.session_id)
        if not history:
            return "No conversation to branch — send a message first."

        branch_name = event.get_command_args().strip()

        # Generate the new session ID
        from datetime import datetime as _dt
        now = _dt.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = _uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        # Determine branch title
        if branch_name:
            branch_title = branch_name
        else:
            current_title = self._session_db.get_session_title(current_entry.session_id)
            base = current_title or "branch"
            branch_title = self._session_db.get_next_title_in_lineage(base)

        parent_session_id = current_entry.session_id

        # Create the new session with parent link
        try:
            self._session_db.create_session(
                session_id=new_session_id,
                source=source.platform.value if source.platform else "gateway",
                model=(self.config.get("model", {}) or {}).get("default") if isinstance(self.config, dict) else None,
                parent_session_id=parent_session_id,
            )
        except Exception as e:
            logger.error("Failed to create branch session: %s", e)
            return f"Failed to create branch: {e}"

        # Copy conversation history to the new session
        for msg in history:
            try:
                self._session_db.append_message(
                    session_id=new_session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name") or msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    reasoning=msg.get("reasoning"),
                )
            except Exception:
                pass  # Best-effort copy

        # Set title
        try:
            self._session_db.set_session_title(new_session_id, branch_title)
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._handle_branch_command", e, exc_info=True)
            pass

        # Switch the session store entry to the new session
        new_entry = self.session_store.switch_session(session_key, new_session_id)
        if not new_entry:
            return "Branch created but failed to switch to it."

        # Evict any cached agent for this session
        self._evict_cached_agent(session_key)

        msg_count = len([m for m in history if m.get("role") == "user"])
        return (
            f"⑂ Branched to **{branch_title}**"
            f" ({msg_count} message{'s' if msg_count != 1 else ''} copied)\n"
            f"Original: `{parent_session_id}`\n"
            f"Branch: `{new_session_id}`\n"
            f"Use `/resume` to switch back to the original."
        )

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the current session.

        Checks both _running_agents (mid-turn) and _agent_cache (between turns)
        so that rate limits, cost estimates, and detailed token breakdowns are
        available whenever the user asks, not only while the agent is running.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        # Try running agent first (mid-turn), then cached agent (between turns)
        agent = self._running_agents.get(session_key)
        if not agent or agent is _AGENT_PENDING_SENTINEL:
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    cached = _cache.get(session_key)
                    if cached:
                        agent = cached[0]

        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = []

            # Rate limits (when available from provider headers)
            rl_state = agent.get_rate_limit_state()
            if rl_state and rl_state.has_data:
                from agent.rate_limit_tracker import format_rate_limit_compact
                lines.append(f"⏱️ **Rate Limits:** {format_rate_limit_compact(rl_state)}")
                lines.append("")

            # Session token usage — detailed breakdown matching CLI
            input_tokens = getattr(agent, "session_input_tokens", 0) or 0
            output_tokens = getattr(agent, "session_output_tokens", 0) or 0
            cache_read = getattr(agent, "session_cache_read_tokens", 0) or 0
            cache_write = getattr(agent, "session_cache_write_tokens", 0) or 0

            lines.append("📊 **Session Token Usage**")
            lines.append(f"Model: `{agent.model}`")
            lines.append(f"Input tokens: {input_tokens:,}")
            if cache_read:
                lines.append(f"Cache read tokens: {cache_read:,}")
            if cache_write:
                lines.append(f"Cache write tokens: {cache_write:,}")
            lines.append(f"Output tokens: {output_tokens:,}")
            lines.append(f"Total: {agent.session_total_tokens:,}")
            lines.append(f"API calls: {agent.session_api_calls}")

            # Cost estimation
            try:
                from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
                cost_result = estimate_usage_cost(
                    agent.model,
                    CanonicalUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read,
                        cache_write_tokens=cache_write,
                    ),
                    provider=getattr(agent, "provider", None),
                    base_url=getattr(agent, "base_url", None),
                )
                if cost_result.amount_usd is not None:
                    prefix = "~" if cost_result.status == "estimated" else ""
                    lines.append(f"Cost: {prefix}${float(cost_result.amount_usd):.4f}")
                elif cost_result.status == "included":
                    lines.append("Cost: included")
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._handle_usage_command", e, exc_info=True)
                pass

            # Context window and compressions
            ctx = agent.context_compressor
            if ctx.last_prompt_tokens:
                pct = min(100, ctx.last_prompt_tokens / ctx.context_length * 100) if ctx.context_length else 0
                lines.append(f"Context: {ctx.last_prompt_tokens:,} / {ctx.context_length:,} ({pct:.0f}%)")
            if ctx.compression_count:
                lines.append(f"Compressions: {ctx.compression_count}")

            return "\n".join(lines)

        # No agent at all -- check session history for a rough count
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough
            msgs = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
            approx = estimate_messages_tokens_rough(msgs)
            return (
                f"📊 **Session Info**\n"
                f"Messages: {len(msgs)}\n"
                f"Estimated context: ~{approx:,} tokens\n"
                f"_(Detailed usage available after the first agent response)_"
            )
        return "No usage data available for this session."

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        import asyncio as _asyncio

        args = event.get_command_args().strip()
        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return f"Invalid --days value: {parts[i + 1]}"
                    i += 2
                elif parts[i] == "--source" and i + 1 < len(parts):
                    source = parts[i + 1]
                    i += 2
                elif parts[i].isdigit():
                    days = int(parts[i])
                    i += 1
                else:
                    i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            loop = _asyncio.get_event_loop()

            def _run_insights():
                db = SessionDB()
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                result = engine.format_gateway(report)
                db.close()
                return result

            return await loop.run_in_executor(None, _run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return f"Error generating insights: {e}"

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> str:
        """Handle /reload-mcp command -- disconnect and reconnect all MCP servers."""
        loop = asyncio.get_event_loop()
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _load_mcp_config, _servers, _lock

            # Capture old server names before shutdown
            with _lock:
                old_servers = set(_servers.keys())

            # Read new config before shutting down, so we know what will be added/removed
            # Shutdown existing connections
            await loop.run_in_executor(None, shutdown_mcp_servers)

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await loop.run_in_executor(None, discover_mcp_tools)

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = ["🔄 **MCP Servers Reloaded**\n"]
            if reconnected:
                lines.append(f"♻️ Reconnected: {', '.join(sorted(reconnected))}")
            if added:
                lines.append(f"➕ Added: {', '.join(sorted(added))}")
            if removed:
                lines.append(f"➖ Removed: {', '.join(sorted(removed))}")
            if not connected_servers:
                lines.append("No MCP servers connected.")
            else:
                lines.append(f"\n🔧 {len(new_tools)} tool(s) available from {len(connected_servers)} server(s)")

            # Inject a message at the END of the session history so the
            # model knows tools changed on its next turn.  Appended after
            # all existing messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[SYSTEM: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = self.session_store.get_or_create_session(event.source)
                self.session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return f"❌ MCP reload failed: {e}"

    async def _handle_approve_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /approve command — unblock waiting agent thread(s).

        The agent thread(s) are blocked inside tools/approval.py waiting for
        the user to respond.  This handler signals the event so the agent
        resumes and the terminal_tool executes the command inline — the same
        flow as the CLI's synchronous input() approval.

        Supports multiple concurrent approvals (parallel subagents,
        execute_code).  ``/approve`` resolves the oldest pending command;
        ``/approve all`` resolves every pending command at once.

        Usage:
            /approve              — approve oldest pending command once
            /approve all          — approve ALL pending commands at once
            /approve session      — approve oldest + remember for session
            /approve all session  — approve all + remember for session
            /approve always       — approve oldest + remember permanently
            /approve all always   — approve all + remember permanently
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )

        if not has_blocking_approval(session_key):
            if session_key in self._pending_approvals:
                self._pending_approvals.pop(session_key)
                return "⚠️ Approval expired (agent is no longer waiting). Ask the agent to try again."
            return "No pending command to approve."

        # Parse args: support "all", "all session", "all always", "session", "always"
        args = event.get_command_args().strip().lower().split()
        resolve_all = "all" in args
        remaining = [a for a in args if a != "all"]

        if any(a in ("always", "permanent", "permanently") for a in remaining):
            choice = "always"
            scope_msg = " (pattern approved permanently)"
        elif any(a in ("session", "ses") for a in remaining):
            choice = "session"
            scope_msg = " (pattern approved for this session)"
        else:
            choice = "once"
            scope_msg = ""

        count = resolve_gateway_approval(session_key, choice, resolve_all=resolve_all)
        if not count:
            return "No pending command to approve."

        # Resume typing indicator — agent is about to continue processing.
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        count_msg = f" ({count} commands)" if count > 1 else ""
        logger.info("User approved %d dangerous command(s) via /approve%s", count, scope_msg)
        return f"✅ Command{'s' if count > 1 else ''} approved{scope_msg}{count_msg}. The agent is resuming..."

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — reject pending dangerous command(s).

        Signals blocked agent thread(s) with a 'deny' result so they receive
        a definitive BLOCKED message, same as the CLI deny flow.

        ``/deny`` denies the oldest; ``/deny all`` denies everything.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )

        if not has_blocking_approval(session_key):
            if session_key in self._pending_approvals:
                self._pending_approvals.pop(session_key)
                return "❌ Command denied (approval was stale)."
            return "No pending command to deny."

        args = event.get_command_args().strip().lower()
        resolve_all = "all" in args

        count = resolve_gateway_approval(session_key, "deny", resolve_all=resolve_all)
        if not count:
            return "No pending command to deny."

        # Resume typing indicator — agent continues (with BLOCKED result).
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        count_msg = f" ({count} commands)" if count > 1 else ""
        logger.info("User denied %d dangerous command(s) via /deny", count)
        return f"❌ Command{'s' if count > 1 else ''} denied{count_msg}."

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug — upload debug report + logs and return paste URLs."""
        import asyncio
        from hermes_cli.debug import (
            _capture_dump, collect_debug_report, _read_full_log,
            upload_to_pastebin,
        )

        loop = asyncio.get_running_loop()

        # Run blocking I/O (dump capture, log reads, uploads) in a thread.
        def _collect_and_upload():
            dump_text = _capture_dump()
            report = collect_debug_report(log_lines=200, dump_text=dump_text)
            agent_log = _read_full_log("agent")
            gateway_log = _read_full_log("gateway")

            if agent_log:
                agent_log = dump_text + "\n\n--- full agent.log ---\n" + agent_log
            if gateway_log:
                gateway_log = dump_text + "\n\n--- full gateway.log ---\n" + gateway_log

            urls = {}
            failures = []

            try:
                urls["Report"] = upload_to_pastebin(report)
            except Exception as exc:
                return f"✗ Failed to upload debug report: {exc}"

            if agent_log:
                try:
                    urls["agent.log"] = upload_to_pastebin(agent_log)
                except Exception:
                    failures.append("agent.log")

            if gateway_log:
                try:
                    urls["gateway.log"] = upload_to_pastebin(gateway_log)
                except Exception:
                    failures.append("gateway.log")

            lines = ["**Debug report uploaded:**", ""]
            label_width = max(len(k) for k in urls)
            for label, url in urls.items():
                lines.append(f"`{label:<{label_width}}`  {url}")

            if failures:
                lines.append(f"\n_(failed to upload: {', '.join(failures)})_")

            lines.append("\nShare these links with the Hermes team for support.")
            return "\n".join(lines)

        return await loop.run_in_executor(None, _collect_and_upload)

