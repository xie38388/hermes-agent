"""Session configuration, voice modes, runtime status, agent config resolution, and drain management."""
from __future__ import annotations
from pathlib import Path
import json

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import asyncio
import os
import shlex
import time
logger = logging.getLogger(__name__)

from gateway._helpers import _resolve_gateway_model, _resolve_hermes_bin, _resolve_runtime_agent_kwargs


from gateway.config import Platform
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, merge_pending_message_event
from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT, parse_restart_drain_timeout
from gateway._helpers import _hermes_home

class GwSessionConfigMixin:
    _VOICE_MODE_PATH = _hermes_home / "gateway_voice_mode.json"

    """Session configuration, voice modes, runtime status, agent config resolution, and drain management."""

    def _load_voice_modes(self) -> Dict[str, str]:
        try:
            data = json.loads(self._VOICE_MODE_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}

        valid_modes = {"off", "voice_only", "all"}
        return {
            str(chat_id): mode
            for chat_id, mode in data.items()
            if mode in valid_modes
        }

    def _save_voice_modes(self) -> None:
        try:
            self._VOICE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._VOICE_MODE_PATH.write_text(
                json.dumps(self._voice_mode, indent=2)
            )
        except OSError as e:
            logger.warning("Failed to save voice modes: %s", e)

    def _set_adapter_auto_tts_disabled(self, adapter, chat_id: str, disabled: bool) -> None:
        """Update an adapter's in-memory auto-TTS suppression set if present."""
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if not isinstance(disabled_chats, set):
            return
        if disabled:
            disabled_chats.add(chat_id)
        else:
            disabled_chats.discard(chat_id)

    def _sync_voice_mode_state_to_adapter(self, adapter) -> None:
        """Restore persisted /voice off state into a live platform adapter."""
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if not isinstance(disabled_chats, set):
            return
        disabled_chats.clear()
        disabled_chats.update(
            chat_id for chat_id, mode in self._voice_mode.items() if mode == "off"
        )

    def _flush_memories_for_session(
        self,
        old_session_id: str,
        session_key: Optional[str] = None,
    ):
        """Prompt the agent to save memories/skills before context is lost.

        Synchronous worker — meant to be called via run_in_executor from
        an async context so it doesn't block the event loop.
        """
        # Skip cron sessions — they run headless with no meaningful user
        # conversation to extract memories from.
        if old_session_id and old_session_id.startswith("cron_"):
            logger.debug("Skipping memory flush for cron session: %s", old_session_id)
            return

        try:
            history = self.session_store.load_transcript(old_session_id)
            if not history or len(history) < 4:
                return

            from run_agent import AIAgent
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                session_key=session_key,
            )
            if not runtime_kwargs.get("api_key"):
                return

            tmp_agent = AIAgent(
                **runtime_kwargs,
                model=model,
                max_iterations=8,
                quiet_mode=True,
                skip_memory=True,  # Flush agent — no memory provider
                enabled_toolsets=["memory", "skills"],
                session_id=old_session_id,
            )
            # Fully silence the flush agent — quiet_mode only suppresses init
            # messages; tool call output still leaks to the terminal through
            # _safe_print → _print_fn.  Set a no-op to prevent that.
            tmp_agent._print_fn = lambda *a, **kw: None

            # Build conversation history from transcript
            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]

            # Read live memory state from disk so the flush agent can see
            # what's already saved and avoid overwriting newer entries.
            _current_memory = ""
            try:
                from tools.memory_tool import get_memory_dir
                _mem_dir = get_memory_dir()
                for fname, label in [
                    ("MEMORY.md", "MEMORY (your personal notes)"),
                    ("USER.md", "USER PROFILE (who the user is)"),
                ]:
                    fpath = _mem_dir / fname
                    if fpath.exists():
                        content = fpath.read_text(encoding="utf-8").strip()
                        if content:
                            _current_memory += f"\n\n## Current {label}:\n{content}"
            except Exception:
                pass  # Non-fatal — flush still works, just without the guard

            # Give the agent a real turn to think about what to save
            flush_prompt = (
                "[System: This session is about to be automatically reset due to "
                "inactivity or a scheduled daily reset. The conversation context "
                "will be cleared after this turn.\n\n"
                "Review the conversation above and:\n"
                "1. Save any important facts, preferences, or decisions to memory "
                "(user profile or your notes) that would be useful in future sessions.\n"
                "2. If you discovered a reusable workflow or solved a non-trivial "
                "problem, consider saving it as a skill.\n"
                "3. If nothing is worth saving, that's fine — just skip.\n\n"
            )

            if _current_memory:
                flush_prompt += (
                    "IMPORTANT — here is the current live state of memory. Other "
                    "sessions, cron jobs, or the user may have updated it since this "
                    "conversation ended. Do NOT overwrite or remove entries unless "
                    "the conversation above reveals something that genuinely "
                    "supersedes them. Only add new information that is not already "
                    "captured below."
                    f"{_current_memory}\n\n"
                )

            flush_prompt += (
                "Do NOT respond to the user. Just use the memory and skill_manage "
                "tools if needed, then stop.]"
            )

            tmp_agent.run_conversation(
                user_message=flush_prompt,
                conversation_history=msgs,
            )
            logger.info("Pre-reset memory flush completed for session %s", old_session_id)
        except Exception as e:
            logger.debug("Pre-reset memory flush failed for session %s: %s", old_session_id, e)

    async def _async_flush_memories(
        self,
        old_session_id: str,
        session_key: Optional[str] = None,
    ):
        """Run the sync memory flush in a thread pool so it won't block the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._flush_memories_for_session,
            old_session_id,
            session_key,
        )

    @property
    def should_exit_cleanly(self) -> bool:
        return self._exit_cleanly

    @property
    def should_exit_with_failure(self) -> bool:
        return self._exit_with_failure

    @property
    def exit_reason(self) -> Optional[str]:
        return self._exit_reason

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        if hasattr(self, "session_store") and self.session_store is not None:
            try:
                session_key = self.session_store._generate_session_key(source)
                if isinstance(session_key, str) and session_key:
                    return session_key
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._session_key_for_source", e, exc_info=True)
                pass
        config = getattr(self, "config", None)
        return build_session_key(
            source,
            group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
        )

    def _resolve_session_agent_runtime(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session, honoring session-scoped /model overrides.

        If the session override already contains a complete provider bundle
        (provider/api_key/base_url/api_mode), prefer it directly instead of
        resolving fresh global runtime state first.
        """
        resolved_session_key = session_key
        if not resolved_session_key and source is not None:
            try:
                resolved_session_key = self._session_key_for_source(source)
            except Exception:
                resolved_session_key = None

        model = _resolve_gateway_model(user_config)
        override = self._session_model_overrides.get(resolved_session_key) if resolved_session_key else None
        if override:
            override_model = override.get("model", model)
            override_runtime = {
                "provider": override.get("provider"),
                "api_key": override.get("api_key"),
                "base_url": override.get("base_url"),
                "api_mode": override.get("api_mode"),
            }
            if override_runtime.get("api_key"):
                logger.debug(
                    "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                    (resolved_session_key or "")[:30], model, override_model,
                    override_runtime.get("provider"),
                )
                return override_model, override_runtime
            # Override exists but has no api_key — fall through to env-based
            # resolution and apply model/provider from the override on top.
            logger.debug(
                "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
                (resolved_session_key or "")[:30], model, override_model,
            )
        else:
            logger.debug(
                "No session model override: session=%s config_model=%s override_keys=%s",
                (resolved_session_key or "")[:30], model,
                list(self._session_model_overrides.keys())[:5] if self._session_model_overrides else "[]",
            )

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        if override and resolved_session_key:
            model, runtime_kwargs = self._apply_session_model_override(
                resolved_session_key, model, runtime_kwargs
            )

        # When the config has no model.default but a provider was resolved
        # (e.g. user ran `hermes auth add openai-codex` without `hermes model`),
        # fall back to the provider's first catalog model so the API call
        # doesn't fail with "model must be a non-empty string".
        if not model and runtime_kwargs.get("provider"):
            try:
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        model, runtime_kwargs["provider"],
                    )
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._resolve_session_agent_runtime", e, exc_info=True)
                pass

        return model, runtime_kwargs

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        from agent.smart_model_routing import resolve_turn_route
        from hermes_cli.models import resolve_fast_mode_overrides

        primary = {
            "model": model,
            "api_key": runtime_kwargs.get("api_key"),
            "base_url": runtime_kwargs.get("base_url"),
            "provider": runtime_kwargs.get("provider"),
            "api_mode": runtime_kwargs.get("api_mode"),
            "command": runtime_kwargs.get("command"),
            "args": list(runtime_kwargs.get("args") or []),
            "credential_pool": runtime_kwargs.get("credential_pool"),
        }
        route = resolve_turn_route(user_message, getattr(self, "_smart_model_routing", {}), primary)

        service_tier = getattr(self, "_service_tier", None)
        if not service_tier:
            route["request_overrides"] = None
            return route

        try:
            overrides = resolve_fast_mode_overrides(route.get("model"))
        except Exception:
            overrides = None
        route["request_overrides"] = overrides
        return route

    async def _handle_adapter_fatal_error(self, adapter: BasePlatformAdapter) -> None:
        """React to an adapter failure after startup.

        If the error is retryable (e.g. network blip, DNS failure), queue the
        platform for background reconnection instead of giving up permanently.
        """
        logger.error(
            "Fatal %s adapter error (%s): %s",
            adapter.platform.value,
            adapter.fatal_error_code or "unknown",
            adapter.fatal_error_message or "unknown error",
        )
        self._update_platform_runtime_status(
            adapter.platform.value,
            platform_state="retrying" if adapter.fatal_error_retryable else "fatal",
            error_code=adapter.fatal_error_code,
            error_message=adapter.fatal_error_message,
        )

        existing = self.adapters.get(adapter.platform)
        if existing is adapter:
            try:
                await adapter.disconnect()
            finally:
                self.adapters.pop(adapter.platform, None)
                self.delivery_router.adapters = self.adapters

        # Queue retryable failures for background reconnection
        if adapter.fatal_error_retryable:
            platform_config = self.config.platforms.get(adapter.platform)
            if platform_config and adapter.platform not in self._failed_platforms:
                self._failed_platforms[adapter.platform] = {
                    "config": platform_config,
                    "attempts": 0,
                    "next_retry": time.monotonic() + 30,
                }
                logger.info(
                    "%s queued for background reconnection",
                    adapter.platform.value,
                )

        if not self.adapters and not self._failed_platforms:
            self._exit_reason = adapter.fatal_error_message or "All messaging adapters disconnected"
            if adapter.fatal_error_retryable:
                self._exit_with_failure = True
                logger.error("No connected messaging platforms remain. Shutting down gateway for service restart.")
            else:
                logger.error("No connected messaging platforms remain. Shutting down gateway cleanly.")
            await self.stop()
        elif not self.adapters and self._failed_platforms:
            # All platforms are down and queued for background reconnection.
            # If the error is retryable, exit with failure so systemd Restart=on-failure
            # can restart the process. Otherwise stay alive and keep retrying in background.
            if adapter.fatal_error_retryable:
                self._exit_reason = adapter.fatal_error_message or "All messaging platforms failed with retryable errors"
                self._exit_with_failure = True
                logger.error(
                    "All messaging platforms failed with retryable errors. "
                    "Shutting down gateway for service restart (systemd will retry)."
                )
                await self.stop()
            else:
                logger.warning(
                    "No connected messaging platforms remain, but %d platform(s) queued for reconnection",
                    len(self._failed_platforms),
                )

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()

    def _running_agent_count(self) -> int:
        return len(self._running_agents)

    def _status_action_label(self) -> str:
        return "restart" if self._restart_requested else "shutdown"

    def _status_action_gerund(self) -> str:
        return "restarting" if self._restart_requested else "shutting down"

    def _queue_during_drain_enabled(self) -> bool:
        return self._restart_requested and self._busy_input_mode == "queue"

    def _update_runtime_status(self, gateway_state: Optional[str] = None, exit_reason: Optional[str] = None) -> None:
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                gateway_state=gateway_state,
                exit_reason=exit_reason,
                restart_requested=self._restart_requested,
                active_agents=self._running_agent_count(),
            )
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._update_runtime_status", e, exc_info=True)
            pass

    def _update_platform_runtime_status(
        self,
        platform: str,
        *,
        platform_state: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                platform=platform,
                platform_state=platform_state,
                error_code=error_code,
                error_message=error_message,
            )
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._update_platform_runtime_status", e, exc_info=True)
            pass

    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.
        
        Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
        the prefill_messages_file key in ~/.hermes/config.yaml.
        Relative paths are resolved from ~/.hermes/.
        """
        import json as _json
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    file_path = cfg.get("prefill_messages_file", "")
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._load_prefill_messages", e, exc_info=True)
                pass
        if not file_path:
            return []
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = _hermes_home / path
        if not path.exists():
            logger.warning("Prefill messages file not found: %s", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt from config or env var.
        
        Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
        agent.system_prompt in ~/.hermes/config.yaml.
        """
        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return (cfg.get("agent", {}).get("system_prompt", "") or "").strip()
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._load_ephemeral_system_prompt", e, exc_info=True)
            pass
        return ""

    @staticmethod
    def _load_reasoning_config() -> dict | None:
        """Load reasoning effort from config.yaml.

        Reads agent.reasoning_effort from config.yaml. Valid: "none",
        "minimal", "low", "medium", "high", "xhigh". Returns None to use
        default (medium).
        """
        from hermes_constants import parse_reasoning_effort
        effort = ""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                effort = str(cfg.get("agent", {}).get("reasoning_effort", "") or "").strip()
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._load_reasoning_config", e, exc_info=True)
            pass
        result = parse_reasoning_effort(effort)
        if effort and effort.strip() and result is None:
            logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
        return result

    @staticmethod
    def _load_service_tier() -> str | None:
        """Load Priority Processing setting from config.yaml.

        Reads agent.service_tier from config.yaml. Accepted values mirror the CLI:
        "fast"/"priority"/"on" => "priority", while "normal"/"off" disables it.
        Returns None when unset or unsupported.
        """
        raw = ""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                raw = str(cfg.get("agent", {}).get("service_tier", "") or "").strip()
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._load_service_tier", e, exc_info=True)
            pass

        value = raw.lower()
        if not value or value in {"normal", "default", "standard", "off", "none"}:
            return None
        if value in {"fast", "priority", "on"}:
            return "priority"
        logger.warning("Unknown service_tier '%s', ignoring", raw)
        return None

    @staticmethod
    def _load_show_reasoning() -> bool:
        """Load show_reasoning toggle from config.yaml display section."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return bool(cfg.get("display", {}).get("show_reasoning", False))
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._load_show_reasoning", e, exc_info=True)
            pass
        return False

    @staticmethod
    def _load_busy_input_mode() -> str:
        """Load gateway drain-time busy-input behavior from config/env."""
        mode = os.getenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "").strip().lower()
        if not mode:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    mode = str(cfg.get("display", {}).get("busy_input_mode", "") or "").strip().lower()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._load_busy_input_mode", e, exc_info=True)
                pass
        return "queue" if mode == "queue" else "interrupt"

    @staticmethod
    def _load_restart_drain_timeout() -> float:
        """Load graceful gateway restart/stop drain timeout in seconds."""
        raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
        if not raw:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    raw = str(cfg.get("agent", {}).get("restart_drain_timeout", "") or "").strip()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._load_restart_drain_timeout", e, exc_info=True)
                pass
        value = parse_restart_drain_timeout(raw)
        if raw and value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT:
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid restart_drain_timeout '%s', using default %.0fs",
                    raw,
                    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
                )
        return value

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Load background process notification mode from config or env var.

        Modes:
          - ``all``    — push running-output updates *and* the final message (default)
          - ``result`` — only the final completion message (regardless of exit code)
          - ``error``  — only the final message when exit code is non-zero
          - ``off``    — no watcher messages at all
        """
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    raw = cfg.get("display", {}).get("background_process_notifications")
                    if raw is False:
                        mode = "off"
                    elif raw not in (None, ""):
                        mode = str(raw)
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._load_background_notifications_mode", e, exc_info=True)
                pass
        mode = (mode or "all").strip().lower()
        valid = {"all", "result", "error", "off"}
        if mode not in valid:
            logger.warning(
                "Unknown background_process_notifications '%s', defaulting to 'all'",
                mode,
            )
            return "all"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """Load OpenRouter provider routing preferences from config.yaml."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return cfg.get("provider_routing", {}) or {}
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._load_provider_routing", e, exc_info=True)
            pass
        return {}

    @staticmethod
    def _load_fallback_model() -> list | dict | None:
        """Load fallback provider chain from config.yaml.

        Returns a list of provider dicts (``fallback_providers``), a single
        dict (legacy ``fallback_model``), or None if not configured.
        AIAgent.__init__ normalizes both formats into a chain.
        """
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                fb = cfg.get("fallback_providers") or cfg.get("fallback_model") or None
                if fb:
                    return fb
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._load_fallback_model", e, exc_info=True)
            pass
        return None

    @staticmethod
    def _load_smart_model_routing() -> dict:
        """Load optional smart cheap-vs-strong model routing config."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return cfg.get("smart_model_routing", {}) or {}
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run._load_smart_model_routing", e, exc_info=True)
            pass
        return {}

    def _snapshot_running_agents(self) -> Dict[str, Any]:
        return {
            session_key: agent
            for session_key, agent in self._running_agents.items()
            if agent is not _AGENT_PENDING_SENTINEL
        }

    def _queue_or_replace_pending_event(self, session_key: str, event: MessageEvent) -> None:
        adapter = self.adapters.get(event.source.platform)
        if not adapter:
            return
        merge_pending_message_event(adapter._pending_messages, session_key, event)

    async def _handle_active_session_busy_message(self, event: MessageEvent, session_key: str) -> bool:
        if not self._draining:
            return False

        adapter = self.adapters.get(event.source.platform)
        if not adapter:
            return True

        thread_meta = {"thread_id": event.source.thread_id} if event.source.thread_id else None
        if self._queue_during_drain_enabled():
            self._queue_or_replace_pending_event(session_key, event)
            message = f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
        else:
            message = f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."

        await adapter._send_with_retry(
            chat_id=event.source.chat_id,
            content=message,
            reply_to=event.message_id,
            metadata=thread_meta,
        )
        return True

    async def _drain_active_agents(self, timeout: float) -> tuple[Dict[str, Any], bool]:
        snapshot = self._snapshot_running_agents()
        last_active_count = self._running_agent_count()
        last_status_at = 0.0

        def _maybe_update_status(force: bool = False) -> None:
            nonlocal last_active_count, last_status_at
            now = asyncio.get_running_loop().time()
            active_count = self._running_agent_count()
            if force or active_count != last_active_count or (now - last_status_at) >= 1.0:
                self._update_runtime_status("draining")
                last_active_count = active_count
                last_status_at = now

        if not self._running_agents:
            _maybe_update_status(force=True)
            return snapshot, False

        _maybe_update_status(force=True)
        if timeout <= 0:
            return snapshot, True

        deadline = asyncio.get_running_loop().time() + timeout
        while self._running_agents and asyncio.get_running_loop().time() < deadline:
            _maybe_update_status()
            await asyncio.sleep(0.1)
        timed_out = bool(self._running_agents)
        _maybe_update_status(force=True)
        return snapshot, timed_out

    def _interrupt_running_agents(self, reason: str) -> None:
        for session_key, agent in list(self._running_agents.items()):
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                agent.interrupt(reason)
                logger.debug("Interrupted running agent for session %s during shutdown", session_key[:20])
            except Exception as e:
                logger.debug("Failed interrupting agent during shutdown: %s", e)

    def _finalize_shutdown_agents(self, active_agents: Dict[str, Any]) -> None:
        for agent in active_agents.values():
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "on_session_finalize",
                    session_id=getattr(agent, "session_id", None),
                    platform="gateway",
                )
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._finalize_shutdown_agents", e, exc_info=True)
                pass
            try:
                if hasattr(agent, "shutdown_memory_provider"):
                    agent.shutdown_memory_provider()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._finalize_shutdown_agents", e, exc_info=True)
                pass
            # Close tool resources (terminal sandboxes, browser daemons,
            # background processes, httpx clients) to prevent zombie
            # process accumulation.
            try:
                if hasattr(agent, 'close'):
                    agent.close()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._finalize_shutdown_agents", e, exc_info=True)
                pass

    async def _launch_detached_restart_command(self) -> None:
        import shutil
        import subprocess

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            logger.error("Could not locate hermes binary for detached /restart")
            return

        current_pid = os.getpid()
        cmd = " ".join(shlex.quote(part) for part in hermes_cmd)
        shell_cmd = (
            f"while kill -0 {current_pid} 2>/dev/null; do sleep 0.2; done; "
            f"{cmd} gateway restart"
        )
        setsid_bin = shutil.which("setsid")
        if setsid_bin:
            subprocess.Popen(
                [setsid_bin, "bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            subprocess.Popen(
                ["bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    def request_restart(self, *, detached: bool = False, via_service: bool = False) -> bool:
        if self._restart_task_started:
            return False
        self._restart_requested = True
        self._restart_detached = detached
        self._restart_via_service = via_service
        self._restart_task_started = True

        async def _run_restart() -> None:
            await asyncio.sleep(0.05)
            await self.stop(restart=True, detached_restart=detached, service_restart=via_service)

        task = asyncio.create_task(_run_restart())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return True

