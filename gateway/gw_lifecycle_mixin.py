"""Gateway lifecycle: start, stop, watchers, adapter creation, and shutdown coordination."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import asyncio
import os
import signal
import time
logger = logging.getLogger(__name__)

from gateway._helpers import _hermes_home


from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
class GwLifecycleMixin:
    """Gateway lifecycle: start, stop, watchers, adapter creation, and shutdown coordination."""

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        logger.info("Session storage: %s", self.config.sessions_dir)
        try:
            from hermes_cli.profiles import get_active_profile_name
            _profile = get_active_profile_name()
            if _profile and _profile != "default":
                logger.info("Active profile: %s", _profile)
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run.start", e, exc_info=True)
            pass
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(gateway_state="starting", exit_reason=None)
        except Exception as e:
            logger.warning("Suppressed exception in %s: %s", "run.start", e, exc_info=True)
            pass
        
        # Warn if no user allowlists are configured and open access is not opted in
        _any_allowlist = any(
            os.getenv(v)
            for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                       "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
                       "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
                       "EMAIL_ALLOWED_USERS",
                       "SMS_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
                       "MATRIX_ALLOWED_USERS", "DINGTALK_ALLOWED_USERS",
                       "FEISHU_ALLOWED_USERS",
                       "WECOM_ALLOWED_USERS",
                       "WECOM_CALLBACK_ALLOWED_USERS",
                       "WEIXIN_ALLOWED_USERS",
                       "BLUEBUBBLES_ALLOWED_USERS",
                       "GATEWAY_ALLOWED_USERS")
        )
        _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes") or any(
            os.getenv(v, "").lower() in ("true", "1", "yes")
            for v in ("TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS",
                       "WHATSAPP_ALLOW_ALL_USERS", "SLACK_ALLOW_ALL_USERS",
                       "SIGNAL_ALLOW_ALL_USERS", "EMAIL_ALLOW_ALL_USERS",
                       "SMS_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS",
                       "MATRIX_ALLOW_ALL_USERS", "DINGTALK_ALLOW_ALL_USERS",
                       "FEISHU_ALLOW_ALL_USERS",
                       "WECOM_ALLOW_ALL_USERS",
                       "WECOM_CALLBACK_ALLOW_ALL_USERS",
                       "WEIXIN_ALLOW_ALL_USERS",
                       "BLUEBUBBLES_ALLOW_ALL_USERS")
        )
        if not _any_allowlist and not _allow_all:
            logger.warning(
                "No user allowlists configured. All unauthorized users will be denied. "
                "Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access, "
                "or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id)."
            )
        
        # Discover and load event hooks
        self.hooks.discover_and_load()
        
        # Recover background processes from checkpoint (crash recovery)
        try:
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        except Exception as e:
            logger.warning("Process checkpoint recovery: %s", e)

        # Suspend sessions that were active when the gateway last exited.
        # This prevents stuck sessions from being blindly resumed on restart,
        # which can create an unrecoverable loop (#7536).  Suspended sessions
        # auto-reset on the next incoming message, giving the user a clean start.
        #
        # SKIP suspension after a clean (graceful) shutdown — the previous
        # process already drained active agents, so sessions aren't stuck.
        # This prevents unwanted auto-resets after `hermes update`,
        # `hermes gateway restart`, or `/restart`.
        _clean_marker = _hermes_home / ".clean_shutdown"
        if _clean_marker.exists():
            logger.info("Previous gateway exited cleanly — skipping session suspension")
            try:
                _clean_marker.unlink()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run.start", e, exc_info=True)
                pass
        else:
            try:
                suspended = self.session_store.suspend_recently_active()
                if suspended:
                    logger.info("Suspended %d in-flight session(s) from previous run", suspended)
            except Exception as e:
                logger.warning("Session suspension on startup failed: %s", e)

        connected_count = 0
        enabled_platform_count = 0
        startup_nonretryable_errors: list[str] = []
        startup_retryable_errors: list[str] = []
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
            enabled_platform_count += 1
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                logger.warning("No adapter available for %s", platform.value)
                continue
            
            # Set up message + fatal error handlers
            adapter.set_message_handler(self._handle_message)
            adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
            adapter.set_session_store(self.session_store)
            adapter.set_busy_session_handler(self._handle_active_session_busy_message)
            
            # Try to connect
            logger.info("Connecting to %s...", platform.value)
            self._update_platform_runtime_status(
                platform.value,
                platform_state="connecting",
                error_code=None,
                error_message=None,
            )
            try:
                success = await adapter.connect()
                if success:
                    self.adapters[platform] = adapter
                    self._sync_voice_mode_state_to_adapter(adapter)
                    connected_count += 1
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="connected",
                        error_code=None,
                        error_message=None,
                    )
                    logger.info("✓ %s connected", platform.value)
                else:
                    logger.warning("✗ %s failed to connect", platform.value)
                    if adapter.has_fatal_error:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying" if adapter.fatal_error_retryable else "fatal",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message,
                        )
                        target = (
                            startup_retryable_errors
                            if adapter.fatal_error_retryable
                            else startup_nonretryable_errors
                        )
                        target.append(
                            f"{platform.value}: {adapter.fatal_error_message}"
                        )
                        # Queue for reconnection if the error is retryable
                        if adapter.fatal_error_retryable:
                            self._failed_platforms[platform] = {
                                "config": platform_config,
                                "attempts": 1,
                                "next_retry": time.monotonic() + 30,
                            }
                    else:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying",
                            error_code=None,
                            error_message="failed to connect",
                        )
                        startup_retryable_errors.append(
                            f"{platform.value}: failed to connect"
                        )
                        # No fatal error info means likely a transient issue — queue for retry
                        self._failed_platforms[platform] = {
                            "config": platform_config,
                            "attempts": 1,
                            "next_retry": time.monotonic() + 30,
                        }
            except Exception as e:
                logger.error("✗ %s error: %s", platform.value, e)
                self._update_platform_runtime_status(
                    platform.value,
                    platform_state="retrying",
                    error_code=None,
                    error_message=str(e),
                )
                startup_retryable_errors.append(f"{platform.value}: {e}")
                # Unexpected exceptions are typically transient — queue for retry
                self._failed_platforms[platform] = {
                    "config": platform_config,
                    "attempts": 1,
                    "next_retry": time.monotonic() + 30,
                }
        
        if connected_count == 0:
            if startup_nonretryable_errors:
                reason = "; ".join(startup_nonretryable_errors)
                logger.error("Gateway hit a non-retryable startup conflict: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
                except Exception as e:
                    logger.warning("Suppressed exception in %s: %s", "run.start", e, exc_info=True)
                    pass
                self._request_clean_exit(reason)
                return True
            if enabled_platform_count > 0:
                reason = "; ".join(startup_retryable_errors) or "all configured messaging platforms failed to connect"
                logger.error("Gateway failed to connect any configured messaging platform: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
                except Exception as e:
                    logger.warning("Suppressed exception in %s: %s", "run.start", e, exc_info=True)
                    pass
                return False
            logger.warning("No messaging platforms enabled.")
            logger.info("Gateway will continue running for cron job execution.")
        
        # Update delivery router with adapters
        self.delivery_router.adapters = self.adapters
        
        self._running = True
        self._update_runtime_status("running")
        
        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters.keys()],
        })
        
        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)
        
        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)
        
        # Check if we're restarting after a /update command. If the update is
        # still running, keep watching so we notify once it actually finishes.
        notified = await self._send_update_notification()
        if not notified and any(
            path.exists()
            for path in (
                _hermes_home / ".update_pending.json",
                _hermes_home / ".update_pending.claimed.json",
            )
        ):
            self._schedule_update_notification_watch()

        # Drain any recovered process watchers (from crash recovery checkpoint)
        try:
            from tools.process_registry import process_registry
            while process_registry.pending_watchers:
                watcher = process_registry.pending_watchers.pop(0)
                asyncio.create_task(self._run_process_watcher(watcher))
                logger.info("Resumed watcher for recovered process %s", watcher.get("session_id"))
        except Exception as e:
            logger.error("Recovered watcher setup error: %s", e)

        # Start background session expiry watcher for proactive memory flushing
        asyncio.create_task(self._session_expiry_watcher())

        # Start background reconnection watcher for platforms that failed at startup
        if self._failed_platforms:
            logger.info(
                "Starting reconnection watcher for %d failed platform(s): %s",
                len(self._failed_platforms),
                ", ".join(p.value for p in self._failed_platforms),
            )
        asyncio.create_task(self._platform_reconnect_watcher())

        logger.info("Press Ctrl+C to stop")
        
        return True

    async def _session_expiry_watcher(self, interval: int = 300):
        """Background task that proactively flushes memories for expired sessions.
        
        Runs every `interval` seconds (default 5 min).  For each session that
        has expired according to its reset policy, flushes memories in a thread
        pool and marks the session so it won't be flushed again.

        This means memories are already saved by the time the user sends their
        next message, so there's no blocking delay.
        """
        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        _flush_failures: dict[str, int] = {}  # session_id -> consecutive failure count
        _MAX_FLUSH_RETRIES = 3
        while self._running:
            try:
                self.session_store._ensure_loaded()
                # Collect expired sessions first, then log a single summary.
                _expired_entries = []
                for key, entry in list(self.session_store._entries.items()):
                    if entry.memory_flushed:
                        continue
                    if not self.session_store._is_session_expired(entry):
                        continue
                    _expired_entries.append((key, entry))

                if _expired_entries:
                    # Extract platform names from session keys for a compact summary.
                    # Keys look like "agent:main:telegram:dm:12345" — platform is field [2].
                    _platforms: dict[str, int] = {}
                    for _k, _e in _expired_entries:
                        _parts = _k.split(":")
                        _plat = _parts[2] if len(_parts) > 2 else "unknown"
                        _platforms[_plat] = _platforms.get(_plat, 0) + 1
                    _plat_summary = ", ".join(
                        f"{p}:{c}" for p, c in sorted(_platforms.items())
                    )
                    logger.info(
                        "Session expiry: %d sessions to flush (%s)",
                        len(_expired_entries), _plat_summary,
                    )

                for key, entry in _expired_entries:
                    try:
                        await self._async_flush_memories(entry.session_id, key)
                        # Shut down memory provider and close tool resources
                        # on the cached agent.  Idle agents live in
                        # _agent_cache (not _running_agents), so look there.
                        _cached_agent = None
                        _cache_lock = getattr(self, "_agent_cache_lock", None)
                        if _cache_lock is not None:
                            with _cache_lock:
                                _cached = self._agent_cache.get(key)
                                _cached_agent = _cached[0] if isinstance(_cached, tuple) else _cached if _cached else None
                        # Fall back to _running_agents in case the agent is
                        # still mid-turn when the expiry fires.
                        if _cached_agent is None:
                            _cached_agent = self._running_agents.get(key)
                        if _cached_agent and _cached_agent is not _AGENT_PENDING_SENTINEL:
                            try:
                                if hasattr(_cached_agent, 'shutdown_memory_provider'):
                                    _cached_agent.shutdown_memory_provider()
                            except Exception as e:
                                logger.warning("Suppressed exception in %s: %s", "run._session_expiry_watcher", e, exc_info=True)
                                pass
                            try:
                                if hasattr(_cached_agent, 'close'):
                                    _cached_agent.close()
                            except Exception as e:
                                logger.warning("Suppressed exception in %s: %s", "run._session_expiry_watcher", e, exc_info=True)
                                pass
                        # Mark as flushed and persist to disk so the flag
                        # survives gateway restarts.
                        with self.session_store._lock:
                            entry.memory_flushed = True
                            self.session_store._save()
                        logger.debug(
                            "Memory flush completed for session %s",
                            entry.session_id,
                        )
                        _flush_failures.pop(entry.session_id, None)
                    except Exception as e:
                        failures = _flush_failures.get(entry.session_id, 0) + 1
                        _flush_failures[entry.session_id] = failures
                        if failures >= _MAX_FLUSH_RETRIES:
                            logger.warning(
                                "Memory flush gave up after %d attempts for %s: %s. "
                                "Marking as flushed to prevent infinite retry loop.",
                                failures, entry.session_id, e,
                            )
                            with self.session_store._lock:
                                entry.memory_flushed = True
                                self.session_store._save()
                            _flush_failures.pop(entry.session_id, None)
                        else:
                            logger.debug(
                                "Memory flush failed (%d/%d) for %s: %s",
                                failures, _MAX_FLUSH_RETRIES, entry.session_id, e,
                            )

                if _expired_entries:
                    _flushed = sum(
                        1 for _, e in _expired_entries if e.memory_flushed
                    )
                    _failed = len(_expired_entries) - _flushed
                    if _failed:
                        logger.info(
                            "Session expiry done: %d flushed, %d pending retry",
                            _flushed, _failed,
                        )
                    else:
                        logger.info(
                            "Session expiry done: %d flushed", _flushed,
                        )
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

    async def _platform_reconnect_watcher(self) -> None:
        """Background task that periodically retries connecting failed platforms.

        Uses exponential backoff: 30s → 60s → 120s → 240s → 300s (cap).
        Stops retrying a platform after 20 failed attempts or if the error
        is non-retryable (e.g. bad auth token).
        """
        _MAX_ATTEMPTS = 20
        _BACKOFF_CAP = 300  # 5 minutes max between retries

        await asyncio.sleep(10)  # initial delay — let startup finish
        while self._running:
            if not self._failed_platforms:
                # Nothing to reconnect — sleep and check again
                for _ in range(30):
                    if not self._running:
                        return
                    await asyncio.sleep(1)
                continue

            now = time.monotonic()
            for platform in list(self._failed_platforms.keys()):
                if not self._running:
                    return
                info = self._failed_platforms[platform]
                if now < info["next_retry"]:
                    continue  # not time yet

                if info["attempts"] >= _MAX_ATTEMPTS:
                    logger.warning(
                        "Giving up reconnecting %s after %d attempts",
                        platform.value, info["attempts"],
                    )
                    del self._failed_platforms[platform]
                    continue

                platform_config = info["config"]
                attempt = info["attempts"] + 1
                logger.info(
                    "Reconnecting %s (attempt %d/%d)...",
                    platform.value, attempt, _MAX_ATTEMPTS,
                )

                try:
                    adapter = self._create_adapter(platform, platform_config)
                    if not adapter:
                        logger.warning(
                            "Reconnect %s: adapter creation returned None, removing from retry queue",
                            platform.value,
                        )
                        del self._failed_platforms[platform]
                        continue

                    adapter.set_message_handler(self._handle_message)
                    adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
                    adapter.set_session_store(self.session_store)
                    adapter.set_busy_session_handler(self._handle_active_session_busy_message)

                    success = await adapter.connect()
                    if success:
                        self.adapters[platform] = adapter
                        self._sync_voice_mode_state_to_adapter(adapter)
                        self.delivery_router.adapters = self.adapters
                        del self._failed_platforms[platform]
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="connected",
                            error_code=None,
                            error_message=None,
                        )
                        logger.info("✓ %s reconnected successfully", platform.value)

                        # Rebuild channel directory with the new adapter
                        try:
                            from gateway.channel_directory import build_channel_directory
                            build_channel_directory(self.adapters)
                        except Exception as e:
                            logger.warning("Suppressed exception in %s: %s", "run._platform_reconnect_watcher", e, exc_info=True)
                            pass
                    else:
                        # Check if the failure is non-retryable
                        if adapter.has_fatal_error and not adapter.fatal_error_retryable:
                            self._update_platform_runtime_status(
                                platform.value,
                                platform_state="fatal",
                                error_code=adapter.fatal_error_code,
                                error_message=adapter.fatal_error_message,
                            )
                            logger.warning(
                                "Reconnect %s: non-retryable error (%s), removing from retry queue",
                                platform.value, adapter.fatal_error_message,
                            )
                            del self._failed_platforms[platform]
                        else:
                            self._update_platform_runtime_status(
                                platform.value,
                                platform_state="retrying",
                                error_code=adapter.fatal_error_code,
                                error_message=adapter.fatal_error_message or "failed to reconnect",
                            )
                            backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                            info["attempts"] = attempt
                            info["next_retry"] = time.monotonic() + backoff
                            logger.info(
                                "Reconnect %s failed, next retry in %ds",
                                platform.value, backoff,
                            )
                except Exception as e:
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        error_code=None,
                        error_message=str(e),
                    )
                    backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                    info["attempts"] = attempt
                    info["next_retry"] = time.monotonic() + backoff
                    logger.warning(
                        "Reconnect %s error: %s, next retry in %ds",
                        platform.value, e, backoff,
                    )

            # Check every 10 seconds for platforms that need reconnection
            for _ in range(10):
                if not self._running:
                    return
                await asyncio.sleep(1)

    async def stop(
        self,
        *,
        restart: bool = False,
        detached_restart: bool = False,
        service_restart: bool = False,
    ) -> None:
        """Stop the gateway and disconnect all adapters."""
        if restart:
            self._restart_requested = True
            self._restart_detached = detached_restart
            self._restart_via_service = service_restart
        if self._stop_task is not None:
            await self._stop_task
            return

        async def _stop_impl() -> None:
            logger.info(
                "Stopping gateway%s...",
                " for restart" if self._restart_requested else "",
            )
            self._running = False
            self._draining = True

            timeout = self._restart_drain_timeout
            active_agents, timed_out = await self._drain_active_agents(timeout)
            if timed_out:
                logger.warning(
                    "Gateway drain timed out after %.1fs with %d active agent(s); interrupting remaining work.",
                    timeout,
                    self._running_agent_count(),
                )
                self._interrupt_running_agents(
                    "Gateway restarting" if self._restart_requested else "Gateway shutting down"
                )
                interrupt_deadline = asyncio.get_running_loop().time() + 5.0
                while self._running_agents and asyncio.get_running_loop().time() < interrupt_deadline:
                    self._update_runtime_status("draining")
                    await asyncio.sleep(0.1)

            if self._restart_requested and self._restart_detached:
                try:
                    await self._launch_detached_restart_command()
                except Exception as e:
                    logger.error("Failed to launch detached gateway restart: %s", e)

            self._finalize_shutdown_agents(active_agents)

            for platform, adapter in list(self.adapters.items()):
                try:
                    await adapter.cancel_background_tasks()
                except Exception as e:
                    logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
                try:
                    await adapter.disconnect()
                    logger.info("✓ %s disconnected", platform.value)
                except Exception as e:
                    logger.error("✗ %s disconnect error: %s", platform.value, e)

            for _task in list(self._background_tasks):
                if _task is self._stop_task:
                    continue
                _task.cancel()
            self._background_tasks.clear()

            self.adapters.clear()
            self._running_agents.clear()
            self._pending_messages.clear()
            self._pending_approvals.clear()
            self._shutdown_event.set()

            # Global cleanup: kill any remaining tool subprocesses not tied
            # to a specific agent (catch-all for zombie prevention).
            try:
                from tools.process_registry import process_registry
                process_registry.kill_all()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._stop_impl", e, exc_info=True)
                pass
            try:
                from tools.terminal_tool import cleanup_all_environments
                cleanup_all_environments()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._stop_impl", e, exc_info=True)
                pass
            try:
                from tools.browser_tool import cleanup_all_browsers
                cleanup_all_browsers()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._stop_impl", e, exc_info=True)
                pass

            from gateway.status import remove_pid_file
            remove_pid_file()

            # Write a clean-shutdown marker so the next startup knows this
            # wasn't a crash.  suspend_recently_active() only needs to run
            # after unexpected exits — graceful shutdowns already drain
            # active agents, so there's no stuck-session risk.
            try:
                (_hermes_home / ".clean_shutdown").touch()
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run._stop_impl", e, exc_info=True)
                pass

            if self._restart_requested and self._restart_via_service:
                self._exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE
                self._exit_reason = self._exit_reason or "Gateway restart requested"

            self._draining = False
            self._update_runtime_status("stopped", self._exit_reason)
            logger.info("Gateway stopped")

        self._stop_task = asyncio.create_task(_stop_impl())
        await self._stop_task

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def _create_adapter(
        self, 
        platform: Platform, 
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        """Create the appropriate adapter for a platform."""
        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault(
                "group_sessions_per_user",
                self.config.group_sessions_per_user,
            )
            config.extra.setdefault(
                "thread_sessions_per_user",
                getattr(self.config, "thread_sessions_per_user", False),
            )

        if platform == Platform.TELEGRAM:
            from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements
            if not check_telegram_requirements():
                logger.warning("Telegram: python-telegram-bot not installed")
                return None
            return TelegramAdapter(config)
        
        elif platform == Platform.DISCORD:
            from gateway.platforms.discord import DiscordAdapter, check_discord_requirements
            if not check_discord_requirements():
                logger.warning("Discord: discord.py not installed")
                return None
            return DiscordAdapter(config)
        
        elif platform == Platform.WHATSAPP:
            from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                logger.warning("WhatsApp: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)
        
        elif platform == Platform.SLACK:
            from gateway.platforms.slack import SlackAdapter, check_slack_requirements
            if not check_slack_requirements():
                logger.warning("Slack: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'")
                return None
            return SlackAdapter(config)

        elif platform == Platform.SIGNAL:
            from gateway.platforms.signal import SignalAdapter, check_signal_requirements
            if not check_signal_requirements():
                logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
                return None
            return SignalAdapter(config)

        elif platform == Platform.HOMEASSISTANT:
            from gateway.platforms.homeassistant import HomeAssistantAdapter, check_ha_requirements
            if not check_ha_requirements():
                logger.warning("HomeAssistant: aiohttp not installed or HASS_TOKEN not set")
                return None
            return HomeAssistantAdapter(config)

        elif platform == Platform.EMAIL:
            from gateway.platforms.email import EmailAdapter, check_email_requirements
            if not check_email_requirements():
                logger.warning("Email: EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set")
                return None
            return EmailAdapter(config)

        elif platform == Platform.SMS:
            from gateway.platforms.sms import SmsAdapter, check_sms_requirements
            if not check_sms_requirements():
                logger.warning("SMS: aiohttp not installed or TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set")
                return None
            return SmsAdapter(config)

        elif platform == Platform.DINGTALK:
            from gateway.platforms.dingtalk import DingTalkAdapter, check_dingtalk_requirements
            if not check_dingtalk_requirements():
                logger.warning("DingTalk: dingtalk-stream not installed or DINGTALK_CLIENT_ID/SECRET not set")
                return None
            return DingTalkAdapter(config)

        elif platform == Platform.FEISHU:
            from gateway.platforms.feishu import FeishuAdapter, check_feishu_requirements
            if not check_feishu_requirements():
                logger.warning("Feishu: lark-oapi not installed or FEISHU_APP_ID/SECRET not set")
                return None
            return FeishuAdapter(config)

        elif platform == Platform.WECOM_CALLBACK:
            from gateway.platforms.wecom_callback import (
                WecomCallbackAdapter,
                check_wecom_callback_requirements,
            )
            if not check_wecom_callback_requirements():
                logger.warning("WeComCallback: aiohttp/httpx not installed")
                return None
            return WecomCallbackAdapter(config)

        elif platform == Platform.WECOM:
            from gateway.platforms.wecom import WeComAdapter, check_wecom_requirements
            if not check_wecom_requirements():
                logger.warning("WeCom: aiohttp not installed or WECOM_BOT_ID/SECRET not set")
                return None
            return WeComAdapter(config)

        elif platform == Platform.WEIXIN:
            from gateway.platforms.weixin import WeixinAdapter, check_weixin_requirements
            if not check_weixin_requirements():
                logger.warning("Weixin: aiohttp/cryptography not installed")
                return None
            return WeixinAdapter(config)

        elif platform == Platform.MATTERMOST:
            from gateway.platforms.mattermost import MattermostAdapter, check_mattermost_requirements
            if not check_mattermost_requirements():
                logger.warning("Mattermost: MATTERMOST_TOKEN or MATTERMOST_URL not set, or aiohttp missing")
                return None
            return MattermostAdapter(config)

        elif platform == Platform.MATRIX:
            from gateway.platforms.matrix import MatrixAdapter, check_matrix_requirements
            if not check_matrix_requirements():
                logger.warning("Matrix: mautrix not installed or credentials not set. Run: pip install 'mautrix[encryption]'")
                return None
            return MatrixAdapter(config)

        elif platform == Platform.API_SERVER:
            from gateway.platforms.api_server import APIServerAdapter, check_api_server_requirements
            if not check_api_server_requirements():
                logger.warning("API Server: aiohttp not installed")
                return None
            return APIServerAdapter(config)

        elif platform == Platform.WEBHOOK:
            from gateway.platforms.webhook import WebhookAdapter, check_webhook_requirements
            if not check_webhook_requirements():
                logger.warning("Webhook: aiohttp not installed")
                return None
            adapter = WebhookAdapter(config)
            adapter.gateway_runner = self  # For cross-platform delivery
            return adapter

        elif platform == Platform.BLUEBUBBLES:
            from gateway.platforms.bluebubbles import BlueBubblesAdapter, check_bluebubbles_requirements
            if not check_bluebubbles_requirements():
                logger.warning("BlueBubbles: aiohttp/httpx missing or BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD not configured")
                return None
            return BlueBubblesAdapter(config)

        return None

