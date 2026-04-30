from __future__ import annotations
"""
LLM Client Mixin - Manages OpenAI/Anthropic client lifecycle and credentials.

Extracted from run_agent.py to isolate connection management from agent logic.
This mixin provides:
  - Client creation, destruction, and health checks
  - Credential rotation and refresh
  - Thread-safe client access
  - URL-based provider detection

It does NOT own:
  - Streaming/response parsing (stays in agent for now)
  - Tool execution
  - Context management
"""

import logging
import threading
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)


class LLMClientMixin:
    """Mixin providing LLM client lifecycle and credential management.

    Expects the host class (AIAgent) to have these attributes set in its __init__:
      - self.client: Any
      - self.api_key: str
      - self.base_url: str
      - self._base_url_lower: str
      - self.provider: str
      - self.api_mode: str
      - self._client_kwargs: dict
      - self._client_lock: threading.RLock
      - self._credential_pool: list
      - self._anthropic_api_key: Optional[str]
      - self._anthropic_base_url: Optional[str]
      - self._anthropic_client: Any
      - self._is_anthropic_oauth: bool
    """

    # ─── URL Detection ───────────────────────────────────────────────────

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        url = (base_url or self._base_url_lower).lower()
        return "api.openai.com" in url and "openrouter" not in url

    def _is_openrouter_url(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return "openrouter" in self._base_url_lower

    # ─── Thread Identity & Logging ────────────────────────────────────

    def _thread_identity(self) -> str:
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """Check if an OpenAI client is closed.

        Handles both property and method forms of is_closed:
        - httpx.Client.is_closed is a bool property
        - openai.OpenAI.is_closed is a method returning bool

        Prior bug: getattr(client, "is_closed", False) returned the bound method,
        which is always truthy, causing unnecessary client recreation on every call.
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # Handle method (openai SDK) vs property (httpx)
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        if self.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """Force-close underlying TCP sockets to prevent CLOSE-WAIT accumulation.

        When a provider drops a connection mid-stream, httpx's ``client.close()``
        performs a graceful shutdown which leaves sockets in CLOSE-WAIT until the
        OS times them out (often minutes).  This method walks the httpx transport
        pool and issues ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()`` to
        force an immediate TCP RST, freeing the file descriptors.

        Returns the number of sockets force-closed.
        """
        import socket as _socket

        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return 0
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return 0
            # httpx uses httpcore connection pools; connections live in
            # _connections (list) or _pool (list) depending on version.
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            for conn in list(connections):
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        # Force-close TCP sockets first to prevent CLOSE-WAIT accumulation,
        # then do the graceful SDK-level close.
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(self._client_kwargs, reason=reason, shared=True)
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _cleanup_dead_connections(self) -> bool:
        """Detect and clean up dead TCP connections on the primary client.

        Inspects the httpx connection pool for sockets in unhealthy states
        (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
        and rebuilds the primary client from scratch.

        Returns True if dead connections were found and cleaned up.
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # Check for connections that are idle but have closed sockets
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # Probe socket health with a non-blocking recv peek
                import socket as _socket
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # No data available — socket is healthy
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    def _create_request_openai_client(self, *, reason: str) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        self._close_openai_client(client, reason=reason, shared=False)


    # ─── Credential Management ───────────────────────────────────────

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous requests should not inherit OpenRouter-only attribution headers.
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # Only refresh credentials for the native Anthropic provider.
        # Other anthropic_messages providers (MiniMax, Alibaba, etc.) use their own keys.
        if self.provider != "anthropic":
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(new_token, getattr(self, "_anthropic_base_url", None))
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # Update OAuth flag — token type may have changed (API key ↔ OAuth)
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token)
        return True

    def _apply_client_headers_for_base_url(self, base_url: str) -> None:
        from agent.auxiliary_client import _OR_HEADERS

        normalized = (base_url or "").lower()
        if "openrouter" in normalized:
            self._client_kwargs["default_headers"] = dict(_OR_HEADERS)
        elif "api.githubcopilot.com" in normalized:
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif "api.kimi.com" in normalized:
            self._client_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.30.0"}
        elif "portal.qwen.ai" in normalized:
            self._client_kwargs["default_headers"] = _qwen_portal_headers()
        else:
            self._client_kwargs.pop("default_headers", None)

    def _swap_credential(self, entry) -> None:
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base
            self._anthropic_client = build_anthropic_client(runtime_key, runtime_base)
            self._is_anthropic_oauth = _is_oauth_token(runtime_key)
            self.api_key = runtime_key
            self.base_url = runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(self.base_url)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        classified_reason: Optional[FailoverReason] = None,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """Attempt credential recovery via pool rotation.

        Returns (recovered, has_retried_429).
        On rate limits: first occurrence retries same credential (sets flag True).
                        second consecutive failure rotates to next credential.
        On billing exhaustion: immediately rotates.
        On auth failures: attempts token refresh before rotating.

        `classified_reason` lets the recovery path honor the structured error
        classifier instead of relying only on raw HTTP codes. This matters for
        providers that surface billing/rate-limit/auth conditions under a
        different status code, such as Anthropic returning HTTP 400 for
        "out of extra usage".
        """
        pool = self._credential_pool
        if pool is None:
            return False, has_retried_429

        effective_reason = classified_reason
        if effective_reason is None:
            if status_code == 402:
                effective_reason = FailoverReason.billing
            elif status_code == 429:
                effective_reason = FailoverReason.rate_limit
            elif status_code == 401:
                effective_reason = FailoverReason.auth

        if effective_reason == FailoverReason.billing:
            rotate_status = status_code if status_code is not None else 402
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (billing) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, has_retried_429

        if effective_reason == FailoverReason.rate_limit:
            if not has_retried_429:
                return False, True
            rotate_status = status_code if status_code is not None else 429
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (rate limit) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, True

        if effective_reason == FailoverReason.auth:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                logger.info(f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}")
                self._swap_credential(refreshed)
                return True, has_retried_429
            # Refresh failed — rotate to next credential instead of giving up.
            # The failed entry is already marked exhausted by try_refresh_current().
            rotate_status = status_code if status_code is not None else 401
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (auth refresh failed) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False

        return False, has_retried_429

