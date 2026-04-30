"""Auto-generated mixin for AIAgent."""
from __future__ import annotations
import logging
from datetime import datetime
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from agent.shared_utils import _install_safe_stdio, _qwen_portal_headers, IterationBudget
from agent.model_metadata import fetch_model_metadata
from hermes_constants import get_hermes_home

from hermes_constants import OPENROUTER_BASE_URL
from openai import OpenAI
from tools.browser_tool import cleanup_browser
from tools.terminal_tool import cleanup_vm
from tools.terminal_tool import is_persistent_env
from agent.context_compressor import ContextCompressor
from agent.model_metadata import is_local_endpoint
from agent.model_metadata import query_ollama_num_ctx
from agent.prompt_builder import AGENT_LOOP_PROTOCOL
from agent.prompt_builder import BRIEF_PARAMETER_DISCIPLINE
from agent.prompt_builder import DEBUG_DELEGATION_PROTOCOL
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
from agent.prompt_builder import DELIVERY_GATE_PROTOCOL
from agent.prompt_builder import DEPTH_EXECUTION_PROTOCOL
from agent.prompt_builder import DISCLOSURE_PROHIBITION
from agent.prompt_builder import EXTERNAL_SERVICE_PROTOCOL
from agent.prompt_builder import FILE_OPERATION_CONSTRAINTS
from agent.prompt_builder import GOOGLE_MODEL_OPERATIONAL_GUIDANCE
from agent.prompt_builder import IMAGE_GENERATION_GUIDANCE
from agent.prompt_builder import MEMORY_GUIDANCE
from agent.prompt_builder import MESSAGE_DISCIPLINE_PROTOCOL
from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
from agent.prompt_builder import OUTPUT_FORMAT_PROTOCOL
from agent.prompt_builder import PARALLEL_SUBTASKS_GUIDANCE
from agent.prompt_builder import PLAN_DISCIPLINE_PROTOCOL
from agent.prompt_builder import PLATFORM_HINTS
from agent.prompt_builder import REQUIREMENT_CHANGE_TRIGGER
from agent.prompt_builder import SESSION_SEARCH_GUIDANCE
from agent.prompt_builder import SKILLS_GUIDANCE
from agent.prompt_builder import STRUCTURED_VARIATION_PROTOCOL
from agent.prompt_builder import TOOL_AVAILABILITY_PROTOCOL
from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
from agent.prompt_builder import VERIFICATION_PROTOCOL
from agent.prompt_builder import build_context_files_prompt
from agent.prompt_builder import build_environment_hints
from agent.prompt_builder import build_nous_subscription_prompt
from agent.prompt_builder import build_skills_system_prompt
from agent.prompt_builder import load_soul_md
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.trajectory import save_trajectory as _save_trajectory_to_file
from agent.trajectory import convert_scratchpad_to_think
from hermes_constants import get_hermes_home
from model_tools import check_toolset_requirements
from model_tools import get_tool_definitions
from model_tools import get_toolset_for_tool
_hermes_home = get_hermes_home()
_hermes_home = get_hermes_home()

logger = logging.getLogger(__name__)


class InitMixin:
    """AIAgent mixin: init methods."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        acp_command: str = None,
        acp_args: list[str] | None = None,
        command: str = None,
        args: list[str] | None = None,
        model: str = "",
        max_iterations: int = 90,  # Default tool-calling iterations (shared with subagents)
        tool_delay: float = 1.0,
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        full_schema_mode: bool = False,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
        provider_require_parameters: bool = False,
        provider_data_collection: str = None,
        session_id: str = None,
        tool_progress_callback: callable = None,
        tool_start_callback: callable = None,
        tool_complete_callback: callable = None,
        thinking_callback: callable = None,
        reasoning_callback: callable = None,
        clarify_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None,
        status_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        service_tier: str = None,
        request_overrides: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        user_id: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db=None,
        parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None,
        fallback_model: Dict[str, Any] = None,
        credential_pool=None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 50,
        pass_session_id: bool = False,
        persist_session: bool = True,
    ):
        """
        Initialize the AI Agent.

        Args:
            base_url (str): Base URL for the model API (optional)
            api_key (str): API key for authentication (optional, uses env var if not provided)
            provider (str): Provider identifier (optional; used for telemetry/routing hints)
            api_mode (str): API mode override: "chat_completions" or "codex_responses"
            model (str): Model name to use (default: "anthropic/claude-opus-4.6")
            max_iterations (int): Maximum number of tool calling iterations (default: 90)
            tool_delay (float): Delay between tool calls in seconds (default: 1.0)
            enabled_toolsets (List[str]): Only enable tools from these toolsets (optional)
            disabled_toolsets (List[str]): Disable tools from these toolsets (optional)
            save_trajectories (bool): Whether to save conversation trajectories to JSONL files (default: False)
            verbose_logging (bool): Enable verbose logging for debugging (default: False)
            quiet_mode (bool): Suppress progress output for clean CLI experience (default: False)
            ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
            log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 100)
            log_prefix (str): Prefix to add to all log messages for identification in parallel processing (default: "")
            providers_allowed (List[str]): OpenRouter providers to allow (optional)
            providers_ignored (List[str]): OpenRouter providers to ignore (optional)
            providers_order (List[str]): OpenRouter providers to try in order (optional)
            provider_sort (str): Sort providers by price/throughput/latency (optional)
            session_id (str): Pre-generated session ID for logging (optional, auto-generated if not provided)
            tool_progress_callback (callable): Callback function(tool_name, args_preview) for progress notifications
            clarify_callback (callable): Callback function(question, choices) -> str for interactive user questions.
                Provided by the platform layer (CLI or gateway). If None, the clarify tool returns an error.
            max_tokens (int): Maximum tokens for model responses (optional, uses model default if not set)
            reasoning_config (Dict): OpenRouter reasoning configuration override (e.g. {"effort": "none"} to disable thinking).
                If None, defaults to {"enabled": True, "effort": "medium"} for OpenRouter. Set to disable/customize reasoning.
            prefill_messages (List[Dict]): Messages to prepend to conversation history as prefilled context.
                Useful for injecting a few-shot example or priming the model's response style.
                Example: [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
            platform (str): The interface platform the user is on (e.g. "cli", "telegram", "discord", "whatsapp").
                Used to inject platform-specific formatting hints into the system prompt.
            skip_context_files (bool): If True, skip auto-injection of SOUL.md, AGENTS.md, and .cursorrules
                into the system prompt. Use this for batch processing and data generation to avoid
                polluting trajectories with user-specific persona or project instructions.
        """
        _install_safe_stdio()

        self.model = model
        self.max_iterations = max_iterations
        # Shared iteration budget — parent creates, children inherit.
        # Consumed by every LLM turn across parent + all subagents.
        self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform  # "cli", "telegram", "discord", "whatsapp", etc.
        self._user_id = user_id  # Platform user identifier (gateway sessions)
        # Pluggable print function — CLI replaces this with _cprint so that
        # raw ANSI status lines are routed through prompt_toolkit's renderer
        # instead of going directly to stdout where patch_stdout's StdoutProxy
        # would mangle the escape sequences.  None = use builtins.print.
        self._print_fn = None
        self.background_review_callback = None  # Optional sync callback for gateway delivery
        self.skip_context_files = skip_context_files
        self.pass_session_id = pass_session_id
        self.persist_session = persist_session
        self._credential_pool = credential_pool
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        # Store effective base URL for feature detection (prompt caching, reasoning, etc.)
        self.base_url = base_url or ""
        provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        self.provider = provider_name or ""
        self.acp_command = acp_command or command
        self.acp_args = list(acp_args or args or [])
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif (provider_name is None) and "chatgpt.com/backend-api/codex" in self._base_url_lower:
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif self.provider == "anthropic" or (provider_name is None and "api.anthropic.com" in self._base_url_lower):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        elif self._base_url_lower.rstrip("/").endswith("/anthropic"):
            # Third-party Anthropic-compatible endpoints (e.g. MiniMax, DashScope)
            # use a URL convention ending in /anthropic. Auto-detect these so the
            # Anthropic Messages API adapter is used instead of chat completions.
            self.api_mode = "anthropic_messages"
        else:
            self.api_mode = "chat_completions"

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS,
                normalize_model_for_provider,
            )

            if self.provider not in _AGGREGATOR_PROVIDERS:
                self.model = normalize_model_for_provider(self.model, self.provider)
        except Exception:
            pass

        # GPT-5.x models require the Responses API path — they are rejected
        # on /v1/chat/completions by both OpenAI and OpenRouter.  Also
        # auto-upgrade for direct OpenAI URLs (api.openai.com) since all
        # newer tool-calling models prefer Responses there.
        if self.api_mode == "chat_completions" and (
            self._is_direct_openai_url()
            or self._model_requires_responses_api(self.model)
        ):
            self.api_mode = "codex_responses"

        # Pre-warm OpenRouter model metadata cache in a background thread.
        # fetch_model_metadata() is cached for 1 hour; this avoids a blocking
        # HTTP request on the first API response when pricing is estimated.
        if self.provider == "openrouter" or self._is_openrouter_url():
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.suppress_status_output = False
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self.interim_assistant_callback = interim_assistant_callback
        self.status_callback = status_callback
        self.tool_gen_callback = tool_gen_callback

        
        # Tool execution state — allows _vprint during tool execution
        # even when stream consumers are registered (no tokens streaming then)
        self._executing_tools = False

        # Interrupt mechanism for breaking out of tool loops
        self._interrupt_requested = False
        self._interrupt_message = None  # Optional message that triggered interrupt
        self._execution_thread_id: int | None = None  # Set at run_conversation() start
        self._client_lock = threading.RLock()
        
        # Subagent delegation state
        self._delegate_depth = 0        # 0 = top-level agent, incremented for children
        self._active_children = []      # Running child AIAgents (for interrupt propagation)
        self._active_children_lock = threading.Lock()
        
        # Store OpenRouter provider preferences
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.provider_require_parameters = provider_require_parameters
        self.provider_data_collection = provider_data_collection

        # Store toolset filtering options
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        self.full_schema_mode = full_schema_mode
        
        # Model response configuration
        self.max_tokens = max_tokens  # None = use model default
        self.reasoning_config = reasoning_config  # None = use default (medium for OpenRouter)
        self.service_tier = service_tier
        self.request_overrides = dict(request_overrides or {})
        self.prefill_messages = prefill_messages or []  # Prefilled conversation turns
        
        # Anthropic prompt caching: auto-enabled for Claude models via OpenRouter.
        # Reduces input costs by ~75% on multi-turn conversations by caching the
        # conversation prefix. Uses system_and_3 strategy (4 breakpoints).
        is_openrouter = self._is_openrouter_url()
        is_claude = "claude" in self.model.lower()
        is_native_anthropic = self.api_mode == "anthropic_messages" and self.provider == "anthropic"
        self._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic
        self._cache_ttl = "5m"  # Default 5-minute TTL (1.25x write cost)
        
        # Iteration budget: the LLM is only notified when it actually exhausts
        # the iteration budget (api_call_count >= max_iterations).  At that
        # point we inject ONE message, allow one final API call, and if the
        # model doesn't produce a text response, force a user-message asking
        # it to summarise.  No intermediate pressure warnings — they caused
        # models to "give up" prematurely on complex tasks (#7915).
        self._budget_exhausted_injected = False
        self._budget_grace_call = False

        # Context pressure warnings: notify the USER (not the LLM) as context
        # fills up.  Purely informational — displayed in CLI output and sent via
        # status_callback for gateway platforms.  Does NOT inject into messages.
        # Tiered: fires at 85% and again at 95% of compaction threshold.
        self._context_pressure_warned_at = 0.0  # highest tier already shown

        # Activity tracking — updated on each API call, tool execution, and
        # stream chunk.  Used by the gateway timeout handler to report what the
        # agent was doing when it was killed, and by the "still working"
        # notifications to show progress.
        self._last_activity_ts: float = time.time()
        self._last_activity_desc: str = "initializing"
        self._current_tool: str | None = None
        self._api_call_count: int = 0

        # Rate limit tracking — updated from x-ratelimit-* response headers
        # after each API call.  Accessed by /usage slash command.
        self._rate_limit_state: Optional["RateLimitState"] = None

        # Centralized logging — agent.log (INFO+) and errors.log (WARNING+)
        # both live under ~/.hermes/logs/.  Idempotent, so gateway mode
        # (which creates a new AIAgent per message) won't duplicate handlers.
        from hermes_logging import setup_logging, setup_verbose_logging
        setup_logging(hermes_home=_hermes_home)

        if self.verbose_logging:
            setup_verbose_logging()
            logger.info("Verbose logging enabled (third-party library logs suppressed)")
        else:
            if self.quiet_mode:
                # In quiet mode (CLI default), suppress all tool/infra log
                # noise on the *console*. The TUI has its own rich display
                # for status; logger INFO/WARNING messages just clutter it.
                # File handlers (agent.log, errors.log) still capture everything.
                for quiet_logger in [
                    'tools',               # all tools.* (terminal, browser, web, file, etc.)
                    'run_agent',            # agent runner internals
                    'trajectory_compressor',
                    'cron',                 # scheduler (only relevant in daemon mode)
                    'hermes_cli',           # CLI helpers
                ]:
                    logging.getLogger(quiet_logger).setLevel(logging.ERROR)
        
        # Internal stream callback (set during streaming TTS).
        # Initialized here so _vprint can reference it before run_conversation.
        self._stream_callback = None
        # Deferred paragraph break flag — set after tool iterations so a
        # single "\n\n" is prepended to the next real text delta.
        self._stream_needs_break = False
        # Visible assistant text already delivered through live token callbacks
        # during the current model response. Used to avoid re-sending the same
        # commentary when the provider later returns it as a completed interim
        # assistant message.
        self._current_streamed_assistant_text = ""

        # Optional current-turn user-message override used when the API-facing
        # user message intentionally differs from the persisted transcript
        # (e.g. CLI voice mode adds a temporary prefix for the live call only).
        self._persist_user_message_idx = None
        self._persist_user_message_override = None

        # Cache anthropic image-to-text fallbacks per image payload/URL so a
        # single tool loop does not repeatedly re-run auxiliary vision on the
        # same image history.
        self._anthropic_image_fallback_cache: Dict[str, str] = {}

        # Initialize LLM client via centralized provider router.
        # The router handles auth resolution, base URL, headers, and
        # Codex/Anthropic wrapping for all known providers.
        # raw_codex=True because the main agent needs direct responses.stream()
        # access for Codex Responses API streaming.
        self._anthropic_client = None
        self._is_anthropic_oauth = False

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
            # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own API key.
            # Falling back would send Anthropic credentials to third-party endpoints (Fixes #1739, #minimax-401).
            _is_native_anthropic = self.provider == "anthropic"
            effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url
            from agent.anthropic_adapter import _is_oauth_token as _is_oat
            self._is_anthropic_oauth = _is_oat(effective_key)
            self._anthropic_client = build_anthropic_client(effective_key, base_url)
            # No OpenAI client needed for Anthropic mode
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)")
                if effective_key and len(effective_key) > 12:
                    print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        else:
            if api_key and base_url:
                # Explicit credentials from CLI/gateway — construct directly.
                # The runtime provider resolver already handled auth for us.
                client_kwargs = {"api_key": api_key, "base_url": base_url}
                if self.provider == "copilot-acp":
                    client_kwargs["command"] = self.acp_command
                    client_kwargs["args"] = self.acp_args
                effective_base = base_url
                if "openrouter" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-OpenRouter-Title": "Hermes Agent",
                        "X-OpenRouter-Categories": "productivity,cli-agent",
                    }
                elif "api.githubcopilot.com" in effective_base.lower():
                    from hermes_cli.models import copilot_default_headers

                    client_kwargs["default_headers"] = copilot_default_headers()
                elif "api.kimi.com" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "User-Agent": "KimiCLI/1.30.0",
                    }
                elif "portal.qwen.ai" in effective_base.lower():
                    client_kwargs["default_headers"] = _qwen_portal_headers()
            else:
                # No explicit creds — use the centralized provider router
                from agent.auxiliary_client import resolve_provider_client
                _routed_client, _ = resolve_provider_client(
                    self.provider or "auto", model=self.model, raw_codex=True)
                if _routed_client is not None:
                    client_kwargs = {
                        "api_key": _routed_client.api_key,
                        "base_url": str(_routed_client.base_url),
                    }
                    # Preserve any default_headers the router set
                    if hasattr(_routed_client, '_default_headers') and _routed_client._default_headers:
                        client_kwargs["default_headers"] = dict(_routed_client._default_headers)
                else:
                    # When the user explicitly chose a non-OpenRouter provider
                    # but no credentials were found, fail fast with a clear
                    # message instead of silently routing through OpenRouter.
                    _explicit = (self.provider or "").strip().lower()
                    if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_explicit.upper()}_API_KEY environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )
                    # Final fallback: try raw OpenRouter key
                    client_kwargs = {
                        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
                        "base_url": OPENROUTER_BASE_URL,
                        "default_headers": {
                            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                            "X-OpenRouter-Title": "Hermes Agent",
                            "X-OpenRouter-Categories": "productivity,cli-agent",
                        },
                    }
            
            self._client_kwargs = client_kwargs  # stored for rebuilding after interrupt

            # Enable fine-grained tool streaming for Claude on OpenRouter.
            # Without this, Anthropic buffers the entire tool call and goes
            # silent for minutes while thinking — OpenRouter's upstream proxy
            # times out during the silence.  The beta header makes Anthropic
            # stream tool call arguments token-by-token, keeping the
            # connection alive.
            _effective_base = str(client_kwargs.get("base_url", "")).lower()
            if "openrouter" in _effective_base and "claude" in (self.model or "").lower():
                headers = client_kwargs.get("default_headers") or {}
                existing_beta = headers.get("x-anthropic-beta", "")
                _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
                if _FINE_GRAINED not in existing_beta:
                    if existing_beta:
                        headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                    else:
                        headers["x-anthropic-beta"] = _FINE_GRAINED
                    client_kwargs["default_headers"] = headers

            self.api_key = client_kwargs.get("api_key", "")
            self.base_url = client_kwargs.get("base_url", self.base_url)
            try:
                self.client = self._create_openai_client(client_kwargs, reason="agent_init", shared=True)
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model}")
                    if base_url:
                        print(f"🔗 Using custom base URL: {base_url}")
                    # Always show API key info (masked) for debugging auth issues
                    key_used = client_kwargs.get("api_key", "none")
                    if key_used and key_used != "dummy-key" and len(key_used) > 12:
                        print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                    else:
                        print(f"⚠️  Warning: API key appears invalid or missing (got: '{key_used[:20] if key_used else 'none'}...')")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        # Provider fallback chain — ordered list of backup providers tried
        # when the primary is exhausted (rate-limit, overload, connection
        # failure).  Supports both legacy single-dict ``fallback_model`` and
        # new list ``fallback_providers`` format.
        if isinstance(fallback_model, list):
            self._fallback_chain = [
                f for f in fallback_model
                if isinstance(f, dict) and f.get("provider") and f.get("model")
            ]
        elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
            self._fallback_chain = [fallback_model]
        else:
            self._fallback_chain = []
        self._fallback_index = 0
        self._fallback_activated = False
        # Legacy attribute kept for backward compat (tests, external callers)
        self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None
        if self._fallback_chain and not self.quiet_mode:
            if len(self._fallback_chain) == 1:
                fb = self._fallback_chain[0]
                print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
            else:
                print(f"🔄 Fallback chain ({len(self._fallback_chain)} providers): " +
                      " → ".join(f"{f['model']} ({f['provider']})" for f in self._fallback_chain))

        # Get available tools with filtering
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=self.quiet_mode,
            full_schema_mode=full_schema_mode,
        )
        
        # Show tool configuration and store valid tool names for validation
        self.valid_tool_names = set()
        if self.tools:
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
            if not self.quiet_mode:
                print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")
                
                # Show filtering info if applied
                if enabled_toolsets:
                    print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
                if disabled_toolsets:
                    print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        elif not self.quiet_mode:
            print("🛠️  No tools loaded (all tools filtered out or unavailable)")
        
        # Check tool requirements
        if self.tools and not self.quiet_mode:
            requirements = check_toolset_requirements()
            missing_reqs = [name for name, available in requirements.items() if not available]
            if missing_reqs:
                print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")
        
        # Show trajectory saving status
        if self.save_trajectories and not self.quiet_mode:
            print("📝 Trajectory saving enabled")
        
        # Show ephemeral system prompt status
        if self.ephemeral_system_prompt and not self.quiet_mode:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")
        
        # Show prompt caching status
        if self._use_prompt_caching and not self.quiet_mode:
            source = "native Anthropic" if is_native_anthropic else "Claude via OpenRouter"
            print(f"💾 Prompt caching: ENABLED ({source}, {self._cache_ttl} TTL)")
        
        # Session logging setup - auto-save conversation trajectories for debugging
        self.session_start = datetime.now()
        if session_id:
            # Use provided session ID (e.g., from CLI)
            self.session_id = session_id
        else:
            # Generate a new session ID
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        
        # Session logs go into ~/.hermes/sessions/ alongside gateway sessions
        hermes_home = get_hermes_home()
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
        
        # Track conversation messages for session logging
        self._session_messages: List[Dict[str, Any]] = []
        
        # Cached system prompt -- built once per session, only rebuilt on compression
        self._cached_system_prompt: Optional[str] = None
        
        # Filesystem checkpoint manager (transparent — not a tool)
        from tools.checkpoint_manager import CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
        )
        
        # SQLite session store (optional -- provided by CLI or gateway)
        self._session_db = session_db
        self._parent_session_id = parent_session_id
        self._last_flushed_db_idx = 0  # tracks DB-write cursor to prevent duplicate writes
        if self._session_db:
            try:
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_iterations,
                        "reasoning_config": reasoning_config,
                        "max_tokens": max_tokens,
                    },
                    user_id=None,
                    parent_session_id=self._parent_session_id,
                )
            except Exception as e:
                # Transient SQLite lock contention (e.g. CLI and gateway writing
                # concurrently) must NOT permanently disable session_search for
                # this agent.  Keep _session_db alive — subsequent message
                # flushes and session_search calls will still work once the
                # lock clears.  The session row may be missing from the index
                # for this run, but that is recoverable (flushes upsert rows).
                logger.warning(
                    "Session DB create_session failed (session_search still available): %s", e
                )
        
        # In-memory todo list for task planning (one per agent/session)
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()
        
        # Load config once for memory, skills, and compression sections
        try:
            from hermes_cli.config import load_config as _load_agent_config
            _agent_cfg = _load_agent_config()
        except Exception:
            _agent_cfg = {}

        # Persistent memory (MEMORY.md + USER.md) -- loaded from disk
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 10
        self._memory_flush_min_turns = 6
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        if not skip_memory:
            try:
                mem_config = _agent_cfg.get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                self._memory_flush_min_turns = int(mem_config.get("flush_min_turns", 6))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass  # Memory is optional -- don't break agent init
        


        # Memory provider plugin (external — one at a time, alongside built-in)
        # Reads memory.provider from config to select which plugin to activate.
        self._memory_manager = None
        if not skip_memory:
            try:
                _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

                # Auto-migrate: if Honcho was actively configured (enabled +
                # credentials) but memory.provider is not set, activate the
                # honcho plugin automatically.  Just having the config file
                # is not enough — the user may have disabled Honcho or the
                # file may be from a different tool.
                if not _mem_provider_name:
                    try:
                        from plugins.memory.honcho.client import HonchoClientConfig as _HCC
                        _hcfg = _HCC.from_global_config()
                        if _hcfg.enabled and (_hcfg.api_key or _hcfg.base_url):
                            _mem_provider_name = "honcho"
                            # Persist so this only auto-migrates once
                            try:
                                from hermes_cli.config import load_config as _lc, save_config as _sc
                                _cfg = _lc()
                                _cfg.setdefault("memory", {})["provider"] = "honcho"
                                _sc(_cfg)
                            except Exception:
                                pass
                            if not self.quiet_mode:
                                print("  ✓ Auto-migrated Honcho to memory provider plugin.")
                                print("    Your config and data are preserved.\n")
                    except Exception:
                        pass

                if _mem_provider_name:
                    from agent.memory_manager import MemoryManager as _MemoryManager
                    from plugins.memory import load_memory_provider as _load_mem
                    self._memory_manager = _MemoryManager()
                    _mp = _load_mem(_mem_provider_name)
                    if _mp and _mp.is_available():
                        self._memory_manager.add_provider(_mp)
                    if self._memory_manager.providers:
                        from hermes_constants import get_hermes_home as _ghh
                        _init_kwargs = {
                            "session_id": self.session_id,
                            "platform": platform or "cli",
                            "hermes_home": str(_ghh()),
                            "agent_context": "primary",
                        }
                        # Thread gateway user identity for per-user memory scoping
                        if self._user_id:
                            _init_kwargs["user_id"] = self._user_id
                        # Profile identity for per-profile provider scoping
                        try:
                            from hermes_cli.profiles import get_active_profile_name
                            _profile = get_active_profile_name()
                            _init_kwargs["agent_identity"] = _profile
                            _init_kwargs["agent_workspace"] = "hermes"
                        except Exception:
                            pass
                        self._memory_manager.initialize_all(**_init_kwargs)
                        logger.info("Memory provider '%s' activated", _mem_provider_name)
                    else:
                        logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                        self._memory_manager = None
            except Exception as _mpe:
                logger.warning("Memory provider plugin init failed: %s", _mpe)
                self._memory_manager = None

        # Inject memory provider tool schemas into the tool surface
        if self._memory_manager and self.tools is not None:
            for _schema in self._memory_manager.get_all_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    self.valid_tool_names.add(_tname)

        # Skills config: nudge interval for skill creation reminders
        self._skill_nudge_interval = 10
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
        except Exception:
            pass

        # Tool-use enforcement config: "auto" (default — matches hardcoded
        # model list), true (always), false (never), or list of substrings.
        _agent_section = _agent_cfg.get("agent", {})
        if not isinstance(_agent_section, dict):
            _agent_section = {}
        self._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

        # Initialize context compressor for automatic context management
        # Compresses conversation when approaching model's context limit
        # Configuration via config.yaml (compression section)
        _compression_cfg = _agent_cfg.get("compression", {})
        if not isinstance(_compression_cfg, dict):
            _compression_cfg = {}
        compression_threshold = float(_compression_cfg.get("threshold", 0.50))
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in ("true", "1", "yes")
        compression_summary_model = _compression_cfg.get("summary_model") or None
        compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))
        compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))

        # Read explicit context_length override from model config
        _model_cfg = _agent_cfg.get("model", {})
        if isinstance(_model_cfg, dict):
            _config_context_length = _model_cfg.get("context_length")
        else:
            _config_context_length = None
        if _config_context_length is not None:
            try:
                _config_context_length = int(_config_context_length)
            except (TypeError, ValueError):
                _config_context_length = None

        # Store for reuse in switch_model (so config override persists across model switches)
        self._config_context_length = _config_context_length

        # Check custom_providers per-model context_length
        if _config_context_length is None:
            _custom_providers = _agent_cfg.get("custom_providers")
            if isinstance(_custom_providers, list):
                for _cp_entry in _custom_providers:
                    if not isinstance(_cp_entry, dict):
                        continue
                    _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                    if _cp_url and _cp_url == self.base_url.rstrip("/"):
                        _cp_models = _cp_entry.get("models", {})
                        if isinstance(_cp_models, dict):
                            _cp_model_cfg = _cp_models.get(self.model, {})
                            if isinstance(_cp_model_cfg, dict):
                                _cp_ctx = _cp_model_cfg.get("context_length")
                                if _cp_ctx is not None:
                                    try:
                                        _config_context_length = int(_cp_ctx)
                                    except (TypeError, ValueError):
                                        pass
                        break
        
        # Select context engine: config-driven (like memory providers).
        # 1. Check config.yaml context.engine setting
        # 2. Check plugins/context_engine/<name>/ directory (repo-shipped)
        # 3. Check general plugin system (user-installed plugins)
        # 4. Fall back to built-in ContextCompressor
        _selected_engine = None
        _engine_name = "compressor"  # default
        try:
            _ctx_cfg = _agent_cfg.get("context", {}) if isinstance(_agent_cfg, dict) else {}
            _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
        except Exception:
            pass

        if _engine_name != "compressor":
            # Try loading from plugins/context_engine/<name>/
            try:
                from plugins.context_engine import load_context_engine
                _selected_engine = load_context_engine(_engine_name)
            except Exception as _ce_load_err:
                logger.debug("Context engine load from plugins/context_engine/: %s", _ce_load_err)

            # Try general plugin system as fallback
            if _selected_engine is None:
                try:
                    from hermes_cli.plugins import get_plugin_context_engine
                    _candidate = get_plugin_context_engine()
                    if _candidate and _candidate.name == _engine_name:
                        _selected_engine = _candidate
                except Exception:
                    pass

            if _selected_engine is None:
                logger.warning(
                    "Context engine '%s' not found — falling back to built-in compressor",
                    _engine_name,
                )
        # else: config says "compressor" — use built-in, don't auto-activate plugins

        if _selected_engine is not None:
            self.context_compressor = _selected_engine
            if not self.quiet_mode:
                logger.info("Using context engine: %s", _selected_engine.name)
        else:
            self.context_compressor = ContextCompressor(
                model=self.model,
                threshold_percent=compression_threshold,
                protect_first_n=3,
                protect_last_n=compression_protect_last,
                summary_target_ratio=compression_target_ratio,
                summary_model_override=compression_summary_model,
                quiet_mode=self.quiet_mode,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
                api_mode=self.api_mode,
            )
        self.compression_enabled = compression_enabled

        # Reject models whose context window is below the minimum required
        # for reliable tool-calling workflows (64K tokens).
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        _ctx = getattr(self.context_compressor, "context_length", 0)
        if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Model {self.model} has a context window of {_ctx:,} tokens, "
                f"which is below the minimum {MINIMUM_CONTEXT_LENGTH:,} required "
                f"by Hermes Agent.  Choose a model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context, or set "
                f"model.context_length in config.yaml to override."
            )

        # Inject context engine tool schemas (e.g. lcm_grep, lcm_describe, lcm_expand)
        self._context_engine_tool_names: set = set()
        if hasattr(self, "context_compressor") and self.context_compressor and self.tools is not None:
            for _schema in self.context_compressor.get_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    self.valid_tool_names.add(_tname)
                    self._context_engine_tool_names.add(_tname)

        # Notify context engine of session start
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_start(
                    self.session_id,
                    hermes_home=str(get_hermes_home()),
                    platform=self.platform or "cli",
                    model=self.model,
                    context_length=getattr(self.context_compressor, "context_length", 0),
                )
            except Exception as _ce_err:
                logger.debug("Context engine on_session_start: %s", _ce_err)

        self._subdirectory_hints = SubdirectoryHintTracker(
            working_dir=os.getenv("TERMINAL_CWD") or None,
        )
        self._user_turn_count = 0

        # Cumulative token usage for the session
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # ── Ollama num_ctx injection ──
        # Ollama defaults to 2048 context regardless of the model's capabilities.
        # When running against an Ollama server, detect the model's max context
        # and pass num_ctx on every chat request so the full window is used.
        # User override: set model.ollama_num_ctx in config.yaml to cap VRAM use.
        self._ollama_num_ctx: int | None = None
        _ollama_num_ctx_override = None
        if isinstance(_model_cfg, dict):
            _ollama_num_ctx_override = _model_cfg.get("ollama_num_ctx")
        if _ollama_num_ctx_override is not None:
            try:
                self._ollama_num_ctx = int(_ollama_num_ctx_override)
            except (TypeError, ValueError):
                logger.debug("Invalid ollama_num_ctx config value: %r", _ollama_num_ctx_override)
        if self._ollama_num_ctx is None and self.base_url and is_local_endpoint(self.base_url):
            try:
                _detected = query_ollama_num_ctx(self.model, self.base_url)
                if _detected and _detected > 0:
                    self._ollama_num_ctx = _detected
            except Exception as exc:
                logger.debug("Ollama num_ctx detection failed: %s", exc)
        if self._ollama_num_ctx and not self.quiet_mode:
            logger.info(
                "Ollama num_ctx: will request %d tokens (model max from /api/show)",
                self._ollama_num_ctx,
            )

        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")

        # Check immediately so CLI users see the warning at startup.
        # Gateway status_callback is not yet wired, so any warning is stored
        # in _compression_warning and replayed in the first run_conversation().
        self._compression_warning = None
        self._check_compression_model_feasibility()

        # Snapshot primary runtime for per-turn restoration.  When fallback
        # activates during a turn, the next turn restores these values so the
        # preferred model gets a fresh attempt each time.  Uses a single dict
        # so new state fields are easy to add without N individual attributes.
        _cc = self.context_compressor
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            # Context engine state that _try_activate_fallback() overwrites.
            # Use getattr for model/base_url/api_key/provider since plugin
            # engines may not have these (they're ContextCompressor-specific).
            "compressor_model": getattr(_cc, "model", self.model),
            "compressor_base_url": getattr(_cc, "base_url", self.base_url),
            "compressor_api_key": getattr(_cc, "api_key", ""),
            "compressor_provider": getattr(_cc, "provider", self.provider),
            "compressor_context_length": _cc.context_length,
            "compressor_threshold_tokens": _cc.threshold_tokens,
        }
        if self.api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })



    def _current_main_runtime(self) -> Dict[str, str]:
        """Return the live main runtime for session-scoped auxiliary routing."""
        return {
            "model": getattr(self, "model", "") or "",
            "provider": getattr(self, "provider", "") or "",
            "base_url": getattr(self, "base_url", "") or "",
            "api_key": getattr(self, "api_key", "") or "",
            "api_mode": getattr(self, "api_mode", "") or "",
        }



    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        url = (base_url or self._base_url_lower).lower()
        return "api.openai.com" in url and "openrouter" not in url



    def _is_openrouter_url(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return "openrouter" in self._base_url_lower



    def _strip_think_blocks(self, content: str) -> str:
        """Remove reasoning/thinking blocks from content, returning only visible text."""
        if not content:
            return ""
        # Strip all reasoning tag variants: <think>, <thinking>, <THINKING>,
        # <reasoning>, <REASONING_SCRATCHPAD>, <thought> (Gemma 4)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*', '', content, flags=re.IGNORECASE)
        return content



    def _cleanup_task_resources(self, task_id: str) -> None:
        """Clean up VM and browser resources for a given task.

        Skips ``cleanup_vm`` when the active terminal environment is marked
        persistent (``persistent_filesystem=True``) so that long-lived sandbox
        containers survive between turns. The idle reaper in
        ``terminal_tool._cleanup_inactive_envs`` still tears them down once
        ``terminal.lifetime_seconds`` is exceeded. Non-persistent backends are
        torn down per-turn as before to prevent resource leakage (the original
        intent of this hook for the Morph backend, see commit fbd3a2fd).
        """
        try:
            if is_persistent_env(task_id):
                if self.verbose_logging:
                    logging.debug(
                        f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                        f"idle reaper will handle it."
                    )
            else:
                cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # Background memory/skill review
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "Review the conversation above and consider saving to memory if appropriate.\n\n"
        "Focus on:\n"
        "1. Has the user revealed things about themselves — their persona, desires, "
        "preferences, or personal details worth remembering?\n"
        "2. Has the user expressed expectations about how you should behave, their work "
        "style, or ways they want you to operate?\n\n"
        "If something stands out, save it using the memory tool. "
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _SKILL_REVIEW_PROMPT = (
        "Review the conversation above and consider saving or updating a skill if appropriate.\n\n"
        "Focus on: was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome?\n\n"
        "If a relevant skill already exists, update it with what you learned. "
        "Otherwise, create a new skill if the approach is reusable.\n"
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and consider two things:\n\n"
        "**Memory**: Has the user revealed things about themselves — their persona, "
        "desires, preferences, or personal details? Has the user expressed expectations "
        "about how you should behave, their work style, or ways they want you to operate? "
        "If so, save using the memory tool.\n\n"
        "**Skills**: Was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome? If a relevant skill "
        "already exists, update it. Otherwise, create a new one if the approach is reusable.\n\n"
        "Only act if there's something genuinely worth saving. "
        "If nothing stands out, just say 'Nothing to save.' and stop."
    )



    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        Get messages up to (but not including) the last assistant turn.
        
        This is used when we need to "roll back" to the last successful point
        in the conversation, typically when the final assistant message is
        incomplete or malformed.
        
        Args:
            messages: Full message list
            
        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []
        
        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()
        
        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]
    


    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """
        Convert internal message format to trajectory format for saving.
        
        Args:
            messages (List[Dict]): Internal message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
            
        Returns:
            List[Dict]: Messages in trajectory format
        """
        trajectory = []
        
        # Add system message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({
            "from": "system",
            "value": system_msg
        })
        
        # Add the actual user prompt (from the dataset) as the first human message
        trajectory.append({
            "from": "human",
            "value": user_query
        })
        
        # Skip the first message (the user query) since we already added it above.
        # Prefill messages are injected at API-call time only (not in the messages
        # list), so no offset adjustment is needed here.
        i = 1
        
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                # Check if this message has tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Format assistant message with tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    if msg.get("content") and msg["content"].strip():
                        # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                        # (used when native thinking is disabled and model reasons via XML)
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"
                    
                    # Add tool calls wrapped in XML tags
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        # Parse arguments - should always succeed since we validate during conversation
                        # but keep try-except as safety net
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            # This shouldn't happen since we validate and retry during conversation,
                            # but if it does, log warning and use empty dict
                            logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    # so the format is consistent for training data
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })
                    
                    # Collect all subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # Format tool response with XML tags
                        tool_response = "<tool_response>\n"
                        
                        # Try to parse tool content as JSON if it looks like JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # Keep as string if not valid JSON
                        
                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": tool_name,
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    # Add all tool responses as a single message
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1  # Skip the tool messages we just processed
                
                else:
                    # Regular assistant message without tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)
                    
                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
                    })
            
            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })
            
            i += 1
        
        return trajectory
    


    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """
        Save conversation trajectory to JSONL file.
        
        Args:
            messages (List[Dict]): Complete message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
        """
        if not self.save_trajectories:
            return
        
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)
    


    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        Assemble the full system prompt from all layers.
        
        Called once per session (cached on self._cached_system_prompt) and only
        rebuilt after context compression events. This ensures the system prompt
        is stable across all turns in a session, maximizing prefix cache hits.
        """
        # Layers (in order):
        #   1. Agent identity — SOUL.md when available, else DEFAULT_AGENT_IDENTITY
        #   2. User / gateway system prompt (if provided)
        #   3. Persistent memory (frozen snapshot)
        #   4. Skills guidance (if skills tools are loaded)
        #   5. Context files (AGENTS.md, .cursorrules — SOUL.md excluded here when used as identity)
        #   6. Current date & time (frozen at build time)
        #   7. Platform-specific formatting hint

        # Try SOUL.md as primary identity (unless context files are skipped)
        _soul_loaded = False
        if not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            # Fallback to hardcoded identity
            prompt_parts = [DEFAULT_AGENT_IDENTITY]

        # Tool-aware behavioral guidance: only inject when the tools are loaded
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if "parallel_subtasks" in self.valid_tool_names:
            tool_guidance.append(PARALLEL_SUBTASKS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            prompt_parts.append(nous_subscription_prompt)
        # Tool-use enforcement: tells the model to actually call tools instead
        # of describing intended actions.  Controlled by config.yaml
        # agent.tool_use_enforcement:
        #   "auto" (default) — matches TOOL_USE_ENFORCEMENT_MODELS
        #   true  — always inject (all models)
        #   false — never inject
        #   list  — custom model-name substrings to match
        if self.valid_tool_names:
            # --- Depth Enhancement: Universal injection ---
            # TOOL_USE_ENFORCEMENT is now injected for ALL models unconditionally.
            # The old model-gating logic has been removed because shallow execution
            # is a universal LLM problem, not model-specific.
            _enforce = self._tool_use_enforcement
            _skip_enforcement = (
                _enforce is False
                or (isinstance(_enforce, str) and _enforce.lower() in ("false", "never", "no", "off"))
            )
            if not _skip_enforcement:
                prompt_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                prompt_parts.append(BRIEF_PARAMETER_DISCIPLINE)

            # Depth execution protocol — always injected when tools are available.
            # Addresses completion standards, anti-laziness, file-as-memory,
            # anti-fabrication, and error resilience.
            prompt_parts.append(DEPTH_EXECUTION_PROTOCOL)
            prompt_parts.append(FILE_OPERATION_CONSTRAINTS)
            prompt_parts.append(DEBUG_DELEGATION_PROTOCOL)
            prompt_parts.append(STRUCTURED_VARIATION_PROTOCOL)
            prompt_parts.append(OUTPUT_FORMAT_PROTOCOL)
            prompt_parts.append(EXTERNAL_SERVICE_PROTOCOL)
            prompt_parts.append(IMAGE_GENERATION_GUIDANCE)
            prompt_parts.append(PLAN_DISCIPLINE_PROTOCOL)
            prompt_parts.append(MESSAGE_DISCIPLINE_PROTOCOL)
            prompt_parts.append(VERIFICATION_PROTOCOL)
            prompt_parts.append(DELIVERY_GATE_PROTOCOL)
            prompt_parts.append(AGENT_LOOP_PROTOCOL)
            prompt_parts.append(REQUIREMENT_CHANGE_TRIGGER)
            prompt_parts.append(DISCLOSURE_PROHIBITION)
        if getattr(self, 'full_schema_mode', False):
            prompt_parts.append(TOOL_AVAILABILITY_PROTOCOL, IMAGE_GENERATION_GUIDANCE)

            # Model-specific guidance (additive, not gated)
            if not _skip_enforcement:
                _model_lower = (self.model or "").lower()
                # Google model operational guidance (conciseness, absolute
                # paths, parallel tool calls, verify-before-edit, etc.)
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    prompt_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex execution discipline (tool persistence,
                # prerequisite checks, verification, anti-hallucination).
                if "gpt" in _model_lower or "codex" in _model_lower:
                    prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        # so it can refer the user to them rather than reinventing answers.

        # Note: ephemeral_system_prompt is NOT included here. It's injected at
        # API-call time only so it stays out of the cached/stored system prompt.
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md is always included when enabled.
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # External memory provider system prompt block (additive to built-in)
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    prompt_parts.append(_ext_mem_block)
            except Exception:
                pass

        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in self.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            # Use TERMINAL_CWD for context file discovery when set (gateway
            # mode).  The gateway process runs from the hermes-agent install
            # dir, so os.getcwd() would pick up the repo's AGENTS.md and
            # other dev files — inflating token usage by ~10k for no benefit.
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
        # of the requested model. Inject explicit model identity into the system prompt
        # so the agent can correctly report which model it is (workaround for API bug).
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        # Environment hints (WSL, Termux, etc.) — tell the agent about the
        # execution environment so it can translate paths and adapt behavior.
        _env_hints = build_environment_hints()
        if _env_hints:
            prompt_parts.append(_env_hints)

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(p.strip() for p in prompt_parts if p.strip())

    # =========================================================================
    # Pre/post-call guardrails (inspired by PR #1321 — @alireza78a)
    # =========================================================================



    def _invalidate_system_prompt(self):
        """
        Invalidate the cached system prompt, forcing a rebuild on the next turn.
        
        Called after context compression events. Also reloads memory from disk
        so the rebuilt prompt captures any writes from this session.
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()



