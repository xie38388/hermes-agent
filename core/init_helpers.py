"""
InitHelpersMixin — extracted __init__ helper methods.

Contains large initialization blocks that were extracted from AIAgent.__init__
to reduce its complexity. These methods are called during __init__ and should
not be called independently.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from hermes_constants import OPENROUTER_BASE_URL, get_hermes_home
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.shared_utils import _qwen_portal_headers
logger = logging.getLogger(__name__)


class InitHelpersMixin:
    """Mixin providing _init_* helper methods called from AIAgent.__init__."""

    def _init_llm_client(self, api_key: str, base_url: str) -> None:
        """Initialize the LLM client (Anthropic or OpenAI-compatible).
        
        Sets: self._anthropic_client, self._is_anthropic_oauth, self.client,
              self._client_kwargs, self.api_key
        """
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
        

    def _init_memory_system(self, skip_memory: bool, 
                            _agent_cfg: dict, platform: str) -> None:
        """Initialize memory stores (built-in MEMORY.md and external providers).
        
        Sets: self._memory_enabled, self._user_profile_enabled, self._memory_store,
              self._memory_manager, self._memory_nudge_interval, self._memory_flush_min_turns
        """
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
                _mem_provider_name = _agent_cfg.get("memory", {}).get("provider", "")

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


    def _init_context_engine(self, _agent_cfg: dict) -> None:
        """Initialize context compression engine.
        
        Sets: self.context_compressor, self.compression_enabled,
              self._context_engine_tool_names
        """
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
        

