"""Auto-generated mixin for AIAgent."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
import re
logger = logging.getLogger(__name__)

class ModelConfigMixin:
    """AIAgent mixin: model_config methods."""

    @property
    def base_url(self) -> str:
        return self._base_url


    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""


    def switch_model(self, new_model, new_provider, api_key='', base_url='', api_mode=''):
        """Switch the model/provider in-place for a live agent.

        Called by the /model command handlers (CLI and gateway) after
        ``model_switch.switch_model()`` has resolved credentials and
        validated the model.  This method performs the actual runtime
        swap: rebuilding clients, updating caching flags, and refreshing
        the context compressor.

        The implementation mirrors ``_try_activate_fallback()`` for the
        client-swap logic but also updates ``_primary_runtime`` so the
        change persists across turns (unlike fallback which is
        turn-scoped).
        """
        import logging
        from hermes_cli.providers import determine_api_mode

        # ── Determine api_mode if not provided ──
        if not api_mode:
            api_mode = determine_api_mode(new_provider, base_url)

        old_model = self.model
        old_provider = self.provider

        # ── Swap core runtime fields ──
        self.model = new_model
        self.provider = new_provider
        self.base_url = base_url or self.base_url
        self.api_mode = api_mode
        if api_key:
            self.api_key = api_key

        # ── Build new client ──
        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
            # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own
            # API key — falling back would send Anthropic credentials to third-party endpoints.
            _is_native_anthropic = new_provider == "anthropic"
            effective_key = (api_key or self.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or self.api_key or "")
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url or getattr(self, "_anthropic_base_url", None)
            self._anthropic_client = build_anthropic_client(
                effective_key, self._anthropic_base_url,
            )
            self._is_anthropic_oauth = _is_oauth_token(effective_key)
            self.client = None
            self._client_kwargs = {}
        else:
            effective_key = api_key or self.api_key
            effective_base = base_url or self.base_url
            self._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            self.client = self._create_openai_client(
                dict(self._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        # ── Re-evaluate prompt caching ──
        is_native_anthropic = api_mode == "anthropic_messages" and new_provider == "anthropic"
        self._use_prompt_caching = (
            ("openrouter" in (self.base_url or "").lower() and "claude" in new_model.lower())
            or is_native_anthropic
        )

        # ── Update context compressor ──
        if hasattr(self, "context_compressor") and self.context_compressor:
            from agent.model_metadata import get_model_context_length
            new_context_length = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                provider=self.provider,
                config_context_length=getattr(self, "_config_context_length", None),
            )
            self.context_compressor.update_model(
                model=self.model,
                context_length=new_context_length,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                provider=self.provider,
                api_mode=self.api_mode,
            )

        # ── Invalidate cached system prompt so it rebuilds next turn ──
        self._cached_system_prompt = None

        # ── Update _primary_runtime so the change persists across turns ──
        _cc = self.context_compressor if hasattr(self, "context_compressor") and self.context_compressor else None
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "compressor_model": getattr(_cc, "model", self.model) if _cc else self.model,
            "compressor_base_url": getattr(_cc, "base_url", self.base_url) if _cc else self.base_url,
            "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
            "compressor_provider": getattr(_cc, "provider", self.provider) if _cc else self.provider,
            "compressor_context_length": _cc.context_length if _cc else 0,
            "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
        }
        if api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

        # ── Reset fallback state ──
        self._fallback_activated = False
        self._fallback_index = 0

        logging.info(
            "Model switched in-place: %s (%s) -> %s (%s)",
            old_model, old_provider, new_model, new_provider,
        )


    @staticmethod
    def _model_requires_responses_api(model: str) -> bool:
        """Return True for models that require the Responses API path.

        GPT-5.x models are rejected on /v1/chat/completions by both
        OpenAI and OpenRouter (error: ``unsupported_api_for_model``).
        Detect these so the correct api_mode is set regardless of
        which provider is serving the model.
        """
        m = model.lower()
        # Strip vendor prefix (e.g. "openai/gpt-5.4" → "gpt-5.4")
        if "/" in m:
            m = m.rsplit("/", 1)[-1]
        return m.startswith("gpt-5")


    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        Extract reasoning/thinking content from an assistant message.
        
        OpenRouter and various providers can return reasoning in multiple formats:
        1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
        2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
        3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)
        
        Args:
            assistant_message: The assistant message object from the API response
            
        Returns:
            Combined reasoning text, or None if no reasoning found
        """
        reasoning_parts = []
        
        # Check direct reasoning field
        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)
        
        # Check reasoning_content field (alternative name used by some providers)
        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            # Don't duplicate if same as reasoning
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)
        
        # Check reasoning_details array (OpenRouter unified format)
        # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # Extract summary from reasoning detail object
                    summary = (
                        detail.get('summary')
                        or detail.get('thinking')
                        or detail.get('content')
                        or detail.get('text')
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # Some providers embed reasoning directly inside assistant content
        # instead of returning structured reasoning fields.  Only fall back
        # to inline extraction when no structured reasoning was found.
        content = getattr(assistant_message, "content", None)
        if not reasoning_parts and isinstance(content, str) and content:
            inline_patterns = (
                r"<think>(.*?)</think>",
                r"<thinking>(.*?)</thinking>",
                r"<reasoning>(.*?)</reasoning>",
                r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
            )
            for pattern in inline_patterns:
                flags = re.DOTALL | re.IGNORECASE
                for block in re.findall(pattern, content, flags=flags):
                    cleaned = block.strip()
                    if cleaned and cleaned not in reasoning_parts:
                        reasoning_parts.append(cleaned)
        
        # Combine all reasoning parts
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None


    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"


    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down the memory provider and context engine — call at actual session boundaries.

        This calls on_session_end() then shutdown_all() on the memory
        manager, and on_session_end() on the context engine.
        NOT called per-turn — only at CLI exit, /reset, gateway
        session expiry, etc.
        """
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
            try:
                self._memory_manager.shutdown_all()
            except Exception:
                pass
        # Notify context engine of session end (flush DAG, close DBs, etc.)
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass
    

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


    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered."""
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass


    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body fields to upstream providers.
        Some providers/routes reject `reasoning` with 400s, so gate it to
        known reasoning-capable model families and direct Nous Portal.
        """
        if "nousresearch" in self._base_url_lower:
            return True
        if "ai-gateway.vercel.sh" in self._base_url_lower:
            return True
        if "models.github.ai" in self._base_url_lower or "api.githubcopilot.com" in self._base_url_lower:
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if "openrouter" not in self._base_url_lower:
            return False
        if "api.mistral.ai" in self._base_url_lower:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)


    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}


