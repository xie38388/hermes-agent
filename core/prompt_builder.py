"""PromptBuilderMixin - System prompt construction.
Extracted from run_agent.py Step 5.
Handles building the system prompt from soul, tools, and dynamic context.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import os

from agent.prompt_builder import (
    build_skills_system_prompt, build_context_files_prompt, build_environment_hints,
    build_nous_subscription_prompt,
    load_soul_md, TOOL_USE_ENFORCEMENT_GUIDANCE, TOOL_USE_ENFORCEMENT_MODELS,
    DEVELOPER_ROLE_MODELS, GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE, PARALLEL_SUBTASKS_GUIDANCE,
    DEPTH_EXECUTION_PROTOCOL, OUTPUT_FORMAT_PROTOCOL, EXTERNAL_SERVICE_PROTOCOL,
    PLAN_DISCIPLINE_PROTOCOL, MESSAGE_DISCIPLINE_PROTOCOL, VERIFICATION_PROTOCOL,
    BRIEF_PARAMETER_DISCIPLINE, FILE_OPERATION_CONSTRAINTS, DEBUG_DELEGATION_PROTOCOL,
    STRUCTURED_VARIATION_PROTOCOL, DELIVERY_GATE_PROTOCOL, AGENT_LOOP_PROTOCOL,
    REQUIREMENT_CHANGE_TRIGGER, DISCLOSURE_PROHIBITION, TOOL_AVAILABILITY_PROTOCOL,
    IMAGE_GENERATION_GUIDANCE,
    DEFAULT_AGENT_IDENTITY, MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE,
    PLATFORM_HINTS,
)
from model_tools import get_toolset_for_tool

class PromptBuilderMixin:
    """System prompt construction methods."""

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

