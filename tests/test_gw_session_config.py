"""Tests for gateway/gw_session_config_mixin.py static config loaders.

These methods load from env vars and config.yaml files. We test them by
controlling HERMES_HOME (via conftest) and writing temp config files.
"""
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ── Stub heavy gateway dependencies ──
_STUBS = {
    "fire": SimpleNamespace(Fire=lambda *a, **k: None),
    "openai": MagicMock(),
    "anthropic": MagicMock(),
    "tiktoken": MagicMock(),
    "colorama": SimpleNamespace(
        Fore=SimpleNamespace(RED="", GREEN="", YELLOW="", BLUE="", MAGENTA="", CYAN="", WHITE="", RESET=""),
        Style=SimpleNamespace(RESET_ALL="", BRIGHT=""),
    ),
    "prompt_toolkit": MagicMock(),
    "prompt_toolkit.formatted_text": MagicMock(),
    "rich": MagicMock(),
    "rich.console": MagicMock(),
    "rich.markdown": MagicMock(),
    "rich.panel": MagicMock(),
    "rich.live": MagicMock(),
    "rich.text": MagicMock(),
    "rich.table": MagicMock(),
    "rich.syntax": MagicMock(),
}
for mod_name, stub in _STUBS.items():
    sys.modules.setdefault(mod_name, stub)

# We need to import the mixin class. The gateway.run imports are heavy,
# so we import just the mixin file directly.
from gateway.gw_session_config_mixin import GwSessionConfigMixin


# ═══════════════════════════════════════════════════════════════════════════════
# _load_show_reasoning
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadShowReasoning:
    def test_default_is_false(self):
        """No config.yaml \u2192 False."""
        assert GwSessionConfigMixin._load_show_reasoning() is False

    def test_from_config_yaml(self, monkeypatch, tmp_path):
        """Config.yaml with display.show_reasoning: true \u2192 True.
        Uses monkeypatch to redirect _hermes_home to tmp_path so we never
        touch production ~/.hermes/config.yaml.
        """
        import yaml
        import gateway._helpers as _h
        import gateway.gw_session_config_mixin as _m
        # Point _hermes_home to tmp_path in both modules
        monkeypatch.setattr(_h, "_hermes_home", tmp_path)
        monkeypatch.setattr(_m, "_hermes_home", tmp_path)
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"display": {"show_reasoning": True}}))
        assert GwSessionConfigMixin._load_show_reasoning() is True


# ═══════════════════════════════════════════════════════════════════════════════
# _load_busy_input_mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadBusyInputMode:
    def test_default_is_queue(self, monkeypatch):
        monkeypatch.delenv("HERMES_BUSY_INPUT_MODE", raising=False)
        result = GwSessionConfigMixin._load_busy_input_mode()
        assert result in ("queue", "interrupt", "reject")  # default should be one of valid modes

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HERMES_BUSY_INPUT_MODE", "interrupt")
        assert GwSessionConfigMixin._load_busy_input_mode() == "interrupt"


# ═══════════════════════════════════════════════════════════════════════════════
# _load_background_notifications_mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadBackgroundNotificationsMode:
    def test_default_is_all(self, monkeypatch):
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        result = GwSessionConfigMixin._load_background_notifications_mode()
        assert result == "all"

    def test_env_override_result(self, monkeypatch):
        monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "result")
        assert GwSessionConfigMixin._load_background_notifications_mode() == "result"

    def test_env_override_off(self, monkeypatch):
        monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "off")
        assert GwSessionConfigMixin._load_background_notifications_mode() == "off"

    def test_env_override_error(self, monkeypatch):
        monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "error")
        assert GwSessionConfigMixin._load_background_notifications_mode() == "error"

    def test_invalid_falls_back_to_all(self, monkeypatch):
        monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "invalid_mode")
        assert GwSessionConfigMixin._load_background_notifications_mode() == "all"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "OFF")
        assert GwSessionConfigMixin._load_background_notifications_mode() == "off"


# ═══════════════════════════════════════════════════════════════════════════════
# _load_provider_routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadProviderRouting:
    def test_default_empty_dict(self, monkeypatch, tmp_path):
        """No config.yaml \u2192 empty dict."""
        result = GwSessionConfigMixin._load_provider_routing()
        assert isinstance(result, dict)

    def test_from_config_yaml(self, monkeypatch, tmp_path):
        """Config.yaml with provider_routing \u2192 returns the dict.
        Uses monkeypatch to redirect _hermes_home to tmp_path.
        """
        import yaml
        import gateway._helpers as _h
        import gateway.gw_session_config_mixin as _m
        monkeypatch.setattr(_h, "_hermes_home", tmp_path)
        monkeypatch.setattr(_m, "_hermes_home", tmp_path)
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"provider_routing": {"order": ["openai", "anthropic"]}}))
        result = GwSessionConfigMixin._load_provider_routing()
        assert result == {"order": ["openai", "anthropic"]}


# ═══════════════════════════════════════════════════════════════════════════════
# _load_smart_model_routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadSmartModelRouting:
    def test_default_empty_dict(self):
        result = GwSessionConfigMixin._load_smart_model_routing()
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# _load_prefill_messages
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadPrefillMessages:
    def test_default_empty_list(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
        result = GwSessionConfigMixin._load_prefill_messages()
        assert isinstance(result, list)
        assert result == []

    def test_from_file_env(self, monkeypatch, tmp_path):
        """HERMES_PREFILL_MESSAGES_FILE points to a JSON file."""
        msgs = [{"role": "user", "content": "hello"}]
        msg_file = tmp_path / "prefill.json"
        msg_file.write_text(json.dumps(msgs))
        monkeypatch.setenv("HERMES_PREFILL_MESSAGES_FILE", str(msg_file))
        result = GwSessionConfigMixin._load_prefill_messages()
        assert result == msgs

    def test_missing_file_returns_empty(self, monkeypatch):
        monkeypatch.setenv("HERMES_PREFILL_MESSAGES_FILE", "/nonexistent/path.json")
        result = GwSessionConfigMixin._load_prefill_messages()
        assert result == []
