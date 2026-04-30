"""Tests for agent mixin static methods — pure functions with zero mocking.

Covers:
  - api_build_mixin: _deterministic_call_id, _normalize_interim_visible_text
  - tool_mixin: _split_responses_tool_id, _sanitize_tool_calls_for_strict_api, _deduplicate_tool_calls
"""
import hashlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ── Stub heavy dependencies so we can import the mixins without the full stack ──
# These modules are imported at module level by run_agent.py but not used by the
# static methods under test.
_STUBS = {
    "fire": SimpleNamespace(Fire=lambda *a, **k: None),
    "openai": MagicMock(),
    "anthropic": MagicMock(),
    "tiktoken": MagicMock(),
    "colorama": SimpleNamespace(Fore=SimpleNamespace(RED="", GREEN="", YELLOW="", BLUE="", MAGENTA="", CYAN="", WHITE="", RESET=""), Style=SimpleNamespace(RESET_ALL="", BRIGHT="")),
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

from run_agent import AIAgent


# ═══════════════════════════════════════════════════════════════════════════════
# _deterministic_call_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterministicCallId:
    def test_basic_format(self):
        result = AIAgent._deterministic_call_id("web_search", '{"query":"test"}')
        assert result.startswith("call_")
        assert len(result) == 5 + 12  # "call_" + 12 hex chars

    def test_deterministic(self):
        a = AIAgent._deterministic_call_id("fn", "args", 0)
        b = AIAgent._deterministic_call_id("fn", "args", 0)
        assert a == b

    def test_different_inputs_different_ids(self):
        a = AIAgent._deterministic_call_id("fn_a", "args")
        b = AIAgent._deterministic_call_id("fn_b", "args")
        assert a != b

    def test_index_matters(self):
        a = AIAgent._deterministic_call_id("fn", "args", 0)
        b = AIAgent._deterministic_call_id("fn", "args", 1)
        assert a != b

    def test_matches_manual_hash(self):
        seed = "web_search:{\"q\":\"hi\"}:0"
        expected = "call_" + hashlib.sha256(seed.encode()).hexdigest()[:12]
        assert AIAgent._deterministic_call_id("web_search", '{"q":"hi"}', 0) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# _normalize_interim_visible_text
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeInterimVisibleText:
    def test_collapses_whitespace(self):
        assert AIAgent._normalize_interim_visible_text("hello   world") == "hello world"

    def test_strips_edges(self):
        assert AIAgent._normalize_interim_visible_text("  hello  ") == "hello"

    def test_newlines_to_space(self):
        assert AIAgent._normalize_interim_visible_text("line1\n\nline2") == "line1 line2"

    def test_non_string_returns_empty(self):
        assert AIAgent._normalize_interim_visible_text(None) == ""
        assert AIAgent._normalize_interim_visible_text(42) == ""

    def test_empty_string(self):
        assert AIAgent._normalize_interim_visible_text("") == ""


# ═══════════════════════════════════════════════════════════════════════════════
# _split_responses_tool_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitResponsesToolId:
    def test_pipe_separated(self):
        call_id, resp_id = AIAgent._split_responses_tool_id("call_abc|fc_xyz")
        assert call_id == "call_abc"
        assert resp_id == "fc_xyz"

    def test_fc_prefix_only(self):
        call_id, resp_id = AIAgent._split_responses_tool_id("fc_xyz")
        assert call_id is None
        assert resp_id == "fc_xyz"

    def test_call_id_only(self):
        call_id, resp_id = AIAgent._split_responses_tool_id("call_abc")
        assert call_id == "call_abc"
        assert resp_id is None

    def test_empty_string(self):
        call_id, resp_id = AIAgent._split_responses_tool_id("")
        assert call_id is None
        assert resp_id is None

    def test_none_input(self):
        call_id, resp_id = AIAgent._split_responses_tool_id(None)
        assert call_id is None
        assert resp_id is None

    def test_non_string_input(self):
        call_id, resp_id = AIAgent._split_responses_tool_id(12345)
        assert call_id is None
        assert resp_id is None

    def test_pipe_with_empty_parts(self):
        call_id, resp_id = AIAgent._split_responses_tool_id("|fc_xyz")
        assert call_id is None
        assert resp_id == "fc_xyz"

    def test_whitespace_stripped(self):
        call_id, resp_id = AIAgent._split_responses_tool_id("  call_abc | fc_xyz  ")
        assert call_id == "call_abc"
        assert resp_id == "fc_xyz"


# ═══════════════════════════════════════════════════════════════════════════════
# _sanitize_tool_calls_for_strict_api
# ═══════════════════════════════════════════════════════════════════════════════

class TestSanitizeToolCallsForStrictApi:
    def test_strips_call_id_and_response_item_id(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "test"}, "call_id": "extra", "response_item_id": "extra2"},
            ],
        }
        result = AIAgent._sanitize_tool_calls_for_strict_api(msg)
        tc = result["tool_calls"][0]
        assert "call_id" not in tc
        assert "response_item_id" not in tc
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "test"

    def test_no_tool_calls_passthrough(self):
        msg = {"role": "assistant", "content": "hello"}
        result = AIAgent._sanitize_tool_calls_for_strict_api(msg)
        assert result == msg

    def test_non_dict_tool_calls_preserved(self):
        msg = {"role": "assistant", "tool_calls": ["not_a_dict"]}
        result = AIAgent._sanitize_tool_calls_for_strict_api(msg)
        assert result["tool_calls"] == ["not_a_dict"]

    def test_multiple_tool_calls(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "call_id": "x1"},
                {"id": "c2", "response_item_id": "y2"},
                {"id": "c3"},
            ],
        }
        result = AIAgent._sanitize_tool_calls_for_strict_api(msg)
        assert len(result["tool_calls"]) == 3
        for tc in result["tool_calls"]:
            assert "call_id" not in tc
            assert "response_item_id" not in tc


# ═══════════════════════════════════════════════════════════════════════════════
# _deduplicate_tool_calls
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeduplicateToolCalls:
    def _make_tc(self, tc_id, name="test", args="{}"):
        return SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=args))

    def test_no_duplicates(self):
        tcs = [self._make_tc("c1", "fn_a"), self._make_tc("c2", "fn_b")]
        result = AIAgent._deduplicate_tool_calls(tcs)
        assert len(result) == 2

    def test_exact_duplicates_removed(self):
        tc = self._make_tc("c1", "fn_a", '{"x":1}')
        result = AIAgent._deduplicate_tool_calls([tc, tc])
        assert len(result) == 1

    def test_same_name_different_args_kept(self):
        tc1 = self._make_tc("c1", "fn_a", '{"x":1}')
        tc2 = self._make_tc("c2", "fn_a", '{"x":2}')
        result = AIAgent._deduplicate_tool_calls([tc1, tc2])
        assert len(result) == 2

    def test_empty_list(self):
        result = AIAgent._deduplicate_tool_calls([])
        assert result == []
