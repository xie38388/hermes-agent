"""Shared utility functions extracted from run_agent.py.

These module-level helpers are used by multiple mixin files and must be
importable without triggering circular imports with run_agent.py.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any


# =========================================================================
# Safe stdio wrapper — prevents crash on broken pipes (systemd/headless)
# =========================================================================

class _SafeWriter:
    """Transparent stdio wrapper that catches OSError/ValueError from broken pipes."""
    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


# =========================================================================
# Surrogate sanitization — prevents json.dumps crash on invalid UTF-8
# =========================================================================

_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate code points with U+FFFD (replacement character)."""
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


def _sanitize_messages_surrogates(messages: list) -> bool:
    """Sanitize surrogate characters from all string content in a messages list.
    Walks message dicts in-place. Returns True if any surrogates were found
    and replaced.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
        name = msg.get("name")
        if isinstance(name, str) and _SURROGATE_RE.search(name):
            msg["name"] = _SURROGATE_RE.sub('\ufffd', name)
            found = True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and _SURROGATE_RE.search(tc_id):
                    tc["id"] = _SURROGATE_RE.sub('\ufffd', tc_id)
                    found = True
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and _SURROGATE_RE.search(fn_name):
                        fn["name"] = _SURROGATE_RE.sub('\ufffd', fn_name)
                        found = True
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str) and _SURROGATE_RE.search(fn_args):
                        fn["arguments"] = _SURROGATE_RE.sub('\ufffd', fn_args)
                        found = True
    return found


# =========================================================================
# Non-ASCII sanitization — last resort for ASCII-only systems
# =========================================================================

def _strip_non_ascii(text: str) -> str:
    """Remove non-ASCII characters for ASCII-only systems (LANG=C)."""
    return text.encode('ascii', errors='ignore').decode('ascii')


def _sanitize_messages_non_ascii(messages: list) -> bool:
    """Strip non-ASCII characters from all string content in a messages list."""
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            sanitized = _strip_non_ascii(content)
            if sanitized != content:
                msg["content"] = sanitized
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        sanitized = _strip_non_ascii(text)
                        if sanitized != text:
                            part["text"] = sanitized
                            found = True
        name = msg.get("name")
        if isinstance(name, str):
            sanitized = _strip_non_ascii(name)
            if sanitized != name:
                msg["name"] = sanitized
                found = True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        fn_args = fn.get("arguments")
                        if isinstance(fn_args, str):
                            sanitized = _strip_non_ascii(fn_args)
                            if sanitized != fn_args:
                                fn["arguments"] = sanitized
                                found = True
    return found


# =========================================================================
# Destructive command detection
# =========================================================================

_DESTRUCTIVE_PATTERNS = re.compile(
    r'(?:^|[;&|]\s*)(?:sudo\s+)?'
    r'(?:rm\s+-[rRf]*\s|rmdir\s|mv\s|cp\s+-[rRf]*\s|chmod\s|chown\s|'
    r'dd\s|mkfs\s|fdisk\s|parted\s|wipefs\s|shred\s|truncate\s|'
    r'git\s+(?:clean|reset\s+--hard|checkout\s+--\s+\.|push\s+--force)|'
    r'docker\s+(?:rm|rmi|system\s+prune)|'
    r'kubectl\s+delete|'
    r'pip\s+uninstall|npm\s+uninstall|apt\s+(?:remove|purge)|yum\s+(?:remove|erase))',
    re.MULTILINE,
)

_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def _is_destructive_command(cmd: str) -> bool:
    """Heuristic: does this terminal command look like it modifies/deletes files?"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


# =========================================================================
# Parallel tool batch safety check
# =========================================================================

_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

_PARALLEL_SAFE_TOOLS = frozenset({
    "web_search",
    "read_file",
    "write_file",
    "patch",
    "execute_code",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_scroll_down",
    "browser_scroll_up",
    "browser_back",
    "browser_press_key",
    "browser_view_page",
    "browser_move_mouse",
    "browser_select_option",
    "browser_restart",
    "image_generate",
})

_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> "Path | None":
    """Return the normalized file target for path-scoped tools."""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None
    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """Return True when a tool-call batch is safe to run concurrently."""
    if len(tool_calls) <= 1:
        return False
    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False
    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False
        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue
        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False
    return True


# =========================================================================
# Qwen Portal headers
# =========================================================================

_QWEN_CODE_VERSION = "0.14.1"


def _qwen_portal_headers() -> dict:
    """Return default HTTP headers required by Qwen Portal API."""
    import platform as _plat
    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }

import threading

class IterationBudget:
    """Thread-safe iteration counter for an agent."""
    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()
    def consume(self) -> bool:
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True
    def refund(self) -> None:
        with self._lock:
            if self._used > 0:
                self._used -= 1
    @property
    def used(self) -> int:
        return self._used
    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)
