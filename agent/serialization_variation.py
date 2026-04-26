"""
Runtime Serialization Variation — Anti-Few-Shot Pattern Breaking

Applies controlled micro-variations to tool result strings before they
enter the conversation context. This prevents the model from locking
into repetitive action-observation patterns (Manus Context Engineering
Principle 6).

Variations are deterministic per-iteration (seeded by iteration count)
so they are reproducible for debugging, but different across iterations
to break monotony.

IMPORTANT: Variations are cosmetic only — they never change semantic
content, data values, or structure. They only affect:
- JSON key ordering (shuffled)
- Whitespace/indentation style (2-space vs 4-space vs compact)
- Trailing newlines
- Minor wording in status prefixes
"""
import json
import random
import re
from typing import Optional


# Status prefix variants — semantically identical, visually different
_SUCCESS_PREFIXES = [
    "Result:",
    "Output:",
    "Completed:",
    "Done:",
    "✓ Result:",
]

_ERROR_PREFIXES = [
    "Error:",
    "Failed:",
    "⚠ Error:",
    "Issue:",
]

# Indentation styles
_INDENT_STYLES = [2, 4, None]  # None = compact (no indent)


def apply_serialization_variation(
    result_str: str,
    iteration: int,
    tool_name: str = "",
) -> str:
    """
    Apply controlled micro-variation to a tool result string.
    
    Args:
        result_str: The original tool result string
        iteration: Current agent loop iteration (used as seed)
        tool_name: Name of the tool (for context-aware variation)
    
    Returns:
        Slightly varied version of the result string
    """
    if not result_str or len(result_str) < 10:
        return result_str
    
    # Skip variation for tools where JSON key order matters for LLM interpretation
    _NO_VARIATION_TOOLS = {"image_generate", "text_to_speech", "vision_analyze"}
    if tool_name in _NO_VARIATION_TOOLS:
        return result_str
    
    # Use iteration as seed for reproducible but varying results
    rng = random.Random(iteration * 7919 + hash(tool_name) % 1000)
    
    # Try to parse as JSON for structural variation
    varied = _try_vary_json(result_str, rng)
    if varied is not None:
        return varied
    
    # For non-JSON results, apply text-level variation
    return _vary_text(result_str, rng)


def _try_vary_json(text: str, rng: random.Random) -> Optional[str]:
    """Try to parse as JSON and apply key-order + indent variation."""
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    
    # Shuffle dict keys at top level (and one level deep)
    if isinstance(data, dict):
        data = _shuffle_dict_keys(data, rng, depth=0, max_depth=1)
    
    # Pick indent style
    indent = rng.choice(_INDENT_STYLES)
    
    # Serialize with variation
    result = json.dumps(data, indent=indent, ensure_ascii=False)
    
    # Occasionally add/remove trailing newline
    if rng.random() < 0.3:
        result = result.rstrip("\n") + "\n"
    
    return result


def _shuffle_dict_keys(d: dict, rng: random.Random, depth: int, max_depth: int) -> dict:
    """Shuffle dictionary keys at current and child levels."""
    items = list(d.items())
    rng.shuffle(items)
    
    result = {}
    for k, v in items:
        if isinstance(v, dict) and depth < max_depth:
            v = _shuffle_dict_keys(v, rng, depth + 1, max_depth)
        result[k] = v
    
    return result


def _vary_text(text: str, rng: random.Random) -> str:
    """Apply minor text-level variations to non-JSON results."""
    lines = text.split("\n")
    
    # Vary leading status line if present
    if lines and lines[0].strip():
        first = lines[0].strip()
        # Check if it starts with a known status word
        for prefix in ["Result:", "Output:", "Error:", "Success:", "Failed:", "Done:"]:
            if first.startswith(prefix):
                # Replace with a random variant
                content = first[len(prefix):].strip()
                if "error" in prefix.lower() or "fail" in prefix.lower():
                    new_prefix = rng.choice(_ERROR_PREFIXES)
                else:
                    new_prefix = rng.choice(_SUCCESS_PREFIXES)
                lines[0] = f"{new_prefix} {content}"
                break
    
    # Occasionally vary blank line density (add or remove one trailing blank)
    if rng.random() < 0.2:
        if lines and lines[-1].strip() == "":
            lines = lines[:-1]
        else:
            lines.append("")
    
    return "\n".join(lines)
