"""
browser_enhanced.py — ai-manus style enhanced browser capabilities for Camofox backend.

Provides:
  - extract_interactive_elements(task_id) → structured interactive element list
  - extract_visible_content(task_id) → visible page content as text
  - camofox_view_page(task_id) → combined view (interactive_elements + content)
  - camofox_click_enhanced(index/coordinate/ref, task_id) → click by index, coordinate, or ref
  - camofox_type_enhanced(text, index/coordinate/ref, task_id) → type by index, coordinate, or ref
  - camofox_move_mouse(x, y, task_id) → move mouse to coordinate
  - camofox_select_option(index, option, task_id) → select dropdown option
  - camofox_scroll_enhanced(direction, task_id) → enhanced scroll (up/down/to_top/to_bottom)
  - camofox_restart(task_id) → restart browser session

All functions use the existing Camofox REST API (especially /evaluate and /type)
to inject ai-manus style JavaScript for DOM extraction and data-manus-id indexing.
"""

import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JavaScript for extracting interactive elements (ported from ai-manus)
# ---------------------------------------------------------------------------
EXTRACT_INTERACTIVE_ELEMENTS_JS = """() => {
    const interactiveElements = [];
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    // Remove old data-manus-id attributes
    document.querySelectorAll('[data-manus-id]').forEach(el => el.removeAttribute('data-manus-id'));

    const elements = document.querySelectorAll(
        'button, a, input, textarea, select, ' +
        '[role="button"], [role="link"], [role="tab"], [role="menuitem"], ' +
        '[role="checkbox"], [role="radio"], [role="switch"], [role="option"], ' +
        '[tabindex]:not([tabindex="-1"]), [onclick], [contenteditable="true"]'
    );

    let validElementIndex = 0;

    for (let i = 0; i < elements.length; i++) {
        const element = elements[i];
        const rect = element.getBoundingClientRect();

        if (rect.width === 0 || rect.height === 0) continue;
        if (rect.bottom < 0 || rect.top > viewportHeight || rect.right < 0 || rect.left > viewportWidth) continue;

        const style = window.getComputedStyle(element);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || parseFloat(style.opacity) < 0.1) continue;

        let tagName = element.tagName.toLowerCase();
        let text = '';

        if (element.value && ['input', 'textarea', 'select'].includes(tagName)) {
            text = element.value;
            if (tagName === 'input') {
                let labelText = '';
                if (element.id) {
                    const label = document.querySelector('label[for="' + element.id + '"]');
                    if (label) labelText = label.innerText.trim();
                }
                if (!labelText) {
                    const parentLabel = element.closest('label');
                    if (parentLabel) labelText = parentLabel.innerText.trim().replace(element.value, '').trim();
                }
                if (labelText) text = '[Label: ' + labelText + '] ' + text;
                if (element.placeholder) text = text + ' [Placeholder: ' + element.placeholder + ']';
            }
        } else if (element.innerText) {
            text = element.innerText.trim().replace(/\\s+/g, ' ');
        } else if (element.alt) {
            text = element.alt;
        } else if (element.title) {
            text = element.title;
        } else if (element.placeholder) {
            text = '[Placeholder: ' + element.placeholder + ']';
        } else if (element.type) {
            text = '[' + element.type + ']';
            if (tagName === 'input') {
                let labelText = '';
                if (element.id) {
                    const label = document.querySelector('label[for="' + element.id + '"]');
                    if (label) labelText = label.innerText.trim();
                }
                if (!labelText) {
                    const parentLabel = element.closest('label');
                    if (parentLabel) labelText = parentLabel.innerText.trim();
                }
                if (labelText) text = '[Label: ' + labelText + '] ' + text;
                if (element.placeholder) text = text + ' [Placeholder: ' + element.placeholder + ']';
            }
        } else if (element.ariaLabel) {
            text = element.ariaLabel;
        } else {
            text = '[No text]';
        }

        if (text.length > 100) text = text.substring(0, 97) + '...';

        element.setAttribute('data-manus-id', 'manus-element-' + validElementIndex);

        interactiveElements.push({
            index: validElementIndex,
            tag: tagName,
            text: text,
            selector: '[data-manus-id="manus-element-' + validElementIndex + '"]'
        });

        validElementIndex++;
    }

    return interactiveElements;
}"""

# ---------------------------------------------------------------------------
# JavaScript for extracting visible page content
# ---------------------------------------------------------------------------
EXTRACT_VISIBLE_CONTENT_JS = """() => {
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    const visibleTexts = [];
    const seen = new Set();

    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode: function(node) {
                const parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;
                const tag = parent.tagName.toLowerCase();
                if (['script', 'style', 'noscript', 'svg', 'path'].includes(tag)) return NodeFilter.FILTER_REJECT;
                const style = window.getComputedStyle(parent);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return NodeFilter.FILTER_REJECT;
                const rect = parent.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return NodeFilter.FILTER_REJECT;
                if (rect.bottom < 0 || rect.top > viewportHeight || rect.right < 0 || rect.left > viewportWidth) return NodeFilter.FILTER_REJECT;
                const text = node.textContent.trim();
                if (!text || text.length < 2) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
            }
        }
    );

    let node;
    while (node = walker.nextNode()) {
        const text = node.textContent.trim();
        if (!seen.has(text)) {
            seen.add(text);
            visibleTexts.push(text);
        }
    }

    const title = document.title || '';
    const url = window.location.href || '';
    const h1s = Array.from(document.querySelectorAll('h1')).map(h => h.innerText.trim()).filter(Boolean);
    const h2s = Array.from(document.querySelectorAll('h2')).map(h => h.innerText.trim()).filter(Boolean);

    return {
        title: title,
        url: url,
        headings: { h1: h1s, h2: h2s },
        visible_text: visibleTexts.join('\\n'),
        text_length: visibleTexts.join('\\n').length
    };
}"""

# ---------------------------------------------------------------------------
# Helpers — use Camofox REST API via existing _post / _get helpers
# ---------------------------------------------------------------------------

def _evaluate_js(tab_id: str, user_id: str, expression: str, timeout: int = 30) -> Any:
    """Execute JavaScript via Camofox evaluate endpoint.
    Uses the existing _post helper from browser_camofox.
    """
    from tools.browser_camofox import _post
    data = _post(
        f"/tabs/{tab_id}/evaluate",
        {"userId": user_id, "expression": expression},
        timeout=timeout,
    )
    return data.get("result")


def _get_session_for_task(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Get camofox session info for the task."""
    from tools.browser_camofox import _get_session
    return _get_session(task_id)


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def extract_interactive_elements(task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract interactive elements from current page using ai-manus style JS injection.
    
    Returns list of dicts: [{index, tag, text, selector}, ...]
    """
    session = _get_session_for_task(task_id)
    if not session.get("tab_id"):
        return []
    try:
        result = _evaluate_js(session["tab_id"], session["user_id"], EXTRACT_INTERACTIVE_ELEMENTS_JS)
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.warning("extract_interactive_elements failed: %s", e)
        return []


def extract_visible_content(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Extract visible page content as structured text.
    
    Returns dict: {title, url, headings, visible_text, text_length}
    """
    session = _get_session_for_task(task_id)
    if not session.get("tab_id"):
        return {"title": "", "url": "", "headings": {}, "visible_text": "", "text_length": 0}
    try:
        result = _evaluate_js(session["tab_id"], session["user_id"], EXTRACT_VISIBLE_CONTENT_JS)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("extract_visible_content failed: %s", e)
        return {}


def format_interactive_elements(elements: List[Dict[str, Any]]) -> str:
    """Format interactive elements into readable string for Agent consumption.
    
    Format: index:<tag>text</tag>
    """
    lines = []
    for el in elements:
        lines.append(f"{el['index']}:<{el['tag']}>{el['text']}</{el['tag']}>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def camofox_view_page(task_id: Optional[str] = None) -> str:
    """View current page: extract interactive elements + visible content.
    
    Returns structured interactive elements with index numbers for subsequent
    click/type operations, plus visible page content as text.
    """
    try:
        from tools.registry import tool_error
        session = _get_session_for_task(task_id)
        if not session.get("tab_id"):
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        elements = extract_interactive_elements(task_id)
        formatted = format_interactive_elements(elements)
        content = extract_visible_content(task_id)

        visible_text = content.get("visible_text", "")
        if len(visible_text) > 8000:
            visible_text = visible_text[:8000] + "\n... [truncated]"

        return json.dumps({
            "success": True,
            "url": content.get("url", ""),
            "title": content.get("title", ""),
            "interactive_elements": formatted,
            "interactive_elements_count": len(elements),
            "page_content": visible_text,
            "headings": content.get("headings", {}),
        }, ensure_ascii=False)
    except Exception as e:
        from tools.registry import tool_error
        return tool_error(str(e), success=False)


def camofox_click_enhanced(
    ref: Optional[str] = None,
    index: Optional[int] = None,
    coordinate_x: Optional[float] = None,
    coordinate_y: Optional[float] = None,
    task_id: Optional[str] = None,
) -> str:
    """Enhanced click: supports index (ai-manus style), coordinate, or ref (legacy).
    
    Priority: index > coordinate > ref
    """
    try:
        from tools.registry import tool_error
        session = _get_session_for_task(task_id)
        if not session.get("tab_id"):
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        tab_id = session["tab_id"]
        user_id = session["user_id"]

        if index is not None:
            # Click by data-manus-id index
            click_js = f"""() => {{
                const el = document.querySelector('[data-manus-id="manus-element-{index}"]');
                if (!el) return {{ success: false, error: 'Element with index {index} not found. Call browser_view_page first to refresh elements.' }};
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                if (rect.width === 0 || rect.height === 0 || style.display === 'none' || style.visibility === 'hidden') {{
                    el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
                el.click();
                return {{ success: true, tag: el.tagName.toLowerCase(), text: (el.innerText || '').substring(0, 50) }};
            }}"""
            result = _evaluate_js(tab_id, user_id, click_js)
            if isinstance(result, dict) and not result.get("success"):
                return tool_error(result.get("error", "Click failed"), success=False)
            return json.dumps({
                "success": True,
                "clicked_by": "index",
                "index": index,
                "element": result if isinstance(result, dict) else {},
            }, ensure_ascii=False)

        elif coordinate_x is not None and coordinate_y is not None:
            # Click by coordinate
            click_js = f"""() => {{
                const el = document.elementFromPoint({coordinate_x}, {coordinate_y});
                if (!el) return {{ success: false, error: 'No element at ({coordinate_x}, {coordinate_y})' }};
                el.click();
                return {{ success: true, tag: el.tagName.toLowerCase(), text: (el.innerText || '').substring(0, 50) }};
            }}"""
            result = _evaluate_js(tab_id, user_id, click_js)
            if isinstance(result, dict) and not result.get("success"):
                return tool_error(result.get("error", "Click failed"), success=False)
            return json.dumps({
                "success": True,
                "clicked_by": "coordinate",
                "x": coordinate_x,
                "y": coordinate_y,
                "element": result if isinstance(result, dict) else {},
            }, ensure_ascii=False)

        elif ref is not None:
            # Legacy ref click — delegate to original camofox_click
            from tools.browser_camofox import camofox_click
            return camofox_click(ref=ref, task_id=task_id)

        else:
            return tool_error("Must provide one of: index, coordinate (x,y), or ref", success=False)

    except Exception as e:
        from tools.registry import tool_error
        return tool_error(str(e), success=False)


def camofox_type_enhanced(
    text: str,
    ref: Optional[str] = None,
    index: Optional[int] = None,
    coordinate_x: Optional[float] = None,
    coordinate_y: Optional[float] = None,
    press_enter: bool = False,
    clear_first: bool = True,
    task_id: Optional[str] = None,
) -> str:
    """Enhanced type: supports index (ai-manus style), coordinate, or ref (legacy).
    
    Priority: index > coordinate > ref
    """
    try:
        from tools.registry import tool_error
        from tools.browser_camofox import _post as camofox_post
        session = _get_session_for_task(task_id)
        if not session.get("tab_id"):
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        tab_id = session["tab_id"]
        user_id = session["user_id"]

        if index is not None:
            # Use Camofox /type endpoint with selector (it supports selector param)
            selector = f'[data-manus-id="manus-element-{index}"]'
            try:
                camofox_post(
                    f"/tabs/{tab_id}/type",
                    {"userId": user_id, "selector": selector, "text": text},
                    timeout=30,
                )
            except Exception as fill_err:
                # Fallback: focus via JS then use keyboard
                logger.warning("Type via selector failed, trying JS fallback: %s", fill_err)
                focus_js = f"""() => {{
                    const el = document.querySelector('[data-manus-id="manus-element-{index}"]');
                    if (!el) return {{ success: false, error: 'Element not found' }};
                    el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    el.focus();
                    el.click();
                    if ({'true' if clear_first else 'false'}) {{ el.value = ''; el.dispatchEvent(new Event('input', {{ bubbles: true }})); }}
                    return {{ success: true }};
                }}"""
                _evaluate_js(tab_id, user_id, focus_js)
                # Type character by character via press endpoint
                for char in text:
                    camofox_post(
                        f"/tabs/{tab_id}/press",
                        {"userId": user_id, "key": char},
                        timeout=10,
                    )

            if press_enter:
                camofox_post(
                    f"/tabs/{tab_id}/press",
                    {"userId": user_id, "key": "Enter"},
                    timeout=10,
                )

            return json.dumps({
                "success": True,
                "typed_by": "index",
                "index": index,
                "text": text,
                "press_enter": press_enter,
            }, ensure_ascii=False)

        elif coordinate_x is not None and coordinate_y is not None:
            # Click coordinate first via JS, then type via keyboard
            click_js = f"""() => {{
                const el = document.elementFromPoint({coordinate_x}, {coordinate_y});
                if (!el) return {{ success: false, error: 'No element at coordinate' }};
                el.focus();
                el.click();
                return {{ success: true }};
            }}"""
            _evaluate_js(tab_id, user_id, click_js)

            # Type via press endpoint
            for char in text:
                camofox_post(
                    f"/tabs/{tab_id}/press",
                    {"userId": user_id, "key": char},
                    timeout=10,
                )

            if press_enter:
                camofox_post(
                    f"/tabs/{tab_id}/press",
                    {"userId": user_id, "key": "Enter"},
                    timeout=10,
                )

            return json.dumps({
                "success": True,
                "typed_by": "coordinate",
                "x": coordinate_x,
                "y": coordinate_y,
                "text": text,
            }, ensure_ascii=False)

        elif ref is not None:
            # Legacy ref type
            from tools.browser_camofox import camofox_type
            return camofox_type(ref, text, task_id)

        else:
            return tool_error("Must provide one of: index, coordinate (x,y), or ref", success=False)

    except Exception as e:
        from tools.registry import tool_error
        return tool_error(str(e), success=False)


def camofox_move_mouse(
    coordinate_x: float,
    coordinate_y: float,
    task_id: Optional[str] = None,
) -> str:
    """Move mouse to coordinate (hover). Uses JS to dispatch mouseover/mousemove events."""
    try:
        from tools.registry import tool_error
        session = _get_session_for_task(task_id)
        if not session.get("tab_id"):
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        move_js = f"""() => {{
            const el = document.elementFromPoint({coordinate_x}, {coordinate_y});
            if (el) {{
                el.dispatchEvent(new MouseEvent('mouseenter', {{ bubbles: true, clientX: {coordinate_x}, clientY: {coordinate_y} }}));
                el.dispatchEvent(new MouseEvent('mouseover', {{ bubbles: true, clientX: {coordinate_x}, clientY: {coordinate_y} }}));
                el.dispatchEvent(new MouseEvent('mousemove', {{ bubbles: true, clientX: {coordinate_x}, clientY: {coordinate_y} }}));
            }}
            return {{ success: true, element: el ? el.tagName.toLowerCase() : null }};
        }}"""
        result = _evaluate_js(session["tab_id"], session["user_id"], move_js)
        return json.dumps({
            "success": True,
            "moved_to": {"x": coordinate_x, "y": coordinate_y},
            "element": result.get("element") if isinstance(result, dict) else None,
        }, ensure_ascii=False)
    except Exception as e:
        from tools.registry import tool_error
        return tool_error(str(e), success=False)


def camofox_select_option(
    index: int,
    option: int,
    task_id: Optional[str] = None,
) -> str:
    """Select a dropdown option by element index and option index."""
    try:
        from tools.registry import tool_error
        session = _get_session_for_task(task_id)
        if not session.get("tab_id"):
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        select_js = f"""() => {{
            const el = document.querySelector('[data-manus-id="manus-element-{index}"]');
            if (!el) return {{ success: false, error: 'Element with index {index} not found' }};
            if (el.tagName.toLowerCase() !== 'select') return {{ success: false, error: 'Element is not a select dropdown' }};
            if ({option} >= el.options.length) return {{ success: false, error: 'Option index out of range (max: ' + (el.options.length - 1) + ')' }};
            el.selectedIndex = {option};
            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            return {{ success: true, selected: el.options[{option}].text, value: el.options[{option}].value }};
        }}"""
        result = _evaluate_js(session["tab_id"], session["user_id"], select_js)
        if isinstance(result, dict) and not result.get("success"):
            return tool_error(result.get("error", "Select failed"), success=False)
        return json.dumps({
            "success": True,
            "element_index": index,
            "option_index": option,
            "selected": result.get("selected") if isinstance(result, dict) else None,
        }, ensure_ascii=False)
    except Exception as e:
        from tools.registry import tool_error
        return tool_error(str(e), success=False)


def camofox_scroll_enhanced(direction: str = "down", task_id: Optional[str] = None) -> str:
    """Enhanced scroll: supports up, down, to_top, to_bottom."""
    try:
        from tools.registry import tool_error
        session = _get_session_for_task(task_id)
        if not session.get("tab_id"):
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        scroll_map = {
            "up": "window.scrollBy(0, -window.innerHeight)",
            "down": "window.scrollBy(0, window.innerHeight)",
            "to_top": "window.scrollTo(0, 0)",
            "to_bottom": "window.scrollTo(0, document.body.scrollHeight)",
        }
        js_expr = scroll_map.get(direction)
        if not js_expr:
            return tool_error(f"Invalid direction: {direction}. Use: up, down, to_top, to_bottom", success=False)

        _evaluate_js(session["tab_id"], session["user_id"], js_expr)
        return json.dumps({"success": True, "scrolled": direction}, ensure_ascii=False)
    except Exception as e:
        from tools.registry import tool_error
        return tool_error(str(e), success=False)


def camofox_restart(task_id: Optional[str] = None) -> str:
    """Restart browser session: close current tab and clear session."""
    try:
        from tools.registry import tool_error
        from tools.browser_camofox import camofox_close, _drop_session

        camofox_close(task_id)
        _drop_session(task_id)

        return json.dumps({
            "success": True,
            "message": "Browser session restarted. Call browser_navigate to open a new page.",
        }, ensure_ascii=False)
    except Exception as e:
        from tools.registry import tool_error
        return tool_error(str(e), success=False)


# ---------------------------------------------------------------------------
# Enhanced tool schemas (new tools to add)
# ---------------------------------------------------------------------------

ENHANCED_BROWSER_TOOL_SCHEMAS = [
    {
        "name": "browser_view_page",
        "description": "View the current page with structured interactive elements and visible content. Returns a numbered list of clickable/typeable elements (use the index number with browser_click or browser_type) and the visible page text. Call this after navigating to understand what's on the page and what you can interact with.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "browser_select_option",
        "description": "Select an option from a dropdown (<select>) element. First call browser_view_page to get the element index, then use this tool with the element index and the option index (0-based).",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "The index of the select element from browser_view_page output.",
                },
                "option": {
                    "type": "integer",
                    "description": "The option index to select (0-based).",
                },
            },
            "required": ["index", "option"],
        },
    },
    {
        "name": "browser_move_mouse",
        "description": "Move the mouse cursor to a specific coordinate on the page. Useful for hovering over elements to trigger tooltips or dropdown menus.",
        "parameters": {
            "type": "object",
            "properties": {
                "coordinate_x": {
                    "type": "number",
                    "description": "X coordinate (pixels from left edge of viewport).",
                },
                "coordinate_y": {
                    "type": "number",
                    "description": "Y coordinate (pixels from top edge of viewport).",
                },
            },
            "required": ["coordinate_x", "coordinate_y"],
        },
    },
    {
        "name": "browser_restart",
        "description": "Restart the browser session. Closes the current tab and clears the session. Call browser_navigate afterwards to open a new page. Use this when the browser is in a bad state or you need a fresh start.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# Patched schemas for existing tools (click, type, scroll gain new params)
CLICK_SCHEMA_PATCH = {
    "name": "browser_click",
    "description": "Click an element on the page. Supports three modes: (1) index — use the element index from browser_view_page output (PREFERRED), (2) coordinate — click at specific x,y pixel position, (3) ref — legacy accessibility tree reference like @e5. When using index mode, call browser_view_page first to get the numbered element list.",
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Legacy: Accessibility tree element ref (e.g. @e5). Use index mode instead when possible.",
            },
            "index": {
                "type": "integer",
                "description": "PREFERRED: Element index from browser_view_page output.",
            },
            "coordinate_x": {
                "type": "number",
                "description": "X coordinate for coordinate-based click.",
            },
            "coordinate_y": {
                "type": "number",
                "description": "Y coordinate for coordinate-based click.",
            },
        },
        "required": [],
    },
}

TYPE_SCHEMA_PATCH = {
    "name": "browser_type",
    "description": "Type text into an input field. Supports three modes: (1) index — use the element index from browser_view_page output (PREFERRED), (2) coordinate — click at x,y then type, (3) ref — legacy accessibility tree reference. When using index mode, call browser_view_page first.",
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Legacy: Accessibility tree element ref (e.g. @e5).",
            },
            "index": {
                "type": "integer",
                "description": "PREFERRED: Element index from browser_view_page output.",
            },
            "coordinate_x": {
                "type": "number",
                "description": "X coordinate for coordinate-based typing.",
            },
            "coordinate_y": {
                "type": "number",
                "description": "Y coordinate for coordinate-based typing.",
            },
            "text": {
                "type": "string",
                "description": "The text to type.",
            },
            "press_enter": {
                "type": "boolean",
                "default": False,
                "description": "If true, press Enter after typing.",
            },
            "clear_first": {
                "type": "boolean",
                "default": True,
                "description": "If true, clear the input field before typing.",
            },
        },
        "required": ["text"],
    },
}

SCROLL_SCHEMA_PATCH = {
    "name": "browser_scroll",
    "description": "Scroll the page. Supports: 'up' (one viewport up), 'down' (one viewport down), 'to_top' (scroll to page top), 'to_bottom' (scroll to page bottom).",
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["up", "down", "to_top", "to_bottom"],
                "description": "Scroll direction.",
            },
        },
        "required": ["direction"],
    },
}
