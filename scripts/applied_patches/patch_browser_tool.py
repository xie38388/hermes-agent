"""
Patch script: Apply browser_enhanced.py integration to browser_tool.py on EC2.

This script:
1. Adds new tool function routing (browser_view_page, browser_select_option, browser_move_mouse, browser_restart)
2. Patches existing browser_click, browser_type, browser_scroll to support enhanced modes
3. Patches BROWSER_TOOL_SCHEMAS to include new tools and updated schemas
4. Creates a backup of the original file

Run on EC2: python3 patch_browser_tool.py
"""
import os
import shutil
import re

BROWSER_TOOL_PATH = "/home/ubuntu/hermes-agent/tools/browser_tool.py"
BACKUP_PATH = BROWSER_TOOL_PATH + ".bak"

def main():
    # 1. Backup
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(BROWSER_TOOL_PATH, BACKUP_PATH)
        print(f"Backup created: {BACKUP_PATH}")
    else:
        print(f"Backup already exists: {BACKUP_PATH}")

    with open(BROWSER_TOOL_PATH, "r") as f:
        content = f.read()

    # 2. Add import at the top of the tool implementations section
    # Find the line "# Tool implementations" or similar marker
    import_block = '''
# --- browser_enhanced integration ---
def _is_enhanced_mode():
    """Check if enhanced browser mode is available (camofox + browser_enhanced)."""
    try:
        from tools.browser_enhanced import camofox_view_page
        from tools.browser_camofox import is_camofox_mode
        return is_camofox_mode()
    except ImportError:
        return False
'''

    # Insert import block before the first tool function
    if "_is_enhanced_mode" not in content:
        # Find the browser_navigate function and insert before it
        marker = "def browser_navigate("
        idx = content.find(marker)
        if idx == -1:
            print("ERROR: Could not find browser_navigate function")
            return
        content = content[:idx] + import_block + "\n" + content[idx:]
        print("Added _is_enhanced_mode helper")

    # 3. Patch browser_click to support index/coordinate
    old_click = '''def browser_click(ref: str, task_id: Optional[str] = None) -> str:'''
    new_click = '''def browser_click(ref: str = None, task_id: Optional[str] = None, index: int = None, coordinate_x: float = None, coordinate_y: float = None) -> str:'''
    
    if old_click in content and "index: int = None" not in content.split("def browser_click")[1].split("def ")[0]:
        content = content.replace(old_click, new_click)
        
        # Add enhanced routing at the start of browser_click body
        # Find the docstring end and insert after it
        click_func_start = content.find(new_click)
        # Find the first if statement after the docstring
        after_click = content[click_func_start:]
        # Find 'if _is_camofox_mode():' in the click function
        camofox_check = after_click.find("if _is_camofox_mode():")
        if camofox_check != -1:
            insert_pos = click_func_start + camofox_check
            enhanced_routing = '''if _is_enhanced_mode() and (index is not None or coordinate_x is not None):
        from tools.browser_enhanced import camofox_click_enhanced
        return camofox_click_enhanced(ref=ref, index=index, coordinate_x=coordinate_x, coordinate_y=coordinate_y, task_id=task_id)
    '''
            content = content[:insert_pos] + enhanced_routing + content[insert_pos:]
            print("Patched browser_click with enhanced routing")

    # 4. Patch browser_type to support index/coordinate
    old_type = '''def browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:'''
    new_type = '''def browser_type(ref: str = None, text: str = "", task_id: Optional[str] = None, index: int = None, coordinate_x: float = None, coordinate_y: float = None, press_enter: bool = False, clear_first: bool = True) -> str:'''
    
    if old_type in content:
        content = content.replace(old_type, new_type)
        
        type_func_start = content.find(new_type)
        after_type = content[type_func_start:]
        camofox_check = after_type.find("if _is_camofox_mode():")
        if camofox_check != -1:
            insert_pos = type_func_start + camofox_check
            enhanced_routing = '''if _is_enhanced_mode() and (index is not None or coordinate_x is not None):
        from tools.browser_enhanced import camofox_type_enhanced
        return camofox_type_enhanced(text=text, ref=ref, index=index, coordinate_x=coordinate_x, coordinate_y=coordinate_y, press_enter=press_enter, clear_first=clear_first, task_id=task_id)
    '''
            content = content[:insert_pos] + enhanced_routing + content[insert_pos:]
            print("Patched browser_type with enhanced routing")

    # 5. Patch browser_scroll to support to_top/to_bottom
    old_scroll_validate = '''if direction not in ["up", "down"]:'''
    new_scroll_validate = '''if direction not in ["up", "down", "to_top", "to_bottom"]:'''
    
    if old_scroll_validate in content:
        content = content.replace(old_scroll_validate, new_scroll_validate)
        print("Patched browser_scroll validation")

    # Add enhanced scroll routing before camofox_scroll call
    old_scroll_camofox = '''if _is_camofox_mode():
        from tools.browser_camofox import camofox_scroll
        # Camofox REST API doesn't support pixel args; use repeated calls'''
    new_scroll_camofox = '''if _is_enhanced_mode() and direction in ("to_top", "to_bottom"):
        from tools.browser_enhanced import camofox_scroll_enhanced
        return camofox_scroll_enhanced(direction=direction, task_id=task_id)
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_scroll
        # Camofox REST API doesn't support pixel args; use repeated calls'''
    
    if old_scroll_camofox in content and "camofox_scroll_enhanced" not in content:
        content = content.replace(old_scroll_camofox, new_scroll_camofox)
        print("Patched browser_scroll with enhanced routing for to_top/to_bottom")

    # 6. Add new tool functions at the end (before if __name__)
    new_functions = '''

# ---------------------------------------------------------------------------
# Enhanced browser tools (ai-manus style)
# ---------------------------------------------------------------------------

def browser_view_page(task_id: Optional[str] = None) -> str:
    """View current page with interactive elements and visible content."""
    if _is_enhanced_mode():
        from tools.browser_enhanced import camofox_view_page
        return camofox_view_page(task_id)
    # Fallback to snapshot
    return browser_snapshot(task_id=task_id)


def browser_select_option(index: int, option: int, task_id: Optional[str] = None) -> str:
    """Select a dropdown option by element index and option index."""
    if _is_enhanced_mode():
        from tools.browser_enhanced import camofox_select_option
        return camofox_select_option(index=index, option=option, task_id=task_id)
    return json.dumps({"success": False, "error": "browser_select_option requires enhanced mode (Camofox)"})


def browser_move_mouse(coordinate_x: float, coordinate_y: float, task_id: Optional[str] = None) -> str:
    """Move mouse to coordinate."""
    if _is_enhanced_mode():
        from tools.browser_enhanced import camofox_move_mouse
        return camofox_move_mouse(coordinate_x=coordinate_x, coordinate_y=coordinate_y, task_id=task_id)
    return json.dumps({"success": False, "error": "browser_move_mouse requires enhanced mode (Camofox)"})


def browser_restart(task_id: Optional[str] = None) -> str:
    """Restart browser session."""
    if _is_enhanced_mode():
        from tools.browser_enhanced import camofox_restart
        return camofox_restart(task_id)
    return json.dumps({"success": False, "error": "browser_restart requires enhanced mode (Camofox)"})
'''

    if "def browser_view_page" not in content:
        # Insert before the last line or at end
        content = content.rstrip() + new_functions + "\n"
        print("Added new tool functions: browser_view_page, browser_select_option, browser_move_mouse, browser_restart")

    # 7. Patch BROWSER_TOOL_SCHEMAS
    # We need to:
    #   a) Update browser_click schema to include index/coordinate params
    #   b) Update browser_type schema to include index/coordinate/press_enter params
    #   c) Update browser_scroll schema to include to_top/to_bottom
    #   d) Add new tool schemas

    # a) Patch browser_click schema description
    old_click_desc = '"description": "Click on an element identified by its ref ID from the snapshot'
    new_click_desc = '"description": "Click an element on the page. Supports three modes: (1) index — use the element index from browser_view_page output (PREFERRED), (2) coordinate — click at specific x,y pixel position, (3) ref — legacy accessibility tree reference like @e5'
    if old_click_desc in content:
        content = content.replace(old_click_desc, new_click_desc)
        print("Patched browser_click schema description")

    # Add index/coordinate params to click schema
    old_click_params = '''"ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., \'@e5\', \'@e12\')"
                }
            },
            "required": ["ref"]'''
    new_click_params = '''"ref": {
                    "type": "string",
                    "description": "Legacy: Accessibility tree element ref (e.g. @e5). Use index mode instead when possible."
                },
                "index": {
                    "type": "integer",
                    "description": "PREFERRED: Element index from browser_view_page output."
                },
                "coordinate_x": {
                    "type": "number",
                    "description": "X coordinate for coordinate-based click."
                },
                "coordinate_y": {
                    "type": "number",
                    "description": "Y coordinate for coordinate-based click."
                }
            },
            "required": []'''
    if old_click_params in content:
        content = content.replace(old_click_params, new_click_params)
        print("Patched browser_click schema params")

    # b) Patch browser_type schema
    old_type_desc = '"description": "Type text into an input field identified by its ref ID.'
    new_type_desc = '"description": "Type text into an input field. Supports three modes: (1) index — use the element index from browser_view_page output (PREFERRED), (2) coordinate — click at x,y then type, (3) ref — legacy accessibility tree reference.'
    if old_type_desc in content:
        content = content.replace(old_type_desc, new_type_desc)
        print("Patched browser_type schema description")

    old_type_params = '''"ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., \'@e3\')"
                },
                "text": {
                    "type": "string",
                    "description": "The text to type into the field"
                }
            },
            "required": ["ref", "text"]'''
    new_type_params = '''"ref": {
                    "type": "string",
                    "description": "Legacy: Accessibility tree element ref (e.g. @e3). Use index mode instead when possible."
                },
                "index": {
                    "type": "integer",
                    "description": "PREFERRED: Element index from browser_view_page output."
                },
                "coordinate_x": {
                    "type": "number",
                    "description": "X coordinate for coordinate-based typing."
                },
                "coordinate_y": {
                    "type": "number",
                    "description": "Y coordinate for coordinate-based typing."
                },
                "text": {
                    "type": "string",
                    "description": "The text to type."
                },
                "press_enter": {
                    "type": "boolean",
                    "default": false,
                    "description": "If true, press Enter after typing."
                },
                "clear_first": {
                    "type": "boolean",
                    "default": true,
                    "description": "If true, clear the input field before typing."
                }
            },
            "required": ["text"]'''
    if old_type_params in content:
        content = content.replace(old_type_params, new_type_params)
        print("Patched browser_type schema params")

    # c) Patch browser_scroll schema
    old_scroll_enum = '"enum": ["up", "down"],'
    new_scroll_enum = '"enum": ["up", "down", "to_top", "to_bottom"],'
    if old_scroll_enum in content:
        content = content.replace(old_scroll_enum, new_scroll_enum)
        print("Patched browser_scroll schema enum")

    # d) Add new tool schemas to BROWSER_TOOL_SCHEMAS
    # Find the closing ] of BROWSER_TOOL_SCHEMAS
    new_schemas = '''
    {
        "name": "browser_view_page",
        "description": "View the current page with structured interactive elements and visible content. Returns a numbered list of clickable/typeable elements (use the index number with browser_click or browser_type) and the visible page text. Call this after navigating to understand what's on the page and what you can interact with.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "browser_select_option",
        "description": "Select an option from a dropdown (<select>) element. First call browser_view_page to get the element index, then use this tool with the element index and the option index (0-based).",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "The index of the select element from browser_view_page output."
                },
                "option": {
                    "type": "integer",
                    "description": "The option index to select (0-based)."
                }
            },
            "required": ["index", "option"]
        }
    },
    {
        "name": "browser_move_mouse",
        "description": "Move the mouse cursor to a specific coordinate on the page. Useful for hovering over elements to trigger tooltips or dropdown menus.",
        "parameters": {
            "type": "object",
            "properties": {
                "coordinate_x": {
                    "type": "number",
                    "description": "X coordinate (pixels from left edge of viewport)."
                },
                "coordinate_y": {
                    "type": "number",
                    "description": "Y coordinate (pixels from top edge of viewport)."
                }
            },
            "required": ["coordinate_x", "coordinate_y"]
        }
    },
    {
        "name": "browser_restart",
        "description": "Restart the browser session. Closes the current tab and clears the session. Call browser_navigate afterwards to open a new page. Use this when the browser is in a bad state.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },'''

    if '"browser_view_page"' not in content:
        # Find BROWSER_TOOL_SCHEMAS = [ and insert new schemas after the opening [
        schemas_start = content.find("BROWSER_TOOL_SCHEMAS = [")
        if schemas_start != -1:
            bracket_pos = content.find("[", schemas_start)
            content = content[:bracket_pos + 1] + new_schemas + content[bracket_pos + 1:]
            print("Added new tool schemas to BROWSER_TOOL_SCHEMAS")

    # Write patched file
    with open(BROWSER_TOOL_PATH, "w") as f:
        f.write(content)

    print(f"\nPatch applied successfully to {BROWSER_TOOL_PATH}")
    print(f"Backup at: {BACKUP_PATH}")


if __name__ == "__main__":
    main()
