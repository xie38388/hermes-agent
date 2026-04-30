"""
Patch script: Update browser tool registry handlers and add new tool registrations.

This script:
1. Updates browser_click handler lambda to pass index/coordinate params
2. Updates browser_type handler lambda to pass new params
3. Adds new tool registrations (browser_view_page, browser_select_option, browser_move_mouse, browser_restart)
4. Updates model_tools.py browser_tools list

Run on EC2: python3 patch_registry.py
"""
import os
import shutil

BROWSER_TOOL_PATH = "/home/ubuntu/hermes-agent/tools/browser_tool.py"
MODEL_TOOLS_PATH = "/home/ubuntu/hermes-agent/model_tools.py"

def main():
    # --- Patch browser_tool.py registry handlers ---
    with open(BROWSER_TOOL_PATH, "r") as f:
        content = f.read()

    # 1. Patch browser_click handler to pass new params
    old_click_handler = 'handler=lambda args, **kw: browser_click(ref=args.get("ref", ""), task_id=kw.get("task_id")),'
    new_click_handler = 'handler=lambda args, **kw: browser_click(ref=args.get("ref"), task_id=kw.get("task_id"), index=args.get("index"), coordinate_x=args.get("coordinate_x"), coordinate_y=args.get("coordinate_y")),'
    if old_click_handler in content:
        content = content.replace(old_click_handler, new_click_handler)
        print("Patched browser_click handler lambda")
    else:
        print("WARN: browser_click handler not found (may already be patched)")

    # 2. Patch browser_type handler to pass new params
    old_type_handler = 'handler=lambda args, **kw: browser_type(ref=args.get("ref", ""), text=args.get("text", ""), task_id=kw.get("task_id")),'
    new_type_handler = 'handler=lambda args, **kw: browser_type(ref=args.get("ref"), text=args.get("text", ""), task_id=kw.get("task_id"), index=args.get("index"), coordinate_x=args.get("coordinate_x"), coordinate_y=args.get("coordinate_y"), press_enter=args.get("press_enter", False), clear_first=args.get("clear_first", True)),'
    if old_type_handler in content:
        content = content.replace(old_type_handler, new_type_handler)
        print("Patched browser_type handler lambda")
    else:
        print("WARN: browser_type handler not found (may already be patched)")

    # 3. Patch browser_scroll handler to pass direction (already works, just make sure to_top/to_bottom are passed)
    # The existing handler already passes direction, so no change needed

    # 4. Add new tool registrations at the end
    new_registrations = '''
# --- Enhanced browser tools (ai-manus style) ---
registry.register(
    name="browser_view_page",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_view_page"],
    handler=lambda args, **kw: browser_view_page(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="👁️",
)
registry.register(
    name="browser_select_option",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_select_option"],
    handler=lambda args, **kw: browser_select_option(index=args.get("index", 0), option=args.get("option", 0), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="📋",
)
registry.register(
    name="browser_move_mouse",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_move_mouse"],
    handler=lambda args, **kw: browser_move_mouse(coordinate_x=args.get("coordinate_x", 0), coordinate_y=args.get("coordinate_y", 0), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="🖱️",
)
registry.register(
    name="browser_restart",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_restart"],
    handler=lambda args, **kw: browser_restart(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="🔄",
)
'''

    if 'name="browser_view_page"' not in content:
        content = content.rstrip() + "\n" + new_registrations
        print("Added new tool registrations")
    else:
        print("New tool registrations already present")

    with open(BROWSER_TOOL_PATH, "w") as f:
        f.write(content)
    print(f"Patched {BROWSER_TOOL_PATH}")

    # --- Patch model_tools.py browser_tools list ---
    if not os.path.exists(MODEL_TOOLS_PATH + ".bak"):
        shutil.copy2(MODEL_TOOLS_PATH, MODEL_TOOLS_PATH + ".bak")
        print(f"Backup created: {MODEL_TOOLS_PATH}.bak")

    with open(MODEL_TOOLS_PATH, "r") as f:
        mt_content = f.read()

    old_browser_tools = '''    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console"
    ],'''
    new_browser_tools = '''    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console",
        "browser_view_page", "browser_select_option",
        "browser_move_mouse", "browser_restart"
    ],'''

    if old_browser_tools in mt_content:
        mt_content = mt_content.replace(old_browser_tools, new_browser_tools)
        print("Patched model_tools.py browser_tools list")
    else:
        print("WARN: browser_tools list not found in model_tools.py (may already be patched)")

    with open(MODEL_TOOLS_PATH, "w") as f:
        f.write(mt_content)
    print(f"Patched {MODEL_TOOLS_PATH}")

    print("\nAll patches applied successfully!")
    print("Restart hermes-agent to load new tools.")


if __name__ == "__main__":
    main()
