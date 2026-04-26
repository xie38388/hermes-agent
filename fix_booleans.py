"""Fix JSON-style booleans in BROWSER_TOOL_SCHEMAS to Python-style."""
import re

BROWSER_TOOL_PATH = "/home/ubuntu/hermes-agent/tools/browser_tool.py"

with open(BROWSER_TOOL_PATH, "r") as f:
    content = f.read()

# Find the BROWSER_TOOL_SCHEMAS section and fix booleans only within it
# We need to be careful to only fix booleans in the schema definitions,
# not in Python code

# Fix "default": false -> "default": False
# Fix "default": true -> "default": True
# These only appear in the new schema additions

# Fix in the new browser_type schema params (added by patch)
content = content.replace('"default": false,', '"default": False,')
content = content.replace('"default": true,', '"default": True,')

with open(BROWSER_TOOL_PATH, "w") as f:
    f.write(content)

print("Fixed boolean values in BROWSER_TOOL_SCHEMAS")
