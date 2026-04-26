#!/usr/bin/env python3
"""
Patch trajectory_compressor.py to enhance summarization prompt with error preservation.
Both sync and async _generate_summary methods have the same prompt — patch both.
"""

FILE = "/home/ubuntu/hermes-agent/trajectory_compressor.py"

with open(FILE, "r") as f:
    content = f.read()

# The exact text has a blank line between item 4 and "Keep the summary"
OLD_PROMPT_BODY = """Write the summary from a neutral perspective describing what the assistant did and learned. Include:
1. What actions the assistant took (tool calls, searches, file operations)
2. Key information or results obtained
3. Any important decisions or findings
4. Relevant data, file names, values, or outputs

Keep the summary factual and informative."""

NEW_PROMPT_BODY = """Write the summary from a neutral perspective describing what the assistant did and learned. Include:
1. What actions the assistant took (tool calls, searches, file operations)
2. Key information or results obtained
3. Any important decisions or findings
4. Relevant data, file names, values, or outputs
5. ALL errors, failures, and exceptions encountered — include the error message, which tool/action failed, and what was tried
6. Any approaches that were attempted but did NOT work — so the agent avoids repeating them

CRITICAL: Error preservation is mandatory. If any tool call returned an error, any command failed, or any approach was abandoned, you MUST include it in the summary with enough detail to prevent the agent from retrying the same failed approach. Format errors as:
- FAILED: [action] → [error message] → [what was tried instead]

Keep the summary factual and informative."""

count = content.count(OLD_PROMPT_BODY)
if count == 0:
    print("ERROR: Could not find the prompt body to replace. The file may have been modified.")
    exit(1)

content = content.replace(OLD_PROMPT_BODY, NEW_PROMPT_BODY)
print(f"Replaced {count} occurrence(s) of the summarization prompt (sync + async)")

with open(FILE, "w") as f:
    f.write(content)

print("SUCCESS: trajectory_compressor.py patched with error preservation in summarization prompt")
