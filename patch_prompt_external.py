#!/usr/bin/env python3
"""
Patch prompt_builder.py to add EXTERNAL_SERVICE_PROTOCOL constant.
Insert after OUTPUT_FORMAT_PROTOCOL closing paren, before DEVELOPER_ROLE_MODELS.
"""

FILE = "/home/ubuntu/hermes-agent/agent/prompt_builder.py"

with open(FILE, "r") as f:
    content = f.read()

# Find the insertion point: after OUTPUT_FORMAT_PROTOCOL, before DEVELOPER_ROLE_MODELS
ANCHOR = 'DEVELOPER_ROLE_MODELS = ("gpt-5", "codex")'

if ANCHOR not in content:
    print("ERROR: Could not find DEVELOPER_ROLE_MODELS anchor")
    exit(1)

NEW_CONSTANT = '''EXTERNAL_SERVICE_PROTOCOL = (
    "<external_service_protocol>\\n"
    "<error_resilience>\\n"
    "- When a tool call or external service returns an error, do NOT immediately give up\\n"
    "- Retry with corrected parameters at least once before reporting failure\\n"
    "- If the error is a timeout or rate limit, wait briefly and retry\\n"
    "- If the error indicates invalid input, analyze the error message, fix the input, and retry\\n"
    "- Only report an external service as unavailable after 2+ failed attempts with different approaches\\n"
    "- When reporting failure, include: what you tried, the exact error, and what alternatives exist\\n"
    "</error_resilience>\\n"
    "<parameter_validation>\\n"
    "- Before calling any tool or API, validate that all required parameters are present and correctly typed\\n"
    "- Read tool descriptions carefully — do not guess parameter names or formats\\n"
    "- When a tool returns unexpected results, re-read the tool description before retrying\\n"
    "- For file operations: verify the file exists before reading, verify the directory exists before writing\\n"
    "</parameter_validation>\\n"
    "<result_interpretation>\\n"
    "- After every tool call, explicitly analyze the result before deciding the next action\\n"
    "- Do not assume a tool call succeeded — check the return value for errors or unexpected data\\n"
    "- When processing search results, read multiple sources to cross-validate information\\n"
    "- When a command produces output, parse and interpret it before moving on\\n"
    "- Never skip result analysis — every tool call result must be acknowledged and interpreted\\n"
    "</result_interpretation>\\n"
    "<multi_step_orchestration>\\n"
    "- For complex tasks requiring multiple external calls, plan the sequence before starting\\n"
    "- Track which calls succeeded and which failed — maintain a mental checklist\\n"
    "- When one step depends on another, verify the prerequisite succeeded before proceeding\\n"
    "- If a mid-sequence step fails, decide whether to retry, skip, or abort the sequence\\n"
    "- After completing a multi-step operation, verify the end-to-end result\\n"
    "</multi_step_orchestration>\\n"
    "</external_service_protocol>"
)

'''

content = content.replace(ANCHOR, NEW_CONSTANT + ANCHOR, 1)
print("Injected EXTERNAL_SERVICE_PROTOCOL before DEVELOPER_ROLE_MODELS")

with open(FILE, "w") as f:
    f.write(content)

print("SUCCESS: prompt_builder.py patched with EXTERNAL_SERVICE_PROTOCOL")
