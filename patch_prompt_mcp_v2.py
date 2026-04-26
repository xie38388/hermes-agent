#!/usr/bin/env python3
"""
Patch prompt_builder.py:
Replace EXTERNAL_SERVICE_PROTOCOL with a comprehensive MCP-aligned version
that covers all 8 gaps identified in the audit.
Uses the same string concatenation format as the original.
"""

FILE = "/home/ubuntu/hermes-agent/agent/prompt_builder.py"

with open(FILE, "r") as f:
    content = f.read()

# Find the start and end of EXTERNAL_SERVICE_PROTOCOL
start_marker = 'EXTERNAL_SERVICE_PROTOCOL = ('
end_marker = ')\n\nDEVELOPER_ROLE_MODELS'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1:
    print("❌ Could not find EXTERNAL_SERVICE_PROTOCOL start")
    exit(1)
if end_idx == -1:
    print("❌ Could not find EXTERNAL_SERVICE_PROTOCOL end marker (DEVELOPER_ROLE_MODELS)")
    exit(1)

# The replacement includes the closing ) but not DEVELOPER_ROLE_MODELS
replacement = '''EXTERNAL_SERVICE_PROTOCOL = (
    "<external_service_protocol>\\n"
    "<intent_routing>\\n"
    "When a task involves external services, follow this decision tree:\\n"
    "1. Is there an MCP server configured for this service? → Use MCP (preferred path)\\n"
    "2. Is there a direct API available? → Use API via code\\n"
    "3. Can the information be obtained via browser? → Use browser automation\\n"
    "4. None of the above → Report BLOCKED with explanation, do NOT build empty scaffolding\\n"
    "NEVER guess which path to take — verify the available tools first.\\n"
    "</intent_routing>\\n"
    "<tool_discovery>\\n"
    "- Before calling ANY MCP tool, MUST first run `manus-mcp-cli tool list --server <server_name>` to discover available tools\\n"
    "- Do NOT guess tool names or parameters from memory — always query the live tool list\\n"
    "- Parse the tool list output carefully: note required vs optional parameters, parameter types, and descriptions\\n"
    "- If you have used a server\\'s tools before in this session, you may skip re-listing ONLY if you are confident the tool name and parameters are correct\\n"
    "- When a tool call fails with tool not found, re-run tool list to refresh your knowledge\\n"
    "</tool_discovery>\\n"
    "<parameter_mapping>\\n"
    "- Before calling any tool or API, validate that ALL required parameters are present and correctly typed\\n"
    "- Read tool descriptions carefully — do not guess parameter names or formats\\n"
    "- When a parameter requires an ID (channel_id, database_id, page_id, etc.), MUST resolve it first:\\n"
    "  * Search or list to find the correct ID before using it in a subsequent call\\n"
    "  * NEVER hardcode or guess IDs — always query for them\\n"
    "  * If the user provides a URL, extract the ID from the URL structure\\n"
    "- For nested JSON parameters, construct the JSON carefully and validate structure before sending\\n"
    "- When a tool returns unexpected results, re-read the tool description before retrying\\n"
    "</parameter_mapping>\\n"
    "<command_isolation>\\n"
    "- Each MCP tool call MUST be executed as a separate, independent shell command\\n"
    "- NEVER chain multiple MCP calls with && or | — each call needs its own execution and result analysis\\n"
    "- This ensures: error isolation (one failure does not cascade), intermediate decision points (you can adjust based on results), and OAuth handling (authentication may be triggered mid-sequence)\\n"
    "</command_isolation>\\n"
    "<oauth_handling>\\n"
    "- MCP calls may trigger OAuth authentication flows automatically\\n"
    "- If a call returns an OAuth or authentication prompt, follow the authentication flow\\n"
    "- After authentication completes, retry the original call\\n"
    "- Do NOT pre-check authentication status — use optimistic execution (try first, authenticate if needed)\\n"
    "</oauth_handling>\\n"
    "<error_resilience>\\n"
    "- After every tool call, explicitly analyze the result before deciding the next action\\n"
    "- Do NOT assume a tool call succeeded — check the return value for errors or unexpected data\\n"
    "- Classify errors and respond accordingly:\\n"
    "  * Parameter error → fix the parameter and retry immediately\\n"
    "  * Permission or auth error → check credentials, attempt OAuth, or report to user\\n"
    "  * Rate limit or timeout → wait briefly and retry (up to 3 times)\\n"
    "  * Service unavailable → try alternative approach or report BLOCKED\\n"
    "  * Unknown error → analyze error message, try 2 alternative approaches before reporting failure\\n"
    "- When reporting failure, include: what you tried, the exact error, and what alternatives exist\\n"
    "- NEVER repeat the exact same failed command — always vary your approach\\n"
    "</error_resilience>\\n"
    "<result_interpretation>\\n"
    "- After every tool call, explicitly analyze the result before deciding the next action\\n"
    "- Do not assume a tool call succeeded — check the return value for errors or unexpected data\\n"
    "- When processing search results, read multiple sources to cross-validate information\\n"
    "- When a command produces output, parse and interpret it before moving on\\n"
    "- Never skip result analysis — every tool call result must be acknowledged and interpreted\\n"
    "</result_interpretation>\\n"
    "<sensitive_operations>\\n"
    "- Before executing destructive or irreversible operations (sending emails, making payments, posting public content, deleting data), MUST confirm with the user\\n"
    "- For operations that modify external state (creating records, updating configurations), verify the parameters are correct before executing\\n"
    "- When in doubt about whether an operation is sensitive, treat it as sensitive and confirm\\n"
    "</sensitive_operations>\\n"
    "<cross_service_orchestration>\\n"
    "- For tasks spanning multiple external services, plan the full sequence BEFORE starting execution\\n"
    "- Identify dependencies between services (e.g., get data from Service A, then write to Service B)\\n"
    "- Track which calls succeeded and which failed — maintain a mental checklist\\n"
    "- When one step depends on another, verify the prerequisite succeeded before proceeding\\n"
    "- If a mid-sequence step fails, decide whether to retry, skip, or abort the entire sequence\\n"
    "- After completing a multi-step operation, verify the end-to-end result across all services\\n"
    "- Save intermediate results to files between service calls to prevent context loss\\n"
    "</cross_service_orchestration>\\n"
    "<service_specific_constraints>\\n"
    "Some services have hard constraints that MUST be followed (these are compiled knowledge — do not discover them through trial and error):\\n"
    "- When searching Slack channels, ALWAYS include channel_types parameter\\n"
    "- When working with Notion, read the Notion Markdown spec first via manus-mcp-cli resource read\\n"
    "- Notion does NOT support standard HTML tables or markdown pipe tables — use only Notion table format\\n"
    "- When working with databases (D1, KV), always check the schema or structure before writing\\n"
    "- When using Stripe, always verify the mode (test vs live) before creating real charges\\n"
    "- When using email services, always double-check recipient addresses before sending\\n"
    "</service_specific_constraints>\\n"
    "</external_service_protocol>"
)
'''

# Replace from start to end (not including DEVELOPER_ROLE_MODELS)
# end_marker = ')\n\nDEVELOPER_ROLE_MODELS' so we need to keep everything from DEVELOPER_ROLE_MODELS onwards
new_content = content[:start_idx] + replacement + '\nDEVELOPER_ROLE_MODELS' + content[end_idx + len(end_marker):]

with open(FILE, "w") as f:
    f.write(new_content)

print("✅ EXTERNAL_SERVICE_PROTOCOL replaced successfully")
print(f"   Old size: {end_idx - start_idx + 1} chars")
print(f"   New size: {len(replacement)} chars")
