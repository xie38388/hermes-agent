#!/usr/bin/env python3
"""
Patch prompt_builder.py:
1. Replace EXTERNAL_SERVICE_PROTOCOL with a comprehensive MCP-aligned version
   that covers all 8 gaps identified in the audit.
2. Add MCP_EXECUTION_PROTOCOL as a new constant for MCP-specific rules.
"""
import re

FILE = "/home/ubuntu/hermes-agent/agent/prompt_builder.py"

with open(FILE, "r") as f:
    content = f.read()

# --- 1. Replace EXTERNAL_SERVICE_PROTOCOL ---

OLD_EXTERNAL = '''EXTERNAL_SERVICE_PROTOCOL = """
# External Service Protocol
<error_resilience>
- If the error is a timeout or rate limit, wait briefly and retry
- If the error indicates invalid input, analyze the error message, fix the input, and retry
- Only report an external service as unavailable after 2+ failed attempts with different approaches
- When reporting failure, include: what you tried, the exact error, and what alternatives exist
</error_resilience>
<parameter_validation>
- Before calling any tool or API, validate that all required parameters are present and correctly typed
- Read tool descriptions carefully — do not guess parameter names or formats
- When a tool returns unexpected results, re-read the tool description before retrying
- For file operations: verify the file exists before reading, verify the directory exists before writing
</parameter_validation>
<result_interpretation>
- After every tool call, explicitly analyze the result before deciding the next action
- Do not assume a tool call succeeded — check the return value for errors or unexpected data
- When processing search results, read multiple sources to cross-validate information
- When a command produces output, parse and interpret it before moving on
- Never skip result analysis — every tool call result must be acknowledged and interpreted
</result_interpretation>
<multi_step_orchestration>
- For complex tasks requiring multiple external calls, plan the sequence before starting
- Track which calls succeeded and which failed — maintain a mental checklist
- When one step depends on another, verify the prerequisite succeeded before proceeding
- If a mid-sequence step fails, decide whether to retry, skip, or abort the sequence
- After completing a multi-step operation, verify the end-to-end result
</multi_step_orchestration>
</external_service_protocol>
"""'''

NEW_EXTERNAL = '''EXTERNAL_SERVICE_PROTOCOL = """
# External Service Protocol
<intent_routing>
When a task involves external services, follow this decision tree:
1. Is there an MCP server configured for this service? → Use MCP (preferred path)
2. Is there a direct API available? → Use API via code
3. Can the information be obtained via browser? → Use browser automation
4. None of the above → Report BLOCKED with explanation, do NOT build empty scaffolding
NEVER guess which path to take — verify the available tools first.
</intent_routing>
<tool_discovery>
- Before calling ANY MCP tool, MUST first run `manus-mcp-cli tool list --server <server_name>` to discover available tools
- Do NOT guess tool names or parameters from memory — always query the live tool list
- Parse the tool list output carefully: note required vs optional parameters, parameter types, and descriptions
- If you have used a server's tools before in this session, you may skip re-listing ONLY if you are confident the tool name and parameters are correct
- When a tool call fails with "tool not found", re-run tool list to refresh your knowledge
</tool_discovery>
<parameter_mapping>
- Before calling any tool or API, validate that ALL required parameters are present and correctly typed
- Read tool descriptions carefully — do not guess parameter names or formats
- When a parameter requires an ID (channel_id, database_id, page_id, etc.), MUST resolve it first:
  * Search/list to find the correct ID before using it in a subsequent call
  * NEVER hardcode or guess IDs — always query for them
  * If the user provides a URL, extract the ID from the URL structure
- For nested JSON parameters, construct the JSON carefully and validate structure before sending
- When a tool returns unexpected results, re-read the tool description before retrying
</parameter_mapping>
<command_isolation>
- Each MCP tool call MUST be executed as a separate, independent shell command
- NEVER chain multiple MCP calls with && or | — each call needs its own execution and result analysis
- This ensures: error isolation (one failure doesn't cascade), intermediate decision points (you can adjust based on results), and OAuth handling (authentication may be triggered mid-sequence)
</command_isolation>
<oauth_handling>
- MCP calls may trigger OAuth authentication flows automatically
- If a call returns an OAuth/authentication prompt, follow the authentication flow
- After authentication completes, retry the original call
- Do NOT pre-check authentication status — use optimistic execution (try first, authenticate if needed)
</oauth_handling>
<error_resilience>
- After every tool call, explicitly analyze the result before deciding the next action
- Do NOT assume a tool call succeeded — check the return value for errors or unexpected data
- Classify errors and respond accordingly:
  * Parameter error → fix the parameter and retry immediately
  * Permission/auth error → check credentials, attempt OAuth, or report to user
  * Rate limit/timeout → wait briefly and retry (up to 3 times)
  * Service unavailable → try alternative approach or report BLOCKED
  * Unknown error → analyze error message, try 2 alternative approaches before reporting failure
- When reporting failure, include: what you tried, the exact error, and what alternatives exist
- NEVER repeat the exact same failed command — always vary your approach
</error_resilience>
<result_interpretation>
- After every tool call, explicitly analyze the result before deciding the next action
- Do not assume a tool call succeeded — check the return value for errors or unexpected data
- When processing search results, read multiple sources to cross-validate information
- When a command produces output, parse and interpret it before moving on
- Never skip result analysis — every tool call result must be acknowledged and interpreted
</result_interpretation>
<sensitive_operations>
- Before executing destructive or irreversible operations (sending emails, making payments, posting public content, deleting data), MUST confirm with the user
- For operations that modify external state (creating records, updating configurations), verify the parameters are correct before executing
- When in doubt about whether an operation is sensitive, treat it as sensitive and confirm
</sensitive_operations>
<cross_service_orchestration>
- For tasks spanning multiple external services, plan the full sequence BEFORE starting execution
- Identify dependencies between services (e.g., get data from Service A, then write to Service B)
- Track which calls succeeded and which failed — maintain a mental checklist
- When one step depends on another, verify the prerequisite succeeded before proceeding
- If a mid-sequence step fails, decide whether to retry, skip, or abort the entire sequence
- After completing a multi-step operation, verify the end-to-end result across all services
- Save intermediate results to files between service calls to prevent context loss
</cross_service_orchestration>
<service_specific_constraints>
Some services have hard constraints that MUST be followed (these are "compiled knowledge" — do not discover them through trial and error):
- When searching Slack channels, ALWAYS include channel_types parameter
- When working with Notion, read the Notion Markdown spec first via `manus-mcp-cli resource read notion://docs/enhanced-markdown-spec --server notion`
- Notion does NOT support standard HTML tables or markdown pipe tables — use only Notion's table format
- When working with databases (D1, KV), always check the schema/structure before writing
- When using Stripe, always verify the mode (test vs live) before creating real charges
- When using email services, always double-check recipient addresses before sending
</service_specific_constraints>
</external_service_protocol>
"""'''

if OLD_EXTERNAL in content:
    content = content.replace(OLD_EXTERNAL, NEW_EXTERNAL)
    print("✅ EXTERNAL_SERVICE_PROTOCOL replaced successfully")
else:
    # Try a more flexible match
    pattern = r'EXTERNAL_SERVICE_PROTOCOL = """.*?"""'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + NEW_EXTERNAL.split(' = ', 1)[0] + ' = ' + '"""' + NEW_EXTERNAL.split('"""', 1)[1].split('"""')[0] + '"""' + content[match.end():]
        print("✅ EXTERNAL_SERVICE_PROTOCOL replaced (flexible match)")
    else:
        print("❌ Could not find EXTERNAL_SERVICE_PROTOCOL to replace")
        exit(1)

with open(FILE, "w") as f:
    f.write(content)

print("Done. File written.")
