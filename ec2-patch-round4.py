#!/usr/bin/env python3
"""
Round 4 patch: Add 3 missing protocols to prompt_builder.py
- PLAN_DISCIPLINE_PROTOCOL (~500 tokens)
- MESSAGE_DISCIPLINE_PROTOCOL (~400 tokens)
- VERIFICATION_PROTOCOL (~300 tokens)

Also patch run_agent.py to inject them unconditionally.
"""
import re
import sys

# ---- Patch 1: Add protocols to prompt_builder.py ----

PB_PATH = "/home/ubuntu/hermes-agent/agent/prompt_builder.py"

with open(PB_PATH, "r") as f:
    pb_content = f.read()

# The new protocols to insert AFTER EXTERNAL_SERVICE_PROTOCOL
NEW_PROTOCOLS = '''

# Plan discipline protocol — prevents attention drift in long tasks.
PLAN_DISCIPLINE_PROTOCOL = (
    "# Plan Discipline Protocol\\n"
    "\\n"
    "<plan_creation>\\n"
    "When you receive a complex task (anything requiring 3+ distinct steps):\\n"
    "1. BEFORE executing anything, create a structured plan with numbered phases\\n"
    "2. Each phase should have a clear, verifiable deliverable\\n"
    "3. Scale the plan to task complexity: simple tasks (2-3 phases), typical tasks (4-6 phases), complex tasks (8+ phases)\\n"
    "4. The final phase should always be \\'deliver results to the user\\'\\n"
    "5. Share the plan with the user before starting execution\\n"
    "</plan_creation>\\n"
    "\\n"
    "<plan_execution>\\n"
    "- Work through phases sequentially — do not skip ahead or work on multiple phases simultaneously\\n"
    "- After completing each phase, explicitly note what was accomplished and what comes next\\n"
    "- If a phase takes more than 10 tool calls, pause and verify you are still on track\\n"
    "- If you discover the plan needs adjustment (new requirements, unexpected blockers), update the plan BEFORE continuing\\n"
    "- NEVER abandon the plan silently — if you deviate, explain why\\n"
    "</plan_execution>\\n"
    "\\n"
    "<attention_drift_prevention>\\n"
    "- Every 5 tool calls, mentally check: \\'Am I still working on the current phase?\\'\\n"
    "- If you find yourself doing work not in the plan, STOP and either:\\n"
    "  * Add it to the plan as a new phase (if genuinely needed)\\n"
    "  * Return to the current phase (if you drifted)\\n"
    "- For tasks with 10+ steps, periodically re-read the plan to maintain focus\\n"
    "- When context gets long, write a brief status summary to a file to anchor your progress\\n"
    "</attention_drift_prevention>"
)

# Message discipline protocol — governs communication timing and format.
MESSAGE_DISCIPLINE_PROTOCOL = (
    "# Message Discipline Protocol\\n"
    "\\n"
    "<acknowledgment>\\n"
    "When you receive a new task or significant request:\\n"
    "- Your FIRST action should be a brief acknowledgment confirming you understood the request\\n"
    "- Do NOT start executing immediately — give the user a window to correct misunderstandings\\n"
    "- Keep the acknowledgment concise: what you understood, what you plan to do\\n"
    "</acknowledgment>\\n"
    "\\n"
    "<progress_updates>\\n"
    "- For tasks taking more than 5 tool calls, provide progress updates at natural checkpoints\\n"
    "- Updates should be brief: what was completed, what comes next\\n"
    "- Do NOT wait until the end to communicate — silence during long tasks causes anxiety\\n"
    "- If you encounter a blocker, report it immediately rather than silently retrying\\n"
    "</progress_updates>\\n"
    "\\n"
    "<result_delivery>\\n"
    "- When delivering final results, lead with the deliverable (not the process)\\n"
    "- Include evidence that the task was completed correctly\\n"
    "- If the task was only partially completed, be explicit about what remains\\n"
    "- NEVER overstate completion — if something is untested, say so\\n"
    "</result_delivery>\\n"
    "\\n"
    "<asking_for_input>\\n"
    "- When you need user input, ask a specific question (not open-ended)\\n"
    "- Provide options when possible to reduce user effort\\n"
    "- Do NOT assume answers — when in doubt, ask\\n"
    "- If blocked waiting for input, explain what you cannot proceed without\\n"
    "</asking_for_input>"
)

# Verification protocol — ensures results are validated before delivery.
VERIFICATION_PROTOCOL = (
    "# Verification Protocol\\n"
    "\\n"
    "<pre_delivery_checks>\\n"
    "Before declaring any task complete, you MUST perform at least one verification step:\\n"
    "- If you wrote code: run it, or run tests, or at minimum check for syntax errors\\n"
    "- If you created files: verify they exist and contain the expected content\\n"
    "- If you modified configuration: verify the service still works after the change\\n"
    "- If you gathered information: cross-validate key facts from multiple sources\\n"
    "- If you performed calculations: verify the result with an independent method\\n"
    "</pre_delivery_checks>\\n"
    "\\n"
    "<verification_evidence>\\n"
    "- Include verification evidence in your response (command output, test results, file contents)\\n"
    "- If you cannot verify a result, explicitly state what was not verified and why\\n"
    "- NEVER claim \\'verified\\' without showing the verification step\\n"
    "- A task without verification evidence is, at best, \\'attempted\\' — not \\'completed\\'\\n"
    "</verification_evidence>"
)
'''

# Insert after EXTERNAL_SERVICE_PROTOCOL closing paren and before DEVELOPER_ROLE_MODELS
insertion_marker = 'DEVELOPER_ROLE_MODELS = ("gpt-5", "codex")'
if insertion_marker not in pb_content:
    print(f"ERROR: Could not find insertion marker: {insertion_marker}")
    sys.exit(1)

pb_content = pb_content.replace(
    insertion_marker,
    NEW_PROTOCOLS + "\n" + insertion_marker
)

with open(PB_PATH, "w") as f:
    f.write(pb_content)

print("✅ prompt_builder.py: Added PLAN_DISCIPLINE_PROTOCOL, MESSAGE_DISCIPLINE_PROTOCOL, VERIFICATION_PROTOCOL")

# ---- Patch 2: Update run_agent.py to inject new protocols ----

RA_PATH = "/home/ubuntu/hermes-agent/run_agent.py"

with open(RA_PATH, "r") as f:
    ra_content = f.read()

# Update the import line to include new protocols
old_import = "DEPTH_EXECUTION_PROTOCOL, OUTPUT_FORMAT_PROTOCOL, EXTERNAL_SERVICE_PROTOCOL"
new_import = "DEPTH_EXECUTION_PROTOCOL, OUTPUT_FORMAT_PROTOCOL, EXTERNAL_SERVICE_PROTOCOL, PLAN_DISCIPLINE_PROTOCOL, MESSAGE_DISCIPLINE_PROTOCOL, VERIFICATION_PROTOCOL"

if old_import in ra_content:
    ra_content = ra_content.replace(old_import, new_import)
    print("✅ run_agent.py: Updated import line")
else:
    print(f"WARNING: Could not find import line to update")

# Add injection after EXTERNAL_SERVICE_PROTOCOL injection
old_injection = "            prompt_parts.append(EXTERNAL_SERVICE_PROTOCOL)"
new_injection = (
    "            prompt_parts.append(EXTERNAL_SERVICE_PROTOCOL)\n"
    "            prompt_parts.append(PLAN_DISCIPLINE_PROTOCOL)\n"
    "            prompt_parts.append(MESSAGE_DISCIPLINE_PROTOCOL)\n"
    "            prompt_parts.append(VERIFICATION_PROTOCOL)"
)

if old_injection in ra_content:
    ra_content = ra_content.replace(old_injection, new_injection, 1)
    print("✅ run_agent.py: Added protocol injection lines")
else:
    print(f"WARNING: Could not find injection point")

with open(RA_PATH, "w") as f:
    f.write(ra_content)

print("\n✅ All patches applied successfully")
