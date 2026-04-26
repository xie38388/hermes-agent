#!/usr/bin/env python3
"""
Patch run_agent.py to add a continuation nudge at the text_response exit point.
When the model returns a text response (no tool calls) early in the conversation
(< 8 api calls) and tools are available, inject a nudge message instead of breaking.
"""

FILE = "/home/ubuntu/hermes-agent/run_agent.py"

with open(FILE, "r") as f:
    content = f.read()

# 1. Add _depth_nudge_count initialization near the top of the while loop
# Find the line right after the while loop starts
INIT_ANCHOR = "            # Reset per-turn checkpoint dedup so each iteration can take one snapshot"
if INIT_ANCHOR not in content:
    print("ERROR: Could not find the while loop init anchor")
    exit(1)

INIT_INJECTION = """            # Reset per-turn checkpoint dedup so each iteration can take one snapshot
            if api_call_count == 0:
                _depth_nudge_count = 0"""

content = content.replace(INIT_ANCHOR, INIT_INJECTION, 1)
print("Injected _depth_nudge_count initialization")

# 2. Add the continuation nudge before the text_response break
# The exact anchor is the block that sets _turn_exit_reason and breaks
OLD_EXIT_BLOCK = """                    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break"""

NEW_EXIT_BLOCK = """                    # --- Continuation nudge: prevent premature stopping ---
                    # If the agent stops very early (< 8 calls) and has tools available,
                    # nudge it to reconsider before accepting the text response as final.
                    if (
                        api_call_count < 8
                        and self.valid_tool_names
                        and _depth_nudge_count < 2
                        and final_response
                        and final_response != "(empty)"
                    ):
                        _depth_nudge_count += 1
                        nudge_msg = {
                            "role": "user",
                            "content": (
                                f"[System: You responded with text after only {api_call_count} tool calls. "
                                "This seems premature. Please verify: is the task truly complete? "
                                "Have you gathered all necessary information, executed all required actions, "
                                "and verified the results? If there is more work to do, continue using tools. "
                                "If you are genuinely finished, re-output your final answer.]"
                            ),
                        }
                        messages.append(final_msg)
                        messages.append(nudge_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        if not self.quiet_mode:
                            self._safe_print(f"🔄 Depth nudge #{_depth_nudge_count}: agent stopped after {api_call_count} calls, nudging to continue")
                        continue
                    # --- End continuation nudge ---

                    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break"""

if OLD_EXIT_BLOCK not in content:
    print("ERROR: Could not find the text_response exit block to patch")
    exit(1)

content = content.replace(OLD_EXIT_BLOCK, NEW_EXIT_BLOCK, 1)
print("Injected continuation nudge before text_response break")

with open(FILE, "w") as f:
    f.write(content)

print("SUCCESS: run_agent.py patched with continuation nudge")
