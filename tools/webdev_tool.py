#!/usr/bin/env python3
"""
WebDev Tool Module — Structured signals for web development projects.

These tools allow the agent to emit structured project lifecycle events
(init, checkpoint, status check, etc.) that the calling platform can
interpret to render UI cards, trigger deployments, or persist state.

The handlers are intentionally thin: they validate input and return a
JSON acknowledgment. The *real* side-effects happen in the platform layer
(e.g., the webapp's eventTranslator) which reads tool.started/tool.completed
events from the SSE stream and maps them to UI actions.
"""
import json
from tools.registry import registry, tool_result, tool_error

# =============================================================================
# Tool Schemas
# =============================================================================

WEBDEV_INIT_PROJECT_SCHEMA = {
    "name": "webdev_init_project",
    "description": (
        "Initialize a new web development project. Call this ONCE at the start "
        "of any website/webapp creation task. Emits a project_card event to the "
        "platform so the user can see and interact with their project."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence description of the project being created"
            },
            "project_name": {
                "type": "string",
                "description": "Short kebab-case name for the project (e.g., 'my-portfolio-site')"
            },
            "description": {
                "type": "string",
                "description": "Brief description of what the project does"
            },
        },
        "required": ["brief", "project_name"],
    },
}

WEBDEV_SAVE_CHECKPOINT_SCHEMA = {
    "name": "webdev_save_checkpoint",
    "description": (
        "Save a checkpoint of the current project state. Call this after completing "
        "a significant feature or before risky changes. Emits a checkpoint_card event "
        "so the user can review, rollback, or publish."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence description of what this checkpoint represents"
            },
            "description": {
                "type": "string",
                "description": "Detailed description of changes included in this checkpoint"
            },
        },
        "required": ["brief", "description"],
    },
}

WEBDEV_CHECK_STATUS_SCHEMA = {
    "name": "webdev_check_status",
    "description": (
        "Check the current status of the web development project (dev server, "
        "build errors, dependencies). Use after impactful changes or when "
        "environment health is uncertain."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence description of why you're checking status"
            },
        },
        "required": ["brief"],
    },
}

WEBDEV_RESTART_SERVER_SCHEMA = {
    "name": "webdev_restart_server",
    "description": (
        "Restart the development server. Use when the server becomes unresponsive, "
        "after changing environment variables or framework configuration, or after "
        "installing new dependencies."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence description of why the server needs restarting"
            },
        },
        "required": ["brief"],
    },
}

WEBDEV_ADD_FEATURE_SCHEMA = {
    "name": "webdev_add_feature",
    "description": (
        "Add a capability to the project (e.g., database, authentication, payments). "
        "This installs dependencies and scaffolds boilerplate for the requested feature."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence description of the feature being added"
            },
            "feature": {
                "type": "string",
                "enum": ["web-db-user", "stripe"],
                "description": "The feature to add: 'web-db-user' for backend+DB+auth, 'stripe' for payments"
            },
        },
        "required": ["brief", "feature"],
    },
}

WEBDEV_REQUEST_SECRETS_SCHEMA = {
    "name": "webdev_request_secrets",
    "description": (
        "Request environment variables or secrets needed for the project "
        "(API keys, tokens, credentials). The platform will prompt the user "
        "to provide values securely."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "Short explanation of why these secrets are required"
            },
            "message": {
                "type": "string",
                "description": "Message explaining the purpose of these secrets and how to obtain them"
            },
            "secrets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Environment variable name (e.g., OPENAI_API_KEY)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Explanation of how the secret is used"
                        },
                    },
                    "required": ["key", "description"],
                },
                "description": "List of secrets to request from the user"
            },
        },
        "required": ["brief", "message", "secrets"],
    },
}

WEBDEV_ROLLBACK_CHECKPOINT_SCHEMA = {
    "name": "webdev_rollback_checkpoint",
    "description": (
        "Rollback the project to a previous checkpoint. Use when recent changes "
        "have broken functionality and you need to restore a stable state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence description of why rollback is needed"
            },
            "version_id": {
                "type": "string",
                "description": "The version/checkpoint ID to rollback to"
            },
        },
        "required": ["brief", "version_id"],
    },
}

WEBDEV_DEBUG_SCHEMA = {
    "name": "webdev_debug",
    "description": (
        "Request debugging analysis from a specialized debugging agent. "
        "Use when stuck on a persistent bug after multiple fix attempts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence description of the debugging purpose"
            },
            "issue_description": {
                "type": "string",
                "description": "Description of the bug from the user's perspective (WHAT, not WHY)"
            },
            "things_tried": {
                "type": "string",
                "description": "Summary of debugging attempts already made and their results"
            },
        },
        "required": ["brief", "issue_description", "things_tried"],
    },
}

# =============================================================================
# Tool Handlers
# =============================================================================

def handle_webdev_init_project(args, **kwargs):
    """Handle webdev_init_project — validate and acknowledge."""
    project_name = args.get("project_name", "").strip()
    if not project_name:
        return tool_error("project_name is required")
    return tool_result(
        success=True,
        action="webdev_init_project",
        project_name=project_name,
        description=args.get("description", ""),
        message=f"Project '{project_name}' initialization signal emitted.",
    )


def handle_webdev_save_checkpoint(args, **kwargs):
    """Handle webdev_save_checkpoint — validate and acknowledge."""
    description = args.get("description", "").strip()
    if not description:
        return tool_error("description is required")
    return tool_result(
        success=True,
        action="webdev_save_checkpoint",
        description=description,
        message="Checkpoint signal emitted.",
    )


def handle_webdev_check_status(args, **kwargs):
    """Handle webdev_check_status — acknowledge status check request."""
    return tool_result(
        success=True,
        action="webdev_check_status",
        message="Status check signal emitted.",
    )


def handle_webdev_restart_server(args, **kwargs):
    """Handle webdev_restart_server — acknowledge restart request."""
    return tool_result(
        success=True,
        action="webdev_restart_server",
        message="Server restart signal emitted.",
    )


def handle_webdev_add_feature(args, **kwargs):
    """Handle webdev_add_feature — validate and acknowledge."""
    feature = args.get("feature", "").strip()
    if feature not in ("web-db-user", "stripe"):
        return tool_error(f"Invalid feature: '{feature}'. Must be 'web-db-user' or 'stripe'.")
    return tool_result(
        success=True,
        action="webdev_add_feature",
        feature=feature,
        message=f"Feature '{feature}' addition signal emitted.",
    )


def handle_webdev_request_secrets(args, **kwargs):
    """Handle webdev_request_secrets — validate and acknowledge."""
    secrets = args.get("secrets", [])
    if not secrets:
        return tool_error("At least one secret must be specified")
    return tool_result(
        success=True,
        action="webdev_request_secrets",
        secrets=secrets,
        message=f"Secrets request signal emitted for {len(secrets)} variable(s).",
    )


def handle_webdev_rollback_checkpoint(args, **kwargs):
    """Handle webdev_rollback_checkpoint — validate and acknowledge."""
    version_id = args.get("version_id", "").strip()
    if not version_id:
        return tool_error("version_id is required")
    return tool_result(
        success=True,
        action="webdev_rollback_checkpoint",
        version_id=version_id,
        message=f"Rollback signal emitted for version '{version_id}'.",
    )


def handle_webdev_debug(args, **kwargs):
    """Handle webdev_debug — validate and acknowledge."""
    issue = args.get("issue_description", "").strip()
    if not issue:
        return tool_error("issue_description is required")
    return tool_result(
        success=True,
        action="webdev_debug",
        issue_description=issue,
        things_tried=args.get("things_tried", ""),
        message="Debug analysis signal emitted.",
    )


# =============================================================================
# Registry
# =============================================================================

def _check_webdev():
    """WebDev tools are always available when the toolset is enabled."""
    return True


registry.register(
    name="webdev_init_project",
    toolset="webdev",
    schema=WEBDEV_INIT_PROJECT_SCHEMA,
    handler=handle_webdev_init_project,
    check_fn=_check_webdev,
    emoji="🏗️",
)

registry.register(
    name="webdev_save_checkpoint",
    toolset="webdev",
    schema=WEBDEV_SAVE_CHECKPOINT_SCHEMA,
    handler=handle_webdev_save_checkpoint,
    check_fn=_check_webdev,
    emoji="💾",
)

registry.register(
    name="webdev_check_status",
    toolset="webdev",
    schema=WEBDEV_CHECK_STATUS_SCHEMA,
    handler=handle_webdev_check_status,
    check_fn=_check_webdev,
    emoji="🔍",
)

registry.register(
    name="webdev_restart_server",
    toolset="webdev",
    schema=WEBDEV_RESTART_SERVER_SCHEMA,
    handler=handle_webdev_restart_server,
    check_fn=_check_webdev,
    emoji="🔄",
)

registry.register(
    name="webdev_add_feature",
    toolset="webdev",
    schema=WEBDEV_ADD_FEATURE_SCHEMA,
    handler=handle_webdev_add_feature,
    check_fn=_check_webdev,
    emoji="➕",
)

registry.register(
    name="webdev_request_secrets",
    toolset="webdev",
    schema=WEBDEV_REQUEST_SECRETS_SCHEMA,
    handler=handle_webdev_request_secrets,
    check_fn=_check_webdev,
    emoji="🔑",
)

registry.register(
    name="webdev_rollback_checkpoint",
    toolset="webdev",
    schema=WEBDEV_ROLLBACK_CHECKPOINT_SCHEMA,
    handler=handle_webdev_rollback_checkpoint,
    check_fn=_check_webdev,
    emoji="⏪",
)

registry.register(
    name="webdev_debug",
    toolset="webdev",
    schema=WEBDEV_DEBUG_SCHEMA,
    handler=handle_webdev_debug,
    check_fn=_check_webdev,
    emoji="🐛",
)
