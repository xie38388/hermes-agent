"""
Presenton Tool -- Generate PPTX presentations via Presenton REST API.

Bypasses MCP (Presenton's MCP endpoint has incompatible session behavior)
and calls the REST API directly:
  POST https://api.presenton.ai/api/v1/ppt/presentation/generate
"""

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

PRESENTON_API_URL = "https://api.presenton.ai/api/v1/ppt/presentation/generate"
# Timeout: PPT generation takes 1-3 minutes
PRESENTON_TIMEOUT = 300


def _get_api_key() -> str | None:
    """Resolve Presenton API key from env or hermes config."""
    key = os.getenv("PRESENTON_API_KEY")
    if key:
        return key
    # Fallback: read from ~/.hermes/config.yaml mcp_servers.presenton.headers
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        headers = (cfg.get("mcp_servers") or {}).get("presenton", {}).get("headers", {})
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
    except Exception:
        pass
    return None


def check_presenton_requirements() -> bool:
    """Return True if Presenton API key is available."""
    return bool(_get_api_key())


def generate_presentation(
    topic: str = "",
    content: str = "",
    n_slides: int = 5,
    language: str = "english",
) -> str:
    """Call Presenton REST API to generate a PPTX presentation.

    Returns JSON string with presentation_id, download URL, and edit URL.
    """
    api_key = _get_api_key()
    if not api_key:
        return json.dumps({"error": "Presenton API key not configured. Set PRESENTON_API_KEY or add presenton to mcp_servers in config.yaml."})

    # Build request body — content is required by the API
    body: dict = {
        "n_slides": n_slides,
        "language": language,
    }
    if topic:
        body["topic"] = topic
    # content is required; if not provided, use topic as content
    if content:
        body["content"] = content
    elif topic:
        body["content"] = topic
    else:
        return json.dumps({"error": "Either topic or content is required."})

    logger.info("Presenton: generating %d-slide presentation (lang=%s)", n_slides, language)

    try:
        resp = requests.post(
            PRESENTON_API_URL,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=PRESENTON_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Expected keys: presentation_id, path (PPTX URL), edit_path, credits_consumed
        result = {
            "presentation_id": data.get("presentation_id", ""),
            "download_url": data.get("path", ""),
            "edit_url": data.get("edit_path", ""),
            "credits_consumed": data.get("credits_consumed", 0),
        }
        logger.info("Presenton: presentation generated — id=%s", result["presentation_id"])
        return json.dumps(result)
    except requests.exceptions.Timeout:
        return json.dumps({"error": "Presenton API timed out (>5 min). Try again or reduce n_slides."})
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return json.dumps({"error": f"Presenton API error: {detail}"})
    except Exception as e:
        return json.dumps({"error": f"Presenton request failed: {e}"})


# ---------------------------------------------------------------------------
# Schema & Registration
# ---------------------------------------------------------------------------
from tools.registry import registry

GENERATE_PRESENTATION_SCHEMA = {
    "name": "generate_presentation",
    "description": (
        "Generate a professional PPTX presentation using Presenton AI. "
        "Provide a topic and/or detailed content. Returns a download URL for the PPTX file "
        "and an online editor URL. Generation takes 1-3 minutes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence preamble describing the purpose of this operation"
            },
            "topic": {
                "type": "string",
                "description": "The presentation topic / title.",
            },
            "content": {
                "type": "string",
                "description": (
                    "Detailed content or outline for the presentation. "
                    "The more detail you provide, the better the slides. "
                    "If omitted, the topic is used as content."
                ),
            },
            "n_slides": {
                "type": "integer",
                "description": "Number of slides to generate (default: 5).",
                "default": 5,
            },
            "language": {
                "type": "string",
                "description": "Language for the presentation (default: english).",
                "default": "english",
            },
        },
        "required": ["brief", "topic"],
    },
}

registry.register(
    name="generate_presentation",
    toolset="presenton",
    schema=GENERATE_PRESENTATION_SCHEMA,
    handler=lambda args, **kw: generate_presentation(
        topic=args.get("topic", ""),
        content=args.get("content", ""),
        n_slides=args.get("n_slides", 5),
        language=args.get("language", "english"),
    ),
    check_fn=check_presenton_requirements,
    requires_env=[],
    is_async=False,
    emoji="📊",
)
