#!/usr/bin/env python3
"""
Image Generation Tools Module

This module provides image generation tools using FAL.ai's FLUX 2 Pro model with 
automatic upscaling via FAL.ai's Clarity Upscaler for enhanced image quality.

Available tools:
- image_generate_tool: Generate images from text prompts with automatic upscaling

Features:
- High-quality image generation using FLUX 2 Pro model
- Automatic 2x upscaling using Clarity Upscaler for enhanced quality
- Comprehensive parameter control (size, steps, guidance, etc.)
- Proper error handling and validation with fallback to original images
- Debug logging support
- Sync mode for immediate results

Usage:
    from image_generation_tool import image_generate_tool
    import asyncio
    
    # Generate and automatically upscale an image
    result = await image_generate_tool(
        prompt="A serene mountain landscape with cherry blossoms",
        image_size="landscape_4_3",
        num_images=1
    )
"""

import json
import base64
import logging
import os
import datetime
import threading
import uuid
from typing import Dict, Any, Optional, Union
from urllib.parse import urlencode
import fal_client
import requests
from tools.debug_helpers import DebugSession
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import managed_nous_tools_enabled

logger = logging.getLogger(__name__)

# Configuration for image generation
DEFAULT_MODEL = "fal-ai/flux-2-pro"
DEFAULT_ASPECT_RATIO = "landscape"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_NUM_IMAGES = 1
DEFAULT_OUTPUT_FORMAT = "png"

# Safety settings
ENABLE_SAFETY_CHECKER = False
SAFETY_TOLERANCE = "5"  # Maximum tolerance (1-5, where 5 is most permissive)

# Aspect ratio mapping - simplified choices for model to select
ASPECT_RATIO_MAP = {
    "landscape": "landscape_16_9",
    "square": "square_hd",
    "portrait": "portrait_16_9"
}
VALID_ASPECT_RATIOS = list(ASPECT_RATIO_MAP.keys())

# Configuration for automatic upscaling
UPSCALER_MODEL = "fal-ai/clarity-upscaler"
UPSCALER_FACTOR = 2
UPSCALER_SAFETY_CHECKER = False
UPSCALER_DEFAULT_PROMPT = "masterpiece, best quality, highres"
UPSCALER_NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2)"
UPSCALER_CREATIVITY = 0.35
UPSCALER_RESEMBLANCE = 0.6
UPSCALER_GUIDANCE_SCALE = 4
UPSCALER_NUM_INFERENCE_STEPS = 18

# Valid parameter values for validation based on FLUX 2 Pro documentation
VALID_IMAGE_SIZES = [
    "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
]
VALID_OUTPUT_FORMATS = ["jpeg", "png"]
VALID_ACCELERATION_MODES = ["none", "regular", "high"]

_debug = DebugSession("image_tools", env_var="IMAGE_TOOLS_DEBUG")
_managed_fal_client = None
_managed_fal_client_config = None
_managed_fal_client_lock = threading.Lock()


def _resolve_managed_fal_gateway():
    """Return managed fal-queue gateway config when direct FAL credentials are absent."""
    if os.getenv("FAL_KEY"):
        return None
    return resolve_managed_tool_gateway("fal-queue")


def _normalize_fal_queue_url_format(queue_run_origin: str) -> str:
    normalized_origin = str(queue_run_origin or "").strip().rstrip("/")
    if not normalized_origin:
        raise ValueError("Managed FAL queue origin is required")
    return f"{normalized_origin}/"


class _ManagedFalSyncClient:
    """Small per-instance wrapper around fal_client.SyncClient for managed queue hosts."""

    def __init__(self, *, key: str, queue_run_origin: str):
        sync_client_class = getattr(fal_client, "SyncClient", None)
        if sync_client_class is None:
            raise RuntimeError("fal_client.SyncClient is required for managed FAL gateway mode")

        client_module = getattr(fal_client, "client", None)
        if client_module is None:
            raise RuntimeError("fal_client.client is required for managed FAL gateway mode")

        self._queue_url_format = _normalize_fal_queue_url_format(queue_run_origin)
        self._sync_client = sync_client_class(key=key)
        self._http_client = getattr(self._sync_client, "_client", None)
        self._maybe_retry_request = getattr(client_module, "_maybe_retry_request", None)
        self._raise_for_status = getattr(client_module, "_raise_for_status", None)
        self._request_handle_class = getattr(client_module, "SyncRequestHandle", None)
        self._add_hint_header = getattr(client_module, "add_hint_header", None)
        self._add_priority_header = getattr(client_module, "add_priority_header", None)
        self._add_timeout_header = getattr(client_module, "add_timeout_header", None)

        if self._http_client is None:
            raise RuntimeError("fal_client.SyncClient._client is required for managed FAL gateway mode")
        if self._maybe_retry_request is None or self._raise_for_status is None:
            raise RuntimeError("fal_client.client request helpers are required for managed FAL gateway mode")
        if self._request_handle_class is None:
            raise RuntimeError("fal_client.client.SyncRequestHandle is required for managed FAL gateway mode")

    def submit(
        self,
        application: str,
        arguments: Dict[str, Any],
        *,
        path: str = "",
        hint: Optional[str] = None,
        webhook_url: Optional[str] = None,
        priority: Any = None,
        headers: Optional[Dict[str, str]] = None,
        start_timeout: Optional[Union[int, float]] = None,
    ):
        url = self._queue_url_format + application
        if path:
            url += "/" + path.lstrip("/")
        if webhook_url is not None:
            url += "?" + urlencode({"fal_webhook": webhook_url})

        request_headers = dict(headers or {})
        if hint is not None and self._add_hint_header is not None:
            self._add_hint_header(hint, request_headers)
        if priority is not None:
            if self._add_priority_header is None:
                raise RuntimeError("fal_client.client.add_priority_header is required for priority requests")
            self._add_priority_header(priority, request_headers)
        if start_timeout is not None:
            if self._add_timeout_header is None:
                raise RuntimeError("fal_client.client.add_timeout_header is required for timeout requests")
            self._add_timeout_header(start_timeout, request_headers)

        response = self._maybe_retry_request(
            self._http_client,
            "POST",
            url,
            json=arguments,
            timeout=getattr(self._sync_client, "default_timeout", 120.0),
            headers=request_headers,
        )
        self._raise_for_status(response)

        data = response.json()
        return self._request_handle_class(
            request_id=data["request_id"],
            response_url=data["response_url"],
            status_url=data["status_url"],
            cancel_url=data["cancel_url"],
            client=self._http_client,
        )


def _get_managed_fal_client(managed_gateway):
    """Reuse the managed FAL client so its internal httpx.Client is not leaked per call."""
    global _managed_fal_client, _managed_fal_client_config

    client_config = (
        managed_gateway.gateway_origin.rstrip("/"),
        managed_gateway.nous_user_token,
    )
    with _managed_fal_client_lock:
        if _managed_fal_client is not None and _managed_fal_client_config == client_config:
            return _managed_fal_client

        _managed_fal_client = _ManagedFalSyncClient(
            key=managed_gateway.nous_user_token,
            queue_run_origin=managed_gateway.gateway_origin,
        )
        _managed_fal_client_config = client_config
        return _managed_fal_client


def _submit_fal_request(model: str, arguments: Dict[str, Any]):
    """Submit a FAL request using direct credentials or the managed queue gateway."""
    request_headers = {"x-idempotency-key": str(uuid.uuid4())}
    managed_gateway = _resolve_managed_fal_gateway()
    if managed_gateway is None:
        return fal_client.submit(model, arguments=arguments, headers=request_headers)

    managed_client = _get_managed_fal_client(managed_gateway)
    return managed_client.submit(
        model,
        arguments=arguments,
        headers=request_headers,
    )


def _validate_parameters(
    image_size: Union[str, Dict[str, int]], 
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    output_format: str,
    acceleration: str = "none"
) -> Dict[str, Any]:
    """
    Validate and normalize image generation parameters for FLUX 2 Pro model.
    
    Args:
        image_size: Either a preset string or custom size dict
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale value
        num_images: Number of images to generate
        output_format: Output format for images
        acceleration: Acceleration mode for generation speed
    
    Returns:
        Dict[str, Any]: Validated and normalized parameters
    
    Raises:
        ValueError: If any parameter is invalid
    """
    validated = {}
    
    # Validate image_size
    if isinstance(image_size, str):
        if image_size not in VALID_IMAGE_SIZES:
            raise ValueError(f"Invalid image_size '{image_size}'. Must be one of: {VALID_IMAGE_SIZES}")
        validated["image_size"] = image_size
    elif isinstance(image_size, dict):
        if "width" not in image_size or "height" not in image_size:
            raise ValueError("Custom image_size must contain 'width' and 'height' keys")
        if not isinstance(image_size["width"], int) or not isinstance(image_size["height"], int):
            raise ValueError("Custom image_size width and height must be integers")
        if image_size["width"] < 64 or image_size["height"] < 64:
            raise ValueError("Custom image_size dimensions must be at least 64x64")
        if image_size["width"] > 2048 or image_size["height"] > 2048:
            raise ValueError("Custom image_size dimensions must not exceed 2048x2048")
        validated["image_size"] = image_size
    else:
        raise ValueError("image_size must be either a preset string or a dict with width/height")
    
    # Validate num_inference_steps
    if not isinstance(num_inference_steps, int) or num_inference_steps < 1 or num_inference_steps > 100:
        raise ValueError("num_inference_steps must be an integer between 1 and 100")
    validated["num_inference_steps"] = num_inference_steps
    
    # Validate guidance_scale (FLUX 2 Pro default is 4.5)
    if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0.1 or guidance_scale > 20.0:
        raise ValueError("guidance_scale must be a number between 0.1 and 20.0")
    validated["guidance_scale"] = float(guidance_scale)
    
    # Validate num_images
    if not isinstance(num_images, int) or num_images < 1 or num_images > 4:
        raise ValueError("num_images must be an integer between 1 and 4")
    validated["num_images"] = num_images
    
    # Validate output_format
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"Invalid output_format '{output_format}'. Must be one of: {VALID_OUTPUT_FORMATS}")
    validated["output_format"] = output_format
    
    # Validate acceleration
    if acceleration not in VALID_ACCELERATION_MODES:
        raise ValueError(f"Invalid acceleration '{acceleration}'. Must be one of: {VALID_ACCELERATION_MODES}")
    validated["acceleration"] = acceleration
    
    return validated


def _upscale_image(image_url: str, original_prompt: str) -> Dict[str, Any]:
    """
    Upscale an image using FAL.ai's Clarity Upscaler.
    
    Uses the synchronous fal_client API to avoid event loop lifecycle issues
    when called from threaded contexts (e.g. gateway thread pool).
    
    Args:
        image_url (str): URL of the image to upscale
        original_prompt (str): Original prompt used to generate the image
    
    Returns:
        Dict[str, Any]: Upscaled image data or None if upscaling fails
    """
    try:
        logger.info("Upscaling image with Clarity Upscaler...")
        
        # Prepare arguments for upscaler
        upscaler_arguments = {
            "image_url": image_url,
            "prompt": f"{UPSCALER_DEFAULT_PROMPT}, {original_prompt}",
            "upscale_factor": UPSCALER_FACTOR,
            "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
            "creativity": UPSCALER_CREATIVITY,
            "resemblance": UPSCALER_RESEMBLANCE,
            "guidance_scale": UPSCALER_GUIDANCE_SCALE,
            "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
            "enable_safety_checker": UPSCALER_SAFETY_CHECKER
        }
        
        # Use sync API — fal_client.submit() uses httpx.Client (no event loop).
        # The async API (submit_async) caches a global httpx.AsyncClient via
        # @cached_property, which breaks when asyncio.run() destroys the loop
        # between calls (gateway thread-pool pattern).
        handler = _submit_fal_request(
            UPSCALER_MODEL,
            arguments=upscaler_arguments,
        )
        
        # Get the upscaled result (sync — blocks until done)
        result = handler.get()
        
        if result and "image" in result:
            upscaled_image = result["image"]
            logger.info("Image upscaled successfully to %sx%s", upscaled_image.get('width', 'unknown'), upscaled_image.get('height', 'unknown'))
            return {
                "url": upscaled_image["url"],
                "width": upscaled_image.get("width", 0),
                "height": upscaled_image.get("height", 0),
                "upscaled": True,
                "upscale_factor": UPSCALER_FACTOR
            }
        else:
            logger.error("Upscaler returned invalid response")
            return None
            
    except Exception as e:
        logger.error("Error upscaling image: %s", e, exc_info=True)
        return None



# ---------------------------------------------------------------------------
# OpenRouter GPT-5.4 Image 2 backend
# ---------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_IMAGE_MODEL = "openai/gpt-5.4-image-2"


def _upload_base64_image(data_url: str) -> Optional[str]:
    """
    Process a base64 data URL image:
    1. Decode the base64 data
    2. Save to the generated files directory
    3. Try uploading to Fal CDN; if that fails, return gateway-relative path
    4. Last resort: return the raw data URL for direct browser rendering

    The gateway's _handle_serve_file endpoint serves files from /tmp/hermes_generated/.
    """
    try:
        # Parse data URL: data:image/png;base64,iVBOR...
        header, b64data = data_url.split(",", 1)
        # Extract mime type
        mime = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        img_bytes = base64.b64decode(b64data)
        logger.info("Decoded base64 image: %d bytes, mime=%s", len(img_bytes), mime)

        # Save to the generated files directory
        gen_dir = "/tmp/hermes_generated"
        os.makedirs(gen_dir, exist_ok=True)

        ext = mime.split("/")[-1] if "/" in mime else "png"
        if ext == "jpeg":
            ext = "jpg"
        import uuid as _uuid
        filename = f"img_{_uuid.uuid4().hex[:12]}.{ext}"
        filepath = os.path.join(gen_dir, filename)

        with open(filepath, "wb") as fout:
            fout.write(img_bytes)
        logger.info("Saved generated image: %s (%d bytes)", filepath, len(img_bytes))

        # Try uploading via fal_client (best case: get a CDN URL)
        try:
            cdn_url = fal_client.upload(img_bytes, content_type=mime)
            if cdn_url and cdn_url.startswith("http"):
                logger.info("Uploaded to Fal CDN: %s", cdn_url[:80])
                return cdn_url
        except Exception as e:
            logger.warning("Fal CDN upload failed (non-fatal): %s", e)

        # Return the gateway-relative path for /v1/files/ endpoint
        return f"/v1/files/{filename}"

    except Exception as e:
        logger.warning("Failed to process base64 image: %s", e)
        # Last resort: return the raw data URL (browsers can render it directly)
        if data_url and len(data_url) < 10_000_000:  # < 10MB
            return data_url
        return None


def _generate_via_openrouter(prompt: str, aspect_ratio: str = "landscape") -> Optional[str]:
    """
    Generate an image via OpenRouter using GPT-5.4 Image 2.

    GPT-5.4 Image 2 response format:
    - message.content is None
    - message.images contains the generated images as base64 data URLs
    - Format: [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

    Returns the image URL on success, or None on failure.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return None

    # Build size hint for the prompt based on aspect ratio
    size_hints = {
        "landscape": "wide landscape format (16:9)",
        "square": "square format (1:1)",
        "portrait": "tall portrait format (9:16)",
    }
    size_hint = size_hints.get(aspect_ratio, "landscape format")
    enhanced_prompt = f"{prompt}. Generate the image in {size_hint}."

    try:
        logger.info("Generating image via OpenRouter GPT-5.4 Image 2...")
        logger.info("  Model: %s", OPENROUTER_IMAGE_MODEL)
        logger.info("  Prompt: %s", prompt[:80])

        resp = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": OPENROUTER_IMAGE_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": enhanced_prompt},
                        ],
                    }
                ],
            },
            timeout=300,
        )

        if resp.status_code != 200:
            logger.warning("OpenRouter returned %s: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            logger.warning("OpenRouter returned empty choices")
            return None

        message = choices[0].get("message", {})

        # GPT-5.4 Image 2 puts images in message.images (not message.content)
        images = message.get("images") or []

        # Also check message.content as fallback (in case API format changes)
        if not images:
            msg_content = message.get("content")
            if isinstance(msg_content, list):
                images = msg_content

        if not images:
            logger.warning("OpenRouter returned no images (content=%s, images=%s)",
                           type(message.get("content")).__name__,
                           type(message.get("images")).__name__)
            return None

        # Find first image_url part
        for part in images:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            raw_url = part.get("image_url", {}).get("url", "")
            if not raw_url:
                continue

            # If it's a base64 data URL, upload to get HTTP URL
            if raw_url.startswith("data:"):
                logger.info("Got base64 image, processing...")
                result_url = _upload_base64_image(raw_url)
                if result_url:
                    # If it's a relative /v1/files/ path, construct full gateway URL
                    if result_url.startswith("/v1/files/"):
                        gateway_port = os.getenv("API_SERVER_PORT", "8642")
                        gateway_host = os.getenv("API_SERVER_HOST", "0.0.0.0")
                        if gateway_host == "0.0.0.0":
                            gateway_host = "localhost"
                        result_url = f"http://{gateway_host}:{gateway_port}{result_url}"
                        logger.info("Image served via gateway: %s", result_url)
                    elif result_url.startswith("data:"):
                        logger.info("Returning base64 data URL directly")
                    else:
                        logger.info("Image uploaded to CDN: %s", result_url[:80])
                    return result_url
            else:
                # Already an HTTP URL
                logger.info("OpenRouter image generated successfully")
                return raw_url

        logger.warning("OpenRouter response contained no valid image_url parts")
        return None

    except Exception as e:
        logger.warning("OpenRouter image generation failed: %s", e)
        return None


def image_generate_tool(
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_images: int = DEFAULT_NUM_IMAGES,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    seed: Optional[int] = None
) -> str:
    """
    Generate images from text prompts using FAL.ai's FLUX 2 Pro model with automatic upscaling.
    
    Uses the synchronous fal_client API to avoid event loop lifecycle issues.
    The async API's global httpx.AsyncClient (cached via @cached_property) breaks
    when asyncio.run() destroys and recreates event loops between calls, which
    happens in the gateway's thread-pool pattern.
    
    Args:
        prompt (str): The text prompt describing the desired image
        aspect_ratio (str): Image aspect ratio - "landscape", "square", or "portrait" (default: "landscape")
        num_inference_steps (int): Number of denoising steps (1-50, default: 50)
        guidance_scale (float): How closely to follow prompt (0.1-20.0, default: 4.5)
        num_images (int): Number of images to generate (1-4, default: 1)
        output_format (str): Image format "jpeg" or "png" (default: "png")
        seed (Optional[int]): Random seed for reproducible results (optional)
    
    Returns:
        str: JSON string containing minimal generation results:
             {
                 "success": bool,
                 "image": str or None  # URL of the upscaled image, or None if failed
             }
    """
    # Validate and map aspect_ratio to actual image_size
    aspect_ratio_lower = aspect_ratio.lower().strip() if aspect_ratio else DEFAULT_ASPECT_RATIO
    if aspect_ratio_lower not in ASPECT_RATIO_MAP:
        logger.warning("Invalid aspect_ratio '%s', defaulting to '%s'", aspect_ratio, DEFAULT_ASPECT_RATIO)
        aspect_ratio_lower = DEFAULT_ASPECT_RATIO
    image_size = ASPECT_RATIO_MAP[aspect_ratio_lower]
    
    debug_call_data = {
        "parameters": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "seed": seed
        },
        "error": None,
        "success": False,
        "images_generated": 0,
        "generation_time": 0
    }
    
    start_time = datetime.datetime.now()
    
    try:
        logger.info("Generating %s image(s) with FLUX 2 Pro: %s", num_images, prompt[:80])
        
        # Validate prompt
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")
        
        # --- Try OpenRouter GPT-5.4 Image 2 first ---
        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_url = _generate_via_openrouter(prompt.strip(), aspect_ratio_lower)
            if openrouter_url:
                generation_time = (datetime.datetime.now() - start_time).total_seconds()
                logger.info("Image generated via OpenRouter in %.1fs", generation_time)
                response_data = {
                    "success": True,
                    "image": openrouter_url,
                }
                debug_call_data["success"] = True
                debug_call_data["images_generated"] = 1
                debug_call_data["generation_time"] = generation_time
                _debug.log_call("image_generate_tool", debug_call_data)
                _debug.save()
                return json.dumps(response_data, indent=2, ensure_ascii=False)
            else:
                logger.info("OpenRouter failed, falling back to FAL.ai FLUX 2 Pro...")
        
        
        # Check API key availability
        if not (os.getenv("FAL_KEY") or _resolve_managed_fal_gateway()):
            message = "FAL_KEY environment variable not set"
            if managed_nous_tools_enabled():
                message += " and managed FAL gateway is unavailable"
            raise ValueError(message)
        
        # Validate other parameters
        validated_params = _validate_parameters(
            image_size, num_inference_steps, guidance_scale, num_images, output_format, "none"
        )
        
        # Prepare arguments for FAL.ai FLUX 2 Pro API
        arguments = {
            "prompt": prompt.strip(),
            "image_size": validated_params["image_size"],
            "num_inference_steps": validated_params["num_inference_steps"],
            "guidance_scale": validated_params["guidance_scale"],
            "num_images": validated_params["num_images"],
            "output_format": validated_params["output_format"],
            "enable_safety_checker": ENABLE_SAFETY_CHECKER,
            "safety_tolerance": SAFETY_TOLERANCE,
            "sync_mode": True  # Use sync mode for immediate results
        }
        
        # Add seed if provided
        if seed is not None and isinstance(seed, int):
            arguments["seed"] = seed
        
        logger.info("Submitting generation request to FAL.ai FLUX 2 Pro...")
        logger.info("  Model: %s", DEFAULT_MODEL)
        logger.info("  Aspect Ratio: %s -> %s", aspect_ratio_lower, image_size)
        logger.info("  Steps: %s", validated_params['num_inference_steps'])
        logger.info("  Guidance: %s", validated_params['guidance_scale'])
        
        # Submit request to FAL.ai using sync API (avoids cached event loop issues)
        handler = _submit_fal_request(
            DEFAULT_MODEL,
            arguments=arguments,
        )
        
        # Get the result (sync — blocks until done)
        result = handler.get()
        
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Process the response
        if not result or "images" not in result:
            raise ValueError("Invalid response from FAL.ai API - no images returned")
        
        images = result.get("images", [])
        if not images:
            raise ValueError("No images were generated")
        
        # Format image data and upscale images
        formatted_images = []
        for img in images:
            if isinstance(img, dict) and "url" in img:
                original_image = {
                    "url": img["url"],
                    "width": img.get("width", 0),
                    "height": img.get("height", 0)
                }
                
                # Attempt to upscale the image
                upscaled_image = _upscale_image(img["url"], prompt.strip())
                
                if upscaled_image:
                    # Use upscaled image if successful
                    formatted_images.append(upscaled_image)
                else:
                    # Fall back to original image if upscaling fails
                    logger.warning("Using original image as fallback")
                    original_image["upscaled"] = False
                    formatted_images.append(original_image)
        
        if not formatted_images:
            raise ValueError("No valid image URLs returned from API")
        
        upscaled_count = sum(1 for img in formatted_images if img.get("upscaled", False))
        logger.info("Generated %s image(s) in %.1fs (%s upscaled)", len(formatted_images), generation_time, upscaled_count)
        
        # Prepare successful response - minimal format
        response_data = {
            "success": True,
            "image": formatted_images[0]["url"] if formatted_images else None
        }
        
        debug_call_data["success"] = True
        debug_call_data["images_generated"] = len(formatted_images)
        debug_call_data["generation_time"] = generation_time
        
        # Log debug information
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = f"Error generating image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        
        # Include error details so callers can diagnose failures
        response_data = {
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def check_fal_api_key() -> bool:
    """
    Check if the FAL.ai API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("FAL_KEY") or _resolve_managed_fal_gateway())


def check_image_generation_requirements() -> bool:
    """
    Check if all requirements for image generation tools are met.
    
    Returns:
        bool: True if requirements are met, False otherwise
    """
    # OpenRouter GPT-5.4 Image 2 is sufficient — no FAL dependency needed
    if os.getenv("OPENROUTER_API_KEY"):
        return True

    try:
        # Check API key
        if not check_fal_api_key():
            return False
        
        # Check if fal_client is available
        import fal_client  # noqa: F401 — SDK presence check
        return True
        
    except ImportError:
        return False


def get_debug_session_info() -> Dict[str, Any]:
    """
    Get information about the current debug session.
    
    Returns:
        Dict[str, Any]: Dictionary containing debug session information
    """
    return _debug.get_session_info()


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🎨 Image Generation Tools Module - FLUX 2 Pro + Auto Upscaling")
    print("=" * 60)
    
    # Check if API key is available
    api_available = check_fal_api_key()
    
    if not api_available:
        print("❌ FAL_KEY environment variable not set")
        print("Please set your API key: export FAL_KEY='your-key-here'")
        print("Get API key at: https://fal.ai/")
        exit(1)
    else:
        print("✅ FAL.ai API key found")
    
    # Check if fal_client is available
    try:
        import fal_client
        print("✅ fal_client library available")
    except ImportError:
        print("❌ fal_client library not found")
        print("Please install: pip install fal-client")
        exit(1)
    
    print("🛠️ Image generation tools ready for use!")
    print(f"🤖 Using model: {DEFAULT_MODEL}")
    print(f"🔍 Auto-upscaling with: {UPSCALER_MODEL} ({UPSCALER_FACTOR}x)")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/image_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set IMAGE_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from image_generation_tool import image_generate_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      # Generate image with automatic 2x upscaling")
    print("      result = await image_generate_tool(")
    print("          prompt='A serene mountain landscape with cherry blossoms',")
    print("          image_size='landscape_4_3',")
    print("          num_images=1")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nSupported image sizes:")
    for size in VALID_IMAGE_SIZES:
        print(f"  - {size}")
    print("  - Custom: {'width': 512, 'height': 768} (if needed)")
    
    print("\nAcceleration modes:")
    for mode in VALID_ACCELERATION_MODES:
        print(f"  - {mode}")
    
    print("\nExample prompts:")
    print("  - 'A candid street photo of a woman with a pink bob and bold eyeliner'")
    print("  - 'Modern architecture building with glass facade, sunset lighting'")
    print("  - 'Abstract art with vibrant colors and geometric patterns'")
    print("  - 'Portrait of a wise old owl perched on ancient tree branch'")
    print("  - 'Futuristic cityscape with flying cars and neon lights'")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export IMAGE_TOOLS_DEBUG=true")
    print("  # Debug logs capture all image generation calls and results")
    print("  # Logs saved to: ./logs/image_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate",
    "description": "Generate high-quality images from text prompts. Uses OpenRouter GPT-5.4 Image 2 when available, with FLUX 2 Pro as fallback. Returns a single image URL. Display it using markdown: ![description](URL)",
    "parameters": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": "A one-sentence preamble describing the purpose of this operation"
            },
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the desired image. Be detailed and descriptive."
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "square", "portrait"],
                "description": "The aspect ratio of the generated image. 'landscape' is 16:9 wide, 'portrait' is 16:9 tall, 'square' is 1:1.",
                "default": "landscape"
            }
        },
        "required": ["brief", "prompt"]
    }
}


def _handle_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for image generation")
    return image_generate_tool(
        prompt=prompt,
        aspect_ratio=args.get("aspect_ratio", "landscape"),
        num_inference_steps=50,
        guidance_scale=4.5,
        num_images=1,
        output_format="png",
        seed=None,
    )


registry.register(
    name="image_generate",
    toolset="image_gen",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate,
    check_fn=check_image_generation_requirements,
    requires_env=[],
    is_async=False,  # Switched to sync fal_client API to fix "Event loop is closed" in gateway
    emoji="🎨",
)
