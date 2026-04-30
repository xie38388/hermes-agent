"""Background task execution and BTW (by-the-way) task handling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import asyncio
import datetime
import os
logger = logging.getLogger(__name__)


from gateway.session import SessionSource
from gateway.platforms.base import MessageEvent
class GwBackgroundMixin:
    """Background task execution and BTW (by-the-way) task handling."""

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /background <prompt> — run a prompt in a separate background session.

        Spawns a new AIAgent in a background thread with its own session.
        When it completes, sends the result back to the same chat without
        modifying the active session's conversation history.
        """
        prompt = event.get_command_args().strip()
        if not prompt:
            return (
                "Usage: /background <prompt>\n"
                "Example: /background Summarize the top HN stories today\n\n"
                "Runs the prompt in a separate session. "
                "You can keep chatting — the result will appear here when done."
            )

        source = event.source
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"

        # Fire-and-forget the background task
        _task = asyncio.create_task(
            self._run_background_task(prompt, source, task_id)
        )
        self._background_tasks.add(_task)
        _task.add_done_callback(self._background_tasks.discard)

        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
        return f'🔄 Background task started: "{preview}"\nTask ID: {task_id}\nYou can keep chatting — results will appear when done.'

    async def _run_background_task(
        self, prompt: str, source: "SessionSource", task_id: str
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        from run_agent import AIAgent

        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return

        _thread_metadata = {"thread_id": source.thread_id} if source.thread_id else None

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                user_config=user_config,
            )
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            platform_key = _platform_config_key(source.platform)

            from hermes_cli.tools_config import _get_platform_tools
            enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))

            pr = self._provider_routing
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
            reasoning_config = self._load_reasoning_config()
            self._reasoning_config = reasoning_config
            self._service_tier = self._load_service_tier()
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    session_db=self._session_db,
                    fallback_model=self._fallback_model,
                )

                return agent.run_conversation(
                    user_message=prompt,
                    task_id=task_id,
                )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"

            # Extract media files from the response
            if response:
                media_files, response = adapter.extract_media(response)
                images, text_content = adapter.extract_images(response)

                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'

                if text_content:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + text_content,
                        metadata=_thread_metadata,
                    )
                elif not images and not media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + "(No response generated)",
                        metadata=_thread_metadata,
                    )

                # Send extracted images
                for image_url, alt_text in (images or []):
                    try:
                        await adapter.send_image(
                            chat_id=source.chat_id,
                            image_url=image_url,
                            caption=alt_text,
                        )
                    except Exception as e:
                        logger.warning("Suppressed exception in %s: %s", "run.run_sync", e, exc_info=True)
                        pass

                # Send media files
                for media_path in (media_files or []):
                    try:
                        await adapter.send_document(
                            chat_id=source.chat_id,
                            file_path=media_path,
                        )
                    except Exception as e:
                        logger.warning("Suppressed exception in %s: %s", "run.run_sync", e, exc_info=True)
                        pass
            else:
                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f'✅ Background task complete\nPrompt: "{preview}"\n\n(No response generated)',
                    metadata=_thread_metadata,
                )

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run.run_sync", e, exc_info=True)
                pass

    async def _handle_btw_command(self, event: MessageEvent) -> str:
        """Handle /btw <question> — ephemeral side question in the same chat."""
        question = event.get_command_args().strip()
        if not question:
            return (
                "Usage: /btw <question>\n"
                "Example: /btw what module owns session title sanitization?\n\n"
                "Answers using session context. No tools, not persisted."
            )

        source = event.source
        session_key = self._session_key_for_source(source)

        # Guard: one /btw at a time per session
        existing = getattr(self, "_active_btw_tasks", {}).get(session_key)
        if existing and not existing.done():
            return "A /btw is already running for this chat. Wait for it to finish."

        if not hasattr(self, "_active_btw_tasks"):
            self._active_btw_tasks: dict = {}

        import uuid as _uuid
        task_id = f"btw_{datetime.now().strftime('%H%M%S')}_{_uuid.uuid4().hex[:6]}"
        _task = asyncio.create_task(self._run_btw_task(question, source, session_key, task_id))
        self._background_tasks.add(_task)
        self._active_btw_tasks[session_key] = _task

        def _cleanup(task):
            self._background_tasks.discard(task)
            if self._active_btw_tasks.get(session_key) is task:
                self._active_btw_tasks.pop(session_key, None)

        _task.add_done_callback(_cleanup)

        preview = question[:60] + ("..." if len(question) > 60 else "")
        return f'💬 /btw: "{preview}"\nReply will appear here shortly.'

    async def _run_btw_task(
        self, question: str, source, session_key: str, task_id: str,
    ) -> None:
        """Execute an ephemeral /btw side question and deliver the answer."""
        from run_agent import AIAgent

        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.warning("No adapter for platform %s in /btw task %s", source.platform, task_id)
            return

        _thread_meta = {"thread_id": source.thread_id} if source.thread_id else None

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
                user_config=user_config,
            )
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    "❌ /btw failed: no provider credentials configured.",
                    metadata=_thread_meta,
                )
                return

            platform_key = _platform_config_key(source.platform)
            reasoning_config = self._load_reasoning_config()
            self._service_tier = self._load_service_tier()
            turn_route = self._resolve_turn_agent_config(question, model, runtime_kwargs)
            pr = self._provider_routing

            # Snapshot history from running agent or stored transcript
            running_agent = self._running_agents.get(session_key)
            if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
                history_snapshot = list(getattr(running_agent, "_session_messages", []) or [])
            else:
                session_entry = self.session_store.get_or_create_session(source)
                history_snapshot = self.session_store.load_transcript(session_entry.session_id)

            btw_prompt = (
                "[Ephemeral /btw side question. Answer using the conversation "
                "context. No tools available. Be direct and concise.]\n\n"
                + question
            )

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    max_iterations=8,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=[],
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    session_db=None,
                    fallback_model=self._fallback_model,
                    skip_memory=True,
                    skip_context_files=True,
                    persist_session=False,
                )
                return agent.run_conversation(
                    user_message=btw_prompt,
                    conversation_history=history_snapshot,
                    task_id=task_id,
                )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_sync)

            response = (result.get("final_response") or "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"
            if not response:
                response = "(No response generated)"

            media_files, response = adapter.extract_media(response)
            images, text_content = adapter.extract_images(response)
            preview = question[:60] + ("..." if len(question) > 60 else "")
            header = f'💬 /btw: "{preview}"\n\n'

            if text_content:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=header + text_content,
                    metadata=_thread_meta,
                )
            elif not images and not media_files:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=header + "(No response generated)",
                    metadata=_thread_meta,
                )

            for image_url, alt_text in (images or []):
                try:
                    await adapter.send_image(chat_id=source.chat_id, image_url=image_url, caption=alt_text)
                except Exception as e:
                    logger.warning("Suppressed exception in %s: %s", "run.run_sync", e, exc_info=True)
                    pass

            for media_path in (media_files or []):
                try:
                    await adapter.send_file(chat_id=source.chat_id, file_path=media_path)
                except Exception as e:
                    logger.warning("Suppressed exception in %s: %s", "run.run_sync", e, exc_info=True)
                    pass

        except Exception as e:
            logger.exception("/btw task %s failed", task_id)
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ /btw failed: {e}",
                    metadata=_thread_meta,
                )
            except Exception as e:
                logger.warning("Suppressed exception in %s: %s", "run.run_sync", e, exc_info=True)
                pass

