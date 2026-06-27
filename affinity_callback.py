"""
KV Cache Affinity Injection + Active KV Cache Management
=========================================================
1. AFFINITY: inject cache_salt + cache_sharing for jiuwen-enabled models
2. ACTIVE MANAGEMENT: detect context compression, call /release_kv_cache
"""

import hashlib
import logging
import time
import asyncio
from typing import Optional, Union

import httpx
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("affinity_callback")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
    logger.addHandler(h)
    try:
        fh = logging.FileHandler("affinity_callback.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    except:
        pass

# ============================================================
# Configuration
# ============================================================

AFFINITY_MODELS = {
    "glm-4.7-affinity",
    "glm-4.7-jzx",
    "GLM-4.7-w8a8",
    "hosted_vllm/GLM-4.7-w8a8",
}

# vLLM backend address for release_kv_cache calls
VLLM_RELEASE_URL = "http://10.41.217.42:8013/release_kv_cache"

# Session store TTL (seconds)
SESSION_TTL = 7200

# ============================================================
# Helpers
# ============================================================


def _get_session_id_from_header(data: dict) -> str | None:
    """Extract x-session-affinity from request metadata headers."""
    headers = data.get("metadata", {}).get("headers", {})
    return headers.get("x-session-affinity")


def _generate_session_id(data: dict) -> str:
    """Generate session_id from x-session-affinity header."""
    client_session = _get_session_id_from_header(data)
    if client_session:
        return hashlib.sha256(client_session.encode("utf-8")).hexdigest()[:16]
    return "default_session"


def _extract_system_prompt(messages: list) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                return "".join(parts)
    return ""


# ============================================================
# Async release caller
# ============================================================

async def _call_release_kv_cache(
    session_id: str,
    old_messages: list,
    messages_released_index: int = 1,
    tools: list = None,
):
    """Async call to vLLM /release_kv_cache endpoint"""
    payload = {
        "model": "GLM-4.7-w8a8",
        "messages": old_messages,
        "cache_salt": session_id,
        "cache_sharing": True,
        "messages_released_index": messages_released_index,
    }
    if tools:
        payload["tools"] = tools
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(VLLM_RELEASE_URL, json=payload)
            result = resp.json()
            logger.info(
                f"[RELEASE] session={session_id} "
                f"block_released={result.get('block_released', 'N/A')} "
                f"old_messages={len(old_messages)} "
                f"released_index={messages_released_index}"
            )
            return result
    except Exception as e:
        logger.error(f"[RELEASE] Failed for session={session_id}: {e}")
        return None


# ============================================================
# Callback
# ============================================================

class AffinityCallback(CustomLogger):

    def __init__(self):
        super().__init__()
        # {session_id: {"messages": [...], "system_prompt": "...",
        #               "msg_count": int, "last_active": float}}
        self.session_store = {}
        logger.info(
            f"AffinityCallback initialized | "
            f"affinity_models={AFFINITY_MODELS} | "
            f"release_url={VLLM_RELEASE_URL}"
        )

    def _cleanup_expired_sessions(self):
        """Remove sessions older than TTL"""
        now = time.time()
        expired = [
            sid for sid, data in self.session_store.items()
            if now - data.get("last_active", 0) > SESSION_TTL
        ]
        for sid in expired:
            del self.session_store[sid]
            logger.debug(f"[CLEANUP] Expired session {sid}")

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:

        if call_type not in ("completion", "acompletion", "text_completion", "atext_completion"):
            return data

        model = data.get("model", "")
        messages = data.get("messages", [])

        if model not in AFFINITY_MODELS or not messages:
            return data

        # Generate session_id from x-session-affinity header
        session_id = _generate_session_id(data)

        # ---- Affinity Injection ----
        if "extra_body" not in data:
            data["extra_body"] = {}
        data["extra_body"]["cache_sharing"] = True
        data["extra_body"]["cache_salt"] = session_id

        # ---- Active Management: detect compression ----
        current_system = _extract_system_prompt(messages)
        current_msg_count = len(messages)

        if session_id in self.session_store:
            prev = self.session_store[session_id]

            # Skip expired sessions
            if time.time() - prev.get("last_active", 0) > SESSION_TTL:
                logger.debug(f"[SKIP] session={session_id} expired, clearing stale data")
                del self.session_store[session_id]
            else:
                prev_system = prev.get("system_prompt", "")
                prev_msg_count = prev.get("msg_count", 0)
                prev_messages = prev.get("messages", [])

                # Detect compression: system prompt changed or message count dropped significantly
                system_changed = (prev_system != "" and current_system != prev_system)
                msg_dropped = (prev_msg_count > 0 and current_msg_count < prev_msg_count * 0.7)

                if (system_changed or msg_dropped) and prev_messages:
                    logger.info(
                        f"[COMPRESS_DETECTED] session={session_id} "
                        f"system_changed={system_changed} "
                        f"msg_dropped={msg_dropped} "
                        f"prev_msg_count={prev_msg_count} "
                        f"current_msg_count={current_msg_count}"
                    )
                    asyncio.create_task(
                        _call_release_kv_cache(
                            session_id=session_id,
                            old_messages=prev_messages,
                            messages_released_index=1,
                            tools=prev.get("tools"),
                        )
                    )

        logger.info(
            f"[AFFINITY] model={model} session={session_id} "
            f"cache_salt={data['extra_body'].get('cache_salt')} "
            f"cache_sharing={data['extra_body'].get('cache_sharing')} "
            f"msgs={current_msg_count}"
        )

        return data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called after successful response. Update session store."""
        try:
            model = kwargs.get("model", "")
            if model not in AFFINITY_MODELS:
                return

            messages = kwargs.get("messages", [])
            if not messages:
                return

            optional_params = kwargs.get("optional_params", {})
            tools = optional_params.get("tools", None)

            # Extract session_id from litellm_params metadata (same headers path)
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})
            session_id = _generate_session_id({"metadata": metadata})

            if session_id == "default_session":
                return

            system_prompt = _extract_system_prompt(messages)

            self.session_store[session_id] = {
                "messages": messages,
                "system_prompt": system_prompt,
                "msg_count": len(messages),
                "last_active": time.time(),
                "tools": tools,
            }

            logger.debug(
                f"[STORE] session={session_id} msgs={len(messages)} "
                f"system_len={len(system_prompt)}"
            )

            # Periodic cleanup
            if len(self.session_store) > 100:
                self._cleanup_expired_sessions()

        except Exception as e:
            logger.error(f"[STORE] Error updating session store: {e}")


handler_instance = AffinityCallback()
