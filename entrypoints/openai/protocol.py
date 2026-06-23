from typing import ClassVar

from pydantic import field_validator

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class ReleaseKvCacheResponse(OpenAIBaseModel):
    cache_salt: str | None = None
    block_released: int


def register_chat_request():
    """ChatCompletionRequest already allows extra fields (extra="allow")."""


class ReleaseKvCacheRequest(ChatCompletionRequest):
    """Chat completion-shaped body plus fields for partial KV release."""

    field_names: ClassVar[set[str] | None] = None

    messages_released_index: int = 0
    tools_released_index: int | None = None
    cache_sharing: bool | None = None

    @field_validator("messages_released_index")
    @classmethod
    def validate_messages_index(cls, v):
        if v < 0:
            raise ValueError("messages_released_index must be >= 0")
        return v

    @field_validator("tools_released_index")
    @classmethod
    def validate_tools_index(cls, v):
        if v is not None and v < 0:
            raise ValueError("tools_released_index must be >= 0")
        return v
