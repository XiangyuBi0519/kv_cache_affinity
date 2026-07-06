from typing import Callable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request
from kv_cache_affinity.v1.engine.core import unpack_sharing_cache_salt


# 取带 session salt 的 request_id 来源：优先 external_req_id，其次 request_id。
# salt 被 pack 成 "$KV$<salt>:<id>" 编码在其中。
def _packed_request_id_source(request: EngineCoreRequest) -> str:
    if request.external_req_id is not None:
        return request.external_req_id
    return request.request_id


@classmethod
def from_engine_core_request_kv(
    cls,
    request: EngineCoreRequest,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None,
) -> "Request":
    """重建 Request（覆盖 vanilla 版）。

    相比 vanilla，额外从 request_id 解出 session salt 并挂到 req 上；构造时传入
    block_hasher，会自动用同样的算法 + salt 重算 block_hashes（供 release 换算/匹配）。
    注意：下方 cls(...) 的参数列表是复制自 vanilla Request.from_engine_core_request，
    vLLM 升级若改动 Request 构造参数，此处需同步。
    """
    sharing_cache_salt = unpack_sharing_cache_salt(_packed_request_id_source(request))
    req = cls(
        request_id=request.request_id,
        client_index=request.client_index,
        prompt_token_ids=request.prompt_token_ids,
        prompt_embeds=request.prompt_embeds,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        priority=request.priority,
        trace_headers=request.trace_headers,
        block_hasher=block_hasher,
        resumable=request.resumable,
        reasoning_ended=request.reasoning_ended,
    )
    # 把 session 身份挂到 req 上，供后续 allocate 时按 session 记账 block 归属、
    # 以及 release 时定向解绑（见 kv_cache_session_manager）。
    if sharing_cache_salt is not None:
        req.sharing_cache_salt = sharing_cache_salt
    return req


def request_get_sharing_cache_salt(request) -> str | None:
    return (
        None
        if not hasattr(request, "sharing_cache_salt")
        else request.sharing_cache_salt
    )


def register_request():
    Request.from_engine_core_request = from_engine_core_request_kv
