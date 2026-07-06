from typing import Sequence

from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request
from vllm.logger import init_logger
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

logger = init_logger(__name__)

_KV_CACHE_PREFIX = "$KV$"


def pack_request_sharing_cache_salt(request_id: str, sharing_cache_salt: str) -> str:
    if request_id.startswith(_KV_CACHE_PREFIX):
        return request_id
    return f"{_KV_CACHE_PREFIX}{sharing_cache_salt}:{request_id}"


def unpack_sharing_cache_salt(request_id_and_salt: str) -> str | None:
    if not request_id_and_salt.startswith(_KV_CACHE_PREFIX):
        return None
    prefix_removed = request_id_and_salt[len(_KV_CACHE_PREFIX):]
    salt, _, _ = prefix_removed.partition(":")
    return salt if salt else None


def encode_engine_core_request(request: EngineCoreRequest) -> Sequence[bytestr]:
    encoder = MsgpackEncoder()
    return encoder.encode(request)


def decode_engine_core_request(frame: Sequence[bytestr]) -> EngineCoreRequest:
    decoder = MsgpackDecoder(EngineCoreRequest)
    return decoder.decode(frame)


class KvCacheEngineCore(EngineCore):
    def release_kv_cache(
        self, session_id: str, token_requests: list[tuple[Sequence[bytestr], int]]
    ) -> int:
        released_blocks = 0
        for params, release_start_index in token_requests:
            request = decode_engine_core_request(params)
            logger.debug("request decode %s", request)
            req = Request.from_engine_core_request(request, self.request_block_hasher)

            # 空 prompt 时无 token/无 block，无从换算，直接跳过（防止下方除零）。
            if len(req.all_token_ids) == 0:
                continue

            # 把 token 偏移换算成 block 下标：本质是 release_start_index / block_size
            #（用 block数/token数 的比例表达，因为这里只有 block 数和 token 数、没有 block_size）。
            # -1：保守往前退一个 block，确保释放点所在的“跨界块”也被释放（release 是软化老化，多释放无害）。
            # max(0, ...)：防止结果为负导致切片取到错误区间。
            release_block_index = max(
                0,
                (release_start_index * len(req.block_hashes)) // len(req.all_token_ids) - 1,
            )
            released_blocks += self.scheduler.release_kv_cache(
                session_id, req.block_hashes[release_block_index:]
            )
        return released_blocks


def register_engine_core():
    EngineCore.release_kv_cache = KvCacheEngineCore.release_kv_cache
