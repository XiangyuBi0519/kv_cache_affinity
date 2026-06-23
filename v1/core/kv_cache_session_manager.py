from typing import Optional, Sequence

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.logger import init_logger

logger = init_logger(__name__)


class KvCacheSessionManager:

    def __init__(self):
        self.block_to_sessions: dict[int, set[str]] = {}

    def add_blocks(
        self, blocks: Sequence[KVCacheBlock], session_id: Optional[str] = None
    ):
        logger.debug("add %s blocks with session %s", len(blocks), session_id)
        if session_id is None:
            return
        for blk in blocks:
            if blk.block_id < 0:
                continue
            self.block_to_sessions.setdefault(blk.block_id, set()).add(session_id)

    def reset_blocks(
        self, blocks: list[KVCacheBlock], session_id: Optional[str] = None
    ):
        logger.debug("reset %s blocks with session %s", len(blocks), session_id)
        for blk in blocks:
            if blk.block_id < 0:
                continue
            self.block_to_sessions[blk.block_id] = {session_id} if session_id else set()

    def release_blocks(
        self, blocks: list[KVCacheBlock], session_id: Optional[str] = None
    ) -> list[KVCacheBlock]:
        logger.debug("release %s blocks with session %s", len(blocks), session_id)
        if session_id is None:
            return []
        aging_blocks: list[KVCacheBlock] = []
        for blk in blocks:
            if blk.block_id < 0:
                continue
            sessions = self.block_to_sessions.get(blk.block_id)
            if sessions is None:
                continue
            sessions.discard(session_id)
            if len(sessions) == 0:
                del self.block_to_sessions[blk.block_id]
                aging_blocks.append(blk)

        return aging_blocks
