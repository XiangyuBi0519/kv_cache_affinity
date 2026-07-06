from typing import Optional, Sequence

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.logger import init_logger

logger = init_logger(__name__)


class KvCacheSessionManager:
    """维护 block -> session 归属表（vanilla 无此机制，全新增）。

    block_to_sessions: {block_id: {session_id, ...}}，记录每个 block 被哪些 session
    引用。这是“会话亲和 + 安全释放”的核心：release 时按 session 定向解绑，只有当某个
    block 不再被任何 session 引用时才允许老化，从而不会误伤多 session 共享的前缀 block。
    每个 single-type manager 各持有一个本实例。
    """

    def __init__(self):
        self.block_to_sessions: dict[int, set[str]] = {}

    # 追加语义（.add）：用于命中前缀缓存、可能被多 session 共享的 block——
    # 把当前 session 加入其归属集合，不影响已有的其他 session。
    def add_blocks(
        self, blocks: Sequence[KVCacheBlock], session_id: Optional[str] = None
    ):
        logger.debug("add %s blocks with session %s", len(blocks), session_id)
        if session_id is None:
            return
        for blk in blocks:
            if blk.block_id < 0:  # 跳过 null block（id=-1）
                continue
            self.block_to_sessions.setdefault(blk.block_id, set()).add(session_id)

    # 覆盖语义（= 直接重置）：用于【新分配】的 block。新 block 刚从 free 队列取出、
    # 之前不该有任何归属，故直接重置为“只属于当前 session”是安全的（不会抹掉别人）。
    # 与 add_blocks 的区别是刻意的：新块独占用 reset，共享命中块追加用 add。
    def reset_blocks(
        self, blocks: list[KVCacheBlock], session_id: Optional[str] = None
    ):
        logger.debug("reset %s blocks with session %s", len(blocks), session_id)
        for blk in blocks:
            if blk.block_id < 0:  # 跳过 null block
                continue
            self.block_to_sessions[blk.block_id] = {session_id} if session_id else set()

    # 定向解绑（释放路径）：只把当前 session 从每个 block 的归属集合摘掉；
    # 仅当集合变空（没有任何 session 再引用）时，该 block 才进入返回的可老化列表。
    # —— 这是“多 session 共享不被单方误删”的安全保证。
    def release_blocks(
        self, blocks: list[KVCacheBlock], session_id: Optional[str] = None
    ) -> list[KVCacheBlock]:
        logger.debug("release %s blocks with session %s", len(blocks), session_id)
        if session_id is None:
            return []
        aging_blocks: list[KVCacheBlock] = []
        for blk in blocks:
            if blk.block_id < 0:  # 跳过 null block
                continue
            sessions = self.block_to_sessions.get(blk.block_id)
            if sessions is None:
                continue
            sessions.discard(session_id)  # 只摘掉当前 session
            if len(sessions) == 0:        # 没人再引用 -> 可以老化
                del self.block_to_sessions[blk.block_id]
                aging_blocks.append(blk)

        return aging_blocks
