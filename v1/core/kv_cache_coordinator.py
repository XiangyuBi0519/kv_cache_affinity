from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

_orig_allocate_new_computed_blocks = KVCacheCoordinator.allocate_new_computed_blocks
_orig_allocate_new_blocks = KVCacheCoordinator.allocate_new_blocks


# 把 session_id 从 coordinator 转发到每个 single_type_manager——因为真正登记 block
# 归属发生在下层的 single_type_manager，需要它们也能读到当前 session。
def _propagate_session_id(coordinator: KVCacheCoordinator, session_id) -> None:
    for m in coordinator.single_type_managers:
        m.set_kv_cache_session_id(session_id)


def _clear_propagated_session_id(coordinator: KVCacheCoordinator) -> None:
    for m in coordinator.single_type_managers:
        m.clear_kv_cache_session_id()


# 覆盖 vanilla：调原版分配逻辑，外面包一层“传播 session 到下层 + finally 清除”。
def allocate_new_computed_blocks_kv(
    self,
    request_id: str,
    new_computed_blocks,
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None:
    sid = self.get_kv_cache_session_id()
    _propagate_session_id(self, sid)
    try:
        return _orig_allocate_new_computed_blocks(
            self,
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        )
    finally:
        _clear_propagated_session_id(self)


def allocate_new_blocks_kv(
    self,
    request_id: str,
    num_tokens: int,
    num_tokens_main_model: int,
    num_encoder_tokens: int = 0,
):
    sid = self.get_kv_cache_session_id()
    _propagate_session_id(self, sid)
    try:
        return _orig_allocate_new_blocks(
            self, request_id, num_tokens, num_tokens_main_model, num_encoder_tokens
        )
    finally:
        _clear_propagated_session_id(self)


class KvCacheCoordinatorMixin(KVCacheCoordinator):
    def set_kv_cache_session_id(self, session_id: str | None) -> None:
        self.kv_cache_session_id = session_id

    def get_kv_cache_session_id(self) -> str | None:
        return getattr(self, "kv_cache_session_id", None)

    def clear_kv_cache_session_id(self) -> None:
        self.kv_cache_session_id = None

    # 释放路径：扇出到所有 single_type_manager，各自老化并累加老化的 block 数。
    def aging_block(self, session_id, block_hashes) -> int:
        num = 0
        for manager in self.single_type_managers:
            num += manager.aging_block(session_id, block_hashes)
        return num


def register_kv_cache_coordinator():
    KVCacheCoordinator.set_kv_cache_session_id = (
        KvCacheCoordinatorMixin.set_kv_cache_session_id
    )
    KVCacheCoordinator.get_kv_cache_session_id = (
        KvCacheCoordinatorMixin.get_kv_cache_session_id
    )
    KVCacheCoordinator.clear_kv_cache_session_id = (
        KvCacheCoordinatorMixin.clear_kv_cache_session_id
    )
    KVCacheCoordinator.aging_block = KvCacheCoordinatorMixin.aging_block
    KVCacheCoordinator.allocate_new_computed_blocks = allocate_new_computed_blocks_kv
    KVCacheCoordinator.allocate_new_blocks = allocate_new_blocks_kv
