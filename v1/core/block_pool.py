from vllm.distributed.kv_events import KVCacheEvent
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from kv_cache_affinity.v1.core.two_phase_block_queue import TwoPhaseBlockQueue


class KvCacheBlockPool(BlockPool):
    def aging_block(self, blocks: list[KVCacheBlock]) -> int:
        num = 0
        for i in range(len(blocks) - 1, -1, -1):
            num += self.free_block_queue.aging_block(blocks[i])
        return num


def block_pool_init_kv(
    self,
    num_gpu_blocks: int,
    enable_caching: bool,
    hash_block_size: int,
    enable_kv_cache_events: bool = False,
    metrics_collector: KVCacheMetricsCollector | None = None,
):
    if not isinstance(num_gpu_blocks, int) or num_gpu_blocks <= 0:
        raise ValueError(
            f"num_gpu_blocks must be a positive int, got {num_gpu_blocks!r}"
        )
    self.num_gpu_blocks = num_gpu_blocks
    self.enable_caching = enable_caching
    self.hash_block_size = hash_block_size
    # All kv-cache blocks.
    self.blocks: list[KVCacheBlock] = [
        KVCacheBlock(idx) for idx in range(num_gpu_blocks)
    ]

    self.free_block_queue = TwoPhaseBlockQueue(self.blocks)
    # Cache for block lookup
    self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

    self.null_block = self.free_block_queue.popleft()
    self.null_block.is_null = True
    self.enable_kv_cache_events = enable_kv_cache_events
    self.kv_event_queue: list[KVCacheEvent] = []
    self.metrics_collector = metrics_collector


def register_block_pool():
    BlockPool.__init__ = block_pool_init_kv
    BlockPool.aging_block = KvCacheBlockPool.aging_block
