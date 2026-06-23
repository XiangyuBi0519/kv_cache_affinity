from vllm.logger import init_logger

logger = init_logger(__name__)

_patches_applied = False


def register():
    global _patches_applied
    if _patches_applied:
        logger.info("kv_cache_affinity patches already applied, skipping")
        return
    _patches_applied = True
    logger.info("register kv_cache_affinity")
    from kv_cache_affinity.patcher import apply_patches
    apply_patches()
