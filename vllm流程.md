kv_cache_coordinator.py:65→ 遍历 single_type_managers，逐个 manager.aging_block(...)

6. SingleTypeKVCacheManager：session 归属判定 single_type_kv_cache_manager.py:91

真正决定"哪些 block 能释放"的逻辑：

def aging_block(self, session_id, block_hashes):
    aging_blocks = []
    for block_hash in block_hashes:
        cached = self.block_pool.get_cached_block(block_hash, [group_id])
        if cached: aging_blocks.append(cached[0])
        else:      break          # 命中链一旦断开就停
    aging_blocks = self.kv_cache_session_manager.release_blocks(aging_blocks, session_id)  # 关键过滤
    return self.block_pool.aging_block(aging_blocks)

session_manager.release_blocks（kv_cache_session_manager.py:34）：对每个 block 从其 block_to_sessions[block_id] 集合里 discard 当前 session，只有当集合变空（没有别的会话还在用这块）才真正放进待老化列表并从字典删除。这就是"亲和性"保证——被多个会话共享的 block 不会被某一个会话的 release 误删。

7. BlockPool → TwoPhaseBlockQueue：实际"老化" two_phase_block_queue.py:236

block_pool.py:9 倒序遍历，对每块调 free_block_queue.aging_block(block)：

def aging_block(self, block):
    if block.ref_cnt != 0: return 0                       # 还被运行中的请求引用 → 不能动
    if block.prev is None and block.next is None: return 0 # 不在 free list 里 → 跳过(这是上次提交补的防御)
    ...
    # 把该 block 从当前位置摘出，重新插到 self.aging 游标之后，并推进 aging 游标

效果：被释放的 free block 被摘出原位、聚拢成一段连续的"老化区"（由 aging 游标维护），aging_block 返回真正被老化的块数，一路加回去层层上报，最终变成响应里的 block_released。
