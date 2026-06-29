1. 路由层 entrypoints/openai/api_server.py:29

新增的 /release_kv_cache 路由，拿到 chat(raw_request) 这个 handler（即被 patch 过的 OpenAIServingChat），调用 handler.release_kv_cache()，异常包成 500。纯转发。

2. 核心算法层 entrypoints/openai/serving_chat.py:208（最关键的一步）

这一层负责把"保留 N 条消息"翻译成"释放从哪个 token 开始"。做法是 渲染两次 + diff：

1. message_begin = messages_released_index（=1），即保留 messages[:1]（serving_chat.py:236）。
2. before：渲染整份旧 messages → before_token_ids（全量 prompt token）(:296)。
3. after：model_copy 出只含 messages[:message_begin] 的请求，渲染 → after_token_ids（保留部分的 token）(:263、:304)。
4. 校验 len(before) > len(after)，否则报错 (:312)。
5. 算 released_token_index（:317-336）——即 before 序列里"保留前缀"的结束位置：
  - 工具变了：找第一个 token 分叉点；
  - 否则用子序列匹配：把 after 的 token 在 before 里顺序对齐，定位最后一个匹配位置。这里有个 eos_token_count = 5（:327），用来跳过 chat 模板在 after 末尾多加的收尾 token（如 <|im_end|> 等），避免边界算错。

然后（:340-391）把 before prompt 按 engine_prompt 切片，对越过 released_token_index 的那一段重新 input_processor.process_inputs 生成 EngineCoreRequest，并 encode_engine_core_request 编码，连同该段内的相对释放位置 released_token_index - elapsed_tokens 一起塞进 request_params（list[tuple[编码后的请求, 释放token下标]]）。

▎ 注意 sub_request_id 在这里又被 pack_request_sharing_cache_salt 打上了 $KV$<salt>: 前缀（:371），把 session 信息编码进 request_id 以跨进程传递。

最后 :395 调 engine_client.release_kv_cache(cache_salt, request_params)。

3. 逐层下传到引擎核心进程

release_kv_cache(session_id, token_requests) 沿着 patch 链一路透传：

async_llm.py:8        AsyncLLM            → self.engine_core.release_kv_cache(...)
core_client.py:14     AsyncMPClient       → call_utility_async("release_kv_cache", ...)   # 跨进程 RPC
core_client.py:23     InprocClient        → 直接 self.engine_core.release_kv_cache(...)

MP 模式下通过 call_utility_async 把调用打到 EngineCore 子进程里执行。（engine/protocol.py 和 core_client.py:7 那两个返回 0/pass 的 mixin 是给基类/抽象层兜底用的占位实现。）

4. EngineCore：token 下标 → block_hash 区间 v1/engine/core.py:39

这是把"token 位置"换算成"block 区间"的地方：

for params, release_index in token_requests:
    request = decode_engine_core_request(params)                       # 解码
    req = Request.from_engine_core_request(request, self.request_block_hasher)  # 重算 block_hashes
    release_block_index = max(0, (release_index * len(req.block_hashes)) // len(req.all_token_ids) - 1)
    released_blocks += self.scheduler.release_kv_cache(
        session_id, req.block_hashes[release_block_index:])            # 释放【尾部】区间

- Request.from_engine_core_request（被 v1/request.py:16 patch）会重新计算 block_hashes，并从 request_id 解出 sharing_cache_salt。
- 按 token 比例换算出 release_block_index，释放 block_hashes[release_block_index:] 这段尾部 block（减 1 是保守留一块边界）。

5. Scheduler → KVCacheManager → Coordinator

纯透传，逐级下钻：

sched/scheduler.py:6      → self.kv_cache_manager.release_kv_cache(session_id, block_hashes)
kv_cache_manager.py:8     → self.coordinator.aging_block(session_id, block_hashes)
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
