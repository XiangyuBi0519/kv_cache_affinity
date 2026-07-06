import time
from collections.abc import AsyncGenerator, Sequence
from typing import Union

import jinja2
from fastapi import Request

from kv_cache_affinity.v1.engine.core import (
    encode_engine_core_request,
    pack_request_sharing_cache_salt,
)
from kv_cache_affinity.entrypoints.openai.protocol import (
    ReleaseKvCacheRequest,
    ReleaseKvCacheResponse,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.openai.engine.serving import GenerationError
from vllm.entrypoints.utils import get_max_tokens
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.v1.serial_utils import bytestr

logger = init_logger(__name__)

_PROMPT_TOKEN_IDS_KEY = "prompt_token_ids"

# prefix 是单独渲染出来的，chat 模板会在它尾部补上若干“轮次收尾 token”
# （如 <|im_end|> 之类），这些 token 在 complete 里紧接着后续对话、并不出现，
# 会导致子序列匹配在尾部错位。匹配时砍掉 prefix 末尾这么多 token，只用它前段
# 的“稳定前缀”去 complete 里定位。
_EOS_TOKEN_COUNT = 5


def _cache_sharing_on(request: ChatCompletionRequest) -> bool:
    if getattr(request, "cache_sharing", None) is not None:
        return bool(request.cache_sharing)
    extra = request.model_extra or {}
    return bool(extra.get("cache_sharing"))


class KvCacheServingChat(OpenAIServingChat):
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        tokenizer = self.renderer.tokenizer
        if tokenizer is None:
            raise RuntimeError("tokenizer is not initialized")
        reasoning_parser = None
        try:
            if self.reasoning_parser_cls:
                chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                    request.chat_template_kwargs,
                    self.default_chat_template_kwargs,
                )
                reasoning_parser = self.reasoning_parser_cls(
                    tokenizer,
                    chat_template_kwargs=chat_template_kwargs,
                )
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            return self.create_error_response(str(e))
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_prompts = result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(
                request, supports_default_mm_loras=True
            )

            model_name = self.models.model_name(lora_request)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.exception("Error preparing request components")
            return self.create_error_response(e)

        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_token_ids = self._extract_prompt_components(
                    engine_prompt
                ).token_ids

                sub_request_id = (
                    request_id if len(engine_prompts) == 1 else f"{request_id}_{i}"
                )

                max_tokens = get_max_tokens(
                    max_model_len,
                    request.max_completion_tokens
                    if request.max_completion_tokens is not None
                    else request.max_tokens,
                    self._extract_prompt_len(engine_prompt),
                    self.default_sampling_params,
                    self.override_max_tokens,
                )

                sampling_params: SamplingParams | BeamSearchParams
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params
                    )
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.default_sampling_params,
                    )

                sharing_cache_salt = (
                    request.cache_salt if _cache_sharing_on(request) else None
                )
                if sharing_cache_salt is not None:
                    sub_request_id = pack_request_sharing_cache_salt(
                        sub_request_id, sharing_cache_salt
                    )

                self._log_inputs(
                    sub_request_id,
                    engine_prompt,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await self._get_trace_headers(raw_request.headers)
                )

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.beam_search(
                        prompt=engine_prompt,
                        request_id=sub_request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                    )
                else:
                    reasoning_ended = (
                        reasoning_parser.is_reasoning_end(prompt_token_ids or [])
                        if reasoning_parser
                        else None
                    )

                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        sub_request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        data_parallel_rank=data_parallel_rank,
                        reasoning_ended=reasoning_ended,
                    )

                generators.append(generator)
        except ValueError as e:
            return self.create_error_response(e)

        if len(generators) != 1:
            raise RuntimeError(
                f"Expected exactly one generator, got {len(generators)}"
            )
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )

        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )
        except GenerationError as e:
            return self._convert_generation_error_to_response(e)
        except ValueError as e:
            return self.create_error_response(e)

    async def release_kv_cache(
        self, request: ReleaseKvCacheRequest, raw_request: Request
    ) -> Union[ReleaseKvCacheResponse, ErrorResponse]:
        if not request.cache_salt:
            return self.create_error_response(
                "cache_salt is required for release_kv_cache"
            )

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            # 算了 lora_request，但从头到尾没用到，可以删掉
            # lora_request = self._maybe_get_adapters(
            #     request, supports_default_mm_loras=True
            # )

            #tool_dicts = (
            #    None
            #    if request.tools is None
            #    else [tool.model_dump() for tool in request.tools]
            #)

            # retained_message_count = 保留的 message 条数（从头保留几条），其余的对应 KV 被释放
            message_length = len(request.messages)
            retained_message_count = (
                request.messages_released_index
                if request.messages_released_index > 0
                else 0
            )
            retained_message_count = min(retained_message_count, message_length)

            # after_tool_dicts = tool_dicts
            # if after_tool_dicts is not None and request.tools_released_index is not None:
            #     after_tool_dicts = tool_dicts[: request.tools_released_index]
            # tool_changed = after_tool_dicts is not None and tool_dicts is not None and (
            #     len(tool_dicts) != len(after_tool_dicts)
            # )

            # if message_begin >= message_length and not tool_changed:
            #     return ReleaseKvCacheResponse(
            #         cache_salt=request.cache_salt, block_released=0
            #     )

            # 保留有用的短路检查，并且将tools部分去除：
            # 没有要释放的 message（保留数 >= 总数）时直接返回 0，省去两次渲染
            if retained_message_count >= message_length:
                return ReleaseKvCacheResponse(
                    cache_salt=request.cache_salt, block_released=0
                )

            if self.use_harmony:
                logger.error("harmony not supported for release_kv_cache")
                return self.create_error_response("harmony not supported")

            # complete = 释放前的完整对话（full）
            complete_render = await self.render_chat_request(request)
            if isinstance(complete_render, ErrorResponse):
                return complete_render

            # 旧的partial_req里面还保留了有关tools的逻辑，虽然不会走，但是会产生误导
            # partial_req = request.model_copy(
            #     update={
            #         "messages": ([] if message_begin == 0 else request.messages[:message_begin]),
            #         "tools": request.tools[:request.tools_released_index] if (
            #                 request.tools is not None and request.tools_released_index is not None
            #         ) else request.tools,
            #     },
            #     deep=False,
            # )

            # prefix = 释放后保留下来的前缀部分（只保留前 retained_message_count 条 message，tools 原样透传）
            prefix_req = request.model_copy(
                update={"messages": [] if retained_message_count == 0 else request.messages[:retained_message_count]},
                deep=False,
            )
            prefix_render = await self.render_chat_request(prefix_req)
            if isinstance(prefix_render, ErrorResponse):
                return prefix_render

            (_, complete_prompts) = complete_render
            (_, prefix_prompts) = prefix_render
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_id = f"chatcmpl-{self._base_request_id(raw_request, str(time.time()))}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        if len(complete_prompts) != 1 or len(prefix_prompts) != 1:
            error_msg = (
                f"engine prompts should be 1, while complete is {len(complete_prompts)} "
                f"and prefix is {len(prefix_prompts)}"
            )
            logger.error(error_msg)
            return self.create_error_response(error_msg)

        complete_token_ids: list[int] = []
        for engine_prompt in complete_prompts:
            if (
                isinstance(engine_prompt, dict)
                and _PROMPT_TOKEN_IDS_KEY in engine_prompt
            ):
                complete_token_ids.extend(engine_prompt[_PROMPT_TOKEN_IDS_KEY])

        prefix_token_ids: list[int] = []
        for engine_prompt in prefix_prompts:
            if (
                isinstance(engine_prompt, dict)
                and _PROMPT_TOKEN_IDS_KEY in engine_prompt
            ):
                prefix_token_ids.extend(engine_prompt[_PROMPT_TOKEN_IDS_KEY])

        if len(complete_token_ids) <= len(prefix_token_ids):
            raise ValueError(
                f"complete_token_ids length ({len(complete_token_ids)}) must be "
                f"greater than prefix_token_ids length ({len(prefix_token_ids)})"
            )
        # released_token_index = len(after_token_ids)
        # if tool_changed:
        #     for i in range(len(after_token_ids)):
        #         if before_token_ids[i] != after_token_ids[i]:
        #             released_token_index = i
        #             break
        # elif len(after_token_ids) > 0:
        #     last_match_index_after = 0
        #     last_match_index_before = 0
        #     match_count = len(after_token_ids)
        #     eos_token_count = 5
        #     if match_count > eos_token_count:
        #         match_count -= eos_token_count
        #     for idx, val in enumerate(before_token_ids):
        #         if val == after_token_ids[last_match_index_after]:
        #             last_match_index_before = idx
        #             last_match_index_after += 1
        #             if last_match_index_after >= match_count:
        #                 break
        #     released_token_index = last_match_index_before + 1

        # 计算 Token 边界：把 message 级的“保留几条”翻译成 token 级的释放起点。
        # 在 complete 中顺序匹配 prefix 的前缀，匹配到的最后位置 +1 即释放起点：
        # 前 release_start_index 个 token 保留，其后的 token 对应 KV 被释放。
        # prefix 一定非空（至少保留 system）。
        release_start_index = len(prefix_token_ids)
        if len(prefix_token_ids) > 0:
            last_match_index_prefix = 0
            last_match_index_complete = 0
            match_count = len(prefix_token_ids)
            if match_count > _EOS_TOKEN_COUNT:
                match_count -= _EOS_TOKEN_COUNT
            for idx, val in enumerate(complete_token_ids):
                if val == prefix_token_ids[last_match_index_prefix]:
                    last_match_index_complete = idx
                    last_match_index_prefix += 1
                    if last_match_index_prefix >= match_count:
                        break
            release_start_index = last_match_index_complete + 1

        supported_tasks = await self.engine_client.get_supported_tasks()
        request_params: list[tuple[Sequence[bytestr], int]] = []

        # ===== 旧的多段打包循环（保留供参考）=====
        # 该循环为“prompt 可能被拆成多段”的通用场景设计，用 elapsed_tokens 做跨段游标、
        # 把全局释放偏移换算成段内局部偏移。但上面已强制 complete_prompts / prefix_prompts
        # 都只有 1 段，循环实际只执行一次、游标恒为 0，故退化为下方的单段直线逻辑。
        # try:
        #     elapsed_tokens = 0
        #     for i, engine_prompt in enumerate(before_engine_prompts):
        #         if (
        #             not isinstance(engine_prompt, dict)
        #             or _PROMPT_TOKEN_IDS_KEY not in engine_prompt
        #         ):
        #             continue
        #         prompt_len = len(engine_prompt[_PROMPT_TOKEN_IDS_KEY])
        #         if elapsed_tokens + prompt_len <= released_token_index:
        #             elapsed_tokens += prompt_len
        #             continue
        #
        #         sub_request_id = (
        #             request_id
        #             if len(before_engine_prompts) == 1
        #             else f"{request_id}_{i}"
        #         )
        #
        #         sampling_params = SamplingParams()
        #
        #         trace_headers = (
        #             None
        #             if raw_request is None
        #             else await self._get_trace_headers(raw_request.headers)
        #         )
        #
        #         sharing_cache_salt = (
        #             request.cache_salt if _cache_sharing_on(request) else None
        #         )
        #         if sharing_cache_salt is not None:
        #             sub_request_id = pack_request_sharing_cache_salt(
        #                 sub_request_id, sharing_cache_salt
        #             )
        #
        #         engine_request = self.input_processor.process_inputs(
        #             sub_request_id,
        #             engine_prompt,
        #             sampling_params,
        #             supported_tasks=supported_tasks,
        #             trace_headers=trace_headers,
        #             priority=0,
        #         )
        #
        #         request_params.append(
        #             (
        #                 encode_engine_core_request(engine_request),
        #                 released_token_index - elapsed_tokens,
        #             )
        #         )
        #         elapsed_tokens += prompt_len
        #         released_token_index = elapsed_tokens
        # except ValueError as e:
        #     return self.create_error_response(str(e))

        # ===== 单段直线逻辑 =====
        # complete_prompts 只有唯一一段（前面已校验），直接打包这段：
        # 单段下 elapsed_tokens 恒为 0，局部偏移就等于全局的 release_start_index。
        try:
            engine_prompt = complete_prompts[0]

            # 防御：complete_prompts[0] 理论上必是含 prompt_token_ids 的合法 dict
            #（前面已多重校验），若结构异常则优雅返回 0，不往下报错。
            if (
                not isinstance(engine_prompt, dict)
                or _PROMPT_TOKEN_IDS_KEY not in engine_prompt
            ):
                return ReleaseKvCacheResponse(
                    cache_salt=request.cache_salt, block_released=0
                )

            sub_request_id = request_id
            sharing_cache_salt = (
                request.cache_salt if _cache_sharing_on(request) else None
            )
            if sharing_cache_salt is not None:
                sub_request_id = pack_request_sharing_cache_salt(
                    sub_request_id, sharing_cache_salt
                )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            engine_request = self.input_processor.process_inputs(
                sub_request_id,
                engine_prompt,
                SamplingParams(),
                supported_tasks=supported_tasks,
                trace_headers=trace_headers,
                priority=0,
            )

            request_params.append(
                (
                    encode_engine_core_request(engine_request),
                    release_start_index,
                )
            )
        except ValueError as e:
            return self.create_error_response(str(e))

        released_num = await self.engine_client.release_kv_cache(
            request.cache_salt or "", request_params
        )
        return ReleaseKvCacheResponse(
            cache_salt=request.cache_salt, block_released=released_num
        )


def register_openai_serving():
    OpenAIServingChat.release_kv_cache = KvCacheServingChat.release_kv_cache
    OpenAIServingChat.create_chat_completion = KvCacheServingChat.create_chat_completion
