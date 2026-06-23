from vllm.logger import init_logger

logger = init_logger(__name__)

from kv_cache_affinity.engine.protocol import (
    register_engine_protocol,
)
import kv_cache_affinity.entrypoints.openai.api_server
from kv_cache_affinity.entrypoints.openai.protocol import (
    register_chat_request,
)
from kv_cache_affinity.entrypoints.openai.serving_chat import (
    register_openai_serving,
)
from kv_cache_affinity.v1.core.sched.interface import (
    register_scheduler_interface,
)
from kv_cache_affinity.v1.core.sched.scheduler import (
    register_scheduler,
)
from kv_cache_affinity.v1.core.block_pool import register_block_pool
from kv_cache_affinity.v1.core.input_processor import (
    register_input_processor,
)
from kv_cache_affinity.v1.core.kv_cache_coordinator import (
    register_kv_cache_coordinator,
)
from kv_cache_affinity.v1.core.kv_cache_manager import (
    register_kv_cache_manager,
)
from kv_cache_affinity.v1.core.single_type_kv_cache_manager import (
    register_single_type_kv_cache_manager,
)
from kv_cache_affinity.v1.engine.async_llm import (
    register_engine_client,
)
from kv_cache_affinity.v1.engine.core import register_engine_core
from kv_cache_affinity.v1.engine.core_client import (
    register_engine_core_client,
)
from kv_cache_affinity.v1.request import register_request


def apply_patches():
    register_engine_protocol()
    register_chat_request()
    register_input_processor()
    register_openai_serving()
    register_scheduler_interface()
    register_scheduler()
    register_block_pool()
    register_single_type_kv_cache_manager()
    register_kv_cache_coordinator()
    register_kv_cache_manager()
    register_engine_client()
    register_engine_core()
    register_engine_core_client()
    register_request()
