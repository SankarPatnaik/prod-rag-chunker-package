from __future__ import annotations

from prod_rag.llm.llama_cpp_wrapper import LlamaCppLLM
from prod_rag.llm.vllm_wrapper import VLLMLLM
from prod_rag.models import LLMConfig


def get_llm(config: LLMConfig):
    if config.backend == "vllm":
        return VLLMLLM(config)
    return LlamaCppLLM(config)
