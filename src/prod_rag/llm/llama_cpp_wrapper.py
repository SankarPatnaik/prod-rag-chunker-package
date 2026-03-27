from __future__ import annotations

from prod_rag.llm.base import BaseLLM
from prod_rag.models import LLMConfig


class LlamaCppLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install llama_cpp extras to use this backend") from exc
        if not config.model_path:
            raise ValueError("llama_cpp backend requires llm.model_path")
        self.model = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.gpu_layers,
            verbose=False,
        )
        self.config = config

    def generate(self, prompt: str) -> str:
        out = self.model.create_completion(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        return out["choices"][0]["text"].strip()
