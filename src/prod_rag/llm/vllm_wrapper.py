from __future__ import annotations

from prod_rag.llm.base import BaseLLM
from prod_rag.models import LLMConfig


class VLLMLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install vllm extras to use this backend") from exc
        if not config.model_name:
            raise ValueError("vllm backend requires llm.model_name")
        self.model = LLM(model=config.model_name)
        self.sampling = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.model.generate([prompt], self.sampling)
        return outputs[0].outputs[0].text.strip()
