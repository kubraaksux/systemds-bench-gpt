import time
from typing import Any, Dict, List

from mlx_lm import load, generate


class MLXBackend:
    def __init__(self, model: str):
        self.model, self.tokenizer = load(model)

    def generate(self, prompts: List[str], config: Dict[str, Any]):
        max_tokens = int(config.get("max_tokens", 128))
        temperature = float(config.get("temperature", 0.0))

        results = []
        for p in prompts:
            t0 = time.perf_counter()
            out = generate(
                self.model,
                self.tokenizer,
                p,
                max_tokens=max_tokens,
                sampler=greedy_sampler,
            )
            t1 = time.perf_counter()
            results.append({"text": out, "latency_ms": (t1 - t0) * 1000.0})
        return results
