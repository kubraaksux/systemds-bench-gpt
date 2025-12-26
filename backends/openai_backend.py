import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI



class OpenAIBackend:
    """
    Uses the OpenAI Responses API by default (recommended for new projects).
    Stores latency and, when available, usage/cost-related fields in `extra`.
    """

    def __init__(self, api_key: str | None = None):
        load_dotenv()  # loads .env from repo root
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompts: List[str], config: Dict[str, Any]):
        model = config.get("model", "gpt-4.1-mini")  # safe default; override via config
        max_output_tokens = int(config.get("max_output_tokens", 256))
        # For benchmarking, keep temperature deterministic unless you explicitly vary it.
        temperature = config.get("temperature", 0.0)

        # Simple retry/backoff for transient errors / rate limits
        max_retries = int(config.get("max_retries", 5))
        base_sleep = float(config.get("base_sleep_s", 0.5))

        results = []

        for prompt in prompts:
            last_err = None
            for attempt in range(max_retries):
                try:
                    t0 = time.perf_counter()
                    resp = self.client.responses.create(
                        model=model,
                        input=prompt,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    )
                    t1 = time.perf_counter()

                    # Extract text output
                    text = ""
                    try:
                        # SDK commonly provides convenience:
                        text = resp.output_text
                    except Exception:
                        # Fallback: be defensive
                        text = str(resp)

                    extra: Dict[str, Any] = {}

                    # Usage fields vary by endpoint/version; keep raw usage if present
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        # Make it JSON-serializable
                        if hasattr(usage, "model_dump"):
                            extra["usage"] = usage.model_dump()
                        elif hasattr(usage, "dict"):
                            extra["usage"] = usage.dict()
                        else:
                            extra["usage"] = str(usage)


                    # Also store response id for traceability
                    extra["response_id"] = getattr(resp, "id", None)

                    results.append(
                        {
                            "text": text,
                            "latency_ms": (t1 - t0) * 1000.0,
                            "extra": extra,
                        }
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(base_sleep * (2**attempt))

            if last_err is not None:
                # Fail fast per-sample with explicit error stored
                results.append(
                    {
                        "text": "",
                        "latency_ms": 0.0,
                        "extra": {"error": repr(last_err)},
                    }
                )

        return results
