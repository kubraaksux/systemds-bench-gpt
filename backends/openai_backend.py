import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


# Pricing per million tokens (USD)
# Reference: https://openai.com/api/pricing/
PRICING = {
    "gpt-4.1-mini": {
        "input": 0.40,   # $0.40 per 1M input tokens
        "output": 1.60,  # $1.60 per 1M output tokens
    },
    "gpt-4.1-mini-2025-04-14": {
        "input": 0.40,
        "output": 1.60,
    },
    "gpt-4.1": {
        "input": 2.00,   # $2.00 per 1M input tokens
        "output": 8.00,  # $8.00 per 1M output tokens
    },
    "gpt-4.1-2025-04-14": {
        "input": 2.00,
        "output": 8.00,
    },
    "gpt-4.1-nano": {
        "input": 0.10,   # $0.10 per 1M input tokens
        "output": 0.40,  # $0.40 per 1M output tokens
    },
    "gpt-4.1-nano-2025-04-14": {
        "input": 0.10,
        "output": 0.40,
    },
    "gpt-4o": {
        "input": 2.50,   # $2.50 per 1M input tokens
        "output": 10.00, # $10.00 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,   # $0.15 per 1M input tokens
        "output": 0.60,  # $0.60 per 1M output tokens
    },
}


class OpenAIBackend:
    """
    Uses the OpenAI Responses API by default (recommended for new projects).
    Stores latency and, when available, usage/cost-related fields in `extra`.
    """

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()  # loads .env from repo root
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        model = config.get("model", "gpt-4.1-mini")  # safe default; override via config
        max_output_tokens = int(config.get("max_output_tokens", 256))
        # For benchmarking, keep temperature deterministic unless you explicitly vary it.
        temperature = config.get("temperature", 0.0)
        
        # NEW: Check if streaming mode is enabled for TTFT measurement
        use_streaming = config.get("streaming", False)

        # Simple retry/backoff for transient errors / rate limits
        max_retries = int(config.get("max_retries", 5))
        base_sleep = float(config.get("base_sleep_s", 0.5))

        results = []

        for prompt in prompts:
            last_err = None
            for attempt in range(max_retries):
                try:
                    if use_streaming:
                        # STREAMING MODE: Measure TTFT
                        result = self._generate_streaming(
                            prompt, model, max_output_tokens, temperature
                        )
                    else:
                        # NON-STREAMING MODE: Current behavior
                        result = self._generate_non_streaming(
                            prompt, model, max_output_tokens, temperature
                        )
                    
                    results.append(result)
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
    
    def _generate_non_streaming(self, prompt: str, model: str, max_output_tokens: int, temperature: float) -> Dict[str, Any]:
        """Non-streaming mode: measures total latency only (current behavior)"""
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
            text = resp.output_text
        except Exception:
            text = str(resp)

        extra: Dict[str, Any] = {}

        # Usage fields vary by endpoint/version; keep raw usage if present
        usage = getattr(resp, "usage", None)
        usage_data = None
        if usage is not None:
            usage_data = self._extract_usage(usage)
            if usage_data is not None:
                extra["usage"] = usage_data
                # Calculate cost based on usage
                cost = self._calculate_cost(usage_data, model)
                if cost is not None:
                    extra["cost_usd"] = cost

        # Also store response id for traceability
        extra["response_id"] = getattr(resp, "id", None)

        return {
            "text": text,
            "latency_ms": (t1 - t0) * 1000.0,
            "extra": extra,
        }
    
    def _generate_streaming(self, prompt: str, model: str, max_output_tokens: int, temperature: float) -> Dict[str, Any]:
        """Streaming mode: measures TTFT and generation time separately"""
        t0 = time.perf_counter()
        stream = self.client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stream=True,
        )
        
        t_first = None
        t_final = None
        full_text = ""
        response_id = None
        usage_data = None
        
        for event in stream:
            # Listen for the first delta event
            if event.type == "response.output_text.delta":
                if t_first is None:
                    t_first = time.perf_counter()  # ← TTFT!
                full_text += event.delta
            
            elif event.type == "response.completed":
                t_final = time.perf_counter()
                # The response.completed event contains the full response object with usage
                response = getattr(event, "response", None)
                if response is not None:
                    response_id = getattr(response, "id", None)
                    # Extract usage data (same format as non-streaming)
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        usage_data = self._extract_usage(usage)
                else:
                    # Fallback: try to get response_id directly from event
                    response_id = getattr(event, "response_id", None) or getattr(event, "id", None)
                    # Also check if usage is on the event itself
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        usage_data = self._extract_usage(usage)
        
        # Fallback: check if stream object has usage after iteration completes
        # (Some OpenAI SDK versions expose usage on the stream object)
        if usage_data is None:
            stream_usage = getattr(stream, "usage", None)
            if stream_usage is not None:
                usage_data = self._extract_usage(stream_usage)
        
        # Handle case where no tokens were generated
        if t_first is None:
            t_first = time.perf_counter()
        if t_final is None:
            t_final = time.perf_counter()
        
        # Calculate metrics
        ttft_ms = (t_first - t0) * 1000.0
        generation_ms = (t_final - t_first) * 1000.0
        total_latency_ms = (t_final - t0) * 1000.0
        
        extra: Dict[str, Any] = {
            "ttft_ms": ttft_ms,
            "generation_ms": generation_ms,
            "response_id": response_id,
        }
        
        # Add usage data if available (same format as non-streaming)
        if usage_data is not None:
            extra["usage"] = usage_data
            # Calculate cost based on usage
            cost = self._calculate_cost(usage_data, model)
            if cost is not None:
                extra["cost_usd"] = cost
        
        return {
            "text": full_text,
            "latency_ms": total_latency_ms,  # Keep this for backwards compatibility
            "extra": extra,
        }
    
    def _extract_usage(self, usage: Any) -> Optional[Dict[str, Any]]:
        """
        Extract usage data in a consistent format.
        
        Expected structure (when available):
        {
            "total_tokens": int,
            "input_tokens": int,
            "output_tokens": int,
            "input_tokens_details": {...},
            "output_tokens_details": {...}
        }
        """
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        elif hasattr(usage, "dict"):
            return usage.dict()
        elif isinstance(usage, dict):
            return usage
        else:
            # Fallback for unexpected types
            return {"raw": str(usage)}
    
    def _calculate_cost(self, usage_data: Optional[Dict[str, Any]], model: str) -> Optional[float]:
        """
        Calculate cost in USD based on token usage and model pricing.
        
        Returns None if pricing is not available for the model or usage data is missing.
        """
        if usage_data is None:
            return None
        
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        
        if input_tokens == 0 and output_tokens == 0:
            return None
        
        # Look up pricing for the model
        prices = PRICING.get(model)
        if prices is None:
            return None
        
        # Calculate cost: tokens * price_per_million / 1,000,000
        cost = (
            input_tokens * prices["input"] / 1_000_000 +
            output_tokens * prices["output"] / 1_000_000
        )
        return cost