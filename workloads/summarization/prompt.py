from typing import Any, Dict
from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    return (
        "Summarize the following text in exactly 2-3 sentences. "
        "Do not add any facts that are not present in the original text.\n\n"
        f"{sample.text}\n"
    )
