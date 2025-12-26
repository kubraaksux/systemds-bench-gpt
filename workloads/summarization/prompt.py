from typing import Any, Dict
from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    # Keep prompt minimal and stable for PR1
    return (
        "Summarize the following text in 2-3 sentences.\n\n"
        f"{sample.text}\n"
    )
