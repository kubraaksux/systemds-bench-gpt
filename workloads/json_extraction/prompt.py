from typing import Any, Dict

from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    """
    Format a JSON extraction prompt for the model.
    
    Instructs the model to extract structured information from text
    and return valid JSON with specified fields.
    """
    return (
        "Extract information from the following text and return ONLY valid JSON.\n\n"
        f"Text: {sample.text}\n\n"
        f"Return JSON with these fields: {sample.schema}"
    )
