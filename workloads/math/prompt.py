from typing import Any, Dict

from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    """
    Format a GSM8K math problem as a prompt for the model.
    
    Instructs the model to solve step-by-step and provide a clear final answer.
    """
    return (
        "Solve this math problem step-by-step. "
        "Show your work and give the final numerical answer.\n\n"
        f"Problem: {sample.question}"
    )
