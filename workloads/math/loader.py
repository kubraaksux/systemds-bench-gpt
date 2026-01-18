import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class Sample:
    sid: str
    question: str
    reference: str  # The ground truth final numerical answer


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    """
    Load math problems from GSM8K dataset.
    
    GSM8K structure:
    - question: The math word problem
    - answer: Full solution with reasoning, ending with "#### <number>"
    """
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "gsm8k")
    n = int(dataset_cfg.get("n_samples", 10))
    
    if source != "gsm8k":
        raise ValueError(f"Math workload only supports source: gsm8k, got: {source}")
    
    # Load GSM8K dataset from HuggingFace
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Sample N problems (take first N for reproducibility, or use random sampling)
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        
        question = item["question"]
        answer_text = item["answer"]
        
        # Extract the final numerical answer after "####"
        final_answer = extract_final_answer(answer_text)
        
        samples.append(Sample(
            sid=f"gsm8k-{i}",
            question=question,
            reference=final_answer,
        ))
    
    return samples


def extract_final_answer(answer_text: str) -> str:
    """
    Extract the final numerical answer from GSM8K answer format.
    
    GSM8K answers end with "#### <number>" where <number> is the final answer.
    Example: "... So the answer is 42. #### 42"
    """
    # Look for #### followed by the answer
    match = re.search(r"####\s*(.+?)$", answer_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def extract_number_from_prediction(prediction: str) -> Optional[str]:
    """
    Extract the final numerical answer from a model's prediction.
    
    Tries multiple strategies:
    1. Look for "#### <number>" format (if model follows GSM8K format)
    2. Look for LaTeX \boxed{number} format
    3. Look for bold markdown with optional $ and units: **$18**, **540 meters**
    4. Look for "final answer is <number>" or similar patterns
    5. Look for the last number in the response
    """
    prediction = prediction.strip()
    
    # Strategy 1: GSM8K format "#### <number>"
    match = re.search(r"####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", prediction)
    if match:
        return match.group(1).replace(",", "")
    
    # Strategy 2: LaTeX \boxed{number} format
    match = re.search(r"\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\}", prediction)
    if match:
        return match.group(1).replace(",", "")
    
    # Strategy 3: Bold markdown with optional currency/units: **$18**, **540 meters**, **$70,000**
    # Match **$X** or **X units** patterns
    match = re.search(r"\*\*\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*[a-zA-Z]*\s*\*\*", prediction)
    if match:
        return match.group(1).replace(",", "")
    
    # Strategy 4: Common patterns like "the answer is X" or "= X"
    patterns = [
        # "answer is $18" or "answer: 42"
        r"(?:final\s+)?answer\s*(?:is|:)\s*\*{0,2}\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        # "therefore 42" or "thus, the answer is 42"
        r"(?:therefore|thus|so|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?\*{0,2}\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        # "= 42" at end of line
        r"=\s*\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
        # "makes **$18**" or similar
        r"(?:makes?|earns?|gets?|pays?|costs?|is)\s+\*{0,2}\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\*{0,2}",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")
    
    # Strategy 5: Last number in the response (fallback)
    # But prefer numbers that appear after key phrases
    numbers = re.findall(r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", prediction)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def normalize_number(num_str: str) -> Optional[float]:
    """
    Normalize a number string for comparison.
    Handles commas, negative signs, and decimals.
    """
    if not num_str:
        return None
    
    try:
        # Remove commas and whitespace
        cleaned = num_str.replace(",", "").strip()
        return float(cleaned)
    except ValueError:
        return None


def accuracy_check(prediction: str, reference: str) -> bool:
    """
    Check if the prediction's final answer matches the ground truth.
    
    Args:
        prediction: The model's full response text
        reference: The ground truth final answer (from GSM8K)
    
    Returns:
        True if the extracted answer matches the reference, False otherwise
    """
    # Extract the final number from the prediction
    pred_answer = extract_number_from_prediction(prediction)
    
    if pred_answer is None:
        return False
    
    # Normalize both numbers for comparison
    pred_num = normalize_number(pred_answer)
    ref_num = normalize_number(reference)
    
    if pred_num is None or ref_num is None:
        # Fall back to string comparison if normalization fails
        return pred_answer.strip() == reference.strip()
    
    # Compare as floats (handles cases like "42.0" vs "42")
    # Use a small tolerance for floating point comparison
    return abs(pred_num - ref_num) < 1e-9
