import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class Sample:
    sid: str
    question: str
    reference: str  # The final numerical answer


# Toy problems for quick testing
TOY_PROBLEMS = [
    {"question": "What is 15 + 27?", "answer": "42"},
    {"question": "A baker has 48 cupcakes. She sells 23. How many are left?", "answer": "25"},
    {"question": "If a train travels 60 miles per hour for 3 hours, how far does it go?", "answer": "180"},
    {"question": "Tom has 5 apples. He buys 3 more bags with 4 apples each. How many apples does he have?", "answer": "17"},
    {"question": "A rectangle has length 8 and width 5. What is the area?", "answer": "40"},
    {"question": "If 3 notebooks cost $12, how much do 7 notebooks cost?", "answer": "28"},
    {"question": "Sarah has 100 stickers. She gives 15 to each of her 4 friends. How many does she have left?", "answer": "40"},
    {"question": "A bus can hold 45 passengers. How many buses are needed for 200 passengers?", "answer": "5"},
    {"question": "What is 25% of 80?", "answer": "20"},
    {"question": "If you divide 144 by 12, what do you get?", "answer": "12"},
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    """
    Load math problem samples.
    
    Supports multiple sources:
    - "toy": Use built-in toy problems (10 simple problems)
    - "gsm8k": Use GSM8K dataset (grade school math problems)
    """
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "toy")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "toy":
        return _load_toy_samples(n)
    elif source == "gsm8k":
        return _load_gsm8k_samples(n)
    else:
        raise ValueError(f"math supports source: toy, gsm8k. Got: {source}")


def _load_toy_samples(n: int) -> List[Sample]:
    """Load from built-in toy problems."""
    problems = TOY_PROBLEMS[: max(1, min(n, len(TOY_PROBLEMS)))]
    samples: List[Sample] = []
    for i, p in enumerate(problems):
        samples.append(Sample(
            sid=f"toy-{i}",
            question=p["question"],
            reference=p["answer"],
        ))
    return samples


def _load_gsm8k_samples(n: int) -> List[Sample]:
    """
    Load from GSM8K dataset.
    
    GSM8K contains grade school math problems with step-by-step solutions.
    Each problem has a question and a final numerical answer.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        
        question = item["question"]
        answer_text = item["answer"]
        
        # GSM8K answers are formatted as step-by-step solution ending with #### final_answer
        # Extract the final numerical answer after ####
        final_answer = extract_gsm8k_answer(answer_text)
        
        if final_answer is not None:
            samples.append(Sample(
                sid=f"gsm8k-{i}",
                question=question,
                reference=final_answer,
            ))
    
    return samples


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """
    Extract the final numerical answer from GSM8K answer format.
    
    GSM8K answers end with "#### <number>"
    Example: "...The answer is #### 42"
    """
    # Look for #### followed by the answer
    match = re.search(r'####\s*([0-9,.\-]+)', answer_text)
    if match:
        # Remove commas from numbers like "1,000"
        return match.group(1).replace(',', '')
    return None


def extract_number_from_response(text: str) -> Optional[str]:
    """
    Extract the final numerical answer from model response.
    
    Strategies (in order of priority):
    1. Look for explicit answer patterns ("the answer is X", "#### X")
    2. Look for bolded/boxed answers (**X**, \\boxed{X})
    3. Look for "= X" patterns in final lines
    4. Take the last standalone number in the response
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Helper to clean number string (remove trailing periods, commas)
    def clean_num(s: str) -> str:
        s = s.replace(',', '').strip()
        # Remove trailing period if it's not a decimal
        if s.endswith('.') and s.count('.') == 1:
            s = s[:-1]
        return s
    
    # Strategy 1: Look for explicit "answer is" patterns (highest priority)
    answer_patterns = [
        r'####\s*\$?([0-9,]+(?:\.[0-9]+)?)',  # GSM8K format: #### 42
        r'(?:the\s+)?(?:final\s+)?answer\s+(?:is|=)[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)',
        r'(?:final\s+)?answer[:\s]+\$?([0-9,]+(?:\.[0-9]+)?)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the last match (final answer)
            return clean_num(matches[-1])
    
    # Strategy 2: Look for bolded/boxed answers (common LLM format)
    bold_patterns = [
        r'\*\*\$?([0-9,]+(?:\.[0-9]+)?)(?:\s*[a-zA-Z]*)?\*\*',  # **45** or **45 miles** or **$45**
        r'\\boxed\{([0-9,]+(?:\.[0-9]+)?)\}',  # LaTeX boxed
    ]
    
    for pattern in bold_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the last bolded number
            return clean_num(matches[-1])
    
    # Strategy 3: Look for "= X" at end of lines (calculation results)
    # Check last few lines for "= number" patterns
    lines = text.split('\n')
    for line in reversed(lines[-10:]):  # Check last 10 lines
        match = re.search(r'=\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*$', line.strip())
        if match:
            return clean_num(match.group(1))
    
    # Strategy 4: Look for the last number followed by period/end (sentence-ending number)
    # This catches "...makes $18." or "...total of 260."
    matches = re.findall(r'\$?([0-9,]+(?:\.[0-9]+)?)\s*[.!?]?\s*$', text, re.MULTILINE)
    if matches:
        return clean_num(matches[-1])
    
    # Final fallback: any number (take the last one)
    numbers = re.findall(r'\b([0-9,]+(?:\.[0-9]+)?)\b', text)
    if numbers:
        return clean_num(numbers[-1])
    
    return None


def normalize_number(num_str: str) -> Optional[float]:
    """
    Normalize a number string to a float for comparison.
    Handles integers, decimals, and negative numbers.
    """
    if not num_str:
        return None
    try:
        # Remove commas and whitespace
        num_str = num_str.replace(',', '').strip()
        return float(num_str)
    except ValueError:
        return None


def accuracy_check(prediction: str, reference: str) -> bool:
    """
    Check if the predicted answer matches the reference answer.
    
    Extracts the final numerical answer from the prediction
    and compares it with the reference (exact numerical match).
    
    Args:
        prediction: The model's full response
        reference: The correct numerical answer
    
    Returns:
        True if the extracted answer matches, False otherwise
    """
    if not prediction or not reference:
        return False
    
    # Extract number from prediction
    pred_num_str = extract_number_from_response(prediction)
    if pred_num_str is None:
        return False
    
    # Normalize both numbers for comparison
    pred_num = normalize_number(pred_num_str)
    ref_num = normalize_number(reference)
    
    if pred_num is None or ref_num is None:
        return False
    
    # Exact match (with small tolerance for floating point)
    return abs(pred_num - ref_num) < 1e-6
