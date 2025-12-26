from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Sample:
    sid: str
    text: str
    reference: str  # for PR1 we keep a placeholder reference


TOY_TEXTS = [
    "Large language models (LLMs) are widely used in modern applications. They can generate text, summarize documents, and answer questions.",
    "SystemDS is a machine learning system designed for flexible and scalable analytics. It supports declarative ML programming and optimization.",
    "Benchmarking inference systems involves measuring latency, throughput, and quality across tasks and models under controlled conditions.",
    "Speculative decoding is a technique to accelerate autoregressive generation by using a smaller draft model and verifying with a larger model.",
    "Reproducible experiments require fixed seeds, versioned configs, and consistent environments across runs.",
    "A good benchmark suite includes diverse workloads such as summarization, question answering, and reasoning tasks.",
    "Local inference can reduce cost and improve privacy, but may be limited by hardware constraints and model support.",
    "Hosted APIs offer strong model quality and easy scaling, but introduce network latency and variable cost per token.",
    "Throughput is typically measured in requests per second or tokens per second, depending on the benchmark design.",
    "Accuracy for summarization can be approximated with overlap metrics, but human evaluation is often the gold standard.",
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset = cfg.get("dataset", {})
    source = dataset.get("source", "toy")
    n = int(dataset.get("n_samples", 10))

    if source != "toy":
        raise ValueError("PR1 supports only dataset.source: toy")

    texts = TOY_TEXTS[: max(1, min(n, len(TOY_TEXTS)))]
    samples: List[Sample] = []
    for i, t in enumerate(texts):
        samples.append(Sample(sid=f"toy-{i}", text=t, reference=""))
    return samples
