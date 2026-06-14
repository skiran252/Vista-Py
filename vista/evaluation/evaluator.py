import asyncio
from typing import Callable, List, Tuple

from vista.core.predict import Predict
from vista.core.example import Example
from vista.llm.client import LLMClient


class Evaluator:
    """
    Evaluates prompt candidates with minibatch verification.
    Returns per-sample binary outcomes for accurate Δacc computation.
    """

    def __init__(self, llm_client: LLMClient, max_concurrent: int = 4):
        self.llm_client = llm_client
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_single(
        self,
        module: Predict,
        example: Example,
        metric: Callable[[Example, dict], float],
        max_retries: int = 3,
    ) -> Tuple[float, Example]:
        """Evaluates a single example with retry on rate limits."""
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    await asyncio.sleep(1.0)
                    actual_output = await module(self.llm_client, **example.inputs)
                    evaluated_example = example.with_output(actual_output)
                    score = metric(evaluated_example, actual_output)
                    return score, evaluated_example
                except Exception as e:
                    error_str = str(e).lower()
                    is_retryable = (
                        "rate" in error_str
                        or "429" in error_str
                        or "temporarily" in error_str
                    )
                    if is_retryable and attempt < max_retries - 1:
                        wait = 15 * (2**attempt)
                        print(
                            f"Rate limited, retrying in {wait}s "
                            f"(attempt {attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(wait)
                    else:
                        print(f"Error evaluating example: {e}")
                        return 0.0, example
            return 0.0, example

    async def evaluate_minibatch(
        self,
        module: Predict,
        minibatch: List[Example],
        metric: Callable[[Example, dict], float],
    ) -> Tuple[float, List[Example], List[int]]:
        """
        Evaluates a minibatch and returns:
        - avg_score: mean accuracy across the batch
        - evaluated_examples: examples with actual outputs populated
        - per_sample: binary outcomes per sample (1=correct, 0=incorrect)
        """
        tasks = [self.evaluate_single(module, ex, metric) for ex in minibatch]
        results = await asyncio.gather(*tasks)

        scores = [res[0] for res in results]
        evaluated_examples = [res[1] for res in results]
        # Binary outcome: 1 if score >= 1.0 (fully correct), else 0
        per_sample = [1 if s >= 1.0 else 0 for s in scores]

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, evaluated_examples, per_sample

    async def evaluate_on_val(
        self,
        module: Predict,
        val_set: List[Example],
        metric: Callable[[Example, dict], float],
    ) -> Tuple[float, List[int]]:
        """
        Full evaluation on the validation set for acceptance gating.
        Returns (accuracy, per_sample_outcomes).
        """
        score, _, per_sample = await self.evaluate_minibatch(module, val_set, metric)
        return score, per_sample

    def compute_delta_acc(
        self, parent_per_sample: List[int], candidate_per_sample: List[int]
    ) -> float:
        """Compute Δacc = acc(candidate, M) - acc(parent, M) on the same minibatch."""
        if not parent_per_sample or not candidate_per_sample:
            return 0.0
        parent_acc = sum(parent_per_sample) / len(parent_per_sample)
        candidate_acc = sum(candidate_per_sample) / len(candidate_per_sample)
        return candidate_acc - parent_acc
