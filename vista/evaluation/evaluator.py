import asyncio
from typing import Callable, List, Tuple
from vista.core.module import Predict
from vista.core.example import Example
from vista.llm.client import LLMClient

class Evaluator:
    def __init__(self, llm_client: LLMClient, max_concurrent: int = 2):
        self.llm_client = llm_client
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_single(
        self, 
        module: Predict, 
        example: Example, 
        metric: Callable[[Example, dict], float],
        max_retries: int = 3
    ) -> Tuple[float, Example]:
        """Evaluates a single example with retry on rate limits."""
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    await asyncio.sleep(2.0)
                    actual_output = await module(self.llm_client, **example.inputs)
                    evaluated_example = example.with_output(actual_output)
                    score = metric(evaluated_example, actual_output)
                    return score, evaluated_example
                except Exception as e:
                    error_str = str(e).lower()
                    is_retryable = "rate" in error_str or "429" in error_str or "temporarily" in error_str
                    if is_retryable and attempt < max_retries - 1:
                        wait = 2 ** attempt * 5  # 5s, 10s, 20s
                        print(f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"Error evaluating example: {e}")
                        return 0.0, example
            return 0.0, example

    async def evaluate_minibatch(
        self, 
        module: Predict, 
        dataset: List[Example], 
        metric: Callable[[Example, dict], float]
    ) -> Tuple[float, List[Example]]:
        """Evaluates a batch of examples in parallel (parallel minibatch verification)."""
        tasks = [self.evaluate_single(module, ex, metric) for ex in dataset]
        results = await asyncio.gather(*tasks)
        
        scores = [res[0] for res in results]
        evaluated_examples = [res[1] for res in results]
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, evaluated_examples
