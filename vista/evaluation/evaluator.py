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
        metric: Callable[[Example, dict], float]
    ) -> Tuple[float, Example]:
        """Evaluates a single example."""
        async with self.semaphore:
            try:
                # Add a delay to help with GitHub Models burst rate limits
                await asyncio.sleep(2.0)
                actual_output = await module(self.llm_client, **example.inputs)
                evaluated_example = example.with_output(actual_output)
                score = metric(evaluated_example, actual_output)
                return score, evaluated_example
            except Exception as e:
                print(f"Error evaluating example: {e}")
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
