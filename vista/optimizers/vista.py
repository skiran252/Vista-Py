import asyncio
import random
from typing import Callable, List

from vista.core.module import Predict
from vista.core.example import Example
from vista.llm.client import LLMClient
from vista.agents.hypothesis import HypothesisAgent
from vista.agents.reflection import ReflectionAgent
from vista.evaluation.evaluator import Evaluator
from vista.trace.logger import TraceLogger

class VistaOptimizer:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm_client = LLMClient(model_name)
        self.hypothesis_agent = HypothesisAgent(self.llm_client)
        self.reflection_agent = ReflectionAgent(self.llm_client)
        self.evaluator = Evaluator(self.llm_client)
        self.logger = TraceLogger()

    async def optimize(
        self,
        module: Predict,
        trainset: List[Example],
        metric: Callable[[Example, dict], float],
        epochs: int = 5,
        k_hypotheses: int = 3,
        epsilon: float = 0.1,
        patience: int = 2
    ):
        """
        Runs the VISTA optimization loop.
        """
        print("Starting VISTA Optimization...")
        current_instructions = module.signature.instructions
        
        # Initial evaluation
        best_score, evaluated = await self.evaluator.evaluate_minibatch(module, trainset, metric)
        self.logger.history.append({"iteration": 0, "instructions": current_instructions, "score": best_score})
        print(f"Initial Score: {best_score:.2f}")

        no_improve_count = 0

        for epoch in range(1, epochs + 1):
            strategy = "heuristics"
            
            # 1. Identify failure cases
            failed_examples = [ex for ex in evaluated if metric(ex, ex.actual_output or {}) < 1.0]
            
            # Epsilon greedy / Random Restart logic
            if no_improve_count >= patience or random.random() < epsilon:
                strategy = "random_restart"
                self.logger.log_random_restart(epoch, "Stuck in local optimum or epsilon triggered")
                # Random restart: wipe instructions and ask for a totally different approach
                current_instructions = await self._do_random_restart(current_instructions)
                no_improve_count = 0
                
                # Re-evaluate
                module._parameters["instructions"] = current_instructions
                best_score, evaluated = await self.evaluator.evaluate_minibatch(module, trainset, metric)
                continue

            if not failed_examples:
                print("No failed examples found! Perfect score?")
                break

            # 2. Generate Hypotheses (Decoupled Phase 1)
            hypotheses = await self.hypothesis_agent.generate_hypotheses(
                current_instructions=current_instructions,
                failed_examples=failed_examples[:3], # Use a subset to save context
                num_hypotheses=k_hypotheses
            )
            
            # 3. Rewrite Prompt for each hypothesis (Decoupled Phase 2)
            rewrite_tasks = [
                self.reflection_agent.rewrite_prompt(current_instructions, h, failed_examples[:3])
                for h in hypotheses
            ]
            candidate_prompts = await asyncio.gather(*rewrite_tasks)

            # 4. Parallel Minibatch Verification
            eval_tasks = []
            for prompt in candidate_prompts:
                # Create a temporary module with the candidate prompt
                temp_module = Predict(module.signature)
                temp_module._parameters["instructions"] = prompt
                eval_tasks.append(self.evaluator.evaluate_minibatch(temp_module, trainset, metric))
                
            eval_results = await asyncio.gather(*eval_tasks)
            
            # 5. Select Best Candidate
            best_candidate_idx = -1
            epoch_best_score = -1.0
            for i, (score, _) in enumerate(eval_results):
                if score > epoch_best_score:
                    epoch_best_score = score
                    best_candidate_idx = i
                    
            hyp_log = [{"tag": h.tag, "desc": h.description} for h in hypotheses]

            if epoch_best_score > best_score:
                best_score = epoch_best_score
                current_instructions = candidate_prompts[best_candidate_idx]
                module._parameters["instructions"] = current_instructions
                evaluated = eval_results[best_candidate_idx][1]
                no_improve_count = 0
                
                self.logger.log_iteration(
                    epoch, current_instructions, best_score, strategy,
                    hyp_log, current_instructions, best_score, hypotheses[best_candidate_idx].tag
                )
            else:
                no_improve_count += 1
                self.logger.log_iteration(
                    epoch, current_instructions, best_score, strategy,
                    hyp_log, candidate_prompts[best_candidate_idx], epoch_best_score, ""
                )
                print(f"Iteration {epoch} failed to improve (Best candidate scored {epoch_best_score:.2f} <= {best_score:.2f})")

        print("Optimization Complete.")
        return module

    async def _do_random_restart(self, current_prompt: str) -> str:
        prompt = f"The following prompt is stuck in a local optimum and failing. Rewrite it from scratch with a completely new structural and logical approach. Output only the new prompt.\n\n{current_prompt}"
        response = await self.llm_client.generate_structured(
            messages=[{"role": "user", "content": prompt}],
            response_model=dict # Raw generation for random restart could just use a basic string, but we can stick to structure
        )
        # Actually since LLMClient.generate_structured expects BaseModel, let's just use acompletion
        import litellm
        res = await litellm.acompletion(
            model=self.llm_client.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
