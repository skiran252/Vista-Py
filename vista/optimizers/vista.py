import asyncio
import random
from typing import Callable, List, Optional

from vista.core.predict import Predict
from vista.core.example import Example
from vista.core.config import VistaConfig
from vista.core.dataset import Dataset
from vista.llm import LLMClient
from vista.agents.hypothesis import HypothesisAgent, Hypothesis
from vista.agents.reflection import ReflectionAgent
from vista.evaluation import Evaluator
from vista.trace import TraceLogger


class VistaOptimizer:
    """
    VISTA Prompt Optimization — Algorithm 1 from the paper.

    Key differences from the simplified v0:
    - Budget-based termination (metric call counting)
    - Minibatch sampling from D_train
    - Two-layer explore-exploit (Layer 1: restart with prob p, Layer 2: ε-greedy per hypothesis)
    - Δacc-based candidate selection
    - Validation-gated acceptance
    - Semantic trace tree fed to hypothesis agent
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_base: str = None,
        config: VistaConfig = None,
    ):
        self.config = config or VistaConfig()
        self.llm_client = LLMClient(model_name, api_base=api_base, rpm=self.config.requests_per_minute)
        self.hypothesis_agent = HypothesisAgent(self.llm_client)
        self.reflection_agent = ReflectionAgent(self.llm_client)
        self.evaluator = Evaluator(self.llm_client, max_concurrent=self.config.max_workers)
        self.logger = TraceLogger()
        self._rng = random.Random(self.config.random_seed)

    async def optimize(
        self,
        module: Predict,
        trainset: List[Example],
        metric: Callable[[Example, dict], float],
        # Legacy params kept for backward compat; config values take precedence
        epochs: int = None,
        k_hypotheses: int = None,
        epsilon: float = None,
        patience: int = None,
    ) -> Predict:
        """Runs the VISTA optimization loop (Algorithm 1)."""
        cfg = self.config
        K = k_hypotheses or cfg.k
        eps = epsilon if epsilon is not None else cfg.epsilon
        p = cfg.restart_prob
        b = cfg.minibatch_size

        # Split dataset into train and val
        dataset = Dataset(
            trainset,
            train_size=cfg.train_size,
            val_size=cfg.val_size,
            seed=cfg.random_seed,
        )

        print(f"Starting VISTA Optimization (budget={cfg.budget}, K={K}, p={p}, ε={eps})")
        print(f"  Train: {len(dataset.train)} examples, Val: {len(dataset.val)} examples")

        current_instructions = module.signature.instructions
        budget_remaining = cfg.budget

        # Initial evaluation on val set to establish baseline
        init_score, init_per_sample = await self._evaluate_prompt(
            module, current_instructions, dataset.val, metric
        )
        budget_remaining -= len(dataset.val)

        # Initialize trace tree
        current_node_id = self.logger.tree.add_root(
            current_instructions, init_score, init_per_sample
        )
        best_score = init_score
        best_instructions = current_instructions

        print(f"Initial Score: {init_score:.2f} (budget remaining: {budget_remaining})")

        iteration = 0
        while budget_remaining > 0:
            iteration += 1
            budget_cost = 0

            # Sample minibatch M ~ D_train
            minibatch = dataset.sample_minibatch(b)

            # Evaluate parent on this minibatch
            parent_score, parent_evaluated, parent_per_sample = (
                await self.evaluator.evaluate_minibatch(
                    self._make_module(module, current_instructions),
                    minibatch,
                    metric,
                )
            )
            budget_cost += len(minibatch)

            # === Layer 1: Random Restart (probability p) ===
            if self._rng.random() < p:
                print(f"\n[RESTART] Iteration {iteration}: Triggering random restart...")
                restart_result = await self._do_random_restart(
                    module, current_instructions, minibatch, parent_per_sample, metric, dataset
                )

                if restart_result is not None:
                    restart_instructions, restart_score, restart_node_id = restart_result
                    current_instructions = restart_instructions
                    current_node_id = restart_node_id
                    if restart_score > best_score:
                        best_score = restart_score
                        best_instructions = restart_instructions

                    self.logger.log_random_restart(
                        iteration, f"Accepted restart (score: {restart_score:.2f})"
                    )
                else:
                    self.logger.log_random_restart(
                        iteration, "Restart did not improve over parent"
                    )

                # Restart costs: 1 LLM call + minibatch eval
                budget_cost += len(minibatch) + 1
                budget_remaining -= budget_cost
                continue

            # === Layer 2: Hypothesis-based optimization ===
            # Collect failure cases from minibatch
            failed_examples = [
                ex for ex in parent_evaluated if metric(ex, ex.actual_output or {}) < 1.0
            ]

            if not failed_examples:
                # Fallback to single mutation when no failures (per paper traces)
                print(f"\n[FALLBACK] Iteration {iteration}: No failures on minibatch, skipping")
                self.logger.log_no_improvement(
                    iteration, "No failed samples on minibatch"
                )
                budget_remaining -= budget_cost
                continue

            # Get trace context for the hypothesis agent
            trace_context = self.logger.tree.format_trace_context(current_node_id)

            # Generate K hypotheses with ε-greedy sampling
            hypotheses = await self._generate_hypotheses_epsilon_greedy(
                current_instructions, failed_examples, K, eps, trace_context
            )

            # Rewrite prompt for each hypothesis (parallel)
            rewrite_tasks = [
                self.reflection_agent.rewrite_prompt(
                    current_instructions, h, failed_examples[:3]
                )
                for h in hypotheses
            ]
            candidate_prompts = await asyncio.gather(*rewrite_tasks)

            # Evaluate all candidates on the SAME minibatch (parallel minibatch verification)
            eval_tasks = [
                self.evaluator.evaluate_minibatch(
                    self._make_module(module, prompt), minibatch, metric
                )
                for prompt in candidate_prompts
            ]
            eval_results = await asyncio.gather(*eval_tasks)
            budget_cost += len(minibatch) * K

            # === Select best candidate by Δacc > 0 ===
            best_candidate = await self._select_best_candidate(
                hypotheses,
                candidate_prompts,
                eval_results,
                parent_per_sample,
                parent_val_score=best_score,
                module=module,
                current_instructions=current_instructions,
                parent_node_id=current_node_id,
                dataset=dataset,
                metric=metric,
                iteration=iteration,
            )

            if best_candidate is not None:
                new_instructions, new_score, new_node_id = best_candidate
                budget_cost += len(dataset.val)  # val evaluation cost

                current_instructions = new_instructions
                current_node_id = new_node_id
                if new_score > best_score:
                    best_score = new_score
                    best_instructions = new_instructions
            else:
                self.logger.log_no_improvement(
                    iteration, "No candidate improved over parent (Δacc ≤ 0)"
                )

            budget_remaining -= budget_cost
            print(f"  Budget remaining: {budget_remaining}")

        # Apply best instructions found across all iterations
        module._parameters["instructions"] = best_instructions
        module.signature.instructions = best_instructions

        print(f"\nOptimization Complete. Best score: {best_score:.2f}")
        self.logger.tree.save(self.logger.log_file)
        return module

    async def _generate_hypotheses_epsilon_greedy(
        self,
        current_instructions: str,
        failed_examples: List[Example],
        k: int,
        epsilon: float,
        trace_context: str,
    ) -> List[Hypothesis]:
        """
        Paper Section 4.4, Layer 2: ε-greedy hypothesis sampling.
        For each of K slots:
        - With prob (1-ε): heuristic-guided (uses taxonomy)
        - With prob ε: free hypothesis (unconstrained discovery)
        """
        heuristic_count = 0
        free_count = 0

        for _ in range(k):
            if self._rng.random() < epsilon:
                free_count += 1
            else:
                heuristic_count += 1

        hypotheses: List[Hypothesis] = []

        # Generate heuristic-guided hypotheses
        if heuristic_count > 0:
            heuristic_hyps = await self.hypothesis_agent.generate_hypotheses(
                current_instructions=current_instructions,
                failed_examples=failed_examples[:3],
                num_hypotheses=heuristic_count,
                trace_context=trace_context,
            )
            hypotheses.extend(heuristic_hyps)

        # Generate free hypotheses (ε-exploration)
        if free_count > 0:
            free_tasks = [
                self.hypothesis_agent.generate_free_hypothesis(
                    current_instructions=current_instructions,
                    failed_examples=failed_examples[:3],
                    trace_context=trace_context,
                )
                for _ in range(free_count)
            ]
            free_hyps = await asyncio.gather(*free_tasks)
            hypotheses.extend(free_hyps)

        return hypotheses

    async def _do_random_restart(
        self,
        module: Predict,
        current_instructions: str,
        minibatch: List[Example],
        parent_per_sample: List[int],
        metric: Callable,
        dataset: Dataset,
    ) -> Optional[tuple]:
        """
        Paper Section 4.4, Layer 1: blank-prompt look-ahead restart.
        Returns (instructions, val_score, node_id) if restart improves, else None.
        """
        # Pick one training example for the look-ahead
        example = self._rng.choice(dataset.train)

        restart_instructions = await self.reflection_agent.blank_prompt_lookahead(
            self.llm_client, example
        )

        # Evaluate restart candidate on same minibatch
        restart_score, _, restart_per_sample = await self.evaluator.evaluate_minibatch(
            self._make_module(module, restart_instructions), minibatch, metric
        )

        # Accept only if Δacc > 0
        delta = self.evaluator.compute_delta_acc(parent_per_sample, restart_per_sample)
        if delta <= 0:
            return None

        # Evaluate on val set for acceptance
        val_score, val_per_sample = await self.evaluator.evaluate_on_val(
            self._make_module(module, restart_instructions), dataset.val, metric
        )

        node_id = self.logger.tree.add_candidate(
            parent_id="π₀",  # Restart always branches from root conceptually
            instructions=restart_instructions,
            score=val_score,
            hypothesis_tag="random_restart",
            delta_acc=delta,
            accepted=True,
            per_sample=val_per_sample,
        )

        return restart_instructions, val_score, node_id

    async def _select_best_candidate(
        self,
        hypotheses: List[Hypothesis],
        candidate_prompts: List[str],
        eval_results: list,
        parent_per_sample: List[int],
        parent_val_score: float,
        module: Predict,
        current_instructions: str,
        parent_node_id: str,
        dataset: Dataset,
        metric: Callable,
        iteration: int,
    ) -> Optional[tuple]:
        """
        Paper Algorithm 1 lines 13-21:
        i* = argmax_{i : Δacc_i > 0} Δacc_i
        Then evaluate on D_val for acceptance.
        """
        # Compute Δacc for each candidate
        deltas = []
        for i, (score, evaluated, per_sample) in enumerate(eval_results):
            delta = self.evaluator.compute_delta_acc(parent_per_sample, per_sample)
            deltas.append(delta)

        # Log all hypotheses explored
        hyp_log = []
        for i, h in enumerate(hypotheses):
            hyp_log.append({
                "tag": h.tag,
                "desc": h.description,
                "delta_acc": deltas[i] if i < len(deltas) else 0.0,
            })

        # Select i* = argmax_{i : Δacc_i > 0} Δacc_i
        best_idx = -1
        best_delta = 0.0
        for i, d in enumerate(deltas):
            if d > best_delta:
                best_delta = d
                best_idx = i

        if best_idx < 0:
            # Log all candidates as rejected
            for i, h in enumerate(hypotheses):
                self.logger.tree.add_candidate(
                    parent_id=parent_node_id,
                    instructions=candidate_prompts[i] if i < len(candidate_prompts) else "",
                    score=eval_results[i][0] if i < len(eval_results) else 0.0,
                    hypothesis_tag=h.tag,
                    delta_acc=deltas[i] if i < len(deltas) else 0.0,
                    accepted=False,
                )
            return None

        # Evaluate winner on D_val
        winner_prompt = candidate_prompts[best_idx]
        winner_hypothesis = hypotheses[best_idx]

        val_score, val_per_sample = await self.evaluator.evaluate_on_val(
            self._make_module(module, winner_prompt), dataset.val, metric
        )

        # Add all candidates to trace tree
        winner_node_id = None
        for i, h in enumerate(hypotheses):
            is_winner = i == best_idx
            node_id = self.logger.tree.add_candidate(
                parent_id=parent_node_id,
                instructions=candidate_prompts[i] if i < len(candidate_prompts) else "",
                score=val_score if is_winner else (eval_results[i][0] if i < len(eval_results) else 0.0),
                hypothesis_tag=h.tag,
                delta_acc=deltas[i] if i < len(deltas) else 0.0,
                accepted=is_winner,
                per_sample=val_per_sample if is_winner else None,
            )
            if is_winner:
                winner_node_id = node_id

        self.logger.log_iteration(
            iteration,
            current_instructions,
            parent_val_score,
            "heuristics",
            hyp_log,
            winner_prompt,
            val_score,
            winner_hypothesis.tag,
        )

        print(
            f"\n[ACCEPTED] Iteration {iteration}: "
            f"[{winner_hypothesis.tag}] Δacc=+{best_delta:.2f}, "
            f"val_score={val_score:.2f}"
        )

        return winner_prompt, val_score, winner_node_id

    def _make_module(self, base_module: Predict, instructions: str) -> Predict:
        """Create a temporary module with the given instructions."""
        temp = Predict(base_module.signature)
        temp._parameters["instructions"] = instructions
        return temp

    async def _evaluate_prompt(
        self,
        module: Predict,
        instructions: str,
        examples: List[Example],
        metric: Callable,
    ) -> tuple:
        """Helper: evaluate a prompt on a set of examples."""
        temp = self._make_module(module, instructions)
        score, per_sample = await self.evaluator.evaluate_on_val(temp, examples, metric)
        return score, per_sample
