import json
import os
from typing import Any, Dict, List

class TraceLogger:
    """
    Logs the optimization trace to make it interpretable.
    """
    def __init__(self, log_file: str = "vista_trace.json"):
        self.log_file = log_file
        self.history: List[Dict[str, Any]] = []

    def log_iteration(
        self, 
        iteration: int, 
        parent_instructions: str,
        parent_score: float,
        strategy: str,
        hypotheses: List[Dict[str, Any]],
        best_candidate: str,
        best_score: float,
        best_hypothesis: str
    ):
        entry = {
            "iteration": iteration,
            "parent_score": parent_score,
            "strategy": strategy,
            "hypotheses_explored": hypotheses,
            "best_candidate_score": best_score,
            "best_hypothesis_tag": best_hypothesis,
            "new_instructions": best_candidate
        }
        self.history.append(entry)
        self._save()
        
        print(f"\n[{strategy.upper()}] Iteration {iteration}: Score {parent_score:.2f} -> {best_score:.2f}")
        if best_hypothesis:
            print(f"Accepted Fix: {best_hypothesis}")

    def log_random_restart(self, iteration: int, reason: str):
        entry = {
            "iteration": iteration,
            "strategy": "random_restart",
            "reason": reason
        }
        self.history.append(entry)
        self._save()
        print(f"\n[RANDOM RESTART] Iteration {iteration}: {reason}")

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)
