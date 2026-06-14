import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TraceNode:
    """A node in the semantic trace tree — represents a prompt candidate."""

    prompt_id: str
    instructions: str
    score: float
    # Per-sample binary outcomes for dominance checking
    per_sample_outcomes: List[int] = field(default_factory=list)


@dataclass
class TraceEdge:
    """An edge (u → v) annotated with the root-cause label and accuracy gain δ."""

    parent_id: str
    child_id: str
    hypothesis_tag: str
    delta_acc: float
    accepted: bool


class SemanticTraceTree:
    """
    Semantic trace tree T = (V, E) as described in Section 4.3 of the paper.

    Each node stores a prompt candidate, each edge stores (c*, δ).
    The tree is rooted at π₀ and grows by one node per successful round.
    """

    def __init__(self):
        self.nodes: Dict[str, TraceNode] = {}
        self.edges: List[TraceEdge] = []
        self._counter = 0

    def add_root(self, instructions: str, score: float, per_sample: List[int] = None) -> str:
        """Add the seed prompt as root node."""
        node_id = "π₀"
        self.nodes[node_id] = TraceNode(
            prompt_id=node_id,
            instructions=instructions,
            score=score,
            per_sample_outcomes=per_sample or [],
        )
        return node_id

    def add_candidate(
        self,
        parent_id: str,
        instructions: str,
        score: float,
        hypothesis_tag: str,
        delta_acc: float,
        accepted: bool,
        per_sample: List[int] = None,
    ) -> str:
        """Add a candidate node with an edge from parent."""
        self._counter += 1
        prefix = "π" if accepted else "r"
        node_id = f"{prefix}_{self._counter}"

        self.nodes[node_id] = TraceNode(
            prompt_id=node_id,
            instructions=instructions,
            score=score,
            per_sample_outcomes=per_sample or [],
        )
        self.edges.append(
            TraceEdge(
                parent_id=parent_id,
                child_id=node_id,
                hypothesis_tag=hypothesis_tag,
                delta_acc=delta_acc,
                accepted=accepted,
            )
        )
        return node_id

    def get_trajectory(self, node_id: str) -> List[TraceEdge]:
        """Get the root-to-node path τ_t for context."""
        # Build parent map
        parent_map: Dict[str, TraceEdge] = {}
        for edge in self.edges:
            if edge.accepted:
                parent_map[edge.child_id] = edge

        path = []
        current = node_id
        while current in parent_map:
            edge = parent_map[current]
            path.append(edge)
            current = edge.parent_id
        path.reverse()
        return path

    def format_trace_context(self, node_id: str) -> str:
        """
        Format the optimization trajectory as context for the hypothesis agent.
        Enables the agent to avoid already-explored directions and detect oscillation.
        """
        trajectory = self.get_trajectory(node_id)
        if not trajectory:
            return "No prior optimization history."

        lines = ["OPTIMIZATION TRACE (previous accepted changes):"]
        for i, edge in enumerate(trajectory, 1):
            lines.append(
                f"  Step {i}: [{edge.hypothesis_tag}] Δacc={edge.delta_acc:+.1f} "
                f"(from {edge.parent_id} → {edge.child_id})"
            )

        # Summarize category frequency to help detect diminishing returns
        tag_counts: Dict[str, int] = {}
        for edge in trajectory:
            tag_counts[edge.hypothesis_tag] = tag_counts.get(edge.hypothesis_tag, 0) + 1

        lines.append("\nCategory frequency in trajectory:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {tag}: {count} time(s)")

        return "\n".join(lines)

    def save(self, path: str):
        """Persist the trace tree to disk."""
        data = {
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class TraceLogger:
    """High-level logging wrapper around the semantic trace tree."""

    def __init__(self, log_file: str = "vista_trace.json"):
        self.log_file = log_file
        self.tree = SemanticTraceTree()
        # Keep flat history for backward compat
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
        best_hypothesis: str,
    ):
        entry = {
            "iteration": iteration,
            "parent_score": parent_score,
            "strategy": strategy,
            "hypotheses_explored": hypotheses,
            "best_candidate_score": best_score,
            "best_hypothesis_tag": best_hypothesis,
            "new_instructions": best_candidate,
        }
        self.history.append(entry)
        self._save()

        print(
            f"\n[{strategy.upper()}] Iteration {iteration}: "
            f"Score {parent_score:.2f} -> {best_score:.2f}"
        )
        if best_hypothesis:
            print(f"Accepted Fix: {best_hypothesis}")

    def log_random_restart(self, iteration: int, reason: str):
        entry = {
            "iteration": iteration,
            "strategy": "random_restart",
            "reason": reason,
        }
        self.history.append(entry)
        self._save()
        print(f"\n[RANDOM RESTART] Iteration {iteration}: {reason}")

    def log_no_improvement(self, iteration: int, reason: str):
        entry = {
            "iteration": iteration,
            "strategy": "no_improvement",
            "reason": reason,
        }
        self.history.append(entry)
        self._save()
        print(f"\n[SKIP] Iteration {iteration}: {reason}")

    def _save(self):
        # Save both flat history and tree
        self.tree.save(self.log_file)
