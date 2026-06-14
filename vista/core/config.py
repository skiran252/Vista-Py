import os
from dataclasses import dataclass


@dataclass
class VistaConfig:
    """Configuration matching the paper's defaults (Appendix A.1)."""

    # Hypotheses per round
    k: int = 3
    # Random restart probability (Layer 1)
    restart_prob: float = 0.2
    # Exploration rate for epsilon-greedy hypothesis sampling (Layer 2)
    epsilon: float = 0.1
    # Minibatch size for verification
    minibatch_size: int = 8
    # Total metric call budget
    budget: int = 500
    # Train/val split sizes
    train_size: int = 50
    val_size: int = 50
    # Maximum concurrent LLM evaluations
    max_workers: int = 4
    # Reproducibility
    random_seed: int = 0
    # Global LLM rate limit (requests per minute).
    requests_per_minute: float = float(os.environ.get("VISTA_RPM", "15.0"))
