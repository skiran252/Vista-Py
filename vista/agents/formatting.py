import json
from typing import List

from vista.core.example import Example


def format_failed_samples(examples: List[Example]) -> str:
    """Formats a list of failed examples into a readable string for agent prompts."""
    parts = []
    for i, ex in enumerate(examples):
        actual = json.dumps(ex.actual_output, indent=2) if ex.actual_output else "N/A"
        parts.append(
            f"--- Sample {i + 1} ---\n"
            f"INPUTS: {json.dumps(ex.inputs, indent=2)}\n"
            f"EXPECTED OUTPUTS: {json.dumps(ex.target, indent=2)}\n"
            f"ACTUAL OUTPUTS: {actual}\n"
        )
    return "\n".join(parts)
