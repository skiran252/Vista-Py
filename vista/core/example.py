from pydantic import BaseModel
from typing import Any, Dict, Optional

class Example(BaseModel):
    """
    A single example for training or evaluation.
    Holds inputs and the target outputs (labels).
    """
    inputs: Dict[str, Any]
    target: Dict[str, Any]
    
    # Store the actual output during prediction for tracing
    actual_output: Optional[Dict[str, Any]] = None

    def with_output(self, output: Dict[str, Any]) -> "Example":
        """Returns a copy of the example with the actual output populated."""
        return Example(
            inputs=self.inputs,
            target=self.target,
            actual_output=output
        )
