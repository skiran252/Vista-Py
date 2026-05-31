from typing import Any, Dict

from vista.core.signature import Signature


class Module:
    """
    Base class for declarative LLM calls.
    Analogous to dspy.Module.
    """

    def __init__(self):
        self._parameters = {}

    def named_parameters(self) -> Dict[str, Any]:
        """Returns the learnable parameters (prompts) of this module."""
        return self._parameters


class Predict(Module):
    """A module that takes a Signature and uses an LLM to predict the outputs."""

    def __init__(self, signature: Signature | str, instructions: str = ""):
        super().__init__()
        if isinstance(signature, str):
            self.signature = Signature.from_string(signature, instructions)
        else:
            self.signature = signature

        # The instructions are the main learnable parameter
        self._parameters["instructions"] = self.signature.instructions

    async def forward(self, llm_client, **kwargs) -> Dict[str, Any]:
        """Executes the LLM call asynchronously."""
        # Sync signature instructions with optimized parameter
        self.signature.instructions = self._parameters["instructions"]
        return await llm_client.predict(self.signature, kwargs)

    def __call__(self, llm_client, **kwargs):
        return self.forward(llm_client, **kwargs)
