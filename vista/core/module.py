from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class FieldSchema(BaseModel):
    """Describes a single field in a signature."""
    name: str
    description: str
    type_: Any = str

class Signature(BaseModel):
    """
    Defines the input and output structure for an LLM call.
    Similar to dspy.Signature (e.g., 'question -> answer').
    """
    inputs: List[FieldSchema]
    outputs: List[FieldSchema]
    instructions: str = ""

    @classmethod
    def from_string(cls, signature_str: str, instructions: str = ""):
        """
        Parses a string like 'question -> answer, reasoning' into a Signature.
        """
        # Very simple parser for proof of concept
        if "->" not in signature_str:
            raise ValueError("Signature must contain '->'")
        
        in_str, out_str = signature_str.split("->")
        
        in_fields = [f.strip() for f in in_str.split(",") if f.strip()]
        out_fields = [f.strip() for f in out_str.split(",") if f.strip()]
        
        return cls(
            inputs=[FieldSchema(name=f, description=f"The {f}") for f in in_fields],
            outputs=[FieldSchema(name=f, description=f"The {f}") for f in out_fields],
            instructions=instructions
        )

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
    """
    A basic module that takes a Signature and uses an LLM to predict the outputs.
    """
    def __init__(self, signature: Signature | str, instructions: str = ""):
        super().__init__()
        if isinstance(signature, str):
            self.signature = Signature.from_string(signature, instructions)
        else:
            self.signature = signature
            
        # The instructions are the main learnable parameter
        self._parameters["instructions"] = self.signature.instructions
        
    async def forward(self, llm_client, **kwargs) -> Dict[str, Any]:
        """
        Executes the LLM call asynchronously.
        """
        # Update signature instructions if parameter was optimized
        self.signature.instructions = self._parameters["instructions"]
        
        # We delegate the actual LLM call to a client that understands the signature
        return await llm_client.predict(self.signature, kwargs)

    def __call__(self, llm_client, **kwargs):
        return self.forward(llm_client, **kwargs)
