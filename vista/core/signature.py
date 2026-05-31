from typing import Any, List
from pydantic import BaseModel


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
        if "->" not in signature_str:
            raise ValueError("Signature must contain '->'")

        in_str, out_str = signature_str.split("->")

        in_fields = [f.strip() for f in in_str.split(",") if f.strip()]
        out_fields = [f.strip() for f in out_str.split(",") if f.strip()]

        return cls(
            inputs=[FieldSchema(name=f, description=f"The {f}") for f in in_fields],
            outputs=[FieldSchema(name=f, description=f"The {f}") for f in out_fields],
            instructions=instructions,
        )
