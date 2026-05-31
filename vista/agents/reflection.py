from typing import List

from pydantic import BaseModel, Field

from vista.llm.client import LLMClient
from vista.core.example import Example
from vista.agents.hypothesis import Hypothesis
from vista.agents.formatting import format_failed_samples

REFLECTION_PROMPT = """
You are a prompt optimization expert. Given a prompt, a diagnosed root cause, and a set of failure cases, your task is to rewrite the prompt to fix the identified issue.

Root cause label: {label}
Hypothesis: {hypothesis}
Suggested fix: {suggestion}

Current prompt:
{prompt}

Failure cases:
{failure_cases}

Rewrite the prompt to address the identified root cause. Follow these rules:
- Make targeted edits only. Do not change parts of the prompt unrelated to the root cause.
- Preserve the output schema and JSON format unless the root cause is structure.
- Output only the rewritten prompt, with no explanation or preamble.
"""


class RewrittenPrompt(BaseModel):
    new_instructions: str = Field(
        description="The rewritten system prompt instructions, with no explanation or preamble."
    )


class ReflectionAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def rewrite_prompt(
        self,
        current_instructions: str,
        hypothesis: Hypothesis,
        failed_examples: List[Example],
    ) -> str:

        prompt = REFLECTION_PROMPT.format(
            label=hypothesis.tag,
            hypothesis=hypothesis.description,
            suggestion=hypothesis.fix,
            prompt=current_instructions,
            failure_cases=format_failed_samples(failed_examples),
        )

        messages = [{"role": "system", "content": prompt}]

        response: RewrittenPrompt = await self.llm_client.generate_structured(
            messages=messages, response_model=RewrittenPrompt
        )

        return response.new_instructions
