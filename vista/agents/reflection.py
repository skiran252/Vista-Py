from typing import List

from pydantic import BaseModel, Field

from vista.llm.client import LLMClient
from vista.core.example import Example
from vista.agents.hypothesis import Hypothesis
from vista.agents.formatting import format_failed_samples

REFLECTION_PROMPT = """\
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

# Blank prompt look-ahead: build a prompt from the model's natural output
RESTART_PROMPT = """\
You are a prompt engineering expert. A language model was run with NO system prompt on a task example. \
Below is the model's raw output and any parsing errors encountered.

Your job: construct a clean, well-structured system prompt that would make this model produce \
correct, parseable outputs for this type of task. Ground the prompt in what the model naturally \
produces — do NOT copy from any existing prompt.

RAW MODEL OUTPUT:
{raw_output}

PARSING ERROR (if any):
{parse_error}

TASK CONTEXT (input that was given):
{task_input}

Write a complete system prompt that:
1. Guides the model to produce well-structured output (e.g., valid JSON)
2. Leverages the model's natural reasoning style visible in the raw output
3. Addresses any parsing errors shown above
4. Is self-contained — no references to previous prompts

Output only the new system prompt, with no explanation or preamble.
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
        """Standard reflection: rewrite the prompt to address a specific hypothesis."""
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

    async def blank_prompt_lookahead(
        self,
        llm_client: LLMClient,
        example: Example,
        max_iterations: int = 3,
    ) -> str:
        """
        Paper Section 4.4, Layer 1: Random Restart via blank-prompt look-ahead.

        Runs a training example with NO system prompt to see the model's natural behavior,
        then constructs a prompt grounded in that output. Loops until parsing succeeds.
        This conditions on the model's natural behavior, NOT on any inherited seed.
        """
        import json

        raw_output = ""
        parse_error = "Not yet attempted"
        task_input = json.dumps(example.inputs, indent=2)
        candidate_prompt = ""

        for iteration in range(max_iterations):
            if iteration == 0:
                # Run with no prompt to get natural model output
                messages = [{"role": "user", "content": f"Solve this:\n{task_input}"}]
                try:
                    raw_output = await llm_client._call_llm(messages)
                except Exception as e:
                    raw_output = f"[Error: {e}]"

                # Check if output is valid JSON
                try:
                    json.loads(raw_output)
                    parse_error = "None — output is valid JSON"
                except (json.JSONDecodeError, TypeError) as e:
                    parse_error = str(e)
            else:
                # Use the constructed prompt from previous iteration
                from vista.core.signature import Signature
                from vista.core.predict import Predict

                temp_module = Predict(
                    Signature.from_string(
                        " -> ".join(
                            [
                                ",".join(example.inputs.keys()),
                                ",".join(example.target.keys()),
                            ]
                        ),
                        candidate_prompt,
                    )
                )
                try:
                    result = await temp_module(llm_client, **example.inputs)
                    raw_output = json.dumps(result)
                    parse_error = "None — output is valid JSON"
                    break  # e = ∅, prompt is good
                except Exception as e:
                    parse_error = str(e)

            # Build prompt from model's natural output
            restart_messages = [
                {
                    "role": "system",
                    "content": RESTART_PROMPT.format(
                        raw_output=raw_output,
                        parse_error=parse_error,
                        task_input=task_input,
                    ),
                }
            ]

            response: RewrittenPrompt = await self.llm_client.generate_structured(
                messages=restart_messages, response_model=RewrittenPrompt
            )
            candidate_prompt = response.new_instructions

        return candidate_prompt
