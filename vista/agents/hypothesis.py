from typing import List, Literal

from pydantic import AliasChoices, BaseModel, Field

from vista.llm.client import LLMClient
from vista.core.example import Example
from vista.agents.formatting import format_failed_samples

ERROR_TAXONOMY = """\
- id: cot_field_ordering
  name: CoT / Output Field Ordering Defect
  description: The output schema requires the final answer before the reasoning steps, preventing chain-of-thought from influencing the result.
- id: format_and_syntax
  name: Format / Syntax Defect
  description: The prompt does not strictly enforce output schema, key set, or syntax validity.
- id: task_instruction_clarity
  name: Task Instruction / Constraint Defect
  description: Task goals or constraints are ambiguous, contradictory, or incomplete.
- id: reasoning_strategy
  name: Reasoning Strategy / Logic Defect
  description: The prompt implies a flawed or suboptimal reasoning procedure for the task.
- id: missing_domain_knowledge
  name: Missing Domain Knowledge Gap
  description: The prompt lacks necessary domain facts or definitions required for solving.
- id: edge_case_handling
  name: Edge Case / Boundary Defect
  description: The prompt handles common inputs but fails on boundary or atypical cases.
- id: unclassified_custom
  name: Unclassified / Custom Discovery
  description: None of the predefined categories fit; discover and justify a latent failure mode.
"""

# Heuristic-guided prompt — uses the taxonomy (exploitation)
HYPOTHESIS_PROMPT = """\
You are an expert prompt engineer analyzing why a system prompt causes failures on certain inputs.

CURRENT SYSTEM PROMPT:
{curr_instructions}

ERROR TAXONOMY:
{error_taxonomy}

{trace_context}

FAILED SAMPLES:
{failed_samples}

TASK: Analyze the failed samples and generate exactly {num_hypotheses} diverse root-cause hypotheses.
For EACH hypothesis:
1. Select the most fitting category from the Error Taxonomy above (use the exact id field).
2. Provide a concise description of the specific root cause you identified.
3. Suggest a concrete fix direction for the prompt.

IMPORTANT:
- Each hypothesis MUST address a DIFFERENT aspect of the failures.
- Try to cover as many different taxonomy categories as possible.
- Be specific about what exactly in the current prompt causes the failure.
- If the optimization trace shows a category has been tried repeatedly with diminishing returns, \
avoid that category and try unexplored ones.
- If two categories seem to alternate without resolution, consider a joint hypothesis addressing both.
"""

# Free hypothesis prompt — no taxonomy constraint (exploration)
FREE_HYPOTHESIS_PROMPT = """\
You are an expert prompt engineer analyzing why a system prompt causes failures on certain inputs.

CURRENT SYSTEM PROMPT:
{curr_instructions}

{trace_context}

FAILED SAMPLES:
{failed_samples}

TASK: Analyze the failed samples and generate exactly 1 root-cause hypothesis.
You are NOT constrained by any predefined taxonomy. Discover the failure mode freely.

For your hypothesis:
1. Invent a short descriptive tag (snake_case, max 4 words) for the failure mode you identified.
2. Provide a concise description of the specific root cause.
3. Suggest a concrete fix direction for the prompt.

IMPORTANT:
- Think outside the box. Look for failure modes that standard categories might miss.
- Consider structural, semantic, or implicit issues in the prompt.
- Be specific about what exactly in the current prompt causes the failure.
"""

TaxonomyID = Literal[
    "cot_field_ordering",
    "format_and_syntax",
    "task_instruction_clarity",
    "reasoning_strategy",
    "missing_domain_knowledge",
    "edge_case_handling",
    "unclassified_custom",
]


class Hypothesis(BaseModel):
    model_config = {"populate_by_name": True}

    tag: str = Field(
        description="The exact id from the Error Taxonomy, or a custom tag for free hypotheses",
        alias="tag",
        validation_alias=AliasChoices("tag", "category", "id", "taxonomy_id"),
    )
    description: str = Field(
        description="One or two sentences describing the specific root cause"
    )
    fix: str = Field(
        default="",
        description="One or two sentences describing how to fix the prompt",
    )


class HypothesisResponse(BaseModel):
    hypotheses: List[Hypothesis]


class FreeHypothesisResponse(BaseModel):
    hypotheses: List[Hypothesis]


class HypothesisAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def generate_hypotheses(
        self,
        current_instructions: str,
        failed_examples: List[Example],
        num_hypotheses: int = 3,
        trace_context: str = "",
    ) -> List[Hypothesis]:
        """Generate heuristic-guided hypotheses using the error taxonomy."""
        ctx = trace_context if trace_context else "No prior optimization history."

        prompt = HYPOTHESIS_PROMPT.format(
            curr_instructions=current_instructions,
            error_taxonomy=ERROR_TAXONOMY,
            trace_context=ctx,
            failed_samples=format_failed_samples(failed_examples),
            num_hypotheses=num_hypotheses,
        )

        messages = [{"role": "system", "content": prompt}]

        response: HypothesisResponse = await self.llm_client.generate_structured(
            messages=messages, response_model=HypothesisResponse
        )

        return response.hypotheses[:num_hypotheses]

    async def generate_free_hypothesis(
        self,
        current_instructions: str,
        failed_examples: List[Example],
        trace_context: str = "",
    ) -> Hypothesis:
        """
        Generate an unconstrained hypothesis (ε-exploration branch).
        Not bound by the taxonomy — can discover novel failure modes.
        """
        ctx = trace_context if trace_context else "No prior optimization history."

        prompt = FREE_HYPOTHESIS_PROMPT.format(
            curr_instructions=current_instructions,
            trace_context=ctx,
            failed_samples=format_failed_samples(failed_examples),
        )

        messages = [{"role": "system", "content": prompt}]

        response: FreeHypothesisResponse = await self.llm_client.generate_structured(
            messages=messages, response_model=FreeHypothesisResponse
        )

        return response.hypotheses[0] if response.hypotheses else Hypothesis(
            tag="unclassified_custom",
            description="Free hypothesis generation failed",
            fix="",
        )
