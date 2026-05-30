import json
from typing import List, Literal
from pydantic import AliasChoices, BaseModel, Field
from vista.llm.client import LLMClient
from vista.core.example import Example

ERROR_TAXONOMY = """
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

HYPOTHESIS_PROMPT = """
You are an expert prompt engineer analyzing why a system prompt causes failures on certain inputs.

CURRENT SYSTEM PROMPT:
{curr_instructions}

ERROR TAXONOMY:
{error_taxonomy}

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
"""

TaxonomyID = Literal[
    "cot_field_ordering",
    "format_and_syntax",
    "task_instruction_clarity",
    "reasoning_strategy",
    "missing_domain_knowledge",
    "edge_case_handling",
    "unclassified_custom"
]

class Hypothesis(BaseModel):
    model_config = {"populate_by_name": True}
    
    tag: TaxonomyID = Field(
        description="The exact id from the Error Taxonomy",
        alias="tag",
        validation_alias=AliasChoices("tag", "category", "id", "taxonomy_id")
    )
    description: str = Field(description="One or two sentences describing the specific root cause")
    fix: str = Field(default="", description="One or two sentences describing how to fix the prompt")

class HypothesisResponse(BaseModel):
    hypotheses: List[Hypothesis]

class HypothesisAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def generate_hypotheses(
        self, 
        current_instructions: str, 
        failed_examples: List[Example], 
        num_hypotheses: int = 3
    ) -> List[Hypothesis]:
        
        # Format failed samples
        samples_str = ""
        for i, ex in enumerate(failed_examples):
            samples_str += f"--- Sample {i+1} ---\n"
            samples_str += f"INPUTS: {json.dumps(ex.inputs, indent=2)}\n"
            samples_str += f"EXPECTED OUTPUTS: {json.dumps(ex.target, indent=2)}\n"
            samples_str += f"ACTUAL OUTPUTS: {json.dumps(ex.actual_output, indent=2) if ex.actual_output else 'N/A'}\n\n"

        prompt = HYPOTHESIS_PROMPT.format(
            curr_instructions=current_instructions,
            error_taxonomy=ERROR_TAXONOMY,
            failed_samples=samples_str,
            num_hypotheses=num_hypotheses
        )

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response: HypothesisResponse = await self.llm_client.generate_structured(
            messages=messages, 
            response_model=HypothesisResponse
        )
        
        return response.hypotheses[:num_hypotheses]
