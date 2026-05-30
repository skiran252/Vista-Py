import asyncio
import os
import sys

from dotenv import load_dotenv

# Ensure vista package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vista.core.module import Predict
from vista.core.example import Example
from vista.optimizers.vista import VistaOptimizer


async def main():
    print("Initializing Mock VISTA Run...")

    # Load .env variables explicitly from parent directory
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path)

    if "GITHUB_MODELS_TOKEN" in os.environ:
        # litellm uses GITHUB_API_KEY for the github provider
        os.environ["GITHUB_API_KEY"] = os.environ["GITHUB_MODELS_TOKEN"]
        print("Loaded GitHub Models Token from .env")
    else:
        print("WARNING: GITHUB_MODELS_TOKEN not found in .env!")

    module = Predict("question -> answer", instructions="Answer the math question.")

    # GSM8K-like mock dataset
    dataset = [
        Example(inputs={"question": "What is 2+2?"}, target={"answer": "4"}),
        Example(inputs={"question": "What is 3*5?"}, target={"answer": "15"}),
        Example(inputs={"question": "What is 10/2?"}, target={"answer": "5"}),
    ]

    def exact_match(example: Example, actual_output: dict) -> float:
        expected = example.target.get("answer", "").strip()
        actual = str(actual_output.get("answer", "")).strip()
        return 1.0 if expected == actual else 0.0

    # Initialize optimizer with github models
    optimizer = VistaOptimizer(model_name="github/gpt-4o")

    try:
        optimized_module = await optimizer.optimize(
            module=module,
            trainset=dataset,
            metric=exact_match,
            epochs=2,
            k_hypotheses=2,
        )
        print("\nFinal optimized instructions:")
        print(optimized_module.signature.instructions)
    except Exception as e:
        print(f"Error during optimization loop: {e}")


if __name__ == "__main__":
    asyncio.run(main())
