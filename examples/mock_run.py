import asyncio
import os

from dotenv import load_dotenv

from vista import Predict, Example, VistaOptimizer, VistaConfig


async def main():
    print("Initializing Mock VISTA Run (Lightning AI)...")

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path)

    if "LIGHTNING_API_KEY" not in os.environ:
        print("WARNING: LIGHTNING_API_KEY not found in .env!")
        return

    # Lightning AI uses OpenAI-compatible endpoint
    os.environ["OPENAI_API_KEY"] = os.environ["LIGHTNING_API_KEY"]

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

    config = VistaConfig(
        k=2,
        budget=100,
        minibatch_size=3,
        train_size=3,
        val_size=3,
        restart_prob=0.2,
        epsilon=0.1,
    )

    optimizer = VistaOptimizer(
        model_name="openai/gpt-4o",
        api_base="https://lightning.ai/api/v1/",
        config=config,
    )

    try:
        optimized_module = await optimizer.optimize(
            module=module,
            trainset=dataset,
            metric=exact_match,
        )
        print("\nFinal optimized instructions:")
        print(optimized_module.signature.instructions)
    except Exception as e:
        print(f"Error during optimization loop: {e}")


if __name__ == "__main__":
    asyncio.run(main())
