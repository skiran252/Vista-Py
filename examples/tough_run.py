import asyncio
import os

from dotenv import load_dotenv

from vista import Predict, Example, VistaOptimizer, VistaConfig


async def main():
    print("Initializing Tough Reasoning VISTA Run (Lightning AI)...")

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path)

    if "LIGHTNING_API_KEY" not in os.environ:
        print("WARNING: LIGHTNING_API_KEY not found in .env!")
        return

    # Lightning AI uses OpenAI-compatible endpoint
    os.environ["OPENAI_API_KEY"] = os.environ["LIGHTNING_API_KEY"]

    module = Predict(
        "question -> answer", instructions="Answer the following reasoning question."
    )

    # Cognitive Reflection Test + Trick Questions
    dataset = [
        Example(
            inputs={
                "question": "I have a car wash in 10 meters, should I walk or go by car?"
            },
            target={"answer": "car"},
        ),
        Example(
            inputs={
                "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?"
            },
            target={"answer": "5"},
        ),
        Example(
            inputs={
                "question": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?"
            },
            target={"answer": "5"},
        ),
        Example(
            inputs={
                "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake?"
            },
            target={"answer": "47"},
        ),
    ]

    def contains_match(example: Example, actual_output: dict) -> float:
        expected = example.target.get("answer", "").strip().lower()
        actual = str(actual_output.get("answer", "")).strip().lower()
        return 1.0 if expected in actual else 0.0

    config = VistaConfig(
        k=2,
        budget=150,
        minibatch_size=4,
        train_size=4,
        val_size=4,
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
            metric=contains_match,
        )
        print("\nFinal optimized instructions:")
        print(optimized_module.signature.instructions)
    except Exception as e:
        print(f"Error during optimization loop: {e}")


if __name__ == "__main__":
    asyncio.run(main())
