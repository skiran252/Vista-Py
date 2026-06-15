import asyncio
import os

from dotenv import load_dotenv

from vista import Predict, Example, VistaOptimizer, VistaConfig


async def main():
    print("Initializing Customer Support Intent VISTA Run (Lightning AI)...")

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path)

    if "LIGHTNING_API_KEY" not in os.environ:
        print("WARNING: LIGHTNING_API_KEY not found in .env!")
        return

    # Lightning AI uses OpenAI-compatible endpoint
    os.environ["OPENAI_API_KEY"] = os.environ["LIGHTNING_API_KEY"]

    initial_instructions = """You are a customer support AI. Analyze the user's message and extract the intent and the order ID.
Allowed Intents: [Return_Request, Cancel_Order, Track_Package]
Output strictly in this JSON format: {"intent": "...", "order_id": "..."}"""

    module = Predict(
        "customer_message -> intent, order_id", instructions=initial_instructions
    )

    dataset = [
        Example(
            inputs={
                "customer_message": "Where on earth is my package? I bought it three days ago and haven't heard anything."
            },
            target={"intent": "Track_Package", "order_id": None},
        ),
        Example(
            inputs={
                "customer_message": "I received order 4412 instead of my actual order 9981, I need to send 4412 back."
            },
            target={"intent": "Return_Request", "order_id": "4412"},
        ),
        Example(
            inputs={
                "customer_message": "Can you please cancel my return request for order #5521? I decided to keep the shoes."
            },
            target={"intent": "Track_Package", "order_id": "5521"},
        ),
        Example(
            inputs={
                "customer_message": "My order ID is US-8831-X. I want to cancel it before it ships out tonight."
            },
            target={"intent": "Cancel_Order", "order_id": "US-8831-X"},
        ),
        Example(
            inputs={
                "customer_message": "Hey, just checking if order 1012 has shipped yet? If not, cancel it."
            },
            target={"intent": "Track_Package", "order_id": "1012"},
        ),
    ]

    def strict_json_match(example: Example, actual_output: dict) -> float:
        expected_intent = example.target.get("intent")
        expected_order_id = example.target.get("order_id")

        actual_intent = actual_output.get("intent")
        actual_order_id = actual_output.get("order_id")

        score = 0.0
        if actual_intent == expected_intent:
            score += 0.5
        if actual_order_id == expected_order_id:
            score += 0.5

        return score

    config = VistaConfig(
        k=3,
        budget=200,
        minibatch_size=5,
        train_size=5,
        val_size=5,
        restart_prob=0.2,
        epsilon=0.1,
    )

    optimizer = VistaOptimizer(
        model_name="openai/lightning-ai/gpt-oss-20b",
        api_base="https://lightning.ai/api/v1/",
        config=config,
    )

    try:
        optimized_module = await optimizer.optimize(
            module=module,
            trainset=dataset,
            metric=strict_json_match,
        )
        print("\nFinal optimized instructions:")
        print(optimized_module.signature.instructions)
    except Exception as e:
        print(f"Error during optimization loop: {e}")


if __name__ == "__main__":
    asyncio.run(main())
