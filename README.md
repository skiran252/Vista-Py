# VISTA-PY (Work in Progress 🚧)

**Disclaimer:** This is a casual exploration and experimental implementation of an Automated Prompt Optimization (APO) framework inspired by the VISTA research paper and DSPy. **It is currently under active development, just a casual exploration, and is not yet ready for production use.**

## Overview

VISTA-PY is a multi-agent Automated Prompt Optimization (APO) library that decouples hypothesis generation from prompt rewriting. It aims to solve the "black box" problem in prompt optimization by providing semantically labeled hypotheses, parallel minibatch verification, and an interpretable optimization trace.

## Features (Implemented & Planned)

- **Decoupled Hypothesis Generation**: Isolates the diagnosis of prompt failures from the generation of prompt rewrites.
- **Interpretable Traces**: Generates detailed logs of the optimization trajectory (`vista_trace.json`).
- **Parallel Minibatch Verification**: Efficiently verifies candidate prompts across multiple examples.
- **Flexible LLM Backend**: Built on top of `litellm`, allowing seamless integration with any LLM provider (OpenAI, Anthropic, OpenRouter, local models via vLLM/Ollama, etc.).
- **Automatic Retries**: Built-in exponential backoff for handling API rate limits gracefully (useful for free-tier APIs).

## Getting Started

### Prerequisites
- Python 3.12+
- `uv` package manager

### Installation

Clone the repository and sync dependencies using `uv`:

```bash
git clone https://github.com/yourusername/VISTA-PY.git
cd VISTA-PY
uv sync
```

### Configuration

Create a `.env` file in the root directory and add your API keys. You can also specify custom API bases for local/custom endpoints:

```env
OPENROUTER_API_KEY=your_openrouter_key
# CUSTOM_API_BASE=https://your-custom-endpoint.com/v1
```

### Running Examples

You can explore the current capabilities by running the provided examples:

```bash
# Customer support intent classification optimization
uv run examples/support_run.py

# Complex reasoning tasks optimization
uv run examples/tough_run.py
```

## Structure

- `vista/core/`: Contains core abstractions like `Signature`, `Predict`, and `Example`.
- `vista/agents/`: Implementations of the multi-agent system (`HypothesisAgent`, `ReflectionAgent`).
- `vista/evaluation/`: Tools for evaluating prompt performance with parallel minibatch support.
- `vista/optimizers/`: The main VISTA optimization loop.
- `vista/llm/`: A wrapper around `litellm` providing structured output generation and error handling.

## Note on Rate Limits

When testing with free-tier APIs (e.g., OpenRouter free models), you may encounter `429 Too Many Requests` errors. The library has built-in exponential backoff, but optimization runs require a significant number of LLM calls. It is highly recommended to use paid endpoints or local models for stable runs.
