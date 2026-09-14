"""HTTP service for the trained model: text in, generated text out.

From 14-mini-llm:
    ../.venv/bin/python src/api.py
    ../.venv/bin/uvicorn api:app --app-dir src --reload

Then POST to /generate, or open /docs for the generated interface.

    curl -s localhost:8000/generate \
      -H 'content-type: application/json' \
      -d '{"prompt": "Transformers", "max_new_tokens": 15}'

This is a delivery surface, not a new step. Every request is
MiniLLMApp.complete -- tokenizer, model, tokenizer -- with argument
validation in front of it and JSON around it.

Configuration comes from the environment so the service has no flags to
keep in sync with Step 10:

    MINI_LLM_ARTIFACTS  checkpoint directory (default: artifacts/step08)
    MINI_LLM_DEVICE     cpu or mps           (default: cpu)
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from step_10_app import DEFAULT_ARTIFACTS, MiniLLMApp


class GenerateRequest(BaseModel):
    """Pydantic rejects out-of-range values with a 422 before we run."""

    prompt: str = Field(
        min_length=1,
        description="Text to continue. Must encode to at least one token.",
    )
    max_new_tokens: int = Field(
        default=16,
        ge=1,
        le=256,
        description="How many tokens to generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="0.0 picks the top token every time and is reproducible.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Restrict sampling to the k best tokens. Ignored when "
                    "temperature is 0.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Makes sampled output repeatable.",
    )


class GenerateResponse(BaseModel):
    prompt: str = Field(description="The prompt exactly as submitted.")
    completion: str = Field(
        description="Generated continuation only, as normalized text."
    )
    text: str = Field(description="Normalized prompt followed by completion.")
    prompt_tokens: int
    generated_tokens: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    vocab_size: int
    context_window: int
    parameters: int
    training_steps: int
    training_loss: float


@asynccontextmanager
async def lifespan(api: FastAPI):
    """Load and verify once at startup, not per request.

    A missing checkpoint raises here rather than on the first request, so
    the service fails to start instead of failing to answer.
    """
    artifacts = Path(
        os.environ.get("MINI_LLM_ARTIFACTS", DEFAULT_ARTIFACTS)
    )
    device = os.environ.get("MINI_LLM_DEVICE", "cpu")

    torch.set_num_threads(1)

    model_app = MiniLLMApp(artifacts_dir=artifacts, device=device)
    model_app.verify()

    api.state.model_app = model_app

    # Endpoints below are sync, so FastAPI runs them in a threadpool and
    # two requests can overlap. generate() switches the module into eval
    # and restores the previous mode afterwards, so interleaved calls
    # could leave it wrong. One model, one generation at a time.
    api.state.lock = Lock()

    yield

    api.state.model_app = None


app = FastAPI(
    title="14-mini-llm",
    version="1.0.0",
    summary="Next-token generation from the Step 8 checkpoint.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """What is loaded, and what it was trained to do."""
    model_app = request.app.state.model_app

    return HealthResponse(
        status="ok",
        device=str(model_app.device),
        vocab_size=model_app.tokenizer.vocab_size,
        context_window=model_app.config["max_seq_len"],
        parameters=sum(
            p.numel() for p in model_app.model.parameters()
        ),
        training_steps=model_app.metadata["training_steps"],
        training_loss=model_app.metadata["training_loss"],
    )


@app.post("/generate", response_model=GenerateResponse)
def generate_text(
    body: GenerateRequest, request: Request
) -> GenerateResponse:
    """Continue `prompt` and return the continuation.

    `completion` and `text` are normalized text, because decoding
    reverses the byte encoding but not Unicode normalization. For text
    normalization does not alter, they match the input exactly.
    """
    model_app = request.app.state.model_app

    encoded = model_app.tokenizer.encode(body.prompt)
    prompt_ids = encoded.token_ids

    # Whitespace-only input can normalize away to nothing, which the model
    # cannot condition on. Say so here rather than 500 from the forward pass.
    if not prompt_ids:
        raise HTTPException(
            status_code=400,
            detail="Prompt encodes to zero tokens.",
        )

    generator = None
    if body.seed is not None:
        # The generator has to live on the same device as the logits.
        generator = torch.Generator(device=model_app.device)
        generator.manual_seed(body.seed)

    started = time.perf_counter()

    try:
        with request.app.state.lock:
            completion = model_app.complete(
                body.prompt,
                body.max_new_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                include_prompt=False,
                generator=generator,
            )
    except ValueError as error:
        # Bad arguments the model layer catches -- top_k above the
        # vocabulary size is the one pydantic cannot check on its own.
        raise HTTPException(status_code=400, detail=str(error)) from error

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return GenerateResponse(
        prompt=body.prompt,
        completion=completion,
        text=encoded.normalized_text + completion,
        prompt_tokens=len(prompt_ids),
        generated_tokens=body.max_new_tokens,
        elapsed_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("MINI_LLM_HOST", "127.0.0.1"),
        port=int(os.environ.get("MINI_LLM_PORT", "8000")),
    )
