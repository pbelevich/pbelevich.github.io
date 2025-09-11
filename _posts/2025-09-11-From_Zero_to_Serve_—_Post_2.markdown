---
layout: post
title:  "From Zero to Serve — Post 2"
date:   2025-09-11 10:49:44 -0400
# categories:
---
# From Zero to Serve — Post 2

In Post #2, **myserve** graduates from a token-echo toy to a real generator. We wire a Hugging Face causal-LM directly to the same OpenAI-compatible `/v1/chat/completions` endpoint from [Post 1](https://pbelevich.github.io/2025/09/11/From_Zero_to_Serve_-_Post_1.html) and keep streaming via SSE, so your clients don’t change at all. Under the hood, a tiny **model registry** loads and caches `(model, tokenizer)` bundles—pick `sshleifer/tiny-gpt2` for fast CPU checks, or step up to **TinyLlama** / Llama-family when you have the VRAM and (optionally) a HF token.

Generation starts simple and explicit: a **greedy loop** with **no KV cache**—we recompute attention from scratch each step. That’s intentional. This version is slow on long outputs, but it’s a rock-solid **correctness baseline** we’ll optimize in later posts (KV cache, FlashAttention, quantization, etc.). We also add **parity tests** against `transformers.generate()` on a tiny model to ensure our next-token choices match reference behavior, so every future speedup can be measured against known-good outputs.

To follow along, make sure [Post 1](https://pbelevich.github.io/2025/09/11/From_Zero_to_Serve_-_Post_1.html) is running and install a suitable **PyTorch** (CPU or CUDA per your setup); optionally log in with `huggingface_hub` to pull gated weights. With that, your OpenAI-compatible server now “thinks” for real—streaming decoded tokens in real time while keeping the API stable for existing clients.

---

## New/changed repo layout

The repo grows just enough to run real models. Under `server/core/`, **`models.py`** adds a tiny registry to load/cache Hugging Face `(model, tokenizer)` pairs, and **`generate.py`** implements a naïve greedy loop (no KV cache yet). **`main.py`** now calls the model when available while preserving the OpenAI-compatible streaming API. Tests gain **`test_parity.py`**, which compares our next-token outputs to `transformers.generate` for a correctness baseline.

```
myserve/
  server/
    core/
      models.py        # load/cache HF model + tokenizer
      generate.py      # naïve greedy generate (no KV cache)
    main.py            # updated to call the model when available
  tests/
    test_parity.py     # checks parity vs transformers.generate
```

## `pyproject.toml`

We now include **PyTorch** in `pyproject.toml` (`torch>=2.2`) so `myserve` can run real models. PyTorch wheels are **platform-specific** (CPU vs. CUDA), so keep the spec generic here and install the correct build using the command from [pytorch.org](https://pytorch.org) for your OS/CUDA (e.g., extra index URL for cu121/cu122). This keeps dependency resolution simple while letting you pick the right accelerator at install time; everything else (FastAPI, Pydantic, Transformers, Tokenizers, SentencePiece, tests) stays the same.

```toml
[project]
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.30",
  "pydantic>=2.6",
  "transformers>=4.42",
  "tokenizers>=0.15",
  "sentencepiece>=0.1",
  "httpx>=0.27",
  "pytest>=8.2",
  "torch>=2.2"          # install the correct build for your platform
]
```

---

## `server/core/models.py`

`models.py` introduces a tiny **model registry** that loads and caches Hugging Face causal-LMs with their tokenizers. `ModelRegistry.load(model_name, dtype="auto", device="auto")` keys the cache by `(model, dtype, device)`, auto-picks **CUDA** if available (else CPU), and resolves dtype (`bf16`/`fp16`/`fp32`, or lets HF choose). It creates a fast `AutoTokenizer` (backfilling `pad_token` with `eos_token` if missing), then loads `AutoModelForCausalLM` with `low_cpu_mem_usage=True` and optional `trust_remote_code` gated by `TRUST_REMOTE_CODE=1`. The model is moved to the resolved device, set to `eval()`, and returned as a `ModelBundle` (tokenizer, model, device, dtype). A module-level `REGISTRY` singleton makes retrieval trivial across the app.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass(frozen=True)
class ModelBundle:
    tokenizer: any
    model: any
    device: torch.device
    dtype: torch.dtype

class ModelRegistry:
    def __init__(self):
        self._cache: Dict[Tuple[str, str, str], ModelBundle] = {}

    def load(self, model_name: str, dtype: str = "auto", device: str = "auto") -> ModelBundle:
        key = (model_name, dtype, device)
        if key in self._cache:
            return self._cache[key]

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        
        # Resolve dtype
        if dtype == "auto":
            torch_dtype = None  # let HF pick
        elif dtype.lower() in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype.lower() in ("fp16", "float16"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=os.environ.get("TRUST_REMOTE_CODE", "0") == "1",
        )
        model.to(dev)
        model.eval()

        bundle = ModelBundle(tokenizer=tok, model=model, device=dev, dtype=model.dtype)
        self._cache[key] = bundle
        return bundle

REGISTRY = ModelRegistry()
```

---

## `server/core/generate.py`

`generate.py` provides the most literal baseline for decoding: **greedy, one token at a time, no KV cache**. The `@torch.no_grad()` `greedy_generate()` takes a batch of `input_ids` `[B, T]` already on the model’s device, runs a full forward pass each step (slow by design), picks `argmax` on the last timestep, appends it, and repeats up to `max_new_tokens`. It supports optional early stop when **all** sequences emit `eos_token_id`, keeps the model in `eval()` for deterministic layers, and returns the concatenated tensor `[B, T + new]`. This deliberately unoptimized loop is our correctness anchor before we add KV cache, better sampling, and fused attention in later posts.

```python
from __future__ import annotations
from typing import List, Optional
import torch
from torch import nn

@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Extremely simple greedy loop: one token at a time, no KV cache.
    input_ids: [B, T] on the correct device.
    Returns: [B, T + new] tokens.
    """
    model.eval()
    bsz = input_ids.size(0)
    out = input_ids
    for _ in range(max_new_tokens):
        # full forward each step (slow!)
        logits = model(out).logits  # [B, T, V]
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [B]
        next_token = next_token.unsqueeze(-1)                # [B,1]
        out = torch.cat([out, next_token], dim=1)
        if eos_token_id is not None:
            # Stop early only if *all* sequences ended
            if torch.all(next_token.squeeze(-1) == eos_token_id):
                break
    return out
```

---

## `server/main.py` (updated)

The updated `server/main.py` keeps the OpenAI-compatible surface but adds a real-model path with a clean **echo fallback**. At startup it configures CORS and `/healthz`, then reads `MYSERVE_FORCE_ECHO`, `MYSERVE_DTYPE`, and `MYSERVE_DEVICE` to control behavior. On `/v1/chat/completions`, it builds a prompt, tries to **load `(model, tokenizer)` from `REGISTRY`** (auto device/dtype), and—if that fails or echo is forced—falls back to the Post-1 **token-echo**. In **streaming** mode, both backends emit SSE chunks with a role preamble, token deltas, and a `[DONE]` trailer; in non-streaming mode, real generation uses `greedy_generate()` while echo decodes a truncated slice. Small helpers (`_sse_chunk`, `_sse_done`, `_non_stream_payload`) centralize OpenAI-shaped payloads, keeping the endpoint terse and client-compatible.

```python
import asyncio
import json
import time
import uuid
import os
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from myserve.api.openai_types import ChatCompletionRequest
from myserve.core.tokenizer import get_tokenizer, render_messages
from myserve.core.models import REGISTRY
from myserve.core.generate import greedy_generate

app = FastAPI(title="myserve")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

USE_ECHO_FALLBACK = os.getenv("MYSERVE_FORCE_ECHO", "0") == "1"
DEFAULT_DTYPE = os.getenv("MYSERVE_DTYPE", "auto")
DEFAULT_DEVICE = os.getenv("MYSERVE_DEVICE", "auto")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model_name = req.model
    tokenizer = get_tokenizer(model_name)
    prompt = render_messages(tokenizer, req.messages)

    created = int(time.time())
    rid = f"chatcmpl_{uuid.uuid4().hex[:24]}"

    # Try to load a real model unless echo is forced
    bundle = None
    if not USE_ECHO_FALLBACK:
        try:
            bundle = REGISTRY.load(model_name, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
        except Exception as e:
            # fallback silently to echo mode; in real servers you would surface a 400
            bundle = None

    if bundle is None:
        # --- Echo backend (Post 1) ---
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        max_new = max(0, int(req.max_tokens or 0)) or 128
        output_ids = input_ids[:max_new]

        if req.stream:
            async def echo_stream() -> AsyncGenerator[bytes, None]:
                yield _sse_chunk(rid, model_name, role="assistant")
                for tid in output_ids:
                    piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if piece:
                        yield _sse_chunk(rid, model_name, content=piece)
                    await asyncio.sleep(0.0)
                yield _sse_done(rid, model_name)
            return StreamingResponse(echo_stream(), media_type="text/event-stream")

        text = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return JSONResponse(_non_stream_payload(rid, model_name, text))

    # --- Real model backend (this post) ---
    tok = bundle.tokenizer
    eos = tok.eos_token_id

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.device)

    max_new = max(1, int(req.max_tokens or 16))

    if req.stream:
        async def model_stream() -> AsyncGenerator[bytes, None]:
            yield _sse_chunk(rid, model_name, role="assistant")
            # We decode token-by-token to stream pieces.
            generated = input_ids
            for i in range(max_new):
                with torch.no_grad():
                    logits = bundle.model(generated).logits
                    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
                piece = tok.decode(next_id[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if piece:
                    yield _sse_chunk(rid, model_name, content=piece)
                if eos is not None and int(next_id.item()) == eos:
                    break
                await asyncio.sleep(0.0)
            yield _sse_done(rid, model_name)
        return StreamingResponse(model_stream(), media_type="text/event-stream")

    # Non‑stream: run the simple greedy helper for clarity
    out = greedy_generate(bundle.model, input_ids, max_new_tokens=max_new, eos_token_id=eos)
    new_tokens = out[0, input_ids.size(1):]
    text = tok.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return JSONResponse(_non_stream_payload(rid, model_name, text))

# helpers ------------------------------------------------------------

def _sse_chunk(rid: str, model: str, content: str | None = None, role: str | None = None) -> bytes:
    delta = {}
    if role is not None:
        delta["role"] = role
    if content:
        delta["content"] = content
    obj = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


def _sse_done(rid: str, model: str) -> bytes:
    obj = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (f"data: {json.dumps(obj, ensure_ascii=False)}\n\n" + "data: [DONE]\n\n").encode()


def _non_stream_payload(rid: str, model: str, text: str) -> dict:
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }
```

---

### `tests/test_openai_api.py`

`tests/test_openai_api.py` exercises **myserve** through the real **OpenAI Python SDK**—not just raw HTTP—so we verify end-to-end compatibility. A fixture builds an in-memory stack using `httpx.AsyncClient` with `ASGITransport` (async-only in httpx ≥0.28), then instantiates `AsyncOpenAI` pointed at our FastAPI app (`/v1`). The first test calls `chat.completions.create()` non-streaming and asserts schema fields plus the exact text “Paris” for a deterministic prompt at `temperature=0`. The second test requests `stream=True`, iterates SSE events, collects `delta.content` chunks, and checks the concatenated output equals “Paris”, closing resources cleanly. Together, these tests prove **wire protocol parity** with OpenAI clients for both batch and streaming paths.

```python
import pytest
import httpx
from httpx import ASGITransport
from openai import AsyncOpenAI

from myserve.main import app


@pytest.fixture()
async def openai_client():
    """
    Async OpenAI client that sends requests into the FastAPI app in-memory.
    Works with httpx>=0.28 where ASGITransport is async-only.
    """
    transport = ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")

    client = AsyncOpenAI(
        base_url="http://test/v1",   # include /v1 if your routes live there
        api_key="test-key",
        http_client=http_client,
    )
    try:
        yield client
    finally:
        # Close both the client and transport cleanly
        await http_client.aclose()
        await transport.aclose()


@pytest.mark.asyncio
async def test_chat_completions_basic(openai_client: AsyncOpenAI):
    resp = await openai_client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France? Answer with one word."}],
        temperature=0,
        max_tokens=5,
    )

    assert resp.id
    assert resp.object in {"chat.completion", "chat.completion.chunk"}
    assert resp.choices and resp.choices[0].message
    text = resp.choices[0].message.content
    assert isinstance(text, str) and len(text) > 0
    assert "Paris" == text


@pytest.mark.asyncio
async def test_chat_completions_stream(openai_client: AsyncOpenAI):
    stream = await openai_client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France? Answer with one word."}],
        stream=True,
        temperature=0,
        max_tokens=10,
    )

    chunks = []
    try:
        async for event in stream:
            for choice in event.choices:
                if getattr(choice, "delta", None) and choice.delta.content:
                    chunks.append(choice.delta.content)
    finally:
        # close stream if the SDK exposes aclose (newer versions do)
        close = getattr(stream, "aclose", None)
        if callable(close):
            await close()

    out = "".join(chunks)
    assert out == "Paris"
```

---

## Run it

Use the model name *in the request*. Examples:

**CPU‑quick test (tiny model):**

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "sshleifer/tiny-gpt2",
        "stream": true,
        "messages": [
          {"role": "user", "content": "Write three words about space:"}
        ],
        "max_tokens": 16
      }'
```

**Small real model (GPU recommended):**

```bash
export HUGGINGFACE_TOKEN=***   # if the model needs auth
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "stream": true,
        "messages": [
          {"role": "user", "content": "List two reasons to learn Rust."}
        ],
        "max_tokens": 64
      }'
```

---

We keep this post’s shape deliberately simple and testable. A **registry** (not globals) gives us a keyed cache today and a path to a real multi-model manager tomorrow—think embeddings, MoE, or A/B variants. The **naïve greedy loop** that recomputes attention each step is slow by design, but it makes logits easy to inspect and correctness trivial to verify; a KV cache lands in Post 4. **Streaming stays at the edge**: we emit one token’s text per step so the OpenAI wire format remains stable even as the decoding core evolves. And with **deterministic greedy** (`model.eval()`, no sampling), outputs are reproducible—perfect for parity tests and CI.

The full code for this second milestone lives here: [https://github.com/pbelevich/myserve/tree/ver2](https://github.com/pbelevich/myserve/tree/ver2) — clone it, run the parity and streaming tests, and open issues/PRs as you try different models.

In **Post 3**, we’ll move beyond greedy and implement **sampling**—`temperature`, **top-k**, and **nucleus (top-p)**—so OpenAI-style parameters map cleanly onto our generator. We’ll add **logprobs** (including top-logprobs) and **seed control** for reproducible runs, and tighten the request parsing so client options translate **1:1** into a well-defined generation config. The goal: feature parity with common OpenAI clients while keeping outputs debuggable and deterministic when you want them to be.

