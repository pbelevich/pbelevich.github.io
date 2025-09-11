---
layout: post
title:  "From Zero to Serve — Post 1"
date:   2025-09-10 23:11:14 -0400
# categories:
---
# From Zero to Serve — Post 1

Since I work a lot with real inference servers, I’ve always wanted to know how they work under the hood—so I decided to build my own wheel. This series documents that journey: designing and implementing an LLM inference server from first principles, then hardening it into something you could actually run. The goals are simple but ambitious: OpenAI-compatible APIs, Hugging Face model support, and a clear path from tiny dense models to tensor-parallel giants and eventually Mixture-of-Experts (MoE).

Why build it yourself when great servers already exist? Because nothing clarifies trade-offs like owning the constraints: batching vs. latency, KV-cache layouts, prefill vs. decode scheduling, attention kernels, quantization choices, and the realities of memory bandwidth and interconnects. By the end, you’ll understand not just what to tweak, but why it moves the needle.

I’m calling this server **myserve**—a small, opinionated playground that grows into a real system as the series progresses. You can follow along, file issues, or star the repo here: [https://github.com/pbelevich/myserve](https://github.com/pbelevich/myserve).

This first milestone ships a tiny but real server: a FastAPI `/v1/chat/completions` endpoint that speaks the OpenAI wire format, streams via SSE, and “token-echoes” using a Hugging Face tokenizer—no model weights yet. The point is to lock down the protocol and repo plumbing before touching CUDA: OpenAI-style request/response, correct streaming chunks, and a tokenizer path we can later swap for real models without changing the API. You also get a clean, production-leaning scaffold (tests, config, typing) plus runnable scripts. Requirements are modest—Python 3.10+ and `pipx`/`uv`/`pip`; **no GPU needed**.

---

## Repository skeleton

The repo keeps things small and legible: **`myserve/`** holds the build metadata (`pyproject.toml`) and docs, while **`server/`** contains the FastAPI app in `main.py` plus a neat split between **`api/`** (OpenAI-compatible request models in `openai_types.py`, tolerant to extra fields) and **`core/`** (tokenizer wiring in `tokenizer.py`). `__init__.py` files make everything import-friendly. A lightweight **`tests/`** folder ships with `test_smoke.py` covering end-to-end happy paths for both non-streaming and SSE streaming, so you can validate the wire protocol before plugging in real model weights.

```
myserve/
  pyproject.toml
  README.md
  server/
    __init__.py
    main.py                 # FastAPI app & OpenAI-compatible endpoints
    api/
      __init__.py
      openai_types.py       # Pydantic request models; we ignore unknown fields
    core/
      __init__.py
      tokenizer.py          # HF tokenizer loader + helpers
  tests/
    test_smoke.py           # e2e: non-stream + stream happy paths
```
---
## `pyproject.toml`

`pyproject.toml` wires a clean, reproducible setup: setuptools for builds; project metadata for **myserve** (v0.1.0, Python ≥3.10); core deps for an OpenAI-style server (**FastAPI**, **uvicorn**, **pydantic**) plus tokenizer plumbing (**transformers**, **tokenizers**, **sentencepiece**). Test deps (**httpx**, **pytest**, **pytest-asyncio**) enable easy integration/streaming tests, and pytest config keeps async tests simple and output quiet.

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "myserve"
version = "0.1.0"
description = "OpenAI-compatible My Inference Server"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.30",
  "pydantic>=2.6",
  "transformers>=4.42",
  "tokenizers>=0.15",
  "sentencepiece>=0.1",
  "httpx>=0.27",         # tests
  "pytest>=8.2",         # tests
  "pytest-asyncio>=0.23", # tests async
]

[tool.pytest.ini_options]
addopts = "-q"
asyncio_mode = "auto"
```

---

## `server/api/openai_types.py`

`openai_types.py` defines a minimal, OpenAI-style schema with Pydantic: a strict `Role` literal (`system|user|assistant|tool`), a `ChatMessage` (role + content), and a `ChatCompletionRequest` mirroring the Chat Completions API (model, messages, `stream`, sampling knobs like `max_tokens`, `temperature`, `top_p`, `n`, stop sequences, penalties, and optional `logprobs`). Using `BaseModel` gives validation/serialization for free, while `model_config = ConfigDict(extra="ignore")` keeps the server tolerant to unknown fields—handy for client variations without breaking requests.

```python
from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict

Role = Literal["system", "user", "assistant", "tool"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    user: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
```

---

## `server/core/tokenizer.py`

`tokenizer.py` centralizes prompt shaping and fast tokenizer loading. `get_tokenizer()` uses `transformers.AutoTokenizer` with an `@lru_cache(maxsize=8)` so repeated model names don’t thrash disk/network, and it falls back to `"gpt2"` if a model can’t be fetched. It sets `padding_side="left"` to keep token indices stable for **one-token-at-a-time streaming**. `render_messages()` first prefers a model’s native `chat_template` (so Llama/Qwen prompt formatting stays correct) and returns a ready-to-generate string; if no template exists, it **flattens** `system/user/assistant` messages into a simple newline-joined prompt (ignoring `tool` for post #1) while supporting both Pydantic objects and dicts. This gives us a clean, swappable prompt path now—and a place to evolve formatting, tool calls, and stop-sequence handling in later posts.

```python
from functools import lru_cache
from transformers import AutoTokenizer
from typing import Iterable

@lru_cache(maxsize=8)
def get_tokenizer(model_name: str):
    """Load and cache a fast tokenizer. Defaults to gpt2 if a model is missing."""
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    # ensure consistent behavior while streaming one token at a time
    tok.padding_side = "left"
    return tok

def render_messages(tok: AutoTokenizer, messages: Iterable):
    if tok.chat_template is not None:
        return tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
        )
    else:
        """Flatten chat messages into a single prompt string.
        This is deliberately simple for post #1 and will evolve later.
        """
        parts = []
        for m in messages:
            if m.role == "system" if hasattr(m, "role") else m["role"] == "system":
                parts.append(m.content.strip() if hasattr(m, "content") else m["content"].strip())
            elif m.role == "user" if hasattr(m, "role") else m["role"] == "user":
                parts.append(m.content.strip() if hasattr(m, "content") else m["content"].strip())
            elif m.role == "assistant" if hasattr(m, "role") else m["role"] == "assistant":
                parts.append(m.content.strip() if hasattr(m, "content") else m["content"].strip())
            # tool messages ignored in post #1
        return "\n".join(p for p in parts if p).strip()
```

---

## `server/main.py`

`main.py` wires the whole stub together: a FastAPI app with permissive CORS, a `/healthz` probe, and a POST `/v1/chat/completions` that implements the OpenAI-style interface in both **streaming** and **non-streaming** modes. It builds a prompt from incoming chat messages (via the tokenizer-aware `render_messages`), grabs a cached Hugging Face tokenizer, and performs a **token-echo**: encode the prompt, optionally truncate by `max_tokens` (defaulting to a safe 128 cap), and then either stream **SSE chunks** that match OpenAI’s `chat.completion.chunk` schema (role preamble → token deltas → final stop + `[DONE]`) or return a single JSON completion payload. IDs and timestamps are generated (`chatcmpl_*`, `created`), output text is decoded with `skip_special_tokens`, and responses are sent via `StreamingResponse` or `JSONResponse` for a drop-in, client-compatible experience.

```python
import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from myserve.api.openai_types import ChatCompletionRequest
from myserve.core.tokenizer import get_tokenizer, render_messages

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

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Build the prompt from chat messages
    prompt = render_messages(req.messages)
    tokenizer = get_tokenizer(req.model)

    # Token-echo: turn the *prompt tokens* into the assistant's output tokens
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Respect max_tokens by truncating the echo
    max_toks = max(0, int(req.max_tokens or 0))
    if max_toks:
        output_ids = input_ids[:max_toks]
    else:
        # default: cap to 128 to avoid huge responses if user pasted a novel
        output_ids = input_ids[:128]

    created = int(time.time())
    model_name = req.model
    rid = f"chatcmpl_{uuid.uuid4().hex[:24]}"

    if req.stream:
        async def event_stream() -> AsyncGenerator[bytes, None]:
            # first chunk has the role field
            preamble = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(preamble, ensure_ascii=False)}\n\n".encode()

            # stream token-by-token as delta.content
            for tid in output_ids:
                piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if piece == "":
                    continue
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                # tiny delay to make streaming visible in demos
                await asyncio.sleep(0.0)

            # finalizer chunk
            final = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming: assemble the whole string
    text = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    payload = {
        "id": rid,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }
    return JSONResponse(payload)
```

---

## `tests/test_smoke.py`

`test_smoke.py` validates the wire contract end-to-end without a live server: it mounts the FastAPI app in-process via `ASGITransport` and drives it with `httpx.AsyncClient`. The first test (`test_non_stream_basic`) posts a minimal OpenAI-style body (`model`, `messages`, `stream=False`) and asserts a 200 OK, an `object: "chat.completion"`, and an assistant message with string content. The second test (`test_streaming_basic_sse`) flips `stream=True`, then inspects the raw SSE text to ensure it includes the chunk schema (`"object": "chat.completion.chunk"`), begins with the role preamble, and **terminates with `data: [DONE]`**—catching regressions in streaming format early.

```python
import pytest
from httpx import AsyncClient, ASGITransport
from myserve.main import app

@pytest.mark.asyncio
async def test_non_stream_basic():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        body = {
            "model": "gpt2",
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a test."},
                {"role": "user", "content": "Hello world"},
            ],
        }
        r = await ac.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(data["choices"][0]["message"]["content"], str)

@pytest.mark.asyncio
async def test_streaming_basic_sse():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        body = {
            "model": "gpt2",
            "stream": True,
            "messages": [
                {"role": "user", "content": "stream me"}
            ],
        }
        r = await ac.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        text = r.text
        # Should end with [DONE]
        assert text.strip().endswith("data: [DONE]")
        # First event should include role preamble
        assert '"object": "chat.completion.chunk"' in text
```

---

## Running it locally

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -e .
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl -s http://localhost:8000/healthz
```

Non‑streaming request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "gpt2",
        "stream": false,
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Say hello in five words."}
        ]
      }' | jq .
```

Streaming request (SSE):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "gpt2",
        "stream": true,
        "messages": [
          {"role": "user", "content": "Stream these exact words."}
        ]
      }'
```

Simple Python client for streaming:

```python
import requests
import json

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt2",
        "stream": True,
        "messages": [
            {"role": "user", "content": "This will be token-echoed."}
        ],
    },
    stream=True,
)

for line in resp.iter_lines():
    if not line:
        continue
    if line.startswith(b"data: "):
        payload = line[len(b"data: "):]
        if payload == b"[DONE]":
            break
        obj = json.loads(payload)
        delta = obj["choices"][0]["delta"]
        print(delta.get("content", ""), end="")
print()
```

---

In this stub we prioritize client parity and predictability: we use **SSE** because that’s what OpenAI SDKs speak by default (WebSockets can come later), and we emit a **role preamble** so `delta.role="assistant"` appears in the first chunk as many clients expect. Streaming is **token-by-token**, which can look quirky with BPE (leading spaces, partial words)—acceptable for an echo server, and it will feel natural once a real model produces substring chunks. We **ignore unknown fields** to stay forward-compatible with evolving OpenAI params. Finally, **`max_tokens`** is enforced conservatively: we cap echoes (default 128) to avoid runaway responses until we add proper stopping and safety in later posts.

Next up in **Post #2**, we’ll swap the echo trick for a **real small dense model**—think Llama-3-1B or TinyLlama—and run a naïve forward pass to actually generate tokens (still no KV cache). We’ll add **correctness tests** that compare our logits/next-token picks against Hugging Face’s `generate()` on the same prompts, so we can trust the plumbing before optimizing. Until then, we already have a working **OpenAI-compatible server** that existing clients can call—it just “thinks” by echoing tokens.

All code for this first milestone is available here: [https://github.com/pbelevich/myserve/tree/ver1](https://github.com/pbelevich/myserve/tree/ver1) — feel free to clone, run the smoke tests, and open issues or PRs as you follow along.
