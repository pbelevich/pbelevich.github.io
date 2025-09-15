---
layout: post
title:  "Inference Server From Scratch - Part 3: Sampling"
date:   2025-09-14 12:00:00 -0400
# categories:
---

In Part 3, **myserve** trades pure greedy for a pluggable **sampler**: temperature scaling, **top-k** and **top-p (nucleus)** filtering, plus **presence/frequency penalties** that mirror OpenAI’s semantics. We add deterministic **seed control** via a device-matched `torch.Generator`, and compute **logprobs**—both the chosen token and the **top-k alternatives**—available in **streaming** and **non-streaming** responses. A tighter request parser maps OpenAI params 1:1 into a runtime config, and we ship **parity tests** against `transformers.generate` for one-token sampling to lock correctness. We’re still recomputing attention each step (no KV yet); this post is about **API parity and correctness**, not speed.

---

## New/changed repo layout

The repo picks up the pieces needed for sampling and logprobs. In **`server/api/openai_types.py`** we extend the request schema with **sampling params**, **logprobs** fields, and a **seed**. A new **`server/core/sampling.py`** implements temperature, top-k/top-p filtering, presence/frequency penalties, and multinomial draws. **`generate.py`** now delegates token selection to this sampler instead of greedy argmax, while **`main.py`** parses OpenAI-style options and returns **per-token logprobs** (when requested) in both streaming and non-streaming paths. Tests add **`test_sampling_parity.py`** to compare against `transformers.generate` and **`test_logprobs_shape.py`** to validate the logprob tensor shapes and fields.

```
myserve/
  server/
    api/
      openai_types.py     # add logprobs fields + seed
    core/
      sampling.py         # new: filters, penalties, sampling
      generate.py         # now calls sampler instead of greedy argmax
    main.py               # parses sampling params + returns logprobs
  tests/
    test_sampling_parity.py
    test_logprobs_shape.py
```

---

## API surface: request/response updates

We extend the OpenAI-style schema to cover **sampling**, **penalties**, and **logprobs** while staying tolerant to unknown fields. The request model adds `temperature`, `top_p`, optional **`top_k`** (not in OpenAI but handy), `n`, `max_tokens`, and a **`seed`** for reproducibility; penalties (`presence_penalty`, `frequency_penalty`) follow OpenAI semantics. If `logprobs` is true, responses include a `logprobs.content` list with one entry per emitted token—each holding the token string, raw bytes, its **logprob**, and **`top_logprobs`** (size controlled by `top_logprobs`, where `0` means only the chosen token). In **streaming**, each SSE `delta` may carry `content` and, when requested, a one-token `logprobs.content` payload; non-streaming returns the full message plus the aggregated logprobs object.

### `server/api/openai_types.py`

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
    # sampling
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None           # OpenAI doesn’t expose top_k, but we support it
    n: Optional[int] = 1
    max_tokens: Optional[int] = 256
    seed: Optional[int] = None            # not in OpenAI; helpful for tests
    # penalties (OpenAI compatible semantics)
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    # logprobs
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0       # 0 → only chosen token
    # streaming
    stream: Optional[bool] = False
    user: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
```

**Response shape (non‑stream):**

```jsonc
{
  "object": "chat.completion",
  "choices": [{
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "logprobs": {                     // included only if request.logprobs==true
    "content": [
      {"token": "Hello", "bytes": [72,101,...], "logprob": -0.21,
       "top_logprobs": [{"token": "Hi", "logprob": -0.35}, ...]}
      // one entry per emitted token
    ]
  }
}
```

**Response shape (streaming chunks):** each SSE `delta` may include `content` and, if requested, a `logprobs.content` array with one token’s info.

---

## Sampler implementation

`server/core/sampling.py` implements a clean, testable sampler around a `SamplerCfg` (temperature, top-p, optional top-k, presence/frequency penalties, and `top_logprobs`). The flow is: apply **penalties in logit space** using the already-generated token history (presence subtracts a constant if a token appeared; frequency subtracts `count × penalty`), scale by **temperature**, then run a combined **top-k / top-p** filter that masks unlikely tokens to `-inf`. We compute both **logprobs** (via `log_softmax`) and **probs**, then draw the next token with `torch.multinomial`, optionally using a device-matched `torch.Generator` for **seeded determinism**. The sampler returns `(next_ids, chosen_logprobs, full_logprobs)`, enabling OpenAI-style per-token logprobs while keeping the selection logic modular. (Note: the penalty implementation counts tokens per batch naïvely for clarity—good for correctness now, with room to optimize later.)

### `server/core/sampling.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

@dataclass
class SamplerCfg:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    top_logprobs: int = 0

@torch.no_grad()
def apply_penalties(
    logits: torch.Tensor,              # [B, V]
    generated_ids: torch.Tensor,       # [B, T] (prefix + already sampled)
    presence_penalty: float,
    frequency_penalty: float,
) -> torch.Tensor:
    if presence_penalty == 0.0 and frequency_penalty == 0.0:
        return logits
    B, V = logits.shape
    # token counts per batch
    # build counts on CPU for simplicity; move back to device
    counts = torch.zeros((B, V), dtype=torch.int32)
    for b in range(B):
        ids = generated_ids[b].tolist()
        for t in ids:
            if 0 <= t < V:
                counts[b, t] += 1
    counts = counts.to(logits.device)
    # presence penalty subtracts a constant if token ever appeared
    presence_mask = (counts > 0).to(logits.dtype)
    logits = logits - presence_penalty * presence_mask
    # frequency penalty subtracts count * penalty
    logits = logits - frequency_penalty * counts.to(logits.dtype)
    return logits

@torch.no_grad()
def top_k_top_p_filter(logits: torch.Tensor, top_k: Optional[int], top_p: float) -> torch.Tensor:
    # logits: [B, V]
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float('-inf')), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        # shift right to always keep the first token above threshold
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        # unsort
        unsorted = torch.full_like(sorted_logits, float('-inf'))
        unsorted.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = unsorted
    return logits

@torch.no_grad()
def sample_next(
    logits: torch.Tensor,              # [B, V] last‑step logits
    cfg: SamplerCfg,
    generated_ids: torch.Tensor,       # [B, T]
    gen: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # penalties first (operate in logits space)
    logits = apply_penalties(logits, generated_ids, cfg.presence_penalty, cfg.frequency_penalty)
    # temperature
    temperature = max(1e-5, float(cfg.temperature))
    logits = logits / temperature
    # filter and normalize
    logits = top_k_top_p_filter(logits, cfg.top_k, cfg.top_p)
    logprobs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(logprobs)
    # sample
    next_ids = torch.multinomial(probs, num_samples=1, generator=gen)  # [B,1]
    chosen_logprobs = logprobs.gather(-1, next_ids)                    # [B,1]
    return next_ids.squeeze(-1), chosen_logprobs.squeeze(-1), logprobs
```

---

## Generation loop: call the sampler

`sample_generate()` replaces greedy argmax with the new sampler: given a model and `[B, T]` `input_ids`, it iteratively runs a forward pass, takes the last-step logits, and calls `sample_next()` with a `SamplerCfg` (temperature, top-k/top-p, presence/frequency penalties) and an optional seeded `torch.Generator` for reproducibility. It appends the sampled IDs to the sequence, optionally collects per-step telemetry—chosen token ID/logprob and the top-`k` alternatives’ `(id, logprob)`—and stops early if **all** sequences hit `eos_token_id`. The function returns the full token tensor and a `per_step` list structured for easy conversion into OpenAI-style logprobs in both streaming and non-streaming responses.

### `server/core/generate.py` (replace greedy path)

```python
from __future__ import annotations
from typing import Optional
import torch
from torch import nn
from .sampling import SamplerCfg, sample_next

@torch.no_grad()
def sample_generate(
    model: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    cfg: SamplerCfg,
    gen: Optional[torch.Generator] = None,
    collect_logprobs: bool = False,
):
    """Token‑by‑token sampling. Returns (all_ids, per_step) where per_step is a list of dicts
    with keys: {"id": int, "logprob": float, "top_logprobs": List[(id, logprob)]} when requested.
    """
    model.eval()
    B = input_ids.size(0)
    out = input_ids
    per_step = []
    for _ in range(max_new_tokens):
        logits = model(out).logits[:, -1, :]  # [B, V]
        next_ids, chosen_logprobs, logprobs = sample_next(logits, cfg, out, gen)
        out = torch.cat([out, next_ids.unsqueeze(1)], dim=1)
        if collect_logprobs:
            k = int(cfg.top_logprobs or 0)
            step = []
            for b in range(B):
                item = {"id": int(next_ids[b]), "logprob": float(chosen_logprobs[b])}
                if k > 0:
                    topv, topi = torch.topk(logprobs[b], k)
                    item["top_logprobs"] = [(int(topi[j]), float(topv[j])) for j in range(topv.numel())]
                step.append(item)
            per_step.append(step)
        if eos_token_id is not None and torch.all(next_ids == eos_token_id):
            break
    return out, per_step
```

---

## Wiring it into the server

This update threads **sampling, logprobs, and seed control** straight into the OpenAI endpoint without changing the client contract. On each `/v1/chat/completions` call, the server builds the prompt, loads a model from the registry, and constructs a `SamplerCfg` from OpenAI-style params (`temperature`, `top_p`, optional `top_k`, presence/frequency penalties, `top_logprobs`). If a `seed` is provided, it initializes a device-matched `torch.Generator` for reproducible draws. In **streaming** mode, the loop samples one token at a time and now attaches optional **per-token logprobs** via an `extra` payload merged into the SSE `delta`; in **non-streaming**, it uses `sample_generate()` to return the full text plus an aggregated `logprobs.content` array when requested. Echo fallback from earlier posts remains intact, but the “real model” path now speaks the full sampling + logprobs dialect end-to-end.

### `server/main.py` (sampling + logprobs + seed)

```python
# ... imports as in Post 2
import torch
from server.core.sampling import SamplerCfg
from server.core.generate import sample_generate

# env defaults (keep prior)

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

    # real model path (after bundle is loaded)
    tok = bundle.tokenizer
    eos = tok.eos_token_id
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.device)

    # sampler config
    cfg = SamplerCfg(
        temperature=float(req.temperature or 1.0),
        top_p=float(req.top_p or 1.0),
        top_k=int(req.top_k) if req.top_k else None,
        presence_penalty=float(req.presence_penalty or 0.0),
        frequency_penalty=float(req.frequency_penalty or 0.0),
        top_logprobs=int(req.top_logprobs or 0),
    )
    max_new = max(1, int(req.max_tokens or 16))

    # seeded generator (device‑specific to avoid CPU/CUDA mismatch)
    gen = None
    if req.seed is not None:
        gen = torch.Generator(device=bundle.device.type)
        gen.manual_seed(int(req.seed))

    if req.stream:
        async def stream() -> AsyncGenerator[bytes, None]:
            yield _sse_chunk(rid, model_name, role="assistant")
            generated = input_ids
            for _ in range(max_new):
                logits = bundle.model(generated).logits[:, -1, :]
                next_ids, chosen_lp, logprobs = sample_next(logits, cfg, generated, gen)
                generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
                piece = tok.decode(next_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                extra = None
                if req.logprobs:
                    k = int(req.top_logprobs or 0)
                    content = [{
                        "token": piece,
                        "bytes": list(piece.encode("utf-8", errors="ignore")),
                        "logprob": float(chosen_lp[0]),
                        "top_logprobs": (
                            [{"token": tok.decode([int(i)], skip_special_tokens=True, clean_up_tokenization_spaces=False),
                               "logprob": float(v)} for v, i in zip(*torch.topk(logprobs[0], k))]
                            if k > 0 else []
                        ),
                    }]
                    extra = {"logprobs": {"content": content}}
                yield _sse_chunk(rid, model_name, content=piece, extra=extra)
                if eos is not None and int(next_ids[0]) == eos:
                    break
                await asyncio.sleep(0.0)
            yield _sse_done(rid, model_name)
        return StreamingResponse(stream(), media_type="text/event-stream")

    # non‑stream
    all_ids, per_step = sample_generate(
        bundle.model, input_ids, max_new_tokens=max_new, eos_token_id=eos,
        cfg=cfg, gen=gen, collect_logprobs=bool(req.logprobs)
    )
    new_ids = all_ids[0, input_ids.size(1):]
    text = tok.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    payload = _non_stream_payload(rid, model_name, text)
    if req.logprobs:
        tokens = []
        for step in per_step:
            item = step[0]
            tstr = tok.decode([item["id"]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            toks = {"token": tstr, "bytes": list(tstr.encode("utf-8", errors="ignore")), "logprob": item["logprob"]}
            if cfg.top_logprobs:
                toks["top_logprobs"] = [
                    {"token": tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False), "logprob": (lp if str(math.fabs(lp)) != "inf" else str(lp))}
                    for (tid, lp) in item.get("top_logprobs", [])
                ]
            tokens.append(toks)
        payload["logprobs"] = {"content": tokens}
    return JSONResponse(payload)
```

> The SSE helper `_sse_chunk(...)` now accepts an `extra` dict that it merges into the `choices[0].delta`.

**Helper change:**

```python
def _sse_chunk(rid: str, model: str, content: str | None = None, role: str | None = None, extra: dict | None = None) -> bytes:
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if extra:
        delta.update(extra)
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
```

---

## Tests

The test suite locks in correctness for sampling and logprobs. **One-token parity** (`tests/test_sampling_parity.py`) compares our `sample_generate()` against `transformers.generate()` on `sshleifer/tiny-gpt2` across temperature/top-p/top-k settings, seeding both paths so they emit the **same next token tensor**—a tight equivalence check on sampling math. **Logprobs shape** (`tests/test_logprobs_shape.py`) drives the FastAPI endpoint in-memory and asserts the non-streaming response includes a `logprobs.content` list with per-token entries containing `token`, `logprob`, and a list of `top_logprobs` when requested. Together, these tests ensure protocol compatibility and sampler fidelity before we chase speed.

### 1) One‑token sampling parity vs HF

`tests/test_sampling_parity.py`

```python
import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from myserve.core.sampling import SamplerCfg
from myserve.core.generate import sample_generate

MODEL = "sshleifer/tiny-gpt2"

@pytest.mark.parametrize("prompt,temperature,top_p,top_k", [
    ("Hello", 1.0, 1.0, None),
    ("Hello", 0.7, 1.0, 50),
    ("Hello", 1.0, 0.9, None),
])
@torch.no_grad()
def test_one_token_sampling_parity(prompt, temperature, top_p, top_k):
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    inp = enc["input_ids"]

    # our sample (1 token)
    cfg = SamplerCfg(temperature=temperature, top_p=top_p, top_k=top_k)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    ours, _ = sample_generate(model, inp, 1, tok.eos_token_id, cfg, gen, collect_logprobs=False)

    # HF sample (1 token)
    set_seed(1234)
    hf = model.generate(**enc, max_new_tokens=1, do_sample=True, temperature=temperature,
                        top_p=top_p, top_k=(top_k or 0), pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    assert torch.equal(ours, hf)
```

### 2) Logprobs shape (non‑stream)

`tests/test_logprobs_shape.py`

```python
from httpx import AsyncClient, ASGITransport
import pytest
from myserve.main import app

@pytest.mark.asyncio
async def test_logprobs_shape():
    body = {
        "model": "sshleifer/tiny-gpt2",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": True,
        "top_logprobs": 3,
        "seed": 42,
        "stream": False,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        data = r.json()
        assert "logprobs" in data and "content" in data["logprobs"]
        assert len(data["logprobs"]["content"]) > 0
        first = data["logprobs"]["content"][0]
        assert "token" in first and "logprob" in first
        assert isinstance(first.get("top_logprobs", []), list)
```

---

## Try it

**Non‑stream with seed + logprobs:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "sshleifer/tiny-gpt2",
        "messages": [{"role": "user", "content": "Count to three:"}],
        "max_tokens": 8,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "seed": 1337,
        "logprobs": true,
        "top_logprobs": 5
      }' | jq .
```

**Streaming with logprobs:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "sshleifer/tiny-gpt2",
        "messages": [{"role": "user", "content": "Say a color:"}],
        "max_tokens": 4,
        "temperature": 1.0,
        "top_p": 0.9,
        "logprobs": true,
        "top_logprobs": 3,
        "seed": 7,
        "stream": true
      }'
```

---

A few design choices keep this step practical and compatible. **Penalties** are implemented in logits space to mirror OpenAI’s public guidance—close in spirit, though exact parity may vary at the margins. **Seed control** uses a per-request `torch.Generator` (CPU or CUDA to match the model device), so concurrent requests don’t collide. For **logprobs**, we return both token **text** and **bytes**; the latter is robust for clients that diff at the byte level across tokenizers/locales. And for **back-compat**, when `logprobs=false` the response shape is unchanged from [Part 2](https://pbelevich.github.io/2025/09/11/Inference_Server_From_Scratch_-_Part_2.html), so existing clients keep working without code changes.

The complete Part 3 code is here: [https://github.com/pbelevich/myserve/tree/ver3](https://github.com/pbelevich/myserve/tree/ver3) — pull it, run the sampling/logprobs tests, and let me know where you want to push the API or performance next.

**Post 4 is all about speed.** We’ll introduce a **KV cache** to avoid recomputing attention, split generation into a clear **prefill → step** pipeline, and add **continuous batching** so new requests can join mid-flight without stalling throughput. With those pieces in place, we’ll start measuring the wins like an adult system: **TTFT** (time-to-first-token), **tokens/sec** (steady-state), and utilization under mixed workloads. This is the turning point where myserve moves from “correct” to **fast**.
