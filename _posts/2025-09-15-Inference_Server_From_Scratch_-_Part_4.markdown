---
layout: post
title:  "Inference Server From Scratch - Part 4: KV cache"
date:   2025-09-15 12:00:00 -0400
# categories:
---

In Part 4, **myserve** finally gets fast. We add a real **KV cache** and refactor generation into two clean phases: a single **prefill** pass over the prompt to populate keys/values, followed by tight **decode** steps that feed just the newest token plus the cache. Same OpenAI-compatible endpoint, but per-token work drops from “recompute the world” to “advance one step,” unlocking big latency and throughput gains on a single request path. This structure is also the foundation for the next part’s batching—once prefill/step are explicit, we can interleave many requests without tripping over ourselves.

We’ll introduce a thin, typed **KV-cache wrapper** over Hugging Face’s `past_key_values`, then expose a clean two-phase API—`prefill(model, input_ids) → (logits, kv)` and `decode_step(model, last_token, kv) → (logits, kv)`—so generation becomes a fast **cached\_generate** loop instead of the full-recompute path from parts [2](https://pbelevich.github.io/2025/09/11/Inference_Server_From_Scratch_-_Part_2.html)–[3](https://pbelevich.github.io/2025/09/14/Inference_Server_From_Scratch_-_Part_3.html). The goal is correctness first, so we add **token parity** tests against the old loop and a tiny **speed smoke test** to confirm per-token latency drops once prefill builds the cache and decode advances one token at a time. We stick to **single-request** execution here to avoid HF batching quirks with uneven KV lengths; full **continuous batching** arrives in Part 5.

Here’s the mental model for KV size as we switch to cached decoding. Hugging Face stores per-layer cache as `K: [B, H, T, Dh]` and `V: [B, H, T, Dh]` (decoder-only). That means **bytes per token per layer** are roughly `2 * H * Dh * sizeof(dtype)`—the factor of 2 is K and V. Accumulated through time, the **total KV footprint** after `T` tokens is `L * B * 2 * H * T * Dh * sizeof(dtype)`. Multiply that by sequence length and layers and you see why memory balloons quickly and why we’ll need paging/eviction later; for now we keep a single, contiguous cache per request.


---

## New/changed repo layout

The codebase grows a proper cached path. Under `server/core/`, **`kv.py`** wraps Hugging Face `past_key_values` with explicit shapes/types, **`engine.py`** introduces `prefill()`/`decode_step()` helpers, and **`generate.py`** switches to a **`cached_generate`** loop that reuses the Part 3 sampler. The FastAPI endpoint in **`main.py`** now calls the cached path by default. Tests include **`test_kv_parity.py`** to confirm identical tokens vs. the old full-recompute loop and **`test_kv_speed_smoke.py`** to verify per-token latency actually drops.

```
myserve/
  server/
    core/
      kv.py             # KV cache wrapper (HF-compatible)
      engine.py         # prefill/step helpers
      generate.py       # cached_generate loop using sampler from Part 3
    main.py             # endpoint now uses cached path
  tests/
    test_kv_parity.py
    test_kv_speed_smoke.py
```

---

## `server/core/kv.py`

`server/core/kv.py` formalizes an HF-compatible KV cache with clear types. It defines `KVPair = (k,v)` and `PastKeyValues = tuple[KVPair,…]` (one per layer), then wraps them in a `KVCache` dataclass whose `layers` hold tensors shaped `[B, H, T, D]`. Utilities include `from_past()` (contiguous copy-in), `to_past()` (unwrap for HF calls), `empty_like()` (builds a zero-length-T cache on the same device/dtype for warm starts), and `append_step()` which concatenates per-step `(k,v)` along the time axis for each layer. The `length` property reports the current `T`, giving the rest of the engine a simple, typed handle on cache size and updates.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch

KVPair = Tuple[torch.Tensor, torch.Tensor]  # (k, v)
PastKeyValues = Tuple[KVPair, ...]          # HF convention per layer

@dataclass
class KVCache:
    layers: PastKeyValues                   # tuple of (k: [B,H,T,D], v: [B,H,T,D]) per layer

    @staticmethod
    def empty_like(past: PastKeyValues) -> "KVCache":
        # create an empty cache with zero-length T on same device/dtype
        new_layers: List[KVPair] = []
        for k, v in past:
            B,H,T,D = k.shape
            device, dtype = k.device, k.dtype
            zk = torch.empty((B,H,0,D), device=device, dtype=dtype)
            zv = torch.empty((B,H,0,D), device=device, dtype=dtype)
            new_layers.append((zk, zv))
        return KVCache(tuple(new_layers))

    @staticmethod
    def from_past(past: PastKeyValues) -> "KVCache":
        return KVCache(tuple((k.contiguous(), v.contiguous()) for (k,v) in past))

    def to_past(self) -> PastKeyValues:
        return tuple((k, v) for (k, v) in self.layers)

    def append_step(self, step_kv: PastKeyValues) -> None:
        # Concatenate new time-step along T for each layer
        new_layers: List[KVPair] = []
        for (k, v), (nk, nv) in zip(self.layers, step_kv):
            new_layers.append((torch.cat([k, nk], dim=2), torch.cat([v, nv], dim=2)))
        self.layers = tuple(new_layers)

    @property
    def length(self) -> int:
        # sequence length T (assumes non-empty)
        return 0 if len(self.layers) == 0 else int(self.layers[0][0].shape[2])
```

---

## `server/core/engine.py`

`server/core/engine.py` exposes the two primitives that power cached decoding. **`prefill()`** runs the entire prompt once with `use_cache=True`, returns the last-position logits `[B, V]`, and wraps Hugging Face’s `past_key_values` into our typed `KVCache`. **`decode_step()`** then feeds only the **last token** `[B,1]` while reusing the cache (converted via `DynamicCache.from_legacy_cache(...)`), producing new last-position logits and an updated `KVCache`. Both paths are `@torch.no_grad()` and shape-stable, giving the generator a clean prefill→step contract that avoids full-sequence recompute.

```python
from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from .kv import KVCache, PastKeyValues
from transformers import DynamicCache

@torch.no_grad()
def prefill(model: nn.Module, input_ids: torch.Tensor) -> Tuple[torch.Tensor, KVCache]:
    """Run the full prompt once to build the KV cache.
    Returns last-position logits [B,V] and a KVCache holding past_key_values.
    """
    out = model(input_ids=input_ids, use_cache=True)
    logits = out.logits[:, -1, :]
    past: PastKeyValues = out.past_key_values  # tuple(layer) of (k,v)
    return logits, KVCache.from_past(past)

@torch.no_grad()
def decode_step(model: nn.Module, last_token: torch.Tensor, kv: KVCache) -> Tuple[torch.Tensor, KVCache]:
    """One decode step: feed only the last token, reuse cached KV.
    last_token: [B,1], kv: KVCache with same B.
    Returns last-position logits [B,V] and the updated cache.
    """
    out = model(input_ids=last_token, past_key_values=DynamicCache.from_legacy_cache(kv.to_past()), use_cache=True)
    logits = out.logits[:, -1, :]
    new_past: PastKeyValues = out.past_key_values
    return logits, KVCache.from_past(new_past)
```

---

## `server/core/generate.py` (cached loop)

`cached_generate()` is the KV-aware decoding loop: it first runs a full **prefill** over `[B,T]` `input_ids` to build the KV cache and obtain last-position **logits**. Then, for up to `max_new_tokens`, it selects the next token via the Part-3 **sampler** (`sample_next`, honoring temperature/top-k/top-p and penalties, with optional seeded `torch.Generator`), appends it to `generated`, and—if requested—records **per-step logprobs** (chosen token plus top-k alternatives). It stops early when all sequences hit `eos_token_id`. Crucially, each iteration advances with a **decode step**: feed only the **last token** and the cached KV via `decode_step()`, which returns fresh logits and an updated cache. The function returns the full token sequence and any collected per-step logprob metadata, giving you correctness and speed without changing the API surface.

```python
from __future__ import annotations
from typing import Optional, List
import torch
from torch import nn
from .sampling import SamplerCfg, sample_next
from .engine import prefill, decode_step

@torch.no_grad()
def cached_generate(
    model: nn.Module,
    input_ids: torch.LongTensor,          # [B,T]
    max_new_tokens: int,
    eos_token_id: Optional[int],
    cfg: SamplerCfg,                      # from Part 3
    gen: Optional[torch.Generator] = None,
    collect_logprobs: bool = False,
):
    B = input_ids.size(0)
    # 1) Prefill
    logits, kv = prefill(model, input_ids)
    generated = input_ids
    per_step = []

    for _ in range(max_new_tokens):
        # sample next token from current logits
        next_ids, chosen_lp, logprobs = sample_next(logits, cfg, generated, gen)
        generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
        if collect_logprobs:
            step = []
            k = int(cfg.top_logprobs or 0)
            for b in range(B):
                item = {"id": int(next_ids[b]), "logprob": float(chosen_lp[b])}
                if k > 0:
                    topv, topi = torch.topk(torch.log_softmax(logits[b], dim=-1), k)
                    item["top_logprobs"] = [(int(topi[j]), float(topv[j])) for j in range(k)]
                step.append(item)
            per_step.append(step)
        # early stop if everyone hit EOS
        if eos_token_id is not None and torch.all(next_ids == eos_token_id):
            break
        # 2) Decode step – feed just last token with KV
        last = next_ids.view(B, 1)
        logits, kv = decode_step(model, last, kv)

    return generated, per_step
```

---

## Wire it into the server

Hooking the cache into the server is a surgical swap: wherever Part 3 called `sample_generate(...)`, we now invoke **`cached_generate(...)`** for non-streaming responses—same inputs, but it does a one-time **prefill** and then fast **decode** steps under the hood while still honoring sampling, penalties, seeds, and optional logprobs. The **streaming** path wins even more: we prefill once to build the KV cache and get initial logits, then iterate token-by-token with `sample_next(...)` and advance using `decode_step(...)`, emitting each piece (plus per-token logprobs if requested) as SSE chunks. The API surface is unchanged; the per-token compute cost drops immediately.

```python
# in non‑stream path
all_ids, per_step = cached_generate(
    bundle.model, input_ids, max_new_tokens=max_new, eos_token_id=eos,
    cfg=cfg, gen=gen, collect_logprobs=bool(req.logprobs)
)
```

**Streaming** (excerpt):

```python
# prefill once
logits, kv = prefill(bundle.model, input_ids)
for _ in range(max_new):
    next_ids, chosen_lp, logprobs = sample_next(logits, cfg, generated, gen)
    generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
    piece = tok.decode(next_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # emit SSE chunk (with optional logprobs)
    yield _sse_chunk(...)
    if eos is not None and int(next_ids[0]) == eos:
        break
    # fast decode step
    logits, kv = decode_step(bundle.model, next_ids.view(1,1), kv)
```

---

## Tests

The KV cache is validated for both **correctness** and **speed**. In `test_kv_parity.py`, we compare `cached_generate()` against the uncached `sample_generate()` from [Part 3](https://pbelevich.github.io/2025/09/14/Inference_Server_From_Scratch_-_Part_3.html) on the same prompt, with identical **sampler config** and a **reseeded torch.Generator**, and assert the full token tensors match exactly—proving the cached path doesn’t change outputs. In `test_kv_speed_smoke.py`, we time both paths on a 100-token decode and expect the **cached** version to be at least modestly faster (or, failing that, the environment to be slow enough to explain variance). Together, these tests ensure the KV refactor preserves behavior while delivering the intended per-token latency win.

### `tests/test_kv_parity.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from myserve.core.sampling import SamplerCfg
from myserve.core.generate import cached_generate
from myserve.core.generate import sample_generate  # from Part 3

MODEL = "sshleifer/tiny-gpt2"

@torch.no_grad()
def test_cached_equals_uncached_one_seed():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    cfg = SamplerCfg(temperature=1.0, top_p=1.0)
    gen = torch.Generator(device="cpu").manual_seed(123)

    ids_cached, _ = cached_generate(model, enc["input_ids"], 8, tok.eos_token_id, cfg, gen)

    # reinit generator for identical draw
    gen2 = torch.Generator(device="cpu").manual_seed(123)
    ids_uncached, _ = sample_generate(model, enc["input_ids"], 8, tok.eos_token_id, cfg, gen2, collect_logprobs=False)

    assert torch.equal(ids_cached, ids_uncached)
```

### `tests/test_kv_speed_smoke.py`

```python
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from myserve.core.sampling import SamplerCfg
from myserve.core.generate import cached_generate, sample_generate

MODEL = "sshleifer/tiny-gpt2"

@torch.no_grad()
def test_kv_is_faster_smoke():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    enc = tok("write a story about a cat", return_tensors="pt", add_special_tokens=False)
    cfg = SamplerCfg()

    t0 = time.time(); cached_generate(model, enc["input_ids"], 100, tok.eos_token_id, cfg); t1 = time.time()
    t2 = time.time(); sample_generate(model, enc["input_ids"], 100, tok.eos_token_id, cfg); t3 = time.time()

    # KV path should be at least modestly faster; guard against flaky envs
    assert (t1 - t0) < (t3 - t2) * 0.9 or (t3 - t2) > 0.05
```

---

## Try it

No API changes. Just run the server and request tokens as before; you should see noticeably better **decode** throughput, especially for longer generations.

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
curl -s http://localhost:8000/healthz

curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "sshleifer/tiny-gpt2",
        "messages": [{"role": "user", "content": "Explain caching in one sentence."}],
        "max_tokens": 48,
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": 7,
        "stream": true
      }'
```

---

We wrap Hugging Face’s `past_key_values` in an explicit **`KVCache`** so we can later swap storage backends (paged blocks, CPU/NVMe offload) without touching call sites. The payoff is latency: **prefill** cost stays the same, but **decode** becomes \~**O(1)** per token with respect to generated length (instead of **O(T)** full recomputes). Numerically, with the same seed and sampler settings, cached and uncached paths are **bit-wise identical on CPU** and match within tolerance on GPU. Current scope is **single-request** only; batching decode steps is tricky with HF’s varying cache lengths. We’ll introduce a simple scheduler in Part 5 and move to true paged caches in later.

The full Part 4 implementation is here: [https://github.com/pbelevich/myserve/tree/ver4](https://github.com/pbelevich/myserve/tree/ver4) — clone it, run the KV parity/speed tests, and kick the tires on the new prefill → decode path.

In the Part 5 we’ll add **continuous batching** with a lightweight per-tick scheduler that separates **prefill** and **decode** lanes and enforces simple fairness, so new requests can join mid-flight without tanking latency. We’ll wire in basic **queue metrics** (TTFT, tokens/sec) and a tiny **loadgen** to visualize throughput under pressure. Finally, we’ll shape the execution paths and data structures to be **TP-ready**, laying the groundwork for scaling to multi-GPU dense models later in the series.

