# Module 7 — Capstone: build a tiny GPT

> **Status: in progress.** The finale: assemble everything from Modules 2–6 into a **runnable
> character-level GPT** that trains on a small text and **generates new text**. We build it **twice**, same
> architecture, two depths: **7A — PyTorch** (we write the forward pass; autograd handles backprop; it trains
> fast and scales onto a GPU) and **7B — pure NumPy from scratch** (the same GPT with the forward *and* backward
> hand-written — nothing hidden). Order: **7A first** (quick, testable, and a correctness reference), then
> **7B** (check our hand-derived gradients against 7A). They share **Stage 1 (data & char tokenizer)**.

## What we're building (and what it can/can't do)

A real GPT *architecturally* — char embeddings + positional info → N Transformer blocks (attention,
feed-forward, residuals, layer norm) → logits → softmax → sample — just **small**. Trained on
**TinyShakespeare** (~1 MB), it learns that corpus's structure and **generates new Shakespeare-flavoured text**
(real words, plausible grammar, character names) one character at a time.

It is **not** a general assistant: at this scale it has no world knowledge and can't answer questions — that
capability only emerges at a scale no single machine can reach (Module 5: scale = data + parameters + compute;
Module 6 ⑤: an LLM is a plausible-language generator, not a knowledge oracle). The honest test is: *give it a
prompt, read what it writes, and see it produce text in the style it learned.*

## The plan (stages)

- **Stage 1 — data & character tokenizer** *(shared by 7A & 7B)* — build the 65-char vocab, encode/decode,
  and make (x, y) batches with the sliding window. ✅ done.
- **Stage 2 — the model** — token + positional embeddings → Transformer blocks → final norm → logits.
- **Stage 3 — training** — forward → cross-entropy loss → backprop → update; watch the loss fall.
- **Stage 4 — generation** — feed a prompt, sample the next char (temperature/top-k), append, repeat.
- **Scale-up** — bump size + train on a GPU (any CUDA build) for genuinely readable samples.

## Why two implementations (for the manual)

- **7A (PyTorch)** shows how GPTs are *actually* built and lets the reader see a working, learning model fast.
- **7B (NumPy)** strips away autograd so the reader sees *every gradient* — the backprop from Module 2, scaled
  up to a transformer. Building 7A first gives a known-good reference to validate 7B's hand-written math.

## Setup (already done)

- Python 3.12, `torch 2.12.1+cpu`, `numpy 2.5.0` installed.
- Code is **device-agnostic** (`device = "cuda" if torch.cuda.is_available() else "cpu"`) — runs on CPU now,
  auto-uses a CUDA GPU the moment the CUDA build of PyTorch is installed.
- Dataset: `demo/input.txt` = TinyShakespeare (1,115,394 chars, 65 unique).

## What's next / depends on

- **Pulls together:** tokens & embeddings (M3), the Transformer block — attention, multi-head, positional,
  residuals, layer norm (M4), the training loop + cross-entropy + backprop (M2), inference & sampling (M6 ①).
