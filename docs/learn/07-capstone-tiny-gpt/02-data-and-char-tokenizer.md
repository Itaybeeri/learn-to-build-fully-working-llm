# Stage 1 — data & the character tokenizer (shared by 7A & 7B)

## Why character-level here

Module 3 taught sub-word tokenization (BPE) — the practical choice for big models. For the capstone we pick the
**simplest** scheme instead: **one token per character.** Trade-offs:

- **Pro:** zero setup — no training a tokenizer, no merge rulebook. The vocabulary is literally "the set of
  characters that appear in the file" (here, 65). The model still learns spelling, words, and grammar from
  scratch.
- **Con:** sequences are longer (more tokens per sentence) and the model must learn words letter-by-letter.
  Fine at our scale; it makes the whole pipeline tiny and transparent.

## The three pieces

**1. Vocab + maps.** `chars = sorted(set(text))` gives the 65 unique characters in a stable order. Then two
dictionaries: `stoi` (string→int, e.g. `'a'→39`) and `itos` (int→string). This is exactly the **ID = address**
idea from Module 3 — each character gets an integer that will later index the embedding table.

**2. encode / decode.** `encode(s)` maps each char through `stoi` to a list of ids; `decode(ids)` maps back
through `itos`. Reversible: `decode(encode(s)) == s`.

**3. Batches via the sliding window.** Convert the whole corpus to one long tensor of ids and split 90/10 into
**train** / **validation** (the held-out set from Module 5's overfitting note). `get_batch` then picks
`BATCH_SIZE` random start positions and, for each, slices a chunk of length `BLOCK_SIZE` for **x** and the same
chunk **shifted right by one** for **y**:

```
   block_size = 8
   x = [F, i, r, s, t,  , C, i]
   y = [i, r, s, t,  , C, i, t]    each y[t] = x[t]'s next char  (the free label)
```

`BLOCK_SIZE` is the **context window** (Module 6 ②) — the most characters the model sees at once. One chunk of
length `BLOCK_SIZE` secretly contains `BLOCK_SIZE` training examples (predict char 2 from char 1, char 3 from
chars 1–2, …), exactly the sliding-window efficiency from Module 5.

## What we ran (verified)

```
device           : cpu
corpus characters: 1,115,394
vocab size       : 65
vocabulary       : "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

encode('Hi there!') = [20, 47, 1, 58, 46, 43, 56, 43, 2]
decode(...)       = 'Hi there!'   (round-trip OK: True)

one batch: x shape = (16, 32), y shape = (16, 32)
  x = "to corrupt a man's wife is\nwhen "
  y = "o corrupt a man's wife is\nwhen s"   <- each char is x's next char
```

The `y`-is-`x`-shifted-by-one line is the whole self-supervised setup (Module 5) made concrete: the text labels
itself.

## Check yourself

- What is the vocabulary in a character-level tokenizer? (The set of **unique characters** in the text — here
  65; no training needed.)
- What do `stoi` / `itos` do? (Map char↔integer id — the embedding **address book**; the id will index the
  embedding table.)
- How is a training target `y` made from `x`? (`y` is `x` **shifted right by one**, so each target is the
  **next character** — the sliding-window free label from Module 5.)
- What does `BLOCK_SIZE` correspond to from earlier modules? (The **context window** — the max chars seen at
  once.)

## Questions we worked through

- **Q: Why is `y` set to `x` shifted right by one — what does each `y` value tell the model to predict, and
  which earlier-module idea is this?** A: got the core — char-level tokenization, and *each prediction is the
  next char* (correct, the heart of it). On naming the earlier idea, the *concept* was right ("we have the
  text, we can check the next value, right or wrong" = the label is free) but reached for the wrong terms first
  ("deterministic", then "self-attention"). Clarified the easy-to-confuse pair: **self-attention** (Module 4) =
  the *mechanism* tokens use to look at each other; **self-supervised** (Module 5) = the *training setup* where
  the data labels itself. Full answer: `y` shifted by one makes each target the **next character** →
  **self-supervised next-token prediction**, and the **sliding window** turns one passage into many examples.

## What's next / depends on

- **Depends on:** [Tokens & tokenization](../03-language-as-data/02-tokens-and-tokenization.md),
  [Embeddings](../03-language-as-data/04-embeddings-words-as-vectors.md) (ID = address),
  [Next-token prediction / sliding window](../05-training-llms/01-next-token-prediction.md),
  [Context windows](../06-using-and-aligning-llms/02-context-windows.md) (`BLOCK_SIZE`).
- **Next:** Stage 2 — the model (token + positional embeddings → Transformer blocks → logits).

## Portal companion

The **Module 7 — Tiny GPT** portal tab ([`portal/modules/module7.js`](../portal/modules/module7.js)) has a
**Stage 1 · Tokenizer** sub-tab: type any text and watch it become the exact token ids (the portal hard-codes
the same 65-char vocab, so ids match this Python demo — e.g. `encode("Hi there!") = [20,47,1,58,46,43,56,43,2]`),
plus the encode/decode round-trip and the live **(x, y) shift**. Later stages get their own sub-tabs as we
build them.
