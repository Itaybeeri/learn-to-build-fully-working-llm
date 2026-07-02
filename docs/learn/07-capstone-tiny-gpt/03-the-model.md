# Stage 2 — the model (embeddings → Transformer blocks → logits)

> **Status: complete (7A / PyTorch).** We assemble the actual GPT, and **almost nothing is new** — every
> piece is a Module 4 concept written in PyTorch. The forward path: ① **token embedding** (Module 3 lookup
> table) **+** ② **positional embedding** (Module 4, learned table) → ③ **N Transformer blocks** each doing
> `x = x + attention(norm(x))` then `x = x + feedforward(norm(x))` (Module 4: self-attention Q/K/V, multi-head,
> feed-forward, residuals, layer norm) → ④ **final layer norm** → ⑤ **lm_head** (unembedding) → **logits**
> (B, T, 65). The **one new detail** is the **causal mask**: a GPT predicts the *next* token, so each position
> may attend only to itself and the **past** — we set future scores to −∞ before softmax. Verified: 159,937
> parameters; forward turns `(16, 32)` ids into `(16, 32, 65)` logits; softmax sums to 1; untrained loss ≈
> **4.37 ≈ ln(65)** = uniform random guessing (the number we'll drive down in Stage 3).

## The whole forward path

```
   char ids (B, T)
      │
   ① token embedding    look up each id → a vector            (Module 3: embedding table)
      +
   ② position embedding add "where in the sequence"           (Module 4: positional info, learned)
      │
   ③ N × Transformer block:                                   (Module 4: the block)
        x = x + attention( norm(x) )    mix info across tokens (self-attention Q/K/V, multi-head)
        x = x + feedforward( norm(x) )  think per token        (feed-forward, residuals, layer norm)
      │
   ④ final layer norm
      │
   ⑤ lm_head (unembedding)  project to 65 scores              (→ logits; softmax happens at sampling)
      │
   logits (B, T, 65)
```

`B` = batch size, `T` = sequence length (≤ `BLOCK_SIZE`), `65` = vocab size.

## The one new piece — the causal mask

In Module 4 every token attended to every other token. But a GPT predicts the *next* token, so when deciding
what follows position 5 it must see **only positions 1–5, never ahead** — otherwise it would peek at the answer
it's trying to predict. We enforce this with a **causal mask**: in the `(T, T)` score grid, blank out every
"future" cell (set it to −∞) **before** softmax, so its weight becomes 0.

```
   attention scores, T=4 (✓ allowed, ✗ masked future):
            pos1 pos2 pos3 pos4
     pos1 [  ✓    ✗    ✗    ✗ ]
     pos2 [  ✓    ✓    ✗    ✗ ]
     pos3 [  ✓    ✓    ✓    ✗ ]
     pos4 [  ✓    ✓    ✓    ✓ ]
```

In code: `scores.masked_fill(self.tril[:T,:T] == 0, float("-inf"))`, where `tril` is a lower-triangular matrix
of 1s.

## Where each Module-4 concept lives in the code (`tiny_gpt_pytorch.py`)

| Concept | Class / line |
|---|---|
| Embedding table (ID → vector) | `TinyGPT.token_emb = nn.Embedding(vocab_size, N_EMBD)` |
| Positional info (what + where) | `pos_emb`; `x = tok + pos` |
| Self-attention Q/K/V, scaled, softmax, blend | `Head` |
| Causal mask (only look back) | `masked_fill(tril == 0, -inf)` in `Head.forward` |
| Multi-head (parallel heads + W_O) | `MultiHeadAttention` |
| Feed-forward (Linear → ReLU bend → Linear) | `FeedForward` |
| Residuals + layer norm | `Block.forward`: `x = x + sa(ln1(x))`, `x = x + ff(ln2(x))` |
| Stacking blocks → final norm → logits | `TinyGPT.blocks`, `ln_f`, `lm_head` |

## What we ran (verified)

```
model parameters : 159,937
forward pass     : x (16, 32) -> logits (16, 32, 65)   (B, T, vocab)
softmax at last position sums to 1.0000  (a real probability distribution)
untrained loss   : 4.366   (~ln(65) = 4.174, i.e. random guessing)
```

**Why untrained loss ≈ ln(65):** cross-entropy (Module 2) for a model that spreads probability *uniformly*
over `V` choices is `−ln(1/V) = ln(V)`. With random weights the model knows nothing, so it's near-uniform over
65 chars → loss ≈ ln(65) ≈ 4.17. A perfect sanity check that the wiring is right; Stage 3 makes this number
fall.

## Check yourself

- What does the model output, and what shape? (For each position, **logits** — one score per vocab char;
  shape `(B, T, vocab_size)`.)
- What are the two embeddings added at the start? (**Token** embedding = *what* the char is (Module 3);
  **positional** embedding = *where* it sits (Module 4).)
- What is the causal mask and why is it needed? (It blanks out **future** positions in the attention scores so
  each token attends only to itself + the **past** — otherwise the model would peek at the next token it's
  trying to predict.)
- Why is the untrained loss ≈ ln(65)? (Random weights → near-**uniform** guessing over 65 chars; cross-entropy
  of uniform guessing is `ln(V)`.)

## Questions we worked through

- **Q: What does the causal mask do (mechanically), and why does a GPT need it?** A: got the *why* early —
  without it the model "always knows the answer, nothing changes" (it would peek at the next char and learn
  nothing). Corrected one phrasing: the mask doesn't *remove characters* — it blocks **attention** to certain
  positions. Isolated the mechanics: it forbids attending to **future** positions by setting their scores to
  **−∞** before softmax, so `e^(−∞) = 0` → weight 0 → each token attends only to itself + the past. Full clean
  answer landed: "set future scores to −∞ so they get 0 after softmax, so the GPT must *predict* and can't see
  the answer."

## What's next / depends on

- **Depends on:** [Self-attention (Q/K/V)](../04-the-transformer/02-self-attention-qkv.md),
  [Multi-head attention](../04-the-transformer/03-multi-head-attention.md),
  [Positional information](../04-the-transformer/04-positional-information.md),
  [A transformer block](../04-the-transformer/05-a-transformer-block.md),
  [Embeddings](../03-language-as-data/04-embeddings-words-as-vectors.md),
  [Loss / cross-entropy](../02-neural-networks/03-loss-measuring-wrong.md) (the ln(65) check).
- **Next:** Stage 3 — training (forward → cross-entropy loss → backprop → update; watch the loss fall).
